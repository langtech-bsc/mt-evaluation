"""Score saved MT results with a two-stage GEMBA-ESA LLM judge.

Follows the WMT26 baseline protocol (Lavie et al., 2026, Appendix E):
Stage 1 annotates major/minor error spans, Stage 2 assigns a 0-100 score
conditioned on those spans. By default (stage2_input: raw) Stage 2 receives the
raw Stage 1 answer, as in the original GEMBA-ESA code; stage2_input: parsed gives
it the parsed spans, as described in WMT26 Appendix E.2. The parsed spans, the raw
Stage 1 answers and the lines the parser could not read are saved in the results.
Uses the WMT26 non-reasoning baseline configuration (Section 4.2.1, Appendix E):
greedy decoding in both stages, with 512 (Stage 1) and 64 (Stage 2) output tokens.

Runs in the vLLM (Transformers 5) environment, separately from
neural_scoring.score, and writes its scores into the same results JSON:

    python -m neural_scoring.llm_judge --input results/my_model/*.json
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re

from . import llm_judge_prompts as P
from .score import atomic_write, extract_tasks, print_scores

STAGE1_MAX_TOKENS = 512
STAGE2_MAX_TOKENS = 64

# ---------------------------------------------------------------- parsing

# The Stage 1 answer is free text (WMT26 Figure 7): a "Major:" and a "Minor:" section with one
# 'category/subcategory - "span"' line per error, or "no-error".
# Severity header, optionally followed by an error on the same line,
# e.g. "Major:", "**Minor:**", "minor: fluency/grammar - "x"".
HEADER = re.compile(r'^\W*(critical|major|minor)\W*:\s*(.*)$', re.I)
# Opening quote -> accepted closing quotes: ASCII, typographic, German, French and CJK quotes.
QUOTE_PAIRS = {
    '"': '"”“', '“': '”"', '”': '”"', '„': '“”"',
    "'": "'’", '‘': '’\'', '’': '’\'', '`': '`',
    '«': '»', '»': '«', '「': '」', '『': '』', '＂': '＂',
}
OPENERS = ''.join(QUOTE_PAIRS)
# Error line head: optional bullet, category[/subcategory], separator (-, en/em dash or colon), opening quote.
LINE_HEAD = re.compile(r'^\s*(?:[-*•]|\d+[.)])?\s*([A-Za-z][\w \-]*(?:/[\w \-]+)?)'
                       r'\s*(?:-|–|—|:)\s*(?=[' + re.escape(OPENERS) + '])')
# A line that starts like an error annotation ("category/subcategory" or a top-level category of the prompt);
# used only to count lines the parser could not read (free comments such as "Note: ..." are not counted).
LOOKS_LIKE_ERROR = re.compile(r'^\W*(?:[A-Za-z][\w \-]*/[A-Za-z]|(?:accuracy|fluency|style|terminology|'
                              r'non-translation|other)\b)', re.I)


def _quoted_span(rest):
    """Span inside the quotes at the start of rest, or None.

    If the line ends with a closing quote, the span runs to that LAST quote (inner quotes are kept, as before).
    Otherwise (text after the span, e.g. '"选举" (selections)') it ends at the first closing quote that is
    followed by a non-alphanumeric character or the end of the line, so the apostrophe in "It's" does not end it.
    """
    closers, body = QUOTE_PAIRS[rest[0]], rest[1:]
    stripped = body.rstrip()
    if stripped and stripped[-1] in closers:
        return stripped[:-1]
    for i, ch in enumerate(body):
        if ch in closers and (i + 1 == len(body) or not body[i + 1].isalnum()):
            return body[:i]
    return None


def parse_stage1_lines(text, hyp):
    """Parse Stage 1 output. Returns (errors, unparsed): error dicts with character offsets in hyp, and the
    lines that look like an error annotation but could not be parsed (e.g. an error before any severity header).

    Each span is located at its first occurrence in hyp; start/end are None
    when it is not found (always the case for omissions, which quote the source).
    """
    errors, unparsed, severity = [], [], None
    for raw_line in (text or '').splitlines():
        line = raw_line.strip().replace('**', '').replace('__', '')  # Markdown bold
        header = HEADER.match(line)
        if header:
            # The prompt only allows major/minor; treat a stray "critical" as major.
            severity = 'minor' if header.group(1).lower() == 'minor' else 'major'
            line = header.group(2).strip()
        if not line:
            continue
        head = LINE_HEAD.match(line)
        span = _quoted_span(line[head.end():]) if head else None
        if span is None:
            # "no-error" markers are only honoured on lines without a quoted span, so an explanation that
            # mentions "no error" no longer drops the error on that line.
            lowered = line.lower()
            if 'no-error' in lowered or 'no error' in lowered:
                continue
            if LOOKS_LIKE_ERROR.search(line):
                unparsed.append(raw_line.strip())
            continue
        if severity is None:
            unparsed.append(raw_line.strip())
            continue
        category = head.group(1).strip().lower()
        found = hyp.find(span) if span else -1
        start = None if found == -1 else found
        errors.append({
            'start': start,
            'end': None if start is None else start + len(span),
            'severity': severity,
            'category': category,
            'text': span,
            'omission': 'omission' in category,
        })
    # Remove repeated spans as in WMT26: one error per (start, end), keeping major on conflicts.
    unique = {}
    for error in errors:
        key = ((error['start'], error['end']) if error['start'] is not None
               else ('unlocated', error['category'], error['text']))
        if key not in unique or error['severity'] == 'major':
            unique[key] = error
    return list(unique.values()), unparsed


def parse_stage1(text, hyp):
    """Parse Stage 1 output into error dicts (see parse_stage1_lines)."""
    return parse_stage1_lines(text, hyp)[0]


def format_spans(errors):
    """Render parsed errors in the Stage 1 format, for the Stage 2 prompt."""
    lines = []
    for severity in ('major', 'minor'):
        lines.append(severity.capitalize() + ':')
        items = [f'{e["category"]} - "{e["text"]}"' for e in errors if e['severity'] == severity]
        lines.extend(items or ['no-error'])
    return '\n'.join(lines)


def span_score(errors):
    """WMT26 Task 1 -> Task 2 conversion: 100 - 10 per major - 2.5 per minor, floored at 0."""
    major = sum(e['severity'] == 'major' for e in errors)
    return max(0.0, 100.0 - 10.0 * major - 2.5 * (len(errors) - major))


def parse_stage2(text):
    """Extract the 0-100 score from Stage 2 output, or None if there is none."""
    text = text or ''
    match = (re.search(r'"?score"?\s*[:=]\s*(\d+(?:\.\d+)?)', text, re.I)
             or re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*', text))
    return None if match is None else min(100.0, max(0.0, float(match.group(1))))

# -------------------------------------------------------------- languages

# Languages supported by SalamandraTA (40 languages + 3 varieties), keyed by the
# short codes used in task names ("<src>_<tgt>_<dataset>"). Extend as needed.
LANGUAGE_NAMES = {
    # European languages
    'arg': 'Aragonese', 'ast': 'Asturian', 'eu': 'Basque', 'bg': 'Bulgarian', 'ca': 'Catalan',
    'hr': 'Croatian', 'cs': 'Czech', 'da': 'Danish', 'nl': 'Dutch', 'en': 'English',
    'et': 'Estonian', 'fi': 'Finnish', 'fr': 'French', 'gl': 'Galician', 'de': 'German',
    'el': 'Greek', 'hu': 'Hungarian', 'is': 'Icelandic', 'ga': 'Irish', 'it': 'Italian', 'lv': 'Latvian',
    'lt': 'Lithuanian', 'mt': 'Maltese', 'nb': 'Norwegian Bokmål', 'no': 'Norwegian Bokmål',
    'oc': 'Occitan', 'pl': 'Polish', 'pt': 'Portuguese', 'ro': 'Romanian', 'ru': 'Russian',
    'sr': 'Serbian', 'sk': 'Slovak', 'sl': 'Slovenian', 'es': 'Spanish', 'sv': 'Swedish',
    'uk': 'Ukrainian', 'cy': 'Welsh',
    # Varieties
    'vl': 'Valencian', 'nn': 'Norwegian Nynorsk', 'arn': 'Aranese',
    # Non-European languages
    'ar': 'Arabic', 'ja': 'Japanese', 'hi': 'Hindi', 'ko': 'Korean', 'zh': 'Simplified Chinese',
}

def resolve_languages(task, cfg):
    """Return (source, target) language names from --src-lang/--tgt-lang or the task name."""
    if cfg.get('src_lang') and cfg.get('tgt_lang'):
        return cfg['src_lang'], cfg['tgt_lang']
    src, tgt = (task.split('_') + ['', ''])[:2]
    if src not in LANGUAGE_NAMES or tgt not in LANGUAGE_NAMES:
        raise ValueError(f'{task}: unknown language codes ({src!r}, {tgt!r}); '
                         'add them to LANGUAGE_NAMES or pass --src-lang and --tgt-lang')
    return LANGUAGE_NAMES[src], LANGUAGE_NAMES[tgt]
# ---------------------------------------------------------------- backend


class VLLMBackend:
    """Offline vLLM inference; the whole list of conversations is batched by vLLM."""

    def __init__(self, cfg):
        from vllm import LLM
        # quantization (config, optional): e.g. "fp8" = vLLM online fp8 weights; gemma-4-31B-it then fits on one
        # 64 GB GPU (tensor_parallel_size: 1). Default: none (bf16 weights).
        extra = {'quantization': cfg['quantization']} if cfg.get('quantization') else {}
        self.llm = LLM(model=cfg['model'],
                       tensor_parallel_size=cfg.get('tensor_parallel_size', 1),
                       max_model_len=cfg.get('max_model_len', 16384),
                       disable_custom_all_reduce=True,  # 2026-11-24: NCCL all-reduce for TP>1 (custom all-reduce can hang on MN5)
                       seed=0, **extra)

    def chat(self, conversations, max_tokens):
        from vllm import SamplingParams
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=0)
        outputs = self.llm.chat(conversations, params, use_tqdm=True)
        # finish_reason 'length' = the answer hit max_tokens (truncated); read by judge_task.
        self.last_finish_reasons = [o.outputs[0].finish_reason for o in outputs]
        return [o.outputs[0].text for o in outputs]

# ---------------------------------------------------------------- scoring


def judge_task(backend, cfg, sources, targets, translations, src_name, tgt_name):
    """Run both stages on one task. Returns a dict with scores, span_scores, errors, fallbacks,
    stage1_raw, stage1_unparsed and stage1_truncated.

    stage2_input (config): 'raw' (default; as in the original GEMBA-ESA code) puts the Stage 1 answer into the
    Stage 2 prompt verbatim, so Stage 2 does not depend on the parser; 'parsed' (WMT26 Appendix E.2) puts the
    parsed spans re-rendered by format_spans. The parsed spans are stored in both cases.
    """
    refs = targets if cfg.get('use_reference', False) else [None] * len(translations)
    extra = {'ref_type': cfg.get('reference_type', 'human'),
             'domain_req': cfg.get('domain_requirements', '')}
    stage2_input = cfg.get('stage2_input', 'raw')
    if stage2_input not in ('raw', 'parsed'):
        raise ValueError(f"stage2_input must be 'raw' or 'parsed', got {stage2_input!r}")

    stage1 = [P.stage1_messages(s, h, r, src_name, tgt_name, **extra)
              for s, h, r in zip(sources, translations, refs)]
    outputs1 = backend.chat(stage1, STAGE1_MAX_TOKENS)
    truncated1 = sum(f == 'length' for f in getattr(backend, 'last_finish_reasons', []))
    parsed1 = [parse_stage1_lines(text, hyp) for text, hyp in zip(outputs1, translations)]
    errors = [e for e, _ in parsed1]
    unparsed = [u for _, u in parsed1]

    spans_in = ([(text or '').strip() or format_spans([]) for text in outputs1] if stage2_input == 'raw'
                else [format_spans(e) for e in errors])
    stage2 = [P.stage2_messages(s, h, r, src_name, tgt_name, spans, **extra)
              for s, h, r, spans in zip(sources, translations, refs, spans_in)]
    parsed = [parse_stage2(text) for text in backend.chat(stage2, STAGE2_MAX_TOKENS)]

    # Results are written with allow_nan=False, so an unparsable Stage 2 answer
    # falls back to the span-converted score and is counted.
    span_scores = [span_score(e) for e in errors]
    scores = [p if p is not None else s for p, s in zip(parsed, span_scores)]
    return dict(scores=scores, span_scores=span_scores, errors=errors,
                fallbacks=sum(p is None for p in parsed), stage1_raw=list(outputs1),
                stage1_unparsed=unparsed, stage1_truncated=truncated1)


def judge_payload(payload, cfg, backend):
    """Return a copy of payload with this judge's scores added to every MT task."""
    output, name = copy.deepcopy(payload), cfg['name']
    for task, sources, targets, translations in extract_tasks(payload):
        result = output['results'][task]
        # Drop results of a previous run of the same judge.
        for key in [k for k in result if k.split(',')[0] == name or k.startswith(name + '_')]:
            del result[key]
        src_name, tgt_name = resolve_languages(task, cfg)
        r = judge_task(backend, cfg, sources, targets, translations, src_name, tgt_name)
        scores, span_scores = r['scores'], r['span_scores']
        result[f'{name},none'] = sum(scores) / len(scores)
        result[f'{name}_segments,none'] = scores
        result[f'{name}_span_score,none'] = sum(span_scores) / len(span_scores)
        result[f'{name}_span_score_segments,none'] = span_scores
        result[f'{name}_error_spans,none'] = r['errors']
        result[f'{name}_fallbacks,none'] = r['fallbacks']
        # For inspection: the raw Stage 1 answers, the lines the parser could not read, truncated answers.
        result[f'{name}_stage1_raw_segments,none'] = r['stage1_raw']
        result[f'{name}_stage1_unparsed_segments,none'] = r['stage1_unparsed']
        result[f'{name}_stage1_unparsed_lines,none'] = sum(len(u) for u in r['stage1_unparsed'])
        result[f'{name}_stage1_truncated,none'] = r['stage1_truncated']
        n_unparsed = result[f'{name}_stage1_unparsed_lines,none']
        if n_unparsed or r['stage1_truncated'] or r['fallbacks']:
            print(f'WARNING {task}: {n_unparsed} unparsed Stage 1 line(s), {r["stage1_truncated"]} truncated '
                  f'Stage 1 answer(s), {r["fallbacks"]} Stage 2 fallback(s); see {name}_stage1_* in the results')
        better = output.setdefault('higher_is_better', {}).setdefault(task, {})
        better[name] = better[f'{name}_span_score'] = True
        # Keep text arrays after all metric scores, as neural_scoring.score does.
        for key in [k for k in result if k.split(',')[0] in (
                'sources', 'sources_stderr', 'targets', 'targets_stderr',
                'translations', 'translations_stderr')]:
            result[key] = result.pop(key)
    prompts = P.SYSTEM + P.STAGE1 + P.STAGE2
    output.setdefault('llm_judge', {})[name] = dict(
        cfg, prompt_sha256=hashlib.sha256(prompts.encode()).hexdigest())
    return output

# -------------------------------------------------------------------- CLI


def apply_precision(cfg):
    """Normalise the judge precision and add it to the result name, so bf16 and fp8 scores never share a name.
    name: auto (default) builds the name from the config (auto_name); any other name is kept as given.

    quantization: none/null/"bf16" -> bf16 weights (name suffix _bf16); "fp8" -> vLLM fp8 weights (suffix _fp8).
    gemma-4-31B-it needs tensor_parallel_size 2 in bf16 and fits on 1 GPU in fp8.
    """
    q = str(cfg.get('quantization') or 'none').lower()
    if q in ('none', 'null', 'bf16'):
        cfg['quantization'], suffix = None, 'bf16'
    elif q == 'fp8':
        cfg['quantization'], suffix = 'fp8', 'fp8'
    else:
        raise ValueError(f"quantization must be none (bf16) or fp8, got {cfg.get('quantization')!r}")
    cfg['precision'] = suffix
    name = str(cfg.get('name') or 'auto')
    if name == 'auto':
        name = auto_name(cfg)
    if not name.endswith('_' + suffix):
        name = f"{name}_{suffix}"
    cfg['name'] = name
    return cfg


def auto_name(cfg):
    """Result name built from the settings that change the score (precision is added by apply_precision):
    gemba_esa_<model>_<noref | ref[_<reference_type>]>_<stage2_input>[_dom<hash>],
    e.g. gemma-4-31B-it, no reference, raw -> gemba_esa_gemma_4_31b_noref_raw."""
    model = Path(str(cfg['model']).rstrip('/')).name.lower()
    model = re.sub(r'-(it|instruct|chat)$', '', model)
    model = '_'.join(re.findall(r'[a-z0-9]+', model))
    if cfg.get('use_reference', False):
        ref_type = cfg.get('reference_type', 'human')
        ref = 'ref' if ref_type == 'human' else f'ref_{ref_type}'
    else:
        ref = 'noref'
    name = f"gemba_esa_{model}_{ref}_{cfg.get('stage2_input', 'raw')}"
    domain = (cfg.get('domain_requirements') or '').strip()
    if domain:  # different prompt -> different name; the text itself is in the results metadata
        name += '_dom' + hashlib.sha1(domain.encode()).hexdigest()[:6]
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', nargs='+', required=True, type=Path,
                        help='results JSON files; the judge model is loaded once for all of them')
    parser.add_argument('--config', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'lm_eval/extra_metrics/mt_metrics_config.yaml')
    parser.add_argument('--src-lang', help='source language name, overrides inference')
    parser.add_argument('--tgt-lang', help='target language name, overrides inference')
    parser.add_argument('--force', action='store_true', help='rescore files that already have this judge')
    parser.add_argument('--dry-run', type=int, default=0, metavar='N',
                        help='print the first N Stage 1 prompts of the first file and exit')
    args = parser.parse_args()

    import yaml
    cfg = yaml.safe_load(args.config.read_text())['mt_metrics']['llm_judge']
    cfg['model'] = os.path.expandvars(cfg['model'])
    cfg.setdefault('stage2_input', 'raw')  # recorded in the results metadata (llm_judge.<name>)
    apply_precision(cfg)
    if args.src_lang and args.tgt_lang:
        cfg.update(src_lang=args.src_lang, tgt_lang=args.tgt_lang)

    if args.dry_run:
        payload = json.loads(args.input[0].read_text())
        task, sources, targets, translations = extract_tasks(payload)[0]
        src_name, tgt_name = resolve_languages(task, cfg)
        for i in range(min(args.dry_run, len(sources))):
            ref = targets[i] if cfg.get('use_reference', False) else None
            messages = P.stage1_messages(sources[i], translations[i], ref, src_name, tgt_name,
                                         cfg.get('reference_type', 'human'), cfg.get('domain_requirements', ''))
            print(f'===== {task} #{i}')
            for message in messages:
                print(f'[{message["role"]}]\n{message["content"]}\n')
        return

    backend = None
    for path in args.input:
        raw = path.read_bytes()
        payload = json.loads(raw)
        tasks = [task for task, *_ in extract_tasks(payload)]
        if not args.force and all(f'{cfg["name"]},none' in payload['results'][t] for t in tasks):
            print(f'Already scored with {cfg["name"]}, skipping: {path}')
            continue
        if backend is None:  # load the model only when there is something to score
            backend = VLLMBackend(cfg)
        result = judge_payload(payload, cfg, backend)
        try:
            result['llm_judge'][cfg['name']]['vllm_version'] = importlib.metadata.version('vllm')
        except importlib.metadata.PackageNotFoundError:
            pass
        if path.read_bytes() != raw:
            raise RuntimeError(f'{path} changed during judging; refusing to overwrite it')
        atomic_write(path, result)
        print_scores(result, tasks)
        print(f'LLM judge scores saved to {path}')


if __name__ == '__main__':
    main()