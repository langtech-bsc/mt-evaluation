"""Prompt templates for the two-stage GEMBA-ESA LLM judge.

Verbatim from Appendix E of Lavie et al. (2026), "Findings of the WMT26
Shared Task on Automated Translation Quality Evaluation Systems"
(Figures 7 and 8, Tables 63 and 64), which builds on GEMBA
(Kocmi and Federmann, 2023). Typographic quotes and backticks from the
PDF are restored to ASCII; literal braces are escaped as {{ }}.

  Stage 1: list error spans with a category and a major/minor severity.
  Stage 2: given the Stage 1 annotations, assign a 0-100 quality score.
"""

# Shared system prompt for both stages.
SYSTEM = "Your task is to identify machine translation errors and assess the quality of the translation."

# Reference block headers, by reference type (Table 64).
REF_LABELS = {
    'human': 'Human reference translation',
    'postedit': 'Post-edited reference translation',
    'pseudo': 'High-quality machine translation reference',
}

# Domain requirement text appended to both stages (Table 63).
# FLORES-style data has no WMT26 domain, so 'general' is the default.
DOMAIN_REQUIREMENTS = {
    'news': 'The source segment is from a news article. The translation should use a formal register consistent with journalistic standards and preserve the source HTML formatting.',
    'speech': "The source segment is a transcript of spoken content from a video. The translation should preserve the speaker's flow and colloquial style. It should omit non-linguistic sounds, such as laughter, groans, and hesitation sounds, while retaining interjections. Interrupted words should be completed when they can be inferred from context; otherwise, they should be omitted. Foreign words should remain unchanged. Each sentence should be placed on a separate line.",
    'social': 'The source segment is user-generated content from a social media platform. Source spelling mistakes should not be reproduced. Meaningful expressiveness, such as capitalization or elongation, should be reproduced naturally in the target language. URLs and user handles should be copied unchanged, while hashtags should be translated when appropriate. Source punctuation should be followed as closely as possible, with additional punctuation only when needed to prevent serious loss of comprehension. The translation should use an informal style, like close friends talking, even if this changes the original tone, and preserve the source HTML formatting.',
    'software': 'The source segment contains software data from a JSON. Only JSON content or values should be translated; keys and placeholders should be copied unchanged. The translation should contain only valid JSON content matching the input format.',
    'edu': 'The source segment consists of biology, chemistry, and geography exercises from an educational web portal for children aged 9-16. The translation should be suitable for this educational context and age range, and preserve the source HTML formatting.',
    'general': 'The translation should be accurate and fluent.',
    'biomed': 'The source segment consists of biomedical text. The translation should use suitable biomedical terminology and preserve the source formatting.',
}
DOMAIN_REQUIREMENTS['factchecking'] = DOMAIN_REQUIREMENTS['news']

# Stage 1 user turn (Figure 7). {ref_block} is empty without a reference;
# {inputs_desc} is the wording that alternates with reference use.
STAGE1 = """{src_name} source:
```{src_text}```
{tgt_name} translation:
```{hyp_text}```
{ref_block}Based on the source segment{inputs_desc} surrounded by triple backticks, identify error types in the translation and classify them. The categories of errors are: accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, punctuation, register, spelling), style (awkward), terminology (inappropriate for context, inconsistent use), non-translation, other, or no-error.
Each error is classified as one of two categories: major or minor. Major errors disrupt the flow and make the understandability of the text difficult or impossible. Minor errors are errors that do not disrupt the flow significantly, and what the text is trying to say is still understandable.
Return only the annotations in this format:
Major:
category/subcategory - "error span"
Minor:
category/subcategory - "error span"
Use one error per line and write no-error when a section is empty. Quote spans from the translation; for omissions, quote the omitted source span.
Domain requirements: {domain_req}"""

# Stage 2 user turn (Figure 8). {error_spans} is the parsed Stage 1 output.
STAGE2 = """Given the translation from {src_name} to {tgt_name} and the annotated error spans, assign a score on a continuous scale from 0 to 100. The scale has the following reference points: 0="No meaning preserved", 33="Some meaning preserved", 66="Most meaning preserved and few grammar mistakes", up to 100="Perfect meaning and grammar".
Domain requirements: {domain_req}
Score the following translation:
{src_name} source:
```{src_text}```
{tgt_name} translation:
```{hyp_text}```
{ref_block}Annotated error spans:
```{error_spans}```
Respond with ONLY a valid JSON object and nothing else: {{"score": N}}
where N is an integer from 0 to 100."""


def _fields(src, hyp, ref, src_name, tgt_name, ref_type, domain_req):
    """Placeholder values shared by both stages. ref=None means reference-free."""
    if ref is not None and ref_type not in REF_LABELS:
        raise ValueError(f'Unknown reference_type {ref_type!r}; expected one of {sorted(REF_LABELS)}')
    return {
        'src_name': src_name,
        'tgt_name': tgt_name,
        'src_text': src,
        'hyp_text': hyp,
        'ref_block': f'{REF_LABELS[ref_type]}:\n```{ref}```\n' if ref is not None else '',
        'domain_req': domain_req or DOMAIN_REQUIREMENTS['general'],
    }


def stage1_messages(src, hyp, ref, src_name, tgt_name, ref_type='human', domain_req=''):
    """Chat messages asking for error-span annotations (Task 1)."""
    fields = _fields(src, hyp, ref, src_name, tgt_name, ref_type, domain_req)
    fields['inputs_desc'] = (', machine translation, and reference translation'
                             if ref is not None else ' and machine translation')
    return [
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user', 'content': STAGE1.format(**fields)},
    ]


def stage2_messages(src, hyp, ref, src_name, tgt_name, error_spans, ref_type='human', domain_req=''):
    """Chat messages asking for a 0-100 score conditioned on the Stage 1 errors (Task 2)."""
    fields = _fields(src, hyp, ref, src_name, tgt_name, ref_type, domain_req)
    return [
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user', 'content': STAGE2.format(error_spans=error_spans, **fields)},
    ]