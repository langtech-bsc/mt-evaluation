"""Score saved MT results without importing lm_eval or regenerating translations."""
import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import tempfile

METRICS = {
    'comet': ('comet', 'BaseCOMET'),
    'comet_kiwi': ('comet_kiwi', 'COMETKiwi'),
    'bleurt': ('bleurt', 'BLEURT'),
    'xcomet': ('xcomet', 'XCOMET'),
    'xcomet_qe': ('xcomet', 'XCOMET_QE'),
    'metricx': ('metricx', 'RefMetricX'),
    'metricx_qe': ('metricx', 'QEMetricX'),
}


def extract_tasks(payload):
    tasks = []
    for task, result in payload.get('results', {}).items():
        keys = ['sources,none', 'targets,none', 'translations,none']
        present = [key in result for key in keys]
        if not any(present):  # Group summaries do not contain segment data.
            continue
        if not all(present):
            raise ValueError(f'{task}: incomplete source/reference/translation arrays')
        arrays = [result[key] for key in keys]
        if not all(isinstance(a, list) and a and all(isinstance(s, str) for s in a) for a in arrays):
            raise ValueError(f'{task}: expected nonempty lists of strings')
        if len({len(a) for a in arrays}) != 1:
            raise ValueError(f'{task}: source/reference/translation lengths differ')
        tasks.append((task, *arrays))
    if not tasks:
        raise ValueError('No MT segment arrays found in results')
    return tasks


def compute(name, config, sources, targets, translations):
    module, cls = METRICS[name]
    factory = getattr(importlib.import_module(f'neural_scoring.legacy_metrics.{module}.metric'), cls)
    # checkpoint/tokenizer paths may use ${MT_MODELS_DIR} so no cluster-specific
    # absolute path is committed; expand it from the environment here.
    checkpoint = os.path.expandvars(config['checkpoint'])
    if name.startswith('metricx'):
        model = factory(os.path.expandvars(config['tokenizer']), checkpoint)
        return model.evaluate(sources=sources, hypotheses=translations, references=targets)
    model = factory(checkpoint)
    batch = config.get('batch_size', 8)
    if name == 'bleurt':
        return model.evaluate(translations, targets, batch)
    if name == 'comet_kiwi':
        return model.evaluate(translations, sources, batch)
    return model.evaluate(translations, [] if name == 'xcomet_qe' else targets, sources, batch)


def score_payload(payload, configs, scorer=compute):
    tasks = extract_tasks(payload)
    enabled = [name for name in METRICS if configs.get(name, {}).get('compute', False)]
    if not enabled:
        raise ValueError('No neural metrics enabled in configuration')
    output = copy.deepcopy(payload)
    output['neural_scoring'] = {
        'status': 'complete',
        'metrics_config': {n: configs[n] for n in enabled},
        'group_neural_scores_aggregated': False,
    }
    # Old neural fields must not survive a new scoring configuration.
    def is_neural(key):
        base = key.split(',')[0]
        return any(base == n or base.startswith(n + '_') for n in METRICS)
    for section in ('results', 'groups', 'higher_is_better'):
        for values in output.get(section, {}).values():
            if isinstance(values, dict):
                for key in list(values):
                    if is_neural(key):
                        del values[key]
    for task, sources, targets, translations in tasks:
        result = output['results'][task]
        for name in enabled:
            prediction = scorer(name, configs[name], sources, targets, translations)
            scores = [float(x) for x in prediction['segments_scores']]
            system = float(prediction['system_score'])
            if len(scores) != len(sources) or not all(math.isfinite(x) for x in scores + [system]):
                raise ValueError(f'{task}/{name}: invalid or misaligned metric scores')
            output.setdefault('higher_is_better', {}).setdefault(task, {})[name] = not name.startswith('metricx')
            result[f'{name},none'] = system
            result[f'{name}_segments,none'] = scores
            if 'error_spans' in prediction:
                result[f'{name}_error_spans,none'] = prediction['error_spans']
            # Release each model before loading the next one.
            import gc
            gc.collect()
            if scorer is compute:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    # Keep text arrays (and their stderr fields) after all metric scores.
    for task, *_ in tasks:
        result = output['results'][task]
        text_keys = [key for key in result if key.split(',')[0] in (
            'sources', 'sources_stderr', 'targets', 'targets_stderr',
            'translations', 'translations_stderr')]
        for key in text_keys:
            result[key] = result.pop(key)
    return output


def atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'lm_eval/extra_metrics/mt_metrics_config.yaml')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    raw = args.input.read_bytes()
    payload = json.loads(raw)
    tasks = extract_tasks(payload)
    if args.validate_only:
        print(json.dumps({t: len(s) for t, s, _, _ in tasks}))
        return
    if args.output is None:
        args.output = args.input
    import transformers
    if transformers.__version__.split('.')[0] != '4':
        raise RuntimeError('Neural scoring requires the separate Transformers v4 environment')
    import yaml
    configs = yaml.safe_load(args.config.read_text())['mt_metrics']
    result = score_payload(payload, configs)
    packages = {}
    for name in ('transformers', 'torch', 'datasets', 'tokenizers', 'sentencepiece', 'unbabel-comet', 'bleurt-pytorch', 'numpy'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result['neural_scoring'].update(input_file=str(args.input.resolve()), input_sha256=hashlib.sha256(raw).hexdigest(), python=platform.python_version(), packages=packages)
    if args.input.read_bytes() != raw:
        raise RuntimeError('Input changed during scoring; refusing to overwrite results')
    atomic_write(args.output, result)
    print(f'Neural scores saved to {args.output}')


if __name__ == '__main__':
    main()
