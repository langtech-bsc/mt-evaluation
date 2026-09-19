"""Run from the v3 repository root. Adds opt-in generation tracing, not new stop rules."""
from pathlib import Path
import ast
p = Path('lm_eval/models/huggingface.py')
s = p.read_text()
if 'MT_GENERATION_TRACE' in s:
    raise SystemExit('Trace instrumentation already exists; no changes made.')
needle = '                # Handle integer think_end_token: find last occurrence and strip tokens after it'
insert = '''                # Opt-in diagnostics: preserve tokens before any text processing.
                import os as _trace_os
                _trace_path = _trace_os.environ.get("MT_GENERATION_TRACE")
                if _trace_path:
                    _trace_tokens = list(cont_toks)

'''
assert s.count(needle) == 1
s = s.replace(needle, insert + needle)
needle = '                res.append(s)\n                # BSC: log each generation'
insert = '''                if _trace_path:
                    import json as _trace_json
                    _trace_record = {
                        "context": context,
                        "raw_token_ids": _trace_tokens,
                        "raw_text_with_special_tokens": self.tokenizer.decode(
                            _trace_tokens, skip_special_tokens=False),
                        "raw_text_without_special_tokens": self.tokenizer.decode(
                            _trace_tokens, skip_special_tokens=True),
                        "final_text": s,
                        "stop_strings": until,
                        "padded_input_length": int(context_enc.shape[1]),
                        "requested_max_gen_toks": max_gen_toks,
                        "passed_max_length": max_length,
                        "generation_kwargs": kwargs,
                        "model_generation_config": self.model.generation_config.to_dict(),
                        "tokenizer_eos_token_id": self.tokenizer.eos_token_id,
                        "tokenizer_pad_token_id": self.tokenizer.pad_token_id,
                    }
                    with open(_trace_path, "a", encoding="utf-8") as _trace_file:
                        _trace_file.write(_trace_json.dumps(
                            _trace_record, ensure_ascii=False, default=str) + "\\n")
                res.append(s)
                # BSC: log each generation'''
assert s.count(needle) == 1
s = s.replace(needle, insert)
ast.parse(s)
backup = p.with_name(p.name + '.before_generation_trace')
if backup.exists():
    raise SystemExit(f'Backup already exists: {backup}; inspect before proceeding.')
backup.write_text(p.read_text())
p.write_text(s)
print(f'Added opt-in tracing. Original saved to {backup}')
