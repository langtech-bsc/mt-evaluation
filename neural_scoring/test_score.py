import copy
import json
from pathlib import Path
import tempfile
import unittest
from .score import extract_tasks, score_payload, atomic_write


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.payload = {'results': {'task': {'sources,none': ['s1', 's2'], 'targets,none': ['r1', 'r2'], 'translations,none': ['h1', 'h2']}, 'group': {'bleu,none': 1}}}

    def test_alignment_and_preservation(self):
        original = copy.deepcopy(self.payload)
        def fake(name, cfg, src, ref, hyp):
            self.assertEqual(list(zip(src, ref, hyp)), [('s1', 'r1', 'h1'), ('s2', 'r2', 'h2')])
            return {'system_score': 1.5, 'segments_scores': [1, 2]}
        out = score_payload(self.payload, {'metricx': {'compute': True}}, fake)
        self.assertEqual(out['results']['task']['metricx_segments,none'], [1, 2])
        self.assertEqual(out['results']['group'], original['results']['group'])
        self.assertEqual(self.payload, original)

    def test_misaligned_input_rejected(self):
        self.payload['results']['task']['targets,none'].pop()
        with self.assertRaises(ValueError):
            extract_tasks(self.payload)

    def test_invalid_predictions_rejected(self):
        for scores in ([1], [1, float('nan')]):
            with self.assertRaises(ValueError):
                score_payload(self.payload, {'metricx': {'compute': True}}, lambda *args: {'system_score': 1, 'segments_scores': scores})

    def test_atomic_failure_preserves_previous_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'out.json'
            atomic_write(path, {'old': True})
            with self.assertRaises(ValueError):
                atomic_write(path, {'new': float('nan')})
            self.assertEqual(json.loads(path.read_text()), {'old': True})
            self.assertEqual(len(list(Path(directory).iterdir())), 1)


if __name__ == '__main__':
    unittest.main()
