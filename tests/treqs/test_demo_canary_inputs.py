"""Dependency-free regression checks for the explicit demo adapter."""
import ast
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('adapter', ROOT / 'lda/utils/demo_canary_inputs.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DemoAdapterTests(unittest.TestCase):
    def test_padding_preserves_values_and_source(self):
        sample = dict(state=[list(range(12))], action=[[1] * 29],
                      embodiment_id=32, assigned_task='policy')
        result = module.adapt_demo_canary_examples([sample] * 4, state_dim=58,
                                                   action_dim=29, num_embodiments=32)
        self.assertEqual(len(result), 4)
        for item in result:
            self.assertEqual(item['state'][0], list(range(12)) + [0.0] * 46)
            self.assertEqual(item['embodiment_id'], 4)
            self.assertIs(item['action'], sample['action'])
        self.assertEqual(sample['embodiment_id'], 32)
        self.assertEqual(len(sample['state'][0]), 12)

    def test_rejects_incompatible_inputs(self):
        sample = dict(state=[[0] * 12], action=[[0] * 29],
                      embodiment_id=32, assigned_task='policy')
        for change in ({'state': [[0] * 59]}, {'action': [[0] * 7]},
                       {'embodiment_id': 4}, {'assigned_task': 'video_gen'}):
            with self.assertRaises(ValueError):
                module.adapt_demo_canary_examples([dict(sample, **change)],
                    state_dim=58, action_dim=29, num_embodiments=32)
        with self.assertRaises(ValueError):
            module.adapt_demo_canary_examples([sample], state_dim=58,
                                              action_dim=29, num_embodiments=4)

    def test_forward_failure_keeps_original_exception(self):
        tree = ast.parse((ROOT / 'lda/training/train_LDA.py').read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == '_train_step')
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<trainer>', 'exec'), namespace)
        class Accelerator:
            def accumulate(self, model):
                raise RuntimeError('original failure')
        class Trainer:
            accelerator = Accelerator()
            model = None
        with self.assertRaisesRegex(RuntimeError, 'original failure'):
            namespace['_train_step'](Trainer(), [])


if __name__ == '__main__':
    unittest.main()
