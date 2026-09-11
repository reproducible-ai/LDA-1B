"""Checkpoint-bound regression checks for the explicit demo adapter."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('adapter', ROOT / 'lda/utils/demo_canary_inputs.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
MODEL = yaml.safe_load((ROOT / '.treqs/assets/robocasa-pinned-config.yaml').read_text())['framework']['action_model']


class DemoAdapterTests(unittest.TestCase):
    def test_fixture_matches_input_preparation_hash(self):
        contract = ast.parse((ROOT / '.treqs/scripts/lda_canary_contract.py').read_text())
        expected = next(ast.literal_eval(node.value) for node in contract.body
                        if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == 'BASE_CONFIG_SHA256'
                                for target in node.targets))
        actual = hashlib.sha256((ROOT / '.treqs/assets/robocasa-pinned-config.yaml').read_bytes()).hexdigest()
        self.assertEqual(actual, expected)

    def test_padding_preserves_values_and_source(self):
        sample = dict(state=[list(range(12))] * MODEL['obs_horizon'], action=[[1] * MODEL['action_dim']],
                      embodiment_id=32, assigned_task='policy')
        result = module.adapt_demo_canary_examples([sample] * 4, state_dim=MODEL['state_dim'],
                                                   action_dim=MODEL['action_dim'], num_embodiments=MODEL['max_num_embodiments'])
        self.assertEqual(len(result), 4)
        for item in result:
            self.assertEqual(item['state'][0], list(range(12)) + [0.0] * 46)
            self.assertEqual(item['embodiment_id'], 4)
            self.assertIs(item['action'], sample['action'])
        self.assertEqual(sample['embodiment_id'], 32)
        self.assertEqual(len(sample['state'][0]), 12)

    def test_rejects_incompatible_inputs(self):
        sample = dict(state=[[0] * 12] * MODEL['obs_horizon'], action=[[0] * MODEL['action_dim']],
                      embodiment_id=32, assigned_task='policy')
        for change in ({'state': [[0] * 59] * 2}, {'state': [[0] * 12]}, {'action': [[0] * 7]},
                       {'embodiment_id': 4}, {'assigned_task': 'video_gen'}):
            with self.assertRaises(ValueError):
                module.adapt_demo_canary_examples([dict(sample, **change)],
                    state_dim=MODEL['state_dim'], action_dim=MODEL['action_dim'], num_embodiments=MODEL['max_num_embodiments'])
        with self.assertRaises(ValueError):
            module.adapt_demo_canary_examples([sample], state_dim=58,
                                              action_dim=MODEL['action_dim'], num_embodiments=4)

    def test_history_matches_pinned_checkpoint_without_changing_defaults(self):
        original = {name: SimpleNamespace(delta_indices=indices, modality_keys=[name])
                    for name, indices in {'video': [0], 'state': [0],
                                          'future_video': [5], 'language': [0],
                                          'action': list(range(16))}.items()}
        result = module.adapt_demo_canary_modalities(original)
        for name in ('video', 'state'):
            self.assertEqual(result[name].delta_indices, [-5, 0])
            self.assertEqual(len(result[name].delta_indices), MODEL['obs_horizon'])
            self.assertEqual(original[name].delta_indices, [0])
        self.assertEqual(result['future_video'].delta_indices, [MODEL['future_obs_index']])
        self.assertEqual(result['action'].delta_indices, list(range(MODEL['action_horizon'])))
        self.assertEqual(result['language'].delta_indices, [0])

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
