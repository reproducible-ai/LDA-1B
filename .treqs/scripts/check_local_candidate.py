"""Dependency-free local checks; does not run workloads or contact services."""
import ast
import re
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
workflow = (ROOT / '.treqs/workflows/robocasa-demo-canary.yaml').read_text()
assert re.search(r'^secrets:\n(?:[ \t]*- [^\n]+\n)*[ \t]*- HF_TOKEN\n', workflow, re.M), \
    'workflow must declare the HF_TOKEN secret for roar put'
stages = {}
for match in re.finditer(r'^([a-z]+):\n((?:[ \t].*\n|\n)+)', workflow, re.M):
    name, body = match.groups()
    if '  command: |\n' not in body:
        continue
    command = '\n'.join(line[4:] for line in body.splitlines() if line.startswith('    '))
    subprocess.run(['bash', '-n'], input=command, text=True, check=True)
    stages[name] = (body, command)
assert set(stages) == {'setup', 'fetch', 'train', 'evaluate', 'package', 'label', 'publish'}
# PyTorch's index also serves general packages, sometimes missing our exact pins.
# uv must consider PyPI too rather than stop at the first index containing a name.
installs = [shlex.split(line) for line in stages['setup'][1].splitlines()
            if '-r requirements.txt' in line]
assert len(installs) == 1
install = installs[0]
assert install[install.index('--index-strategy') + 1] == 'unsafe-best-match'
requirements_text = (ROOT / 'requirements.txt').read_text()
assert '--extra-index-url https://download.pytorch.org/whl/cu128' in requirements_text
assert 'certifi==2025.11.12' in requirements_text
print('PASS: requirements install considers both package indexes without changing pins')
assert set(re.findall(r'^([a-z_]+):', workflow, re.M)) == set(stages) | {'name', 'secrets'}
assert 'trace: "run"' in stages['train'][0]
assert 'roar' not in stages['train'][1]
body, command = stages['publish']
assert 'glaas_creds: true' in body and 'trace: "off"' in body
uploads = [line for line in command.splitlines() if 'roar put' in line]
assert len(uploads) == 1
args = shlex.split(uploads[0])
assert args[-1].startswith('hf://')
assert args[-1].endswith('/artifacts/lda-robocasa-canary/release/checkpoints')
assert args[:2] == ["roar", "put"]
assert args[2] == "artifacts/lda-robocasa-canary/release/checkpoints"
assert args[3:-1] == ["--private", "--yes", "--no-tag", "-m", "private reproducibility canary"]
assert all(flag in args for flag in ('--private', '--yes', '--no-tag'))
assert not any(flag in args for flag in ('--public', '--anonymous'))
assert args.count('-m') == 1 and args[args.index('-m') + 1].strip()
for name in ('artifact-manifest.json', 'result.json'):
    assert name in command
for path in list((ROOT / '.treqs/scripts').glob('*.py')) + list((ROOT / 'tests/treqs').glob('*.py')):
    ast.parse(path.read_text(), filename=str(path))
# Exercise destination derivation after a supervisor repository replacement.
contract_path = ROOT / '.treqs/scripts/lda_canary_contract.py'
tree = ast.parse(contract_path.read_text())
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == 'publication_repo_id')
class Workflow:
    def __truediv__(self, other):
        return self
    def read_text(self):
        return re.sub(r'hf://[^/\s]+/[^/\s]+', 'hf://reproducible-ai/replaced-attempt', workflow)
namespace = {'ROOT': Workflow()}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(contract_path), 'exec'), namespace)
assert namespace['publication_repo_id']() == 'reproducible-ai/replaced-attempt'
print('PASS: HF_TOKEN declaration, seven stage shell syntax checks, private upload contract, Python syntax, repository replacement')

# Exercise the real receipt writer without importing GPU or YAML dependencies.
import contextlib
import hashlib
import io
import json
import tempfile
import shutil

package_path = ROOT / '.treqs/scripts/package_robocasa_canary.py'
tree = ast.parse(package_path.read_text())
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in {'sha256_file', 'write_receipts'}]
namespace = {'Path': Path, 'hashlib': hashlib, 'json': json, 'shutil': shutil, 'ROOT': ROOT}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(package_path), 'exec'), namespace)
with tempfile.TemporaryDirectory(dir=ROOT) as directory:
    release = Path(directory) / 'release'
    checkpoint = release / 'checkpoints/canary.pt'
    checkpoint.parent.mkdir(parents=True)
    files = ['checkpoints/canary.pt', 'config.yaml', 'dataset_statistics.json',
             'pretrained/dino/config.json', 'input-manifest.json', 'LICENSE',
             'CC-BY-NC-4.0.md', 'APACHE-2.0.txt', 'DINOv3-LICENSE.md', 'NOTICE']
    for name in files:
        path = release / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture: ' + name)
    evaluation = {'status': 'passed', 'optimizer_steps_completed': 1, 'loadVerified': True, 'checkpoint': {'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        namespace['write_receipts'](release, checkpoint, evaluation)
        namespace['write_receipts'](release, checkpoint, evaluation)
    result_path = checkpoint.parent / 'result.json'
    receipts = output.getvalue().splitlines()
    assert len(receipts) == 4
    artifact = json.loads(receipts[0].removeprefix('E2E_ARTIFACT='))
    result = json.loads(receipts[1].removeprefix('E2E_RESULT='))
    assert result == json.loads(result_path.read_text())
    assert result['schema'] == 'reproai.result/v1'
    assert result['optimizerSteps'] == 1
    assert result['taskMetric'] == {'metric': 'optimizerSteps', 'minimum': 1}
    assert result['artifactSha256'] == artifact['sha256'] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert result['artifactSizeBytes'] == artifact['sizeBytes'] == checkpoint.stat().st_size
    assert result['checkpoint'] == artifact['path']
    assert artifact['schema'] == 'reproai.artifact/v1'
    manifest = json.loads((checkpoint.parent / 'artifact-manifest.json').read_text())
    assert manifest['schema'] == 'reproai.artifact-manifest/v1'
    assert manifest['format'] == 'pytorch-state-dict'
    assert manifest['loadVerified'] is artifact['loadVerified'] is result['loadVerified'] is True
    assert manifest['files'] == artifact['files']
    expected = {'canary.pt'} | {'loader/' + name for name in files if name != 'checkpoints/canary.pt'}
    assert {record['path'] for record in manifest['files']} == expected
    for record in manifest['files']:
        path = Path(record['path'])
        assert not path.is_absolute() and '..' not in path.parts
        data = (checkpoint.parent / path).read_bytes()
        assert record['sizeBytes'] == len(data)
        assert record['sha256'] == hashlib.sha256(data).hexdigest()
    for updates in ({'loadVerified': False}, {'checkpoint': {'sha256': 'wrong'}}):
        try:
            namespace['write_receipts'](release, checkpoint, dict(evaluation, **updates))
        except RuntimeError:
            pass
        else:
            raise AssertionError('Unverified checkpoint accepted')
    for steps in (0, 2, None):
        try:
            namespace['write_receipts'](release, checkpoint,
                                        {'status': 'passed', 'optimizer_steps_completed': steps})
        except RuntimeError:
            pass
        else:
            raise AssertionError('Invalid optimizer step count accepted')
print('PASS: receipt markers, metric, complete package hashes, repeat writes, invalid step rejection')

# Exercise actual runtime validation and launch construction with no CUDA workload.
from types import SimpleNamespace
import os
runner_path = ROOT / '.treqs/scripts/run_robocasa_canary.py'
tree = ast.parse(runner_path.read_text())
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
properties = SimpleNamespace(total_memory=96 * 1024**3, name='RTX PRO 6000 Blackwell')
cuda = SimpleNamespace(device_count=lambda: 1, get_device_properties=lambda _: properties,
                       get_device_capability=lambda _: (12, 0))
fake_torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda='12.8'), __version__='2.9.0+cu128')
calls = []
with tempfile.TemporaryDirectory(dir=ROOT) as directory:
    root = Path(directory)
    paths = dict(BASE_CHECKPOINT=root / 'base/checkpoints/base.pt',
                 BASE_SNAPSHOT=root / 'base', QWEN_SNAPSHOT=root / 'qwen',
                 PRETRAINED_ROOT=root / 'pretrained', ROOT=root,
                 RUN_ROOT=root / 'run', RUN_ID='fixture')
    for path in (paths['BASE_CHECKPOINT'], paths['BASE_SNAPSHOT'] / 'config.yaml',
                 paths['QWEN_SNAPSHOT'] / 'config.json',
                 paths['PRETRAINED_ROOT'] / 'dinov3-vits16-pretrain-lvd1689m/config.json'):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
    namespace = dict(paths, torch=fake_torch, os=os,
                     subprocess=SimpleNamespace(run=lambda *a, **kw: calls.append((a, kw))))
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(runner_path), 'exec'), namespace)
    namespace['main']()
    assert len(calls) == 1
    argv = calls[0][0][0]
    options = dict(zip(argv[argv.index('--config_yaml')::2], argv[argv.index('--config_yaml') + 1::2]))
    assert argv[argv.index('--num_processes') + 1] == '1'
    assert options['--datasets.vla_data.per_device_batch_size'] == '4'
    assert options['--trainer.gradient_accumulation_steps'] == '1'
    assert options['--trainer.max_train_steps'] == '1'
    assert options['--trainer.strict_pretrained_checkpoint'] == 'true'
    assert options['--trainer.freeze_modules'] == 'action_model.vision_encoder,qwen_vl_interface'
    assert options['--datasets.vla_data.training_tasks'] == '["policy"]'
    for obj, key, invalid in ((cuda, 'device_count', lambda: 4),
                              (properties, 'total_memory', 48 * 1024**3),
                              (properties, 'name', 'L40S'),
                              (cuda, 'get_device_capability', lambda _: (8, 9)),
                              (fake_torch.version, 'cuda', '12.4'),
                              (fake_torch, '__version__', '2.6.0')):
        original = getattr(obj, key)
        setattr(obj, key, invalid)
        try:
            namespace['validate_runtime']()
        except RuntimeError:
            pass
        else:
            raise AssertionError('Incompatible runtime accepted: ' + key)
        finally:
            setattr(obj, key, original)
requirements = (ROOT / 'requirements.txt').read_text().splitlines()
assert {'torch==2.9.0+cu128', 'torchvision==0.24.0+cu128', 'torchcodec==0.8.1',
        'sympy==1.14.0', 'triton==3.5.0'} <= set(requirements)
assert not any(line.startswith('nvidia-') and not line.startswith('nvidia-ml-py==') for line in requirements)
assert 'num_processes: 1' in (ROOT / '.treqs/assets/accelerate-zero2-cpu.yaml').read_text()
ds = json.loads((ROOT / '.treqs/assets/deepspeed-zero2-cpu.json').read_text())
assert ds['gradient_accumulation_steps'] == 1 and ds['bf16']['enabled'] is True
assert ds['train_micro_batch_size_per_gpu'] == 4
assert ds['train_batch_size'] == ds['train_micro_batch_size_per_gpu'] * ds['gradient_accumulation_steps'] == 4
print('PASS: explicit DeepSpeed micro/global batch sizes remain four for the custom batch sampler')
assert ds['zero_optimization']['offload_optimizer']['device'] == 'none'
assert 'pin_memory' not in ds['zero_optimization']['offload_optimizer']
accelerate_text = (ROOT / '.treqs/assets/accelerate-zero2-cpu.yaml').read_text()
ds_owned = {'mixed_precision', 'gradient_accumulation_steps', 'gradient_clipping',
            'zero_stage', 'offload_optimizer_device', 'offload_param_device',
            'offload_param_nvme_path', 'offload_optimizer_nvme_path',
            'zero3_save_16bit_model'}
assert not ds_owned.intersection(re.findall(r'^\s*([a-z_0-9]+):', accelerate_text, re.M))
assert ds['fp16']['enabled'] is False and ds['gradient_clipping'] == 1.0
print('PASS: DeepSpeed exclusively owns precision, accumulation, clipping and ZeRO settings')
assert 'test "${GPU_COUNT}" = "1"' in stages['setup'][1]
print('PASS: single Blackwell launch, effective batch four, frozen encoders, strict load, six invalid runtimes, dependency pins, BF16/offload configuration (mocked only)')
assert options['--datasets.vla_data.demo_canary_adapter'] == 'true'
framework = (ROOT / 'lda/model/framework/QwenMMDiT.py').read_text()
assert framework.index('examples = adapt_demo_canary_examples(') < framework.index('batch_images =')
assert 'config["datasets"]["vla_data"]["demo_canary_adapter"] = True' in (ROOT / '.treqs/scripts/package_robocasa_canary.py').read_text()
print('PASS: demo adapter enabled before forward input extraction and retained in loader config')

# Execute the adapter itself without requiring the remote training environment.
import runpy
adapter = runpy.run_path(str(ROOT / 'lda/utils/demo_canary_inputs.py'))
fixture = ROOT / '.treqs/assets/robocasa-pinned-config.yaml'
contract_tree = ast.parse(contract_path.read_text())
expected_hash = next(ast.literal_eval(node.value) for node in contract_tree.body
                     if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == 'BASE_CONFIG_SHA256'
                             for target in node.targets))
assert hashlib.sha256(fixture.read_bytes()).hexdigest() == expected_hash
for name, value in dict(state_dim=58, action_dim=138, max_num_embodiments=32,
                        obs_horizon=2, future_obs_index=16).items():
    assert re.search(rf'^    {name}: {value}$', fixture.read_text(), re.M)
sample = dict(state=[list(range(12)), list(range(20, 32))],
              action=[[1.0] * 138 for _ in range(16)],
              action_mask=[[True] * 7 + [False] * 131 for _ in range(16)],
              embodiment_id=32, assigned_task='policy')
kwargs = dict(state_dim=58, action_dim=138, num_embodiments=32)
adapted = adapter['adapt_demo_canary_examples']([sample] * 4, **kwargs)
assert len(adapted) == 4
for example in adapted:
    assert example['embodiment_id'] == 4
    assert example['action'] is sample['action']
    assert example['action_mask'] is sample['action_mask']
    for source, padded in zip(sample['state'], example['state']):
        assert padded == source + [0.0] * 46
assert sample['embodiment_id'] == 32 and all(len(row) == 12 for row in sample['state'])
for update in ({'state': [[0] * 12]}, {'action': [[0] * 7]},
               {'embodiment_id': 4}, {'assigned_task': 'video_gen'}):
    try:
        adapter['adapt_demo_canary_examples']([dict(sample, **update)], **kwargs)
    except ValueError:
        pass
    else:
        raise AssertionError('Incompatible demo input accepted')
modalities = {name: SimpleNamespace(delta_indices=indices)
              for name, indices in dict(video=[0], state=[0], future_video=[5],
                                        action=list(range(16))).items()}
history = adapter['adapt_demo_canary_modalities'](modalities)
assert history['video'].delta_indices == history['state'].delta_indices == [-5, 0]
assert history['future_video'].delta_indices == [16]
assert history['action'].delta_indices == list(range(16))
assert modalities['video'].delta_indices == modalities['state'].delta_indices == [0]
assert modalities['future_video'].delta_indices == [5]
print('PASS: pinned config hash/dimensions, distinct state histories, zero padding, action/mask preservation, Franka slot, invalid inputs and sampling offsets (synthetic only)')
