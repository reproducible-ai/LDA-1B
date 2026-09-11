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
assert args[2:5] == [
    "artifacts/lda-robocasa-canary/release/checkpoints/" + name
    for name in ("LDA-robocasa-treqs-canary.pt", "artifact-manifest.json", "result.json")
]
assert args[5:-1] == ["--private", "--yes", "--no-tag", "-m", "private reproducibility canary"]
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

package_path = ROOT / '.treqs/scripts/package_robocasa_canary.py'
tree = ast.parse(package_path.read_text())
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in {'sha256_file', 'write_receipts'}]
namespace = {'Path': Path, 'hashlib': hashlib, 'json': json}
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
    evaluation = {'status': 'passed', 'optimizer_steps_completed': 1}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        namespace['write_receipts'](release, checkpoint, evaluation)
        namespace['write_receipts'](release, checkpoint, evaluation)
    result_path = checkpoint.parent / 'result.json'
    assert f'E2E_ARTIFACT={checkpoint}' in output.getvalue()
    assert f'E2E_RESULT={result_path}' in output.getvalue()
    assert json.loads(result_path.read_text())['optimizerSteps'] == 1
    manifest = json.loads((checkpoint.parent / 'artifact-manifest.json').read_text())
    assert manifest['format'] == 'pytorch-state-dict'
    assert manifest['path_base'] == 'release'
    assert {record['path'] for record in manifest['files']} == set(files + ['checkpoints/result.json'])
    for record in manifest['files']:
        data = (release / record['path']).read_bytes()
        assert record['size'] == len(data)
        assert record['sha256'] == hashlib.sha256(data).hexdigest()
        if "content_hex" in record:
            assert bytes.fromhex(record["content_hex"]) == data
        else:
            assert record["path"] in {"checkpoints/canary.pt", "checkpoints/result.json"}
    for steps in (0, 2, None):
        try:
            namespace['write_receipts'](release, checkpoint,
                                        {'status': 'passed', 'optimizer_steps_completed': steps})
        except RuntimeError:
            pass
        else:
            raise AssertionError('Invalid optimizer step count accepted')
print('PASS: receipt markers, metric, complete package hashes, repeat writes, invalid step rejection')
