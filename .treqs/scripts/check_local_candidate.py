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
assert args[:3] == ['roar', 'put', 'artifacts/lda-robocasa-canary/release']
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
