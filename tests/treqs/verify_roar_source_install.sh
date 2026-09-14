#!/usr/bin/env bash
# Linux integration check; use an isolated container. No GPU, model inputs,
# credentials or remote publication. All writes stay under the supplied root.
set -euo pipefail
test_root="${1:?supply an empty validation directory}"
install_root="${2:-$test_root/install}"
test "!" -e "$test_root"
mkdir -p "$test_root/bin" "$test_root/workload"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
installer="$repo_root/.treqs/scripts/install_roar_source.sh"

timeout --signal=TERM --kill-after=30 1200 bash "$installer" "$install_root" "$test_root/bin" > "$test_root/cold.log" 2>&1
receipt="$(find "$install_root" -name build.json -print -quit)"
test -s "$receipt"
cp "$receipt" "$test_root/original-build.json"
timeout --signal=TERM --kill-after=30 120 bash "$installer" "$install_root" "$test_root/bin" > "$test_root/cached.log" 2>&1
cmp "$receipt" "$test_root/original-build.json"
! grep -q 'Installing Python package' "$test_root/cached.log"

uv venv --python 3.11 "$test_root/workload-venv"
export PATH="$test_root/workload-venv/bin:$test_root/bin:/usr/local/bin:/usr/bin:/bin"
cd "$test_root/workload"
git init
git -c user.name='Roar installer test' -c user.email='roar-test@localhost' commit --allow-empty -m 'test: initialize isolated trace'
printf 'source-install-probe\n' > input.txt
printf '.roar/\noutput.txt\n' > .gitignore
git add input.txt .gitignore
git -c user.name='Roar installer test' -c user.email='roar-test@localhost' commit -m 'test: add tracked input'
roar init --no-gitignore
# Match the TReqs agent's normal lineage initialization for temporary inputs.
roar config set filters.ignore_tmp_files false
roar run --tracer preload -- python3 -c 'from pathlib import Path; Path("output.txt").write_bytes(Path("input.txt").read_bytes())'
cmp input.txt output.txt
roar dag
python3 - <<'PY'
from pathlib import Path
import sqlite3
with sqlite3.connect("file:.roar/roar.db?mode=ro", uri=True) as db:
    job_id, exit_code = db.execute("SELECT id, exit_code FROM jobs ORDER BY id DESC LIMIT 1").fetchone()
    assert exit_code == 0
    inputs = {Path(row[0]).name for row in db.execute("SELECT path FROM job_inputs WHERE job_id=?", (job_id,))}
    outputs = {Path(row[0]).name for row in db.execute("SELECT path FROM job_outputs WHERE job_id=?", (job_id,))}
    assert "input.txt" in inputs, inputs
    assert "output.txt" in outputs, outputs
print("PASS: input and output edges recorded by the real preload tracer")
PY

source_dir="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["source"])' "$receipt")"
printf '\ninstaller tamper test\n' >> "$source_dir/README.md"
if timeout --signal=TERM --kill-after=30 120 bash "$installer" "$install_root" "$test_root/bin" > "$test_root/source-tamper.log" 2>&1; then
  echo 'ERROR: changed tracked source was accepted' >&2
  exit 1
fi
git -C "$source_dir" restore -- README.md
cp "$source_dir/rust/target/release/roar-tracer-preload" "$test_root/preload-original"
printf 'tamper' >> "$source_dir/rust/target/release/roar-tracer-preload"
if timeout --signal=TERM --kill-after=30 120 bash "$installer" "$install_root" "$test_root/bin" > "$test_root/native-tamper.log" 2>&1; then
  echo 'ERROR: changed native bytes were accepted' >&2
  exit 1
fi
grep -q 'cached Roar build differs from its receipt' "$test_root/native-tamper.log"
cp "$test_root/preload-original" "$source_dir/rust/target/release/roar-tracer-preload"
cmp "$receipt" "$test_root/original-build.json"
printf 'PASS: Linux source build, unchanged cache receipt, real file trace, source/native tamper rejection\n'
