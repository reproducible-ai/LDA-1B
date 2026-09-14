#!/usr/bin/env bash
# Install the reviewed main revision outside the model environment. Optional
# directories let the same installer run in an isolated local Linux container.
set -euo pipefail

roar_repository=https://github.com/treqs/roar.git
roar_commit=61e5e98ca823a25c19522870cb81d3ce730c391d
roar_version=0.4.7
roar_hub_version=0.36.0
roar_install_root="${1:-$(python3 -m site --user-base)/share/reproai/roar}"
roar_bin_dir="${2:-/usr/local/bin}"
roar_prefix="$roar_install_root/$roar_commit"
roar_source="$roar_prefix/source"
roar_venv="$roar_prefix/venv"

test "$(uname -s)" = Linux
command -v uv >/dev/null
mkdir -p "$roar_prefix"
# A failed build has no success receipt and can resume incrementally. Serialize
# installation so another setup cannot read a partially built environment.
exec 9>"$roar_prefix/install.lock"
flock -w 120 9

as_root() {
  if [ "$(id -u)" = 0 ]; then
    "$@"
  else
    sudo -n "$@"
  fi
}

if ! command -v git >/dev/null || ! command -v curl >/dev/null \
  || ! command -v cc >/dev/null || ! command -v pkg-config >/dev/null \
  || ! pkg-config --exists openssl; then
  as_root env DEBIAN_FRONTEND=noninteractive apt-get update -qq
  as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    git curl ca-certificates build-essential pkg-config libssl-dev
fi

if [ ! -d "$roar_source/.git" ]; then
  git init "$roar_source"
  git -C "$roar_source" remote add origin "$roar_repository"
fi
test "$(git -C "$roar_source" remote get-url origin)" = "$roar_repository"
if ! git -C "$roar_source" rev-parse --verify HEAD >/dev/null 2>&1; then
  GIT_TERMINAL_PROMPT=0 git -C "$roar_source" fetch --depth 1 origin "$roar_commit"
  git -C "$roar_source" checkout --detach FETCH_HEAD
fi
test "$(git -C "$roar_source" rev-parse HEAD)" = "$roar_commit"
git -C "$roar_source" diff --quiet HEAD --

if [ ! -f "$roar_prefix/build.json" ]; then
  if [ -f "${HOME:?}/.cargo/env" ]; then
    . "$HOME/.cargo/env"
  fi
  if ! command -v cargo >/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile minimal
    . "$HOME/.cargo/env"
  fi
  if [ ! -x "$roar_venv/bin/python" ]; then
    uv venv --python 3.11 "$roar_venv"
  fi
  # The upstream installer builds both the Python extension and native tracers.
  # A plain git pip install would omit the separate Rust tracer binaries.
  VIRTUAL_ENV="$roar_venv" PATH="$roar_venv/bin:$PATH" \
    bash "$roar_source/scripts/install-dev.sh"
  uv pip install --python "$roar_venv/bin/python" "huggingface-hub==$roar_hub_version"
fi

# Check the installed import and native bytes on every invocation, including
# cached installs. A version string alone cannot distinguish this unreleased
# commit from the existing 0.4.7 PyPI package.
"$roar_venv/bin/python" -I - "$roar_source" "$roar_commit" "$roar_repository" \
  "$roar_version" "$roar_hub_version" "$roar_prefix/build.json" <<'PY'
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import roar

source, commit, repository, version, hub_version, receipt = sys.argv[1:]
source = Path(source).resolve()
assert Path(roar.__file__).resolve() == source / "roar/__init__.py", "wrong Roar import"
assert importlib.metadata.version("roar-cli") == version, "wrong Roar version"
assert importlib.metadata.version("huggingface-hub") == hub_version, "wrong HF dependency"
required = {"roar-tracer", "roar-tracer-preload", "libroar_tracer_preload.so", "roar-proxy"}
# Editable installs prefer the cargo release directory; wheels use roar/bin.
# Verify both so the receipt covers the binaries that are actually selected.
native_paths = [Path(directory) / name for directory in ("roar/bin", "rust/target/release") for name in sorted(required)]
assert all((source / path).is_file() for path in native_paths), "native tracer build incomplete"
hashes = {str(path): hashlib.sha256((source / path).read_bytes()).hexdigest() for path in native_paths}
assert subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() == commit
subprocess.run(["git", "-C", str(source), "diff", "--exit-code", "HEAD", "--"], check=True)
record = {
    "repository": repository, "commit": commit, "version": version,
    "huggingfaceHubVersion": hub_version, "python": sys.version.split()[0],
    "pythonExecutable": sys.executable, "source": str(source), "nativeSha256": hashes,
}
receipt = Path(receipt)
if receipt.exists():
    assert json.loads(receipt.read_text()) == record, "cached Roar build differs from its receipt"
receipt.with_suffix(".pending.json").write_text(json.dumps(record, sort_keys=True) + "\n")
PY

roar_probe="$(mktemp -d "$roar_prefix/preflight.XXXXXX")"
trap 'rm -rf -- "$roar_probe"' EXIT
(
  cd "$roar_probe"
  "$roar_venv/bin/roar" tracer check preload
)

if [ -w "$roar_bin_dir" ]; then
  ln -sfn "$roar_venv/bin/roar" "$roar_bin_dir/roar"
  ln -sfn "$roar_venv/bin/roar-worker" "$roar_bin_dir/roar-worker"
else
  as_root mkdir -p "$roar_bin_dir"
  as_root ln -sfn "$roar_venv/bin/roar" "$roar_bin_dir/roar"
  as_root ln -sfn "$roar_venv/bin/roar-worker" "$roar_bin_dir/roar-worker"
fi
test "$(env PATH="$roar_bin_dir:/usr/bin:/bin" roar --version)" = "roar, version $roar_version"
mv "$roar_prefix/build.pending.json" "$roar_prefix/build.json"
printf 'ROAR_SOURCE_BUILD='
cat "$roar_prefix/build.json"
