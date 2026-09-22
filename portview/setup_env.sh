#!/usr/bin/env bash
# portview environment bootstrap: venv (native -> uv fallback) + deps
V=/workspaces/codespaces-blank/portview/.venv
cd /workspaces/codespaces-blank/portview

sudo -n apt-get update >/tmp/aptup.log 2>&1
sudo -n apt-get install -y python3.12-venv >/tmp/apt2.log 2>&1

rm -rf .venv
if /usr/bin/python3.12 -m venv .venv >/tmp/venv3.log 2>&1 && \
   $V/bin/python -m ensurepip --version >/dev/null 2>&1; then
  echo VENV_NATIVE
  $V/bin/pip install -q -U pip
  $V/bin/pip install -q mitsuba numpy opencv-python-headless >/tmp/pip2.log 2>&1
else
  echo VENV_FALLBACK_UV
  curl -LsSf https://astral.sh/uv/install.sh | sh >/tmp/uv-install.log 2>&1
  export PATH="$HOME/.local/bin:$PATH"
  rm -rf .venv
  uv venv --python /usr/bin/python3.12 .venv >>/tmp/venv3.log 2>&1
  uv pip install --python $V/bin/python mitsuba numpy opencv-python-headless \
      >>/tmp/pip2.log 2>&1
fi

$V/bin/python -c "import mitsuba, numpy, cv2; print('IMPORT_OK', mitsuba.__version__)" \
    >/tmp/import_check.log 2>&1
echo DONE >/tmp/install.done