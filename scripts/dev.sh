#!/bin/sh
set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_dir/frontend"
npm run build
cd "$repo_dir/backend"
exec .venv/bin/python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8765
