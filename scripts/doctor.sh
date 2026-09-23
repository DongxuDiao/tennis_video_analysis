#!/bin/sh
set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_dir/backend"
.venv/bin/python -c 'from app.store import db; db.init_db(); from app.api.main import system_doctor; import json; print(json.dumps(system_doctor(), ensure_ascii=False, indent=2))'
