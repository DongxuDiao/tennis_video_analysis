#!/bin/sh
# Compose wrapper: preserves this checkout's data paths and host ownership.
set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_dir"
export TENNIS_HOST_DATA_DIR="${TENNIS_HOST_DATA_DIR:-$repo_dir/data}"
export TENNIS_HOST_SECRET_DIR="${TENNIS_HOST_SECRET_DIR:-$repo_dir/.docker-state/secrets}"
export TENNIS_UID="$(id -u)"
export TENNIS_GID="$(id -g)"
if [ ! -d "$TENNIS_HOST_SECRET_DIR" ]; then
    (umask 077; mkdir -p "$TENNIS_HOST_SECRET_DIR"; touch "$TENNIS_HOST_SECRET_DIR/.writable")
fi
mkdir -p "$TENNIS_HOST_DATA_DIR"
if [ "$#" -eq 0 ]; then set -- up -d --build; fi
exec docker compose "$@"
