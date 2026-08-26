#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_DIR="${PROJECT_ROOT}/docker"

export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"
cd "${DOCKER_DIR}"

compose_files=(-f docker-compose.yml)
if [[ "${1:-}" == "--gui" ]]; then
    if [[ ! -d /dev/dri || -z "${DISPLAY:-}" ]]; then
        echo "ERROR: --gui requires /dev/dri and an active X11 DISPLAY." >&2
        exit 1
    fi
    XAUTH_FILE="${XAUTHORITY:-${HOME}/.Xauthority}"
    if [[ ! -r "${XAUTH_FILE}" ]]; then
        echo "ERROR: X11 authority file is not readable: ${XAUTH_FILE}" >&2
        exit 1
    fi
    export XAUTHORITY="${XAUTH_FILE}"
    export VIDEO_GID="$(getent group video | cut -d: -f3)"
    export RENDER_GID="$(getent group render | cut -d: -f3)"
    compose_files+=(-f docker-compose.gui.yml)
elif [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--gui]" >&2
    exit 2
fi

docker compose "${compose_files[@]}" build
docker compose "${compose_files[@]}" run --rm iea_g1_dev
