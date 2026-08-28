#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_DIR="${PROJECT_ROOT}/docker"
SERVICE="iea_g1_dev"

action="shell"
gui=false
action_set=false

for argument in "$@"; do
    case "${argument}" in
        start|shell|stop)
            if [[ "${action_set}" == true ]]; then
                echo "ERROR: choose only one of start, shell, or stop." >&2
                exit 2
            fi
            action="${argument}"
            action_set=true
            ;;
        --gui)
            gui=true
            ;;
        *)
            echo "Usage: $0 [start|shell|stop] [--gui]" >&2
            exit 2
            ;;
    esac
done

export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"
cd "${DOCKER_DIR}"

compose_files=(-f docker-compose.yml)
if [[ "${gui}" == true && "${action}" != "stop" ]]; then
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
fi

if [[ "${action}" == "stop" ]]; then
    docker compose -f docker-compose.yml down --remove-orphans
    exit 0
fi

is_running() {
    docker compose -f docker-compose.yml ps --status running --services |
        grep -qx "${SERVICE}"
}

if [[ "${action}" == "start" ]]; then
    docker compose "${compose_files[@]}" up -d --build "${SERVICE}"
    echo "Development container is running. Open a shell with:"
    echo "  ./scripts/dev.sh shell"
    exit 0
fi

if [[ "${gui}" == true ]]; then
    docker compose "${compose_files[@]}" up -d --build "${SERVICE}"
elif ! is_running; then
    docker compose "${compose_files[@]}" up -d --build "${SERVICE}"
fi

docker compose -f docker-compose.yml exec "${SERVICE}" bash
