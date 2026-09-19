#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_DIR="${PROJECT_ROOT}/docker/development"

ENV_FILE="${DOCKER_DIR}/.env.local"
SERVICE="iea_g1_dev"
use_proxy=false

action="shell"
gui=true
action_set=false
build=true

for argument in "$@"; do
    case "${argument}" in
        build|start|shell|stop)
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
        --headless)
            gui=false
            ;;
        --no-build)
            build=false
            ;;
        --proxy)
            use_proxy=true
            ;;
        *)
            echo "Usage: $0 [build|start|shell|stop] [--gui|--headless] [--no-build] [--proxy]" >&2
            exit 2
            ;;
    esac
done

if [[ "${use_proxy}" == true ]]; then
    if [[ ! -f "${ENV_FILE}" ]]; then
        echo "ERROR: --proxy requested but ${ENV_FILE} does not exist." >&2
        exit 1
    fi

    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
else
    unset HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy || true
fi

export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"
cd "${DOCKER_DIR}"

if [[ "${action}" == "build" ]]; then
    docker compose -f docker-compose.yml build "${SERVICE}"
    exit 0
fi

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
    build_args=()
    if [[ "${build}" == true ]]; then
        build_args+=(--build)
    else
        build_args+=(--no-build)
    fi
    docker compose "${compose_files[@]}" up -d "${build_args[@]}" "${SERVICE}"
    echo "Development container is running. Open a shell with:"
    echo "  ./scripts/dev.sh shell"
    exit 0
fi

if [[ "${gui}" == true ]]; then
    build_args=()
    if [[ "${build}" == true ]]; then
        build_args+=(--build)
    else
        build_args+=(--no-build)
    fi
    docker compose "${compose_files[@]}" up -d "${build_args[@]}" "${SERVICE}"
elif ! is_running; then
    build_args=()
    if [[ "${build}" == true ]]; then
        build_args+=(--build)
    else
        build_args+=(--no-build)
    fi
    docker compose "${compose_files[@]}" up -d "${build_args[@]}" "${SERVICE}"
fi

docker compose -f docker-compose.yml exec "${SERVICE}" bash
