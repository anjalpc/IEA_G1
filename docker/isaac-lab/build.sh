#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_FILE="${SCRIPT_DIR}/.env.local"
USE_PROXY=false

for argument in "$@"; do
    case "${argument}" in
        --proxy)
            USE_PROXY=true
            ;;
        *)
            echo "Usage: $0 [--proxy]" >&2
            exit 2
            ;;
    esac
done

if [[ "${USE_PROXY}" == true ]]; then
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
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE="${IEA_WORKSPACE:-$(dirname "${REPO_ROOT}")}"
ISAACLAB_SOURCE="${ISAACLAB_SOURCE:-${WORKSPACE}/src/IsaacLab}"

test -d "${ISAACLAB_SOURCE}"

docker build \
  --network=host \
  --build-arg HTTP_PROXY="${HTTP_PROXY:-}" \
  --build-arg HTTPS_PROXY="${HTTPS_PROXY:-}" \
  --build-arg NO_PROXY="${NO_PROXY:-}" \
  --build-arg http_proxy="${http_proxy:-${HTTP_PROXY:-}}" \
  --build-arg https_proxy="${https_proxy:-${HTTPS_PROXY:-}}" \
  --build-arg no_proxy="${no_proxy:-${NO_PROXY:-}}" \
  -t isaac-lab-base-fixed:2.3.0 \
  -f "${SCRIPT_DIR}/Dockerfile" \
  "${ISAACLAB_SOURCE}"
