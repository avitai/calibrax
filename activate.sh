#!/usr/bin/env bash

_calibrax_activate_die() {
    echo "error: $*" >&2
    if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
        return 1
    fi
    exit 1
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    _calibrax_activate_die "use 'source ./activate.sh' so the environment stays active"
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTIVATE_SCRIPT="$PROJECT_ROOT/.venv/bin/activate"
MANAGED_ENV_FILE="${CALIBRAX_MANAGED_ENV_FILE:-$PROJECT_ROOT/.calibrax.env}"

if [[ ! -f "$ACTIVATE_SCRIPT" ]]; then
    _calibrax_activate_die "virtual environment not found; run ./setup.sh first"
fi

_calibrax_reset_previous_managed_env() {
    local variable
    for variable in ${CALIBRAX_MANAGED_ENV_VARS:-}; do
        unset "$variable"
    done
    unset CALIBRAX_MANAGED_ENV_VARS
}

_calibrax_filter_cuda_library_path() {
    local current_path="$1"
    local entry
    local entries=()
    local filtered_entries=()
    local IFS=':'

    read -r -a entries <<< "$current_path"
    for entry in "${entries[@]}"; do
        [[ -n "$entry" ]] || continue
        case "$entry" in
            *cuda*/targets/*/lib|*cuda*/targets/*/lib64|\
            *cuda*/targets/*/lib/*|*cuda*/targets/*/lib64/*|\
            *cuda*/lib|*cuda*/lib64|*cuda*/lib/*|*cuda*/lib64/*|\
            *cudnn*/lib|*cudnn*/lib64|*cudnn*/lib/*|*cudnn*/lib64/*)
                continue
                ;;
        esac
        filtered_entries+=("$entry")
    done

    (
        IFS=':'
        printf '%s' "${filtered_entries[*]}"
    )
}

_calibrax_sanitize_cuda_library_path() {
    [[ "${CALIBRAX_BACKEND:-}" == "cuda12" ]] || return
    [[ -n "${LD_LIBRARY_PATH:-}" ]] || return

    local filtered_path
    filtered_path="$(_calibrax_filter_cuda_library_path "$LD_LIBRARY_PATH")"
    if [[ -n "$filtered_path" ]]; then
        export LD_LIBRARY_PATH="$filtered_path"
    else
        unset LD_LIBRARY_PATH
    fi
}

# shellcheck disable=SC1090
source "$ACTIVATE_SCRIPT"

_calibrax_reset_previous_managed_env

if [[ -f "$MANAGED_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$MANAGED_ENV_FILE"
fi

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    # shellcheck disable=SC1090
    source "$PROJECT_ROOT/.env"
fi

if [[ -f "$PROJECT_ROOT/.env.local" ]]; then
    # shellcheck disable=SC1090
    source "$PROJECT_ROOT/.env.local"
fi

_calibrax_sanitize_cuda_library_path

echo "Calibrax environment active (${CALIBRAX_BACKEND:-auto})."
echo "Use 'python3 scripts/setup_env.py show' to inspect the resolved backend and env layering."
