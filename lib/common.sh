#!/usr/bin/env bash
# Reusable helpers for piper_ta_rocm71_train.sh.
#
# This file is meant to be sourced. It only defines functions and has no side
# effects, so it can be exercised in isolation by tests/test_common.sh.

# shellcheck disable=SC2034  # PIPER_COMMON_LOADED is read by callers.
PIPER_COMMON_LOADED=1

log() {
  printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

# Wait until no process holds the apt/dpkg locks (e.g. unattended-upgrades).
# Usage: wait_for_apt_locks [max_wait_seconds]
wait_for_apt_locks() {
  local max_wait="${1:-1800}"
  local waited=0
  command -v fuser >/dev/null 2>&1 || return 0
  while fuser /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock \
    /var/lib/apt/lists/lock /var/cache/apt/archives/lock >/dev/null 2>&1; do
    if ((waited == 0)); then
      log "Waiting for apt/dpkg lock (likely unattended-upgrades) ..."
    fi
    sleep 5
    waited=$((waited + 5))
    if ((waited >= max_wait)); then
      printf 'ERROR: Timed out waiting for apt/dpkg locks.\n' >&2
      return 1
    fi
  done
}

# Run a command, retrying while apt/dpkg is busy.
# Usage: run_apt_cmd <cmd> [args...]
run_apt_cmd() {
  local attempt
  local attempts="${APT_RETRY_ATTEMPTS:-5}"
  local delay="${APT_RETRY_DELAY:-10}"
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    wait_for_apt_locks || return 1
    if "$@"; then
      return 0
    fi
    log "APT command failed (attempt ${attempt}/${attempts}), retrying in ${delay}s ..."
    sleep "${delay}"
  done
  printf 'ERROR: APT command failed after retries: %s\n' "$*" >&2
  return 1
}

# True when the file starts with the local-file-header magic "PK\x03\x04".
# Uses od (coreutils) rather than xxd, which is not installed by default on
# Ubuntu 24.04.
has_zip_magic() {
  local path="$1"
  [[ -s "${path}" ]] || return 1
  [[ "$(head -c 4 "${path}" | od -An -tx1 | tr -d ' \n')" == "504b0304" ]]
}

# True when the file is a readable, complete ZIP archive.
is_valid_zip_archive() {
  local path="$1"
  has_zip_magic "${path}" || return 1
  unzip -tqq "${path}" >/dev/null 2>&1
}

# Load KEY=VALUE pairs from a dotenv-style file without overriding variables
# that are already set in the environment. Blank lines and # comments are
# ignored, as are inline " # comments" after unquoted values; surrounding
# single/double quotes are stripped. Values are taken
# literally (no command substitution or expansion), so the file cannot run code.
# Usage: load_env_file <path>
load_env_file() {
  local path="$1" line key value
  [[ -f "${path}" ]] || return 0
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    line="${line#export }"
    [[ "${line}" == *=* ]] || continue
    key="${line%%=*}"
    value="${line#*=}"
    [[ "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || {
      printf 'WARNING: ignoring invalid line in %s: %s\n' "${path}" "${key}" >&2
      continue
    }
    if [[ "${value}" =~ ^\"(.*)\"$ || "${value}" =~ ^\'(.*)\'$ ]]; then
      value="${BASH_REMATCH[1]}"
    else
      # Unquoted: drop an inline " # comment" and trailing whitespace.
      value="${value%%[[:space:]]#*}"
      value="${value%"${value##*[![:space:]]}"}"
    fi
    [[ -n "${!key+x}" ]] && continue
    printf -v "${key}" '%s' "${value}"
    export "${key?}"
  done <"${path}"
}

# Extract the value of a hidden <input name="..."> field from an HTML page.
# Usage: html_input_value <name> <<<"$html"
html_input_value() {
  local name="$1"
  sed -n "s/.*name=\"${name}\" value=\"\([^\"]*\)\".*/\1/p" | head -n1
}

# Download a (possibly large) Google Drive file to <dest>.
#
# Large files are served behind a "can't scan for viruses" interstitial page
# that carries `confirm` and `uuid` tokens; small files are served directly.
# Both cases are handled. The result is validated as a ZIP archive.
# Usage: download_gdrive_zip <url> <dest>
download_gdrive_zip() {
  local url="$1"
  local dest="$2"
  local page="${dest}.page"
  local tmp="${dest}.part"
  local confirm uuid

  mkdir -p "$(dirname "${dest}")"
  rm -f "${page}"

  log "Requesting dataset archive from Google Drive ..."
  if ! curl -L --fail -sS --retry 5 --retry-delay 5 -o "${page}" "${url}"; then
    rm -f "${page}"
    printf 'ERROR: Failed to fetch the dataset URL.\n' >&2
    return 1
  fi

  if has_zip_magic "${page}"; then
    # Small file: Google served the archive directly.
    mv -f "${page}" "${tmp}"
  else
    # Interstitial confirmation page: pull the tokens out and re-request.
    confirm="$(html_input_value confirm <"${page}")"
    uuid="$(html_input_value uuid <"${page}")"
    rm -f "${page}"
    if [[ -z "${confirm}" || -z "${uuid}" ]]; then
      printf 'ERROR: Could not extract Google Drive confirm token/uuid.\n' >&2
      printf 'Download the archive manually and set DATASET_ARCHIVE, or check DATASET_URL.\n' >&2
      return 1
    fi
    # A previous interrupted run leaves ${tmp} behind; -C - resumes it.
    log "Downloading dataset archive (resumable; this can take a while) ..."
    if ! curl -L --fail --retry 5 --retry-delay 5 -C - -o "${tmp}" \
      "${url}&confirm=${confirm}&uuid=${uuid}"; then
      printf 'ERROR: Dataset download failed; re-run to resume.\n' >&2
      return 1
    fi
  fi

  if ! is_valid_zip_archive "${tmp}"; then
    rm -f "${tmp}"
    printf 'ERROR: Downloaded file is not a valid ZIP archive (removed).\n' >&2
    return 1
  fi
  mv -f "${tmp}" "${dest}"
}

# Print the global step encoded in a Lightning checkpoint URL or filename
# ("...step=839202.ckpt" or URL-encoded "...step%3D839202.ckpt"), or nothing.
checkpoint_step_from_name() {
  local name="$1"
  if [[ "${name}" =~ step(%3D|=)([0-9]+) ]]; then
    printf '%s\n' "${BASH_REMATCH[2]}"
  fi
}

# Lightning's max_steps is a global step count, so fine-tuning from a
# checkpoint that is already past MAX_STEPS would stop immediately. In that
# case treat MAX_STEPS as "additional" steps on top of the base step.
# Usage: effective_max_steps <max_steps> [base_step]
effective_max_steps() {
  local max_steps="$1"
  local base_step="${2:-}"
  [[ "${max_steps}" =~ ^[0-9]+$ ]] || {
    printf 'ERROR: MAX_STEPS must be a non-negative integer, got: %s\n' "${max_steps}" >&2
    return 1
  }
  if [[ -n "${base_step}" ]] && ((max_steps <= base_step)); then
    printf '%s\n' "$((base_step + max_steps))"
  else
    printf '%s\n' "${max_steps}"
  fi
}

# Print the most recently modified *.ckpt under <dir>, ignoring <exclude>
# (the downloaded base checkpoint). Prints nothing if none exist.
# Usage: latest_checkpoint <dir> [exclude_path]
latest_checkpoint() {
  local dir="$1"
  local exclude="${2:-}"
  local newest="" newest_mtime=-1 path mtime
  while IFS= read -r -d '' path; do
    [[ -n "${exclude}" && "${path}" -ef "${exclude}" ]] && continue
    mtime="$(stat -c %Y "${path}")"
    if ((mtime > newest_mtime)); then
      newest="${path}"
      newest_mtime="${mtime}"
    fi
  done < <(find "${dir}" -type f -name '*.ckpt' -print0 2>/dev/null)
  [[ -n "${newest}" ]] && printf '%s\n' "${newest}"
  return 0
}
