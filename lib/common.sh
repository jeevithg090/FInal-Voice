#!/usr/bin/env bash
# Reusable helpers for piper_ta_rocm71_train.sh.
#
# This file is meant to be sourced. It only defines functions and has no side
# effects, so it can be exercised in isolation by tests/test_common.sh.

# shellcheck disable=SC2034  # PIPER_COMMON_LOADED is read by callers.
PIPER_COMMON_LOADED=1

# Terminal styling. Colours are used only when every given file descriptor
# (default: 2, stderr) is a terminal and NO_COLOR (https://no-color.org) is
# unset; otherwise every code is empty.
# Usage: setup_colors [fd...]
setup_colors() {
  local fd use=1
  (($#)) || set -- 2
  for fd in "$@"; do [[ -t "${fd}" ]] || use=0; done
  [[ -z "${NO_COLOR:-}" && "${TERM:-}" != "dumb" ]] || use=0
  if ((use)); then
    C_RESET=$'\e[0m' C_BOLD=$'\e[1m' C_DIM=$'\e[2m'
    C_RED=$'\e[31m' C_GREEN=$'\e[32m' C_YELLOW=$'\e[33m' C_BLUE=$'\e[34m'
  else
    C_RESET='' C_BOLD='' C_DIM='' C_RED='' C_GREEN='' C_YELLOW='' C_BLUE=''
  fi
}
setup_colors

log() {
  printf '%s[%s]%s %s\n' "${C_DIM}" "$(date +%H:%M:%S)" "${C_RESET}" "$*" >&2
}

warn() {
  printf '%sWARNING:%s %s\n' "${C_YELLOW}${C_BOLD}" "${C_RESET}" "$*" >&2
}

# Print an error without exiting (for functions that return a status).
error() {
  printf '%sERROR:%s %s\n' "${C_RED}${C_BOLD}" "${C_RESET}" "$*" >&2
}

die() {
  error "$@"
  exit 1
}

# Print a stage banner such as "==> [2/6] Download dataset".
# Usage: stage_header <index> <total> <title>
stage_header() {
  printf '\n%s==> [%s/%s] %s%s\n' "${C_BLUE}${C_BOLD}" "$1" "$2" "$3" "${C_RESET}" >&2
}

# Format a number of seconds as "45s", "3m 05s" or "2h 03m 05s".
format_duration() {
  local total="$1" h m s
  h=$((total / 3600))
  m=$((total % 3600 / 60))
  s=$((total % 60))
  if ((h > 0)); then
    printf '%dh %02dm %02ds\n' "${h}" "${m}" "${s}"
  elif ((m > 0)); then
    printf '%dm %02ds\n' "${m}" "${s}"
  else
    printf '%ds\n' "${s}"
  fi
}

# --- Value validators (return 0 when valid) ---
is_uint() { [[ "$1" =~ ^[0-9]+$ ]]; }
is_positive_int() { [[ "$1" =~ ^[0-9]+$ ]] && (("10#$1" > 0)); }
is_bool01() { [[ "$1" == 0 || "$1" == 1 ]]; }
# A decimal strictly between 0 and 1, e.g. 0.05.
is_fraction() {
  [[ "$1" =~ ^0?\.[0-9]+$ ]] && [[ ! "$1" =~ ^0?\.0+$ ]]
}

# Print the closest existing ancestor directory of <path> (or <path> itself).
nearest_existing_dir() {
  local path="$1"
  while [[ ! -d "${path}" && "${path}" != "/" && -n "${path}" ]]; do
    path="$(dirname "${path}")"
  done
  printf '%s\n' "${path:-/}"
}

# Print the free space, in whole GiB, of the filesystem that holds <path>
# (or would hold it, if it does not exist yet).
free_space_gib() {
  local dir kib
  dir="$(nearest_existing_dir "$1")"
  kib="$(df -Pk "${dir}" 2>/dev/null | awk 'NR == 2 { print $4 }')"
  [[ -n "${kib}" ]] || return 1
  printf '%s\n' "$((kib / 1024 / 1024))"
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
      error "Timed out after ${max_wait}s waiting for apt/dpkg locks." \
        "Check 'ps aux | grep -E \"apt|dpkg\"' and re-run."
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
  error "APT command failed after ${attempts} attempts: $*"
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
    error "Failed to fetch the dataset URL (${url})." \
      "Check network access to Google Drive, or download the archive yourself and set DATASET_ARCHIVE."
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
      error "Google Drive did not return the file or a download confirmation page." \
        "The link may be private, deleted or over its download quota." \
        "Download the archive manually and set DATASET_ARCHIVE, or check DATASET_URL."
      return 1
    fi
    # A previous interrupted run leaves ${tmp} behind; -C - resumes it.
    log "Downloading dataset archive (resumable; this can take a while) ..."
    if ! curl -L --fail --retry 5 --retry-delay 5 -C - -o "${tmp}" \
      "${url}&confirm=${confirm}&uuid=${uuid}"; then
      error "Dataset download failed; re-run the script to resume from ${tmp}."
      return 1
    fi
  fi

  if ! is_valid_zip_archive "${tmp}"; then
    rm -f "${tmp}"
    error "Downloaded file is not a valid ZIP archive and was removed." \
      "Check DATASET_URL, or download the archive manually and set DATASET_ARCHIVE."
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
    error "MAX_STEPS must be a non-negative integer, got: ${max_steps}"
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
