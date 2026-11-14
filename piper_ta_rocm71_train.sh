#!/usr/bin/env bash
# Fine-tune a Piper TTS voice for Tamil on an AMD GPU (ROCm 7.1, Ubuntu 24.04)
# using the IITM IndicTTS Tamil female single-speaker recordings.
#
# Run with --help for usage. Configuration comes from environment variables or
# a .env file next to this script (see .env.example). Every stage is
# idempotent, so re-running after an interruption resumes where it left off.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

PROG="$(basename "$0")"

# Pipeline stages, in order, with a one-line description each.
STAGES=(packages dataset piper prepare train export)
declare -A STAGE_TITLE=(
  [packages]="Install system packages"
  [dataset]="Download and extract the IndicTTS dataset"
  [piper]="Install Piper and ROCm PyTorch, check the GPU"
  [prepare]="Build audio/ and metadata.csv"
  [train]="Fine-tune from the base checkpoint"
  [export]="Export ONNX voice and synthesise a sample"
)

usage() {
  cat <<EOF
Fine-tune a Piper TTS voice for Tamil on an AMD GPU (ROCm 7.1, Ubuntu 24.04)
using the IITM IndicTTS Tamil female single-speaker recordings.

${C_BOLD}Usage:${C_RESET}
  bash ${PROG} [options]

${C_BOLD}Options:${C_RESET}
  --from STAGE     Start at STAGE and run the rest (earlier outputs must exist)
  --only STAGE     Run only STAGE
  --list-stages    List the pipeline stages and exit
  --print-config   Print the resolved configuration and exit (changes nothing)
  --preflight      Check tools, GPU, disk space and configuration, then exit
  -h, --help       Show this help and exit

${C_BOLD}Stages:${C_RESET}
  ${STAGES[*]}

${C_BOLD}Configuration:${C_RESET}
  Set environment variables, or copy .env.example to .env next to this
  script. Values already set in the shell override .env. Frequently used:
    WORK_ROOT  DATASET_ARCHIVE  MAX_STEPS  BATCH_SIZE  SKIP_APT  SETUP_SCRATCH
  See README.md for every variable.

${C_BOLD}Examples:${C_RESET}
  # Full pipeline
  bash ${PROG}
  # Override settings for one run
  MAX_STEPS=60000 BATCH_SIZE=32 bash ${PROG}
  # Check the host before a long run
  bash ${PROG} --preflight
  # Resume after a crash (training continues from the newest checkpoint)
  bash ${PROG} --from train
  # Re-export the newest checkpoint
  bash ${PROG} --only export
EOF
}

list_stages() {
  local i=1 name
  for name in "${STAGES[@]}"; do
    printf '  %d. %-9s %s\n' "${i}" "${name}" "${STAGE_TITLE[${name}]}"
    i=$((i + 1))
  done
}

is_stage() {
  local name
  for name in "${STAGES[@]}"; do [[ "${name}" == "$1" ]] && return 0; done
  return 1
}

# ---------------- Argument parsing ----------------
setup_colors 1 # --help and the other info actions write to stdout
ACTION=run
FROM_STAGE=""
ONLY_STAGE=""
while (($#)); do
  case "$1" in
    -h | --help)
      usage
      exit 0
      ;;
    --list-stages)
      list_stages
      exit 0
      ;;
    --print-config) ACTION=print-config ;;
    --preflight | --check) ACTION=preflight ;;
    --from | --only)
      [[ $# -ge 2 ]] || die "$1 needs a stage name. Stages: ${STAGES[*]}"
      is_stage "$2" || die "Unknown stage '$2'. Stages: ${STAGES[*]}"
      if [[ "$1" == --from ]]; then FROM_STAGE="$2"; else ONLY_STAGE="$2"; fi
      shift
      ;;
    --from=* | --only=*)
      is_stage "${1#*=}" || die "Unknown stage '${1#*=}'. Stages: ${STAGES[*]}"
      if [[ "$1" == --from=* ]]; then FROM_STAGE="${1#*=}"; else ONLY_STAGE="${1#*=}"; fi
      ;;
    *) die "Unknown option '$1'. Run 'bash ${PROG} --help' for usage." ;;
  esac
  shift
done
[[ -z "${FROM_STAGE}" || -z "${ONLY_STAGE}" ]] || die "Use either --from or --only, not both."
[[ "${ACTION}" != run ]] || setup_colors 1 2

# ---------------- Configuration ----------------
CONFIG_VARS=(
  WORK_ROOT DATA_ZIP_DIR DATASET_ARCHIVE DATASET_URL PIPER_REPO_URL PIPER_REPO_DIR
  TORCH_INDEX_URL BASE_CKPT_URL VOICE_NAME MODEL_BASENAME ESPEAK_VOICE
  INCLUDE_ADDITIONAL BATCH_SIZE MAX_STEPS VAL_SPLIT NUM_TEST_EXAMPLES PRECISION
  VAL_CHECK_INTERVAL MIN_ROWS RESUME_TRAINING SAMPLE_TEXT SKIP_APT SETUP_SCRATCH
  MIN_FREE_GB
)
# Remember where each value comes from (shell, .env or default) for --print-config.
declare -A CONFIG_SOURCE=()
for _var in "${CONFIG_VARS[@]}"; do
  [[ -n "${!_var+x}" ]] && CONFIG_SOURCE[${_var}]=shell
done
[[ -n "${DATASET_URL+x}" || -z "${DATASET_BASE_URL+x}" ]] || CONFIG_SOURCE[DATASET_URL]=shell

ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/.env}"
if [[ -f "${ENV_FILE}" ]]; then
  [[ "${ACTION}" == run ]] && log "Loading configuration from ${ENV_FILE}"
  load_env_file "${ENV_FILE}"
  for _var in "${CONFIG_VARS[@]}"; do
    [[ -n "${!_var+x}" && -z "${CONFIG_SOURCE[${_var}]:-}" ]] && CONFIG_SOURCE[${_var}]=.env
  done
  [[ -n "${CONFIG_SOURCE[DATASET_URL]:-}" || -z "${DATASET_BASE_URL+x}" ]] || CONFIG_SOURCE[DATASET_URL]=.env
fi
unset _var

WORK_ROOT="${WORK_ROOT:-/scratch/piper_tamil}"
DATA_ZIP_DIR="${DATA_ZIP_DIR:-${HOME}/tamil_female_mono}"
DATASET_ARCHIVE="${DATASET_ARCHIVE:-${DATA_ZIP_DIR}/tamil_female_mono.zip}"
# Google Drive download URL for tamil_female_mono.zip. The previous default is
# kept for backwards compatibility; set DATASET_URL to use your own copy.
DATASET_URL="${DATASET_URL:-${DATASET_BASE_URL:-https://drive.usercontent.google.com/download?id=1lTi5J7iX2dlo8GB16OhnDdqXlJyfuHjI&export=download&authuser=0}}"
PIPER_REPO_URL="${PIPER_REPO_URL:-https://github.com/OHF-Voice/piper1-gpl.git}"
PIPER_REPO_DIR="${PIPER_REPO_DIR:-${WORK_ROOT}/piper1-gpl}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/rocm7.1}"
BASE_CKPT_URL="${BASE_CKPT_URL:-https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/te/te_IN/maya/medium/epoch%3D3277-step%3D839202.ckpt}"

VOICE_NAME="${VOICE_NAME:-ta_IN_iitm_female_s1}"
MODEL_BASENAME="${MODEL_BASENAME:-ta_IN-iitm-female-s1-medium}"
ESPEAK_VOICE="${ESPEAK_VOICE:-ta}"

INCLUDE_ADDITIONAL="${INCLUDE_ADDITIONAL:-0}" # 0 = clean Tamil only, 1 = include Phase2 Additional_data
BATCH_SIZE="${BATCH_SIZE:-48}"
MAX_STEPS="${MAX_STEPS:-120000}"
VAL_SPLIT="${VAL_SPLIT:-0.05}"
NUM_TEST_EXAMPLES="${NUM_TEST_EXAMPLES:-16}"
PRECISION="${PRECISION:-bf16-mixed}" # alternatives: 16-mixed, 32-true
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-50}"
MIN_ROWS="${MIN_ROWS:-3000}"
RESUME_TRAINING="${RESUME_TRAINING:-1}" # 1 = continue from the newest trained checkpoint
SAMPLE_TEXT="${SAMPLE_TEXT:-வணக்கம். இது தமிழ் பைப்பர் சோதனை குரல்.}"

SKIP_APT="${SKIP_APT:-0}"           # 1 = do not install system packages
SETUP_SCRATCH="${SETUP_SCRATCH:-1}" # 1 = mount a DOSCRATCH-labelled disk at /scratch
MIN_FREE_GB="${MIN_FREE_GB:-50}"    # preflight warns below this much free disk

PHASE2_ZIP="IndicTTS_Phase2_Tamil_fem_Speaker1_mono.zip"
PHASE3_ZIP="IndicTTS_Phase3_Tamil_fem_Speaker1_mono.zip"

APT_PACKAGES=(
  git build-essential cmake ninja-build python3-venv ffmpeg libsndfile1
  unzip rsync curl tmux espeak-ng psmisc
)
# Commands the pipeline calls directly (all provided by APT_PACKAGES).
REQUIRED_COMMANDS=(git curl unzip rsync python3 espeak-ng ffmpeg)

if [[ "${EUID}" -eq 0 ]]; then
  SUDO=()
else
  SUDO=(sudo)
fi

# Print every configuration problem at once, so they can be fixed in one go.
config_errors() {
  is_uint "${MAX_STEPS}" || echo "MAX_STEPS must be a non-negative integer, got: '${MAX_STEPS}'"
  is_positive_int "${BATCH_SIZE}" || echo "BATCH_SIZE must be a positive integer, got: '${BATCH_SIZE}'"
  is_uint "${NUM_TEST_EXAMPLES}" ||
    echo "NUM_TEST_EXAMPLES must be a non-negative integer, got: '${NUM_TEST_EXAMPLES}'"
  is_positive_int "${VAL_CHECK_INTERVAL}" ||
    echo "VAL_CHECK_INTERVAL must be a positive integer, got: '${VAL_CHECK_INTERVAL}'"
  is_uint "${MIN_ROWS}" || echo "MIN_ROWS must be a non-negative integer, got: '${MIN_ROWS}'"
  is_uint "${MIN_FREE_GB}" || echo "MIN_FREE_GB must be a non-negative integer, got: '${MIN_FREE_GB}'"
  is_fraction "${VAL_SPLIT}" || echo "VAL_SPLIT must be a decimal between 0 and 1 (e.g. 0.05), got: '${VAL_SPLIT}'"
  case "${PRECISION}" in
    bf16-mixed | 16-mixed | 32-true | 32 | 16 | bf16) ;;
    *) echo "PRECISION must be one of bf16-mixed, 16-mixed, 32-true, got: '${PRECISION}'" ;;
  esac
  local flag
  for flag in INCLUDE_ADDITIONAL RESUME_TRAINING SKIP_APT SETUP_SCRATCH; do
    is_bool01 "${!flag}" || echo "${flag} must be 0 or 1, got: '${!flag}'"
  done
  [[ -n "${MODEL_BASENAME}" && "${MODEL_BASENAME}" != */* ]] ||
    echo "MODEL_BASENAME must be a non-empty file name without '/', got: '${MODEL_BASENAME}'"
}

validate_config() {
  local errors
  errors="$(config_errors)"
  [[ -z "${errors}" ]] && return 0
  error "Invalid configuration:"
  printf '  - %s\n' "${errors//$'\n'/$'\n'  - }" >&2
  if [[ -f "${ENV_FILE}" ]]; then
    printf 'Fix the values above (in the shell or %s) and re-run.\n' "${ENV_FILE}" >&2
  else
    printf 'Fix the values above and re-run.\n' >&2
  fi
  exit 1
}

has_doscratch_label() {
  command -v lsblk >/dev/null 2>&1 && lsblk -o LABEL 2>/dev/null | grep -qw 'DOSCRATCH'
}

# Fall back to /opt/piper_tamil when WORK_ROOT is on an unmounted /scratch.
# Side-effect free apart from updating the path variables.
resolve_work_root() {
  if [[ "${WORK_ROOT}" == /scratch/* ]] && ! mountpoint -q /scratch 2>/dev/null; then
    WORK_ROOT_NOTE="/scratch is not mounted; using /opt/piper_tamil instead of ${WORK_ROOT}"
    WORK_ROOT="/opt/piper_tamil"
    [[ -n "${CONFIG_SOURCE[PIPER_REPO_DIR]:-}" ]] || PIPER_REPO_DIR="${WORK_ROOT}/piper1-gpl"
  fi
  DATA_ROOT="${WORK_ROOT}/data_ta_single_speaker"
  RAW_DIR="${DATA_ROOT}/raw"
  AUDIO_DIR="${DATA_ROOT}/audio"
  META_PATH="${DATA_ROOT}/metadata.csv"
  RUN_ROOT="${WORK_ROOT}/runs/${MODEL_BASENAME}"
  BASE_CKPT="${RUN_ROOT}/base.ckpt"
  MODEL_PATH="${RUN_ROOT}/${MODEL_BASENAME}.onnx"
}
WORK_ROOT_NOTE=""

# ---------------- Scratch disk setup ----------------
setup_scratch_disk() {
  if [[ "${SETUP_SCRATCH}" == "1" ]] && has_doscratch_label; then
    log "Mounting DOSCRATCH volume at /scratch"
    "${SUDO[@]}" mkdir -p /scratch
    if ! grep -q 'LABEL=DOSCRATCH' /etc/fstab; then
      echo 'LABEL=DOSCRATCH /scratch ext4 discard,errors=remount-ro 0 2' |
        "${SUDO[@]}" tee -a /etc/fstab >/dev/null
    fi
    mountpoint -q /scratch || "${SUDO[@]}" mount /scratch || true
  fi
  resolve_work_root
  [[ -z "${WORK_ROOT_NOTE}" ]] || log "${WORK_ROOT_NOTE}"
}

# ---------------- --print-config ----------------
print_config() {
  local var value src
  printf '%sResolved configuration%s (source: shell, .env or default)\n\n' "${C_BOLD}" "${C_RESET}"
  for var in "${CONFIG_VARS[@]}"; do
    value="${!var}"
    src="${CONFIG_SOURCE[${var}]:-default}"
    printf '  %-19s %s%-8s%s %s\n' "${var}" "${C_DIM}" "${src}" "${C_RESET}" "${value}"
  done
  printf '\n%sDerived paths%s\n\n' "${C_BOLD}" "${C_RESET}"
  [[ -z "${WORK_ROOT_NOTE}" ]] || printf '  note: %s\n' "${WORK_ROOT_NOTE}"
  printf '  %-19s %s\n' "work root" "${WORK_ROOT}" \
    "metadata" "${META_PATH}" \
    "audio" "${AUDIO_DIR}" \
    "run directory" "${RUN_ROOT}" \
    "voice model" "${MODEL_PATH}" \
    "config file" "${ENV_FILE}$([[ -f "${ENV_FILE}" ]] || echo ' (not present)')"
}

# ---------------- --preflight ----------------
PREFLIGHT_FAILS=0
PREFLIGHT_WARNS=0
check_ok() { printf '  %sok%s    %s\n' "${C_GREEN}" "${C_RESET}" "$*"; }
check_warn() {
  printf '  %swarn%s  %s\n' "${C_YELLOW}" "${C_RESET}" "$*"
  PREFLIGHT_WARNS=$((PREFLIGHT_WARNS + 1))
}
check_note() { printf '  %snote%s  %s\n' "${C_BLUE}" "${C_RESET}" "$*"; }
check_fail() {
  printf '  %sFAIL%s  %s\n' "${C_RED}" "${C_RESET}" "$*"
  PREFLIGHT_FAILS=$((PREFLIGHT_FAILS + 1))
}

preflight() {
  local errors cmd free os_id="" os_version="" py
  printf '%sPreflight checks%s\n\n' "${C_BOLD}" "${C_RESET}"

  errors="$(config_errors)"
  if [[ -z "${errors}" ]]; then
    check_ok "configuration values are valid"
  else
    while IFS= read -r line; do check_fail "${line}"; done <<<"${errors}"
  fi

  if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    os_id="$(. /etc/os-release && echo "${ID:-}")"
    # shellcheck disable=SC1091
    os_version="$(. /etc/os-release && echo "${VERSION_ID:-}")"
  fi
  if [[ "${os_id}" == ubuntu && "${os_version}" == 24.04 ]]; then
    check_ok "Ubuntu 24.04"
  else
    check_warn "OS is '${os_id:-unknown} ${os_version}'; only Ubuntu 24.04 is tested"
  fi

  if [[ "${SKIP_APT}" == 1 ]]; then
    check_ok "SKIP_APT=1: system packages will not be installed"
  elif [[ "${EUID}" -eq 0 ]]; then
    check_ok "running as root (apt-get available)"
  elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    check_ok "passwordless sudo available"
  else
    check_warn "not root and no passwordless sudo; apt-get will prompt or fail (set SKIP_APT=1 to skip)"
  fi

  for cmd in "${REQUIRED_COMMANDS[@]}"; do
    if [[ "${cmd}" == python3 ]] && command -v python3 >/dev/null 2>&1; then
      continue # version checked below
    elif command -v "${cmd}" >/dev/null 2>&1; then
      check_ok "${cmd} found"
    elif [[ "${SKIP_APT}" == 1 ]]; then
      check_fail "${cmd} not found and SKIP_APT=1 (install it, or unset SKIP_APT)"
    else
      check_note "${cmd} missing; the 'packages' stage will install it"
    fi
  done

  if command -v python3 >/dev/null 2>&1; then
    py="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    if python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
      check_ok "python3 ${py}"
    else
      check_fail "python3 ${py} is too old; 3.10 or newer is required"
    fi
  fi

  if command -v rocminfo >/dev/null 2>&1 && rocminfo 2>/dev/null | grep -q 'gfx[0-9]'; then
    check_ok "ROCm GPU agent reported by rocminfo"
  elif command -v rocminfo >/dev/null 2>&1; then
    check_warn "rocminfo found but reports no GPU agent; the 'piper' stage will stop"
  else
    check_warn "rocminfo not found; ROCm may not be installed (the 'piper' stage checks the GPU)"
  fi

  if free="$(free_space_gib "${WORK_ROOT}")"; then
    if ((free >= MIN_FREE_GB)); then
      check_ok "${free} GiB free for ${WORK_ROOT}"
    else
      check_warn "only ${free} GiB free for ${WORK_ROOT} (MIN_FREE_GB=${MIN_FREE_GB})"
    fi
  else
    check_warn "could not determine free disk space for ${WORK_ROOT}"
  fi

  if [[ -s "${DATA_ZIP_DIR}/${PHASE2_ZIP}" && -s "${DATA_ZIP_DIR}/${PHASE3_ZIP}" ]]; then
    check_ok "phase archives already in ${DATA_ZIP_DIR}"
  elif [[ -s "${DATASET_ARCHIVE}" ]]; then
    check_ok "dataset archive present: ${DATASET_ARCHIVE}"
  else
    check_note "dataset will be downloaded to ${DATASET_ARCHIVE}"
  fi

  printf '\n%d failed, %d warning(s)\n' "${PREFLIGHT_FAILS}" "${PREFLIGHT_WARNS}"
  ((PREFLIGHT_FAILS == 0))
}

# ---------------- Stage: packages ----------------
stage_packages() {
  if [[ "${SKIP_APT}" == "1" ]]; then
    log "SKIP_APT=1: not installing system packages"
    local cmd missing=()
    for cmd in "${REQUIRED_COMMANDS[@]}"; do
      command -v "${cmd}" >/dev/null 2>&1 || missing+=("${cmd}")
    done
    ((${#missing[@]} == 0)) || die "SKIP_APT=1 but these commands are missing: ${missing[*]}"
    return 0
  fi
  log "Installing: ${APT_PACKAGES[*]}"
  run_apt_cmd "${SUDO[@]}" apt-get update
  run_apt_cmd "${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive \
    apt-get install -y "${APT_PACKAGES[@]}"
}

# ---------------- Stage: dataset ----------------
stage_dataset() {
  mkdir -p "${DATA_ZIP_DIR}"

  if [[ -s "${DATA_ZIP_DIR}/${PHASE2_ZIP}" && -s "${DATA_ZIP_DIR}/${PHASE3_ZIP}" ]]; then
    log "Phase archives already present in ${DATA_ZIP_DIR}; skipping download"
    return 0
  fi

  if [[ -e "${DATASET_ARCHIVE}" ]] && ! is_valid_zip_archive "${DATASET_ARCHIVE}"; then
    warn "Existing dataset archive is invalid or incomplete; re-downloading"
    rm -f "${DATASET_ARCHIVE}"
  fi
  if [[ ! -s "${DATASET_ARCHIVE}" ]]; then
    download_gdrive_zip "${DATASET_URL}" "${DATASET_ARCHIVE}"
  else
    log "Using existing dataset archive: ${DATASET_ARCHIVE}"
  fi

  log "Extracting phase archives from ${DATASET_ARCHIVE}"
  unzip -j -o "${DATASET_ARCHIVE}" "${PHASE2_ZIP}" "${PHASE3_ZIP}" -d "${DATA_ZIP_DIR}" >/dev/null ||
    die "Could not extract ${PHASE2_ZIP} and ${PHASE3_ZIP} from ${DATASET_ARCHIVE}." \
      "Check that the archive contains both files ('unzip -l ${DATASET_ARCHIVE}')."
  if [[ ! -s "${DATA_ZIP_DIR}/${PHASE2_ZIP}" || ! -s "${DATA_ZIP_DIR}/${PHASE3_ZIP}" ]]; then
    die "Failed to extract ${PHASE2_ZIP} and ${PHASE3_ZIP} from ${DATASET_ARCHIVE}"
  fi
}

# ---------------- Stage: piper ----------------
stage_piper() {
  if [[ ! -d "${PIPER_REPO_DIR}/.git" ]]; then
    log "Cloning Piper into ${PIPER_REPO_DIR}"
    git clone "${PIPER_REPO_URL}" "${PIPER_REPO_DIR}"
  else
    log "Using existing Piper checkout: ${PIPER_REPO_DIR}"
  fi

  cd "${PIPER_REPO_DIR}"
  [[ -x .venv/bin/python ]] || python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate

  log "Installing Python packages into ${PIPER_REPO_DIR}/.venv"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install --index-url "${TORCH_INDEX_URL}" torch
  python -m pip install \
    lightning tensorboard tensorboardX "jsonargparse[signatures]>=4.27.7" \
    pathvalidate onnx onnxscript "pysilero-vad<3" cython scikit-build "librosa<1" onnxruntime
  python -m pip install -e . --no-deps

  log "Building monotonic_align extension"
  ./build_monotonic_align.sh
  python setup.py build_ext --inplace

  python - <<'PY'
import torch
print("torch:", torch.__version__)
print("gpu visible:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit(
        "ERROR: ROCm GPU is not visible to PyTorch. Check 'rocminfo', that your user is in "
        "the 'render' and 'video' groups, and that TORCH_INDEX_URL matches your ROCm version."
    )
print("device:", torch.cuda.get_device_name(0))
PY
}

# Activate the Piper virtualenv when starting after the 'piper' stage.
ensure_piper_env() {
  [[ "${VIRTUAL_ENV:-}" == "${PIPER_REPO_DIR}/.venv" ]] && return 0
  [[ -x "${PIPER_REPO_DIR}/.venv/bin/python" ]] ||
    die "Piper is not installed in ${PIPER_REPO_DIR}. Run the 'piper' stage first" \
      "(bash ${PROG} --from piper)."
  cd "${PIPER_REPO_DIR}"
  # shellcheck disable=SC1091
  source .venv/bin/activate
}

# ---------------- Stage: prepare ----------------
stage_prepare() {
  ensure_piper_env
  mkdir -p "${RAW_DIR}" "${AUDIO_DIR}"

  local f
  for f in "${PHASE2_ZIP}" "${PHASE3_ZIP}"; do
    [[ -f "${DATA_ZIP_DIR}/${f}" ]] ||
      die "Missing ${DATA_ZIP_DIR}/${f}. Run the 'dataset' stage first (bash ${PROG} --from dataset)."
    rsync -a "${DATA_ZIP_DIR}/${f}" "${RAW_DIR}/${f}"
  done

  local extra=()
  [[ "${INCLUDE_ADDITIONAL}" == "1" ]] && extra+=(--include-additional)

  log "Pairing transcripts with audio (this copies every WAV once)"
  python "${SCRIPT_DIR}/scripts/prepare_metadata.py" \
    --raw-dir "${RAW_DIR}" \
    --audio-dir "${AUDIO_DIR}" \
    --metadata "${META_PATH}" \
    --phase2-zip "${PHASE2_ZIP}" \
    --phase3-zip "${PHASE3_ZIP}" \
    --min-rows "${MIN_ROWS}" \
    "${extra[@]}"
}

# ---------------- Stage: train ----------------
stage_train() {
  ensure_piper_env
  [[ -s "${META_PATH}" ]] ||
    die "No metadata at ${META_PATH}. Run the 'prepare' stage first (bash ${PROG} --from prepare)."
  mkdir -p "${RUN_ROOT}"

  if [[ ! -s "${BASE_CKPT}" ]]; then
    log "Downloading base checkpoint"
    curl -L --retry 5 --retry-delay 5 --fail -o "${BASE_CKPT}.part" "${BASE_CKPT_URL}"
    mv -f "${BASE_CKPT}.part" "${BASE_CKPT}"
  fi

  local base_step effective_steps train_log start_ckpt resume_ckpt=""
  base_step="$(checkpoint_step_from_name "${BASE_CKPT_URL}")"
  effective_steps="$(effective_max_steps "${MAX_STEPS}" "${base_step}")"
  if [[ "${effective_steps}" != "${MAX_STEPS}" ]]; then
    log "MAX_STEPS=${MAX_STEPS} is <= base checkpoint step ${base_step};" \
      "training to step ${effective_steps} (base + ${MAX_STEPS} additional)."
  fi

  start_ckpt="${BASE_CKPT}"
  if [[ "${RESUME_TRAINING}" == "1" ]]; then
    resume_ckpt="$(latest_checkpoint "${RUN_ROOT}" "${BASE_CKPT}")"
  fi
  if [[ -n "${resume_ckpt}" ]]; then
    start_ckpt="${resume_ckpt}"
    log "Resuming from ${resume_ckpt} (set RESUME_TRAINING=0 to start from the base checkpoint)"
  else
    log "Starting from base checkpoint ${BASE_CKPT}"
  fi

  train_log="${RUN_ROOT}/train_$(date +%Y%m%d_%H%M%S).log"
  log "Training to global step ${effective_steps}; log: ${train_log}"
  log "Watch progress: tensorboard --logdir ${RUN_ROOT}"
  python -m piper.train fit \
    --data.voice_name "${VOICE_NAME}" \
    --data.csv_path "${META_PATH}" \
    --data.audio_dir "${AUDIO_DIR}" \
    --data.cache_dir "${RUN_ROOT}/cache" \
    --data.config_path "${RUN_ROOT}/${MODEL_BASENAME}.onnx.json" \
    --data.espeak_voice "${ESPEAK_VOICE}" \
    --model.sample_rate 22050 \
    --data.batch_size "${BATCH_SIZE}" \
    --data.validation_split "${VAL_SPLIT}" \
    --data.num_test_examples "${NUM_TEST_EXAMPLES}" \
    --ckpt_path "${start_ckpt}" \
    --trainer.default_root_dir "${RUN_ROOT}" \
    --trainer.accelerator gpu \
    --trainer.devices 1 \
    --trainer.precision "${PRECISION}" \
    --trainer.max_steps "${effective_steps}" \
    --trainer.val_check_interval "${VAL_CHECK_INTERVAL}" \
    --trainer.log_every_n_steps 20 \
    2>&1 | tee "${train_log}"
}

# ---------------- Stage: export ----------------
stage_export() {
  ensure_piper_env
  local latest_ckpt
  latest_ckpt="$(latest_checkpoint "${RUN_ROOT}" "${BASE_CKPT}")"
  [[ -n "${latest_ckpt}" ]] ||
    die "No trained checkpoint found in ${RUN_ROOT}. Run the 'train' stage first (bash ${PROG} --from train)."
  log "Exporting ${latest_ckpt}"

  python -m piper.train.export_onnx \
    --checkpoint "${latest_ckpt}" \
    --output-file "${MODEL_PATH}"

  log "Synthesising sample.wav"
  printf '%s\n' "${SAMPLE_TEXT}" |
    python -m piper --model "${MODEL_PATH}" --output_file "${RUN_ROOT}/sample.wav"
  EXPORTED=1
}

# ---------------- Main ----------------
CURRENT_STAGE=""
EXPORTED=0

on_exit() {
  local status=$?
  if ((status != 0)) && [[ -n "${CURRENT_STAGE}" ]]; then
    printf '\n%sStage %s failed%s (exit %d) after %s.\n' "${C_RED}${C_BOLD}" "'${CURRENT_STAGE}'" \
      "${C_RESET}" "${status}" "$(format_duration "${SECONDS}")" >&2
    printf 'Fix the error above, then resume with: bash %s --from %s\n' "${PROG}" "${CURRENT_STAGE}" >&2
  fi
}

selected_stages() {
  local name started=0
  if [[ -n "${ONLY_STAGE}" ]]; then
    echo "${ONLY_STAGE}"
    return
  fi
  for name in "${STAGES[@]}"; do
    [[ -z "${FROM_STAGE}" || "${name}" == "${FROM_STAGE}" ]] && started=1
    ((started)) && echo "${name}"
  done
}

print_summary() {
  printf '\n%sFinished in %s.%s\n' "${C_GREEN}${C_BOLD}" "$(format_duration "${SECONDS}")" "${C_RESET}"
  if ((EXPORTED)); then
    printf '  Model:  %s\n' "${MODEL_PATH}"
    printf '  Config: %s\n' "${MODEL_PATH}.json"
    printf '  Sample: %s\n' "${RUN_ROOT}/sample.wav"
    printf '\nTry it:\n  echo "வணக்கம்" | %s/.venv/bin/python -m piper --model %s --output_file out.wav\n' \
      "${PIPER_REPO_DIR}" "${MODEL_PATH}"
  fi
}

main() {
  case "${ACTION}" in
    print-config)
      resolve_work_root
      print_config
      return 0
      ;;
    preflight)
      resolve_work_root
      preflight
      return
      ;;
  esac

  validate_config
  local stages=() name i=0 t0
  mapfile -t stages < <(selected_stages)

  trap on_exit EXIT
  setup_scratch_disk
  mkdir -p "${WORK_ROOT}"
  log "Work root: ${WORK_ROOT}"
  log "Stages: ${stages[*]}"

  for name in "${stages[@]}"; do
    i=$((i + 1))
    CURRENT_STAGE="${name}"
    stage_header "${i}" "${#stages[@]}" "${name}: ${STAGE_TITLE[${name}]}"
    t0=${SECONDS}
    "stage_${name}"
    log "${C_GREEN}done${C_RESET} (${name}, $(format_duration $((SECONDS - t0))))"
  done
  CURRENT_STAGE=""
  print_summary
}

main "$@"
