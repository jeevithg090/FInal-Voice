#!/usr/bin/env bash
# Fine-tune a Piper TTS voice for Tamil on an AMD GPU (ROCm 7.1, Ubuntu 24.04)
# using the IITM IndicTTS Tamil female single-speaker recordings.
#
# Usage:  bash piper_ta_rocm71_train.sh
# Config: environment variables, or a .env file next to this script
#         (see .env.example). Every step is idempotent, so re-running after an
#         interruption resumes where it left off.
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,8s/^# \{0,1\}//p' "$0"
  echo
  echo "See .env.example and README.md for all configuration variables."
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/.env}"
if [[ -f "${ENV_FILE}" ]]; then
  log "Loading configuration from ${ENV_FILE}"
  load_env_file "${ENV_FILE}"
fi

# ---------------- Configuration ----------------
WORK_ROOT="${WORK_ROOT:-/scratch/piper_tamil}"
DATA_ZIP_DIR="${DATA_ZIP_DIR:-${HOME}/tamil_female_mono}"
DATASET_ARCHIVE="${DATASET_ARCHIVE:-${DATA_ZIP_DIR}/tamil_female_mono.zip}"
# Google Drive download URL for tamil_female_mono.zip. The previous default is
# kept for backwards compatibility; set DATASET_URL to use your own copy.
DATASET_URL="${DATASET_URL:-${DATASET_BASE_URL:-https://drive.usercontent.google.com/download?id=1lTi5J7iX2dlo8GB16OhnDdqXlJyfuHjI&export=download&authuser=0}}"
PIPER_REPO_URL="${PIPER_REPO_URL:-https://github.com/OHF-Voice/piper1-gpl.git}"
PIPER_REPO_DIR="${PIPER_REPO_DIR:-${WORK_ROOT}/piper1-gpl}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/rocm7.1}"

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

SKIP_APT="${SKIP_APT:-0}"           # 1 = do not install system packages
SETUP_SCRATCH="${SETUP_SCRATCH:-1}" # 1 = mount a DOSCRATCH-labelled disk at /scratch

BASE_CKPT_URL="${BASE_CKPT_URL:-https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/te/te_IN/maya/medium/epoch%3D3277-step%3D839202.ckpt}"

PHASE2_ZIP="IndicTTS_Phase2_Tamil_fem_Speaker1_mono.zip"
PHASE3_ZIP="IndicTTS_Phase3_Tamil_fem_Speaker1_mono.zip"
SAMPLE_TEXT="${SAMPLE_TEXT:-வணக்கம். இது தமிழ் பைப்பர் சோதனை குரல்.}"

APT_PACKAGES=(
  git build-essential cmake ninja-build python3-venv ffmpeg libsndfile1
  unzip rsync curl tmux espeak-ng psmisc
)

if [[ "${EUID}" -eq 0 ]]; then
  SUDO=()
else
  SUDO=(sudo)
fi

[[ "${MAX_STEPS}" =~ ^[0-9]+$ ]] || die "MAX_STEPS must be an integer, got: ${MAX_STEPS}"
[[ "${BATCH_SIZE}" =~ ^[0-9]+$ ]] || die "BATCH_SIZE must be an integer, got: ${BATCH_SIZE}"

# ---------------- Scratch disk setup ----------------
setup_scratch_disk() {
  if [[ "${SETUP_SCRATCH}" == "1" ]] && command -v lsblk >/dev/null 2>&1 &&
    lsblk -o LABEL | grep -qw 'DOSCRATCH'; then
    log "Mounting DOSCRATCH volume at /scratch"
    "${SUDO[@]}" mkdir -p /scratch
    if ! grep -q 'LABEL=DOSCRATCH' /etc/fstab; then
      echo 'LABEL=DOSCRATCH /scratch ext4 discard,errors=remount-ro 0 2' |
        "${SUDO[@]}" tee -a /etc/fstab >/dev/null
    fi
    mountpoint -q /scratch || "${SUDO[@]}" mount /scratch || true
  fi

  if [[ "${WORK_ROOT}" == /scratch/* ]] && ! mountpoint -q /scratch; then
    log "/scratch is not a mount point; falling back to /opt/piper_tamil"
    WORK_ROOT="/opt/piper_tamil"
    PIPER_REPO_DIR="${WORK_ROOT}/piper1-gpl"
  fi
}

# ---------------- System packages ----------------
install_system_packages() {
  if [[ "${SKIP_APT}" == "1" ]]; then
    log "SKIP_APT=1: not installing system packages"
    return 0
  fi
  log "Installing system packages"
  run_apt_cmd "${SUDO[@]}" apt-get update
  run_apt_cmd "${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive \
    apt-get install -y "${APT_PACKAGES[@]}"
}

# ---------------- Dataset download + extraction ----------------
fetch_dataset() {
  mkdir -p "${DATA_ZIP_DIR}"

  if [[ -s "${DATA_ZIP_DIR}/${PHASE2_ZIP}" && -s "${DATA_ZIP_DIR}/${PHASE3_ZIP}" ]]; then
    log "Required phase archives already present in ${DATA_ZIP_DIR}"
    return 0
  fi

  if [[ -e "${DATASET_ARCHIVE}" ]] && ! is_valid_zip_archive "${DATASET_ARCHIVE}"; then
    log "Existing dataset archive is invalid or incomplete, re-downloading"
    rm -f "${DATASET_ARCHIVE}"
  fi
  if [[ ! -s "${DATASET_ARCHIVE}" ]]; then
    download_gdrive_zip "${DATASET_URL}" "${DATASET_ARCHIVE}"
  else
    log "Using existing dataset archive: ${DATASET_ARCHIVE}"
  fi

  log "Extracting phase archives from ${DATASET_ARCHIVE}"
  unzip -j -o "${DATASET_ARCHIVE}" "${PHASE2_ZIP}" "${PHASE3_ZIP}" -d "${DATA_ZIP_DIR}" >/dev/null
  if [[ ! -s "${DATA_ZIP_DIR}/${PHASE2_ZIP}" || ! -s "${DATA_ZIP_DIR}/${PHASE3_ZIP}" ]]; then
    die "Failed to extract ${PHASE2_ZIP} and ${PHASE3_ZIP} from ${DATASET_ARCHIVE}"
  fi
}

# ---------------- Piper source + Python env ----------------
setup_piper() {
  if [[ ! -d "${PIPER_REPO_DIR}/.git" ]]; then
    log "Cloning Piper into ${PIPER_REPO_DIR}"
    git clone "${PIPER_REPO_URL}" "${PIPER_REPO_DIR}"
  fi

  cd "${PIPER_REPO_DIR}"
  [[ -x .venv/bin/python ]] || python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate

  python -m pip install --upgrade pip setuptools wheel
  python -m pip install --index-url "${TORCH_INDEX_URL}" torch
  python -m pip install \
    lightning tensorboard tensorboardX "jsonargparse[signatures]>=4.27.7" \
    pathvalidate onnx onnxscript "pysilero-vad<3" cython scikit-build "librosa<1" onnxruntime
  python -m pip install -e . --no-deps

  ./build_monotonic_align.sh
  python setup.py build_ext --inplace

  python - <<'PY'
import torch
print("torch:", torch.__version__)
print("gpu visible:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: ROCm GPU is not visible to PyTorch.")
print("device:", torch.cuda.get_device_name(0))
PY
}

# ---------------- Dataset preparation ----------------
prepare_dataset() {
  DATA_ROOT="${WORK_ROOT}/data_ta_single_speaker"
  RAW_DIR="${DATA_ROOT}/raw"
  AUDIO_DIR="${DATA_ROOT}/audio"
  META_PATH="${DATA_ROOT}/metadata.csv"
  mkdir -p "${RAW_DIR}" "${AUDIO_DIR}"

  local f
  for f in "${PHASE2_ZIP}" "${PHASE3_ZIP}"; do
    [[ -f "${DATA_ZIP_DIR}/${f}" ]] || die "Missing ${DATA_ZIP_DIR}/${f}"
    rsync -a "${DATA_ZIP_DIR}/${f}" "${RAW_DIR}/${f}"
  done

  local extra=()
  [[ "${INCLUDE_ADDITIONAL}" == "1" ]] && extra+=(--include-additional)

  log "Preparing metadata"
  python "${SCRIPT_DIR}/scripts/prepare_metadata.py" \
    --raw-dir "${RAW_DIR}" \
    --audio-dir "${AUDIO_DIR}" \
    --metadata "${META_PATH}" \
    --phase2-zip "${PHASE2_ZIP}" \
    --phase3-zip "${PHASE3_ZIP}" \
    --min-rows "${MIN_ROWS}" \
    "${extra[@]}"
}

# ---------------- Training ----------------
train() {
  RUN_ROOT="${WORK_ROOT}/runs/${MODEL_BASENAME}"
  BASE_CKPT="${RUN_ROOT}/base.ckpt"
  mkdir -p "${RUN_ROOT}"

  if [[ ! -s "${BASE_CKPT}" ]]; then
    log "Downloading base checkpoint"
    curl -L --retry 5 --retry-delay 5 --fail -o "${BASE_CKPT}.part" "${BASE_CKPT_URL}"
    mv -f "${BASE_CKPT}.part" "${BASE_CKPT}"
  fi

  local base_step effective_steps train_log
  base_step="$(checkpoint_step_from_name "${BASE_CKPT_URL}")"
  effective_steps="$(effective_max_steps "${MAX_STEPS}" "${base_step}")"
  if [[ "${effective_steps}" != "${MAX_STEPS}" ]]; then
    log "MAX_STEPS=${MAX_STEPS} is <= base checkpoint step ${base_step};" \
      "training to step ${effective_steps} (base + ${MAX_STEPS} additional)."
  fi

  train_log="${RUN_ROOT}/train_$(date +%Y%m%d_%H%M%S).log"
  log "Training; log: ${train_log}"
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
    --ckpt_path "${BASE_CKPT}" \
    --trainer.default_root_dir "${RUN_ROOT}" \
    --trainer.accelerator gpu \
    --trainer.devices 1 \
    --trainer.precision "${PRECISION}" \
    --trainer.max_steps "${effective_steps}" \
    --trainer.val_check_interval "${VAL_CHECK_INTERVAL}" \
    --trainer.log_every_n_steps 20 \
    2>&1 | tee "${train_log}"
}

# ---------------- Export + smoke test ----------------
export_model() {
  local latest_ckpt
  latest_ckpt="$(latest_checkpoint "${RUN_ROOT}" "${BASE_CKPT}")"
  [[ -n "${latest_ckpt}" ]] || die "No trained checkpoint found in ${RUN_ROOT}"
  log "Exporting ${latest_ckpt}"

  python -m piper.train.export_onnx \
    --checkpoint "${latest_ckpt}" \
    --output-file "${RUN_ROOT}/${MODEL_BASENAME}.onnx"

  printf '%s\n' "${SAMPLE_TEXT}" |
    python -m piper --model "${RUN_ROOT}/${MODEL_BASENAME}.onnx" --output_file "${RUN_ROOT}/sample.wav"

  echo "Done."
  echo "Model:  ${RUN_ROOT}/${MODEL_BASENAME}.onnx"
  echo "Config: ${RUN_ROOT}/${MODEL_BASENAME}.onnx.json"
  echo "Sample: ${RUN_ROOT}/sample.wav"
}

main() {
  setup_scratch_disk
  install_system_packages
  mkdir -p "${WORK_ROOT}"
  fetch_dataset
  setup_piper
  prepare_dataset
  train
  export_model
}

main "$@"
