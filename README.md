# Piper Tamil TTS Trainer for AMD ROCm

A single, resumable command that turns a fresh Ubuntu 24.04 GPU server with an
AMD card into a trained **Tamil text-to-speech voice** for
[Piper](https://github.com/OHF-Voice/piper1-gpl). It fine-tunes a pretrained
Piper checkpoint on the IIT Madras **IndicTTS** Tamil female single-speaker
recordings using PyTorch on **ROCm 7.1**, then exports an ONNX voice and
synthesises a sample.

```bash
git clone https://github.com/jeevithg090/FInal-Voice.git
cd FInal-Voice
bash piper_ta_rocm71_train.sh
```

## Features

- **One command, end to end**: system packages, dataset download, Piper
  install, dataset preparation, training, ONNX export and a sample WAV.
- **Resumable and idempotent**: re-running skips finished steps; an
  interrupted dataset download resumes from the partial file; a corrupted
  archive is detected and re-downloaded.
- **Google Drive large-file support**: handles both direct downloads and the
  "can't scan for viruses" confirmation page (`confirm` + `uuid` tokens).
- **Robust on fresh cloud servers**: waits for and retries around
  `apt`/`dpkg` locks held by `unattended-upgrades`; optionally mounts a
  `DOSCRATCH`-labelled scratch volume at `/scratch`.
- **Fine-tuning that does what you mean**: the base checkpoint is already at
  global step 839,202, so a `MAX_STEPS` at or below that value is treated as
  *additional* steps instead of stopping immediately.
- **Configurable** through environment variables or a `.env` file.
- **Tested**: shell helpers and the dataset preparation logic have unit tests
  that run without a GPU.

## Stack

| Layer | Technology |
| --- | --- |
| Orchestration | Bash (`piper_ta_rocm71_train.sh`, `lib/common.sh`) |
| Dataset preparation | Python 3 standard library (`scripts/prepare_metadata.py`) |
| TTS model | [Piper](https://github.com/OHF-Voice/piper1-gpl) (VITS) with PyTorch Lightning |
| GPU runtime | PyTorch ROCm 7.1 wheels |
| Phonemisation | eSpeak NG (`ta` voice) |
| Export / inference | ONNX, ONNX Runtime |
| Dev tooling | ShellCheck, shfmt, Ruff, pytest |

## How it works

```
piper_ta_rocm71_train.sh
 ├─ setup_scratch_disk       mount DOSCRATCH volume at /scratch (or fall back to /opt/piper_tamil)
 ├─ install_system_packages  apt-get with lock waiting + retries
 ├─ fetch_dataset            download tamil_female_mono.zip, extract the Phase2 + Phase3 zips
 ├─ setup_piper              clone piper1-gpl, venv, ROCm torch, build monotonic_align, GPU check
 ├─ prepare_dataset          scripts/prepare_metadata.py -> audio/ + metadata.csv
 ├─ train                    python -m piper.train fit, from the Telugu "maya" medium checkpoint
 └─ export_model             latest trained checkpoint -> .onnx + .onnx.json, sample.wav
```

`scripts/prepare_metadata.py` reads the Festival-format `txt.done.data`
transcripts inside each IndicTTS archive (`( utt_id "text" )`), pairs them
with the matching WAV files, normalises whitespace, removes the `|` delimiter
from text, and writes Piper's `relative/path.wav|text` metadata. Utterances
without a transcript are skipped and counted; the run aborts if fewer than
`MIN_ROWS` (default 3000) utterances are found.

Data sources used for training:

- `IndicTTS_Phase2_Tamil_fem_Speaker1_mono.zip` — `mono/Read_speech`
  (plus `mono/Additional_data` when `INCLUDE_ADDITIONAL=1`)
- `IndicTTS_Phase3_Tamil_fem_Speaker1_mono.zip` — `speaker1`

## Requirements

- Ubuntu 24.04 with an AMD GPU supported by ROCm 7.1 and the ROCm driver
  stack already installed (the script checks that PyTorch can see the GPU and
  stops otherwise).
- Root or passwordless `sudo` (for `apt-get` and the optional scratch mount),
  unless you set `SKIP_APT=1` and `SETUP_SCRATCH=0`.
- Plenty of disk space for the dataset archives, extracted audio, the
  Python environment and checkpoints.
- Network access to Google Drive (or your own dataset URL), GitHub,
  `download.pytorch.org`, PyPI and Hugging Face.
- The right to use the IndicTTS data. The corpus is distributed by IIT Madras
  under its own licence terms; obtain it from the official source and point
  `DATASET_URL` / `DATASET_ARCHIVE` at your copy.

## Configuration

Every setting is optional. Set variables on the command line, or copy
`.env.example` to `.env` next to the script (variables already set in the
shell take precedence over `.env`; values are read literally).

```bash
MAX_STEPS=60000 BATCH_SIZE=40 bash piper_ta_rocm71_train.sh
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `WORK_ROOT` | `/scratch/piper_tamil` | Root for Piper checkout, data and runs. Falls back to `/opt/piper_tamil` if `/scratch` is not mounted. |
| `DATA_ZIP_DIR` | `$HOME/tamil_female_mono` | Where the dataset archive and phase zips are stored. |
| `DATASET_ARCHIVE` | `$DATA_ZIP_DIR/tamil_female_mono.zip` | Outer archive. Place it here yourself to skip the download. |
| `DATASET_URL` | project's Google Drive link | Download URL for the outer archive (`DATASET_BASE_URL` is accepted as an alias). |
| `PIPER_REPO_URL` | `https://github.com/OHF-Voice/piper1-gpl.git` | Piper source. |
| `PIPER_REPO_DIR` | `$WORK_ROOT/piper1-gpl` | Piper checkout location. |
| `TORCH_INDEX_URL` | `https://download.pytorch.org/whl/rocm7.1` | PyTorch wheel index. |
| `BASE_CKPT_URL` | Piper `te_IN/maya/medium` checkpoint | Checkpoint to fine-tune from. |
| `VOICE_NAME` | `ta_IN_iitm_female_s1` | Voice name stored in the model config. |
| `MODEL_BASENAME` | `ta_IN-iitm-female-s1-medium` | Output file and run directory name. |
| `ESPEAK_VOICE` | `ta` | eSpeak NG voice used for phonemes. |
| `INCLUDE_ADDITIONAL` | `0` | `1` adds Phase 2 `Additional_data`. |
| `BATCH_SIZE` | `48` | Training batch size; lower it if you run out of GPU memory. |
| `MAX_STEPS` | `120000` | Target global step; treated as additional steps if ≤ the base checkpoint's step. |
| `VAL_SPLIT` | `0.05` | Validation fraction. |
| `NUM_TEST_EXAMPLES` | `16` | Test utterances held out. |
| `PRECISION` | `bf16-mixed` | Lightning precision (`16-mixed`, `32-true`). |
| `VAL_CHECK_INTERVAL` | `50` | Validate every N training batches. |
| `MIN_ROWS` | `3000` | Abort if fewer usable utterances are found. |
| `SAMPLE_TEXT` | a Tamil greeting | Text synthesised into `sample.wav`. |
| `SKIP_APT` | `0` | `1` skips installing system packages. |
| `SETUP_SCRATCH` | `1` | `0` never edits `/etc/fstab` or mounts `/scratch`. |
| `ENV_FILE` | `./.env` | Alternative configuration file. |

## Running

```bash
bash piper_ta_rocm71_train.sh          # full pipeline
bash piper_ta_rocm71_train.sh --help   # usage
```

Training takes a long time; run it inside `tmux` (installed by the script) so
it survives SSH disconnects. Each run writes a timestamped log to
`$WORK_ROOT/runs/<MODEL_BASENAME>/train_YYYYmmdd_HHMMSS.log`. TensorBoard logs
are written under the same run directory:

```bash
$WORK_ROOT/piper1-gpl/.venv/bin/tensorboard --logdir "$WORK_ROOT/runs"
```

### Outputs

In `$WORK_ROOT/runs/ta_IN-iitm-female-s1-medium/`:

- `ta_IN-iitm-female-s1-medium.onnx` — the voice model
- `ta_IN-iitm-female-s1-medium.onnx.json` — its Piper config
- `sample.wav` — a synthesised Tamil sample

Use the voice with Piper:

```bash
echo "வணக்கம்" | python -m piper --model ta_IN-iitm-female-s1-medium.onnx --output_file out.wav
```

## Deployment

This is a training job, not a service. A typical setup is a single-GPU cloud
VM (for example an AMD Instinct droplet with a scratch volume labelled
`DOSCRATCH`): clone the repository, optionally add a `.env`, start `tmux`, and
run the script. Copy the `.onnx` and `.onnx.json` files to wherever Piper runs
inference — no GPU is needed for inference.

## Development

The tests and linters run on any Linux or macOS machine; no GPU, ROCm or
dataset is required.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"   # pytest, ruff, shellcheck, shfmt

make lint     # shellcheck, shfmt --diff, ruff check, ruff format --check
make format   # apply shfmt + ruff formatting
make test     # bash tests/test_common.sh + pytest
make check    # lint + test
```

- `tests/test_common.sh` covers the shell helpers: checkpoint-step parsing,
  the additional-steps rule, ZIP validation, Google Drive token extraction,
  latest-checkpoint selection, `.env` loading, apt retries, and the download
  flow against a stubbed `curl`.
- `tests/test_prepare_metadata.py` builds small synthetic IndicTTS-shaped
  archives and checks transcript parsing, audio/text pairing, skipping,
  duration accounting, CSV output and CLI error handling.

## Project structure

```
.
├── piper_ta_rocm71_train.sh     # entry point: the full pipeline
├── lib/
│   └── common.sh                # sourced helpers (download, validation, retries, step maths)
├── scripts/
│   └── prepare_metadata.py      # IndicTTS archives -> Piper audio dir + metadata.csv
├── tests/
│   ├── test_common.sh           # shell unit tests
│   ├── test_prepare_metadata.py # Python unit tests
│   └── conftest.py
├── .env.example                 # every configuration variable with its default
├── Makefile                     # lint / format / test
├── pyproject.toml               # dev tooling + ruff/pytest config
└── .editorconfig
```

## Acknowledgements

- [Piper](https://github.com/OHF-Voice/piper1-gpl) by the Open Home Foundation
  and the [rhasspy/piper-checkpoints](https://huggingface.co/datasets/rhasspy/piper-checkpoints)
  pretrained models.
- The [IndicTTS](https://www.iitm.ac.in/donlab/indictts/) corpus from IIT
  Madras.
