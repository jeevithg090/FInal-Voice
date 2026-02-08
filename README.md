# Piper Tamil ROCm 7.1 Training (IITM Female Mono)

This repo contains a single script to train Piper Tamil on ROCm 7.1 (Ubuntu 24.04), using:
- `IndicTTS_Phase2_Tamil_fem_Speaker1_mono.zip`
- `IndicTTS_Phase3_Tamil_fem_Speaker1_mono.zip`

The script auto-downloads `tamil_female_mono.zip` from Google Drive using `curl` + confirm token (`confirm` + `uuid`), then extracts only required phase zips.

## One Command (on your server console)

```bash
bash piper_ta_rocm71_train.sh
```

## Fresh Server Setup

```bash
git clone https://github.com/jeevithg090/FInal-Voice.git
cd FInal-Voice
bash piper_ta_rocm71_train.sh
```

## Important Env Vars

- `DATASET_FILE_ID` (default: `1lTi5J7iX2dlo8GB16OhnDdqXlJyfuHjI`)
- `WORK_ROOT` (default: `/scratch/piper_tamil`)
- `MAX_STEPS` (default: `120000`)
- `BATCH_SIZE` (default: `48`)
- `VAL_CHECK_INTERVAL` (default: `50`)
- `INCLUDE_ADDITIONAL` (`0` default, set `1` to include Phase2 Additional_data)

Example:

```bash
MAX_STEPS=60000 BATCH_SIZE=40 VAL_CHECK_INTERVAL=50 bash piper_ta_rocm71_train.sh
```

## Outputs

Final artifacts are written to:

- `/scratch/piper_tamil/runs/ta_IN-iitm-female-s1-medium/` (or `/opt/piper_tamil/...` fallback)

Expected final files:

- `ta_IN-iitm-female-s1-medium.onnx`
- `ta_IN-iitm-female-s1-medium.onnx.json`
- `sample.wav`
