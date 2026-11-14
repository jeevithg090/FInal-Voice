#!/usr/bin/env bash
# Tests for the command-line interface of piper_ta_rocm71_train.sh.
# They only exercise argument parsing, configuration and preflight output, so
# nothing is installed, downloaded or trained. Run: bash tests/test_cli.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="${ROOT}/piper_ta_rocm71_train.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

PASS=0
FAIL=0

# Run the script with a clean, predictable configuration.
# Sets OUT (stdout+stderr) and RC.
run() {
  set +e
  OUT="$(env -u WORK_ROOT -u MAX_STEPS -u BATCH_SIZE -u SKIP_APT NO_COLOR=1 \
    ENV_FILE="${TMP}/none.env" WORK_ROOT="${TMP}/work" "$@" 2>&1)"
  RC=$?
  set -e
}

check() {
  local name="$1" expected_rc="$2" pattern="$3"
  if [[ "${RC}" == "${expected_rc}" ]] && grep -qE -- "${pattern}" <<<"${OUT}"; then
    PASS=$((PASS + 1))
  else
    FAIL=$((FAIL + 1))
    printf 'FAIL: %s\n  expected rc=%s and /%s/\n  got rc=%s:\n%s\n' \
      "${name}" "${expected_rc}" "${pattern}" "${RC}" "${OUT}"
  fi
}

run bash "${SCRIPT}" --help
check "--help prints usage" 0 '^Usage:'
check "--help lists stages" 0 'packages dataset piper prepare train export'

run bash "${SCRIPT}" -h
check "-h is an alias" 0 '--print-config'

run bash "${SCRIPT}" --list-stages
check "--list-stages" 0 '6\. export +Export ONNX voice'

run bash "${SCRIPT}" --bogus
check "unknown option" 1 "Unknown option '--bogus'"

run bash "${SCRIPT}" --from nope
check "unknown stage" 1 "Unknown stage 'nope'"

run bash "${SCRIPT}" --only
check "missing stage name" 1 '--only needs a stage name'

run bash "${SCRIPT}" --from train --only export
check "--from and --only conflict" 1 'either --from or --only'

run env MAX_STEPS=abc BATCH_SIZE=0 VAL_SPLIT=1.5 PRECISION=fp8 bash "${SCRIPT}" --only export
check "invalid config: MAX_STEPS" 1 'MAX_STEPS must be a non-negative integer'
check "invalid config: BATCH_SIZE" 1 'BATCH_SIZE must be a positive integer'
check "invalid config: VAL_SPLIT" 1 'VAL_SPLIT must be a decimal'
check "invalid config: PRECISION" 1 'PRECISION must be one of'

run env MAX_STEPS=60000 bash "${SCRIPT}" --print-config
check "print-config shows shell value" 0 'MAX_STEPS +shell +60000'
check "print-config shows defaults" 0 'BATCH_SIZE +default +48'
check "print-config shows derived paths" 0 'voice model .*ta_IN-iitm-female-s1-medium\.onnx'

printf 'BATCH_SIZE=24\nMAX_STEPS=1000\n' >"${TMP}/test.env"
run env MAX_STEPS=5 ENV_FILE="${TMP}/test.env" bash "${SCRIPT}" --print-config
check "print-config shows .env value" 0 'BATCH_SIZE +\.env +24'
check "shell beats .env" 0 'MAX_STEPS +shell +5'

run env SKIP_APT=1 PATH="/usr/bin:/bin" MIN_FREE_GB=0 bash "${SCRIPT}" --preflight
check "preflight prints a summary" "${RC}" '[0-9]+ failed, [0-9]+ warning'

run env SKIP_APT=maybe bash "${SCRIPT}" --preflight
check "preflight fails on invalid config" 1 'FAIL +SKIP_APT must be 0 or 1'

# A failing stage names itself and prints the resume command.
run env SKIP_APT=1 SETUP_SCRATCH=0 bash "${SCRIPT}" --only export
check "stage failure explains how to resume" 1 "resume with: bash piper_ta_rocm71_train.sh --from export"
check "missing piper install is explained" 1 "Run the 'piper' stage first"

printf '\n%d passed, %d failed\n' "${PASS}" "${FAIL}"
((FAIL == 0))
