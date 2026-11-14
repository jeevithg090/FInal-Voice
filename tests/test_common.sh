#!/usr/bin/env bash
# Unit tests for lib/common.sh. Run: bash tests/test_common.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib/common.sh
source "${ROOT}/lib/common.sh"

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

PASS=0
FAIL=0

assert_eq() {
  local name="$1" expected="$2" actual="$3"
  if [[ "${expected}" == "${actual}" ]]; then
    PASS=$((PASS + 1))
  else
    FAIL=$((FAIL + 1))
    printf 'FAIL: %s\n  expected: [%s]\n  actual:   [%s]\n' "${name}" "${expected}" "${actual}"
  fi
}

assert_ok() {
  local name="$1"
  shift
  if "$@"; then PASS=$((PASS + 1)); else
    FAIL=$((FAIL + 1))
    printf 'FAIL: %s (expected success)\n' "${name}"
  fi
}

assert_fails() {
  local name="$1"
  shift
  if "$@" 2>/dev/null; then
    FAIL=$((FAIL + 1))
    printf 'FAIL: %s (expected failure)\n' "${name}"
  else PASS=$((PASS + 1)); fi
}

make_zip() {
  local dest="$1"
  (cd "${TMP}" && printf 'hello' >payload.txt && zip -q "${dest}" payload.txt && rm payload.txt)
}

# --- checkpoint_step_from_name ---
assert_eq "step from url-encoded name" 839202 \
  "$(checkpoint_step_from_name 'https://x/epoch%3D3277-step%3D839202.ckpt')"
assert_eq "step from plain name" 42 "$(checkpoint_step_from_name 'epoch=1-step=42.ckpt')"
assert_eq "no step" "" "$(checkpoint_step_from_name 'model.ckpt')"

# --- effective_max_steps ---
assert_eq "max above base" 900000 "$(effective_max_steps 900000 839202)"
assert_eq "max below base is additive" 959202 "$(effective_max_steps 120000 839202)"
assert_eq "max equal base is additive" 200 "$(effective_max_steps 100 100)"
assert_eq "no base step" 120000 "$(effective_max_steps 120000)"
assert_fails "non-integer max steps" effective_max_steps abc 10

# --- zip validation ---
make_zip "${TMP}/good.zip"
printf 'not a zip' >"${TMP}/bad.zip"
head -c 40 "${TMP}/good.zip" >"${TMP}/truncated.zip"
: >"${TMP}/empty.zip"
assert_ok "valid zip" is_valid_zip_archive "${TMP}/good.zip"
assert_fails "non-zip file" is_valid_zip_archive "${TMP}/bad.zip"
assert_fails "truncated zip" is_valid_zip_archive "${TMP}/truncated.zip"
assert_fails "empty file" is_valid_zip_archive "${TMP}/empty.zip"
assert_fails "missing file" is_valid_zip_archive "${TMP}/missing.zip"
assert_ok "truncated zip still has magic" has_zip_magic "${TMP}/truncated.zip"

# --- html_input_value ---
html='<form><input type="hidden" name="confirm" value="t"><input type="hidden" name="uuid" value="abc-123"></form>'
assert_eq "confirm token" t "$(html_input_value confirm <<<"${html//></>$'\n'<}")"
assert_eq "uuid token" abc-123 "$(html_input_value uuid <<<"${html//></>$'\n'<}")"
assert_eq "absent token" "" "$(html_input_value nope <<<"${html}")"

# --- latest_checkpoint ---
ckpts="${TMP}/run"
mkdir -p "${ckpts}/lightning_logs/version_0/checkpoints"
touch -d '2024-01-03' "${ckpts}/base.ckpt"
touch -d '2024-01-01' "${ckpts}/lightning_logs/version_0/checkpoints/old.ckpt"
touch -d '2024-01-02' "${ckpts}/lightning_logs/version_0/checkpoints/new.ckpt"
assert_eq "latest excludes base" "${ckpts}/lightning_logs/version_0/checkpoints/new.ckpt" \
  "$(latest_checkpoint "${ckpts}" "${ckpts}/base.ckpt")"
assert_eq "latest includes base when not excluded" "${ckpts}/base.ckpt" "$(latest_checkpoint "${ckpts}")"
mkdir -p "${TMP}/emptyrun"
assert_eq "no checkpoints" "" "$(latest_checkpoint "${TMP}/emptyrun")"

# --- run_apt_cmd retries ---
counter="${TMP}/count"
echo 0 >"${counter}"
flaky() {
  local n
  n=$(($(cat "${counter}") + 1))
  echo "${n}" >"${counter}"
  ((n >= 2))
}
APT_RETRY_DELAY=0 assert_ok "retry until success" run_apt_cmd flaky
assert_eq "retried once" 2 "$(cat "${counter}")"
APT_RETRY_DELAY=0 APT_RETRY_ATTEMPTS=2 assert_fails "gives up" run_apt_cmd false

# --- download_gdrive_zip with a stubbed curl ---
stub_bin="${TMP}/bin"
mkdir -p "${stub_bin}"
make_zip "${TMP}/served.zip"
cat >"${stub_bin}/curl" <<STUB
#!/usr/bin/env bash
out="" url=""
while (( \$# )); do
  case "\$1" in
    -o) out="\$2"; shift 2 ;;
    -C|--retry|--retry-delay) shift 2 ;;
    -*) shift ;;
    *) url="\$1"; shift ;;
  esac
done
echo "\${url}" >>"${TMP}/curl_calls"
case "\${MODE}:\${url}" in
  direct:*) cp "${TMP}/served.zip" "\${out}" ;;
  confirm:*confirm=tok*uuid=u-1*) cp "${TMP}/served.zip" "\${out}" ;;
  confirm:*) printf '<input name="confirm" value="tok">\n<input name="uuid" value="u-1">\n' >"\${out}" ;;
  garbage:*confirm=*) printf 'junk' >"\${out}" ;;
  garbage:*) printf '<input name="confirm" value="tok">\n<input name="uuid" value="u-1">\n' >"\${out}" ;;
  notoken:*) printf '<html>quota exceeded</html>' >"\${out}" ;;
esac
STUB
chmod +x "${stub_bin}/curl"

run_download() {
  MODE="$1" PATH="${stub_bin}:${PATH}" download_gdrive_zip "https://example.test/dl?id=1" "$2" 2>/dev/null
}

assert_ok "direct download" run_download direct "${TMP}/dl1/out.zip"
assert_ok "direct result valid" is_valid_zip_archive "${TMP}/dl1/out.zip"

rm -f "${TMP}/curl_calls"
assert_ok "confirm-page download" run_download confirm "${TMP}/dl2/out.zip"
assert_ok "confirm result valid" is_valid_zip_archive "${TMP}/dl2/out.zip"
assert_eq "two requests made" 2 "$(wc -l <"${TMP}/curl_calls" | tr -d ' ')"
assert_eq "no temp files left" "out.zip" "$(ls "${TMP}/dl2")"

assert_fails "garbage download rejected" run_download garbage "${TMP}/dl3/out.zip"
assert_eq "garbage not kept" "" "$(ls "${TMP}/dl3")"
assert_fails "missing tokens rejected" run_download notoken "${TMP}/dl4/out.zip"

# --- load_env_file ---
cat >"${TMP}/test.env" <<'ENV'
# comment line

T_PLAIN=value
export T_EXPORTED=yes
T_INLINE=120000   # trailing comment
T_DQ="a # not a comment"
T_SQ='$(touch should-not-exist)'
T_URL=https://x.test/dl?id=1&export=download
T_PRESET=from-file
not a valid line
ENV
(
  T_PRESET=from-shell
  cd "${TMP}"
  load_env_file "${TMP}/test.env"
  assert_eq "plain value" value "${T_PLAIN}"
  assert_eq "export prefix" yes "${T_EXPORTED}"
  assert_eq "inline comment stripped" 120000 "${T_INLINE}"
  assert_eq "double-quoted keeps hash" "a # not a comment" "${T_DQ}"
  # shellcheck disable=SC2016  # literal $(...) is the point of this test
  assert_eq "single-quoted literal" '$(touch should-not-exist)' "${T_SQ}"
  assert_eq "url with ampersand" 'https://x.test/dl?id=1&export=download' "${T_URL}"
  assert_eq "shell value wins" from-shell "${T_PRESET}"
  assert_eq "no code executed" "" "$([[ -e "${TMP}/should-not-exist" ]] && echo exists)"
  assert_eq "exported to children" yes "$(bash -c 'echo "${T_EXPORTED}"')"
  printf '%s %s\n' "${PASS}" "${FAIL}" >"${TMP}/subshell_counts"
)
read -r PASS FAIL <"${TMP}/subshell_counts"
assert_ok "missing env file is a no-op" load_env_file "${TMP}/does-not-exist.env"

printf '\n%d passed, %d failed\n' "${PASS}" "${FAIL}"
((FAIL == 0))
