SHELL_FILES := piper_ta_rocm71_train.sh lib/common.sh tests/test_common.sh tests/test_cli.sh
PY_FILES := scripts tests

.PHONY: help lint format test check

help:
	@echo "make lint    - shellcheck, shfmt (diff), ruff check, ruff format --check"
	@echo "make format  - apply shfmt and ruff formatting"
	@echo "make test    - run shell and Python unit tests"
	@echo "make check   - lint + test"

lint:
	shellcheck -x $(SHELL_FILES)
	shfmt -d -i 2 -ci $(SHELL_FILES)
	ruff check $(PY_FILES)
	ruff format --check $(PY_FILES)

format:
	shfmt -w -i 2 -ci $(SHELL_FILES)
	ruff check --fix $(PY_FILES)
	ruff format $(PY_FILES)

test:
	bash tests/test_common.sh
	bash tests/test_cli.sh
	python3 -m pytest

check: lint test
