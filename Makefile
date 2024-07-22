LINT_FILES := domain_matcher tests

.PHONY: format
format:
	if [ -n "${POETRY_ACTIVE}" ]; then make _format $(LINT_FILES); else poetry run make _format $(LINT_FILES); fi

.PHONY: _format
_format:
	ruff format $(LINT_FILES)
	nb-clean clean notebooks --remove-empty-cells --preserve-cell-metadata

	$(MAKE) lint

test: lint mypy unit-test

.PHONY: lint
lint:
	@# calling make _lint within poetry make it so that we only init poetry once
	if [ -n "${POETRY_ACTIVE}" ]; then make _lint $(LINT_FILES); else poetry run make _lint $(LINT_FILES); fi

.PHONY: _lint
_lint:
	ruff check $(LINT_FILES)
	# nb-clean check notebooks --remove-empty-cells --preserve-cell-metadata

.PHONY: mypy
mypy:
	poetry run mypy domain_matcher

.PHONY: unit-test
unit-test:
	poetry run pytest tests --durations 5