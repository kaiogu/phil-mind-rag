.PHONY: test test-paid-api

test:
	uv run pytest

test-paid-api:
	RUN_PAID_API_TESTS=1 uv run pytest -m paid_api tests/test_paid_api_smoke.py -v
