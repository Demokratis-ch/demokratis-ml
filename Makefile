test:
	uv run pre-commit run -a
	PYTHONPATH=. uv run pytest research demokratis_ml
