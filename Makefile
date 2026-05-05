setup:
	pip install -r requirements.txt

run:
	python scripts/run_pipeline.py

test:
	pytest tests/