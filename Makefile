setup:
	pip install -r requirements.txt

run:
	python -m scripts.run_pipeline

test:
	pytest tests/ 