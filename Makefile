.PHONY: dev test lint run run-mock run-nllb gemma-pull docker bench-asr bench-mt bench-gemma cascade

dev:
	pip install -r requirements-dev.txt

test:
	WVC_MOCK_MODELS=1 pytest -q

lint:
	ruff check app eval scripts tests

run-mock:
	WVC_MOCK_MODELS=1 uvicorn app.main:app --reload --port 8000

GEMMA ?= gemma4:e4b

gemma-pull:
	ollama pull $(GEMMA)

run:
	WVC_MT_BACKEND=gemma WVC_GEMMA_MODEL=$(GEMMA) uvicorn app.main:app --host 0.0.0.0 --port 8000

run-nllb:
	WVC_MT_BACKEND=nllb uvicorn app.main:app --host 0.0.0.0 --port 8000

docker:
	docker compose up --build -d

bench-asr:
	python -m eval.asr_benchmark --manifest data/test/manifest.csv \
	  --models facebook/mms-1b-all openai/whisper-large-v3

bench-mt:
	python -m eval.mt_benchmark --flores-dir data/flores200_dataset \
	  --models facebook/nllb-200-distilled-600M facebook/nllb-200-1.3B facebook/nllb-200-3.3B

bench-gemma:
	python -m eval.mt_benchmark --flores-dir data/flores200_dataset --limit 200 \
	  --models $(GEMMA) facebook/nllb-200-distilled-600M

cascade:
	python -m eval.cascade --hyps results/asr/facebook__mms-1b-all/hypotheses.csv --synthetic
