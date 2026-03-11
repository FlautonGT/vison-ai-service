PYTHON ?= python
MODEL_DIR ?= /opt/models
APP_HOST ?= 127.0.0.1
APP_PORT ?= 8000
APP_WORKERS ?= 2

.PHONY: setup migrate run test unit benchmark nist calibrate quantize frvt fairness asian train-select train-fit train-eval

setup:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) scripts/download_models.py --model-dir $(MODEL_DIR)

migrate:
	@echo "No local migrations in vison-ai-service (pure inference mode)."
	@echo "Run migrations from vison-database project."

run:
	MODEL_DIR=$(MODEL_DIR) FACE_AI_WORKERS=$(APP_WORKERS) $(PYTHON) -m uvicorn app.main:app --host $(APP_HOST) --port $(APP_PORT) --workers $(APP_WORKERS)

test:
	MODEL_DIR=$(MODEL_DIR) E2E_HOST=$(APP_HOST) $(PYTHON) scripts/test_e2e.py

unit:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"

benchmark:
	MODEL_DIR=$(MODEL_DIR) BENCHMARK_URL=http://$(APP_HOST):$(APP_PORT) $(PYTHON) scripts/benchmark_real_vs_ai.py --ai-count 200 --max-tests 200

nist:
	MODEL_DIR=$(MODEL_DIR) BENCHMARK_URL=http://$(APP_HOST):$(APP_PORT) $(PYTHON) scripts/benchmark_nist_style.py --real-count 200 --attack-count 200

calibrate:
	MODEL_DIR=$(MODEL_DIR) BENCHMARK_URL=http://$(APP_HOST):$(APP_PORT) $(PYTHON) scripts/calibrate_thresholds.py --real-count 200 --attack-count 200

quantize:
	MODEL_DIR=$(MODEL_DIR) $(PYTHON) scripts/quantize_models.py --model-dir $(MODEL_DIR)

frvt:
	MODEL_DIR=$(MODEL_DIR) BENCHMARK_URL=http://$(APP_HOST):$(APP_PORT) $(PYTHON) scripts/benchmark_frvt.py --dataset-dir ./benchmark_data/lfw --max-pairs 500

fairness:
	MODEL_DIR=$(MODEL_DIR) BENCHMARK_URL=http://$(APP_HOST):$(APP_PORT) $(PYTHON) scripts/benchmark_fairness.py --dataset-dir ./benchmark_data/UTKFace --max-tests 1000

asian:
	MODEL_DIR=$(MODEL_DIR) BENCHMARK_URL=http://$(APP_HOST):$(APP_PORT) $(PYTHON) scripts/benchmark_asian.py --dataset-dir ./benchmark_data/asian_faces --max-pairs 200

train-select:
	$(PYTHON) scripts/select_training_datasets.py --task $(TASK)

train-fit:
	$(PYTHON) scripts/train_pipeline.py --task $(TASK)

train-eval:
	$(PYTHON) scripts/evaluate_pipeline.py --task $(TASK)
