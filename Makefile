# Every number in the report is rebuilt from these targets.
# On Windows without `make`, run the python commands shown on each line directly.
PY ?= python

.PHONY: install data features tune train eval analyze ablate attack explain report serve demo test lint clean

install:
	$(PY) -m pip install -e ".[all]"

data:          ## raw zips -> cleaned, labeled Parquet + P1 split in data/processed
	$(PY) -m flowguard.data.load
	$(PY) -m flowguard.data.clean
	$(PY) -m flowguard.data.split

features:      ## fit the preprocessing pipeline on the P1 training split
	$(PY) -m flowguard.features.pipeline

tune:          ## random hyperparameter search on the validation split (before train)
	$(PY) -m flowguard.models.tune

train:         ## train the P1 classifiers, then build and calibrate the hybrid detector
	$(PY) -m flowguard.models.train
	$(PY) -m flowguard.models.hybrid

eval:          ## run protocols P1-P3 and log to MLflow
	$(PY) -m flowguard.eval.protocols

analyze:       ## calibration, loss and learning curves, drift, adaptation (after eval)
	$(PY) -m flowguard.eval.calibration
	$(PY) -m flowguard.eval.curves
	$(PY) -m flowguard.eval.drift
	$(PY) -m flowguard.eval.adaptation

ablate:        ## ablation study: retrains our models once per variant (slow)
	$(PY) -m flowguard.eval.ablation

attack:        ## protocol P4: evasion attacks and hardened models
	$(PY) -m flowguard.adversarial.attacks

explain:       ## SHAP figures into reports/figures
	$(PY) -m flowguard.explain.shap_explain

report:        ## result tables and figures into reports/
	$(PY) -m flowguard.eval.report

serve:         ## detection API on http://127.0.0.1:8000
	$(PY) -m uvicorn flowguard.serve.api:app --reload

demo:          ## dashboard on http://localhost:8501
	$(PY) -m streamlit run app/dashboard.py

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests app

clean:
	$(PY) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('mlruns', '.pytest_cache', '.ruff_cache')]"
