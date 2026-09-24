PY=.venv/bin/python
install:
	python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-dev.txt
	cd web && npm install && npm run build
run:
	$(PY) -m tulpar_ai
test:
	$(PY) -m pytest -q
keys:
	$(PY) tools/check_keys.py
evals:
	$(PY) evals/run.py program && $(PY) evals/run.py router && $(PY) evals/run.py qa
mcp:
	$(PY) -m tulpar_ai.mcp_server
export:
	$(PY) tools/export_from_tulpar.py --tulpar ~/Projects/tulpar-saas
