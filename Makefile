# Hyderabad labelling sequence — one command per stage, all config-driven.
# Human steps (labelling in QGIS, adjudication) run between `split` and `merge`.
PY = python

.PHONY: labels-skeleton labels-split labels-qgis labels-validate labels-merge labels-test

labels-skeleton:
	$(PY) -m geoeco.labels.pipeline --stage sample

labels-split:
	$(PY) -m geoeco.labels.pipeline --stage split

labels-qgis:
	$(PY) -m geoeco.labels.qgis

labels-validate:
	$(PY) -m geoeco.labels.validate

labels-merge:
	$(PY) -m geoeco.labels.merge

labels-test:
	pytest -q tests/test_labelling.py tests/test_leakage.py -v
