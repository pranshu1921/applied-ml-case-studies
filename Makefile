.PHONY: help churn clean

help:
	@echo "Targets:"
	@echo "  make churn   - run churn case study end-to-end"
	@echo "  make clean   - remove generated artifacts (data/processed, artifacts/, reports/*.html)"

churn:
	python -m case_studies.churn.run_all

clean:
	rm -rf artifacts reports/*.html data/processed/*.csv
