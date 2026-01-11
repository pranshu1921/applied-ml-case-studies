.PHONY: help churn fraud clean

help:
	@echo "Targets:"
	@echo "  make churn   - run churn case study end-to-end"
	@echo "  make fraud   - run fraud detection case study end-to-end"
	@echo "  make clean   - remove generated artifacts"

churn:
	python -m case_studies.churn.run_all

fraud:
	python -m case_studies.fraud.run_all

clean:
	rm -rf artifacts reports/*.html data/processed/*.csv
