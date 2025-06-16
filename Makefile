.PHONY: test clean docs install-pre-commit install-dependencies setup

setup: install-dependencies install-pre-commit
	@echo "==================== Setup Finished ===================="
	@echo "All set! Ready to go!"

docs: clean
	pip install . --quiet -r requirements/docs.txt
	mkdocs serve

clean:
	# clean all temp runs
	rm -rf $(shell find . -name "mlruns")
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf ./docs/build
	rm -rf ./docs/source/**/generated
	rm -rf ./docs/source/api
	rm -rf ./src/*.egg-info
	rm -rf ./build
	rm -rf ./dist

install-dependencies:
	pip install -r requirements.txt
	pip install -r requirements/test.txt
	pip install -r requirements/docs.txt
	pip install -r requirements/extras.txt
	pip install -e .


install-pre-commit:
	pip install pre-commit
	pre-commit install
