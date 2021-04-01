# Project settings
PROJECT := responsibly
PACKAGE := responsibly
REPOSITORY := ResponsiblyAI/responsibly

# Project paths
PACKAGES := $(PACKAGE) tests
CONFIG := $(wildcard *.py)
MODULES := $(wildcard $(PACKAGE)/*.py)

# Virtual environment paths
export PIPENV_VENV_IN_PROJECT=true
export PIPENV_IGNORE_VIRTUALENVS=true
VENV := .venv

# MAIN TASKS ##################################################################

SNIFFER := pipenv run sniffer

.PHONY: all
all: install

.PHONY: ci
ci: check test ## Run all tasks that determine CI status

.PHONY: watch
watch: install .clean-test ## Continuously run all CI tasks when files chanage
	$(SNIFFER)

.PHONY: run ## Start the program
run: install
	pipenv run python $(PACKAGE)/__main__.py

# SYSTEM DEPENDENCIES #########################################################

.PHONY: doctor
doctor:  ## Confirm system dependencies are available
	bin/verchew

# PROJECT DEPENDENCIES ########################################################

DEPENDENCIES = $(VENV)/.pipenv-$(shell bin/checksum Pipfile* setup.py)

.PHONY: install
install: $(DEPENDENCIES)

$(DEPENDENCIES):
	$(SETUP) develop
	pipenv install --dev
	@ touch $@

# FREEZE ######################################################################

.PHONY: freeze
freeze:
	pipenv run pip freeze > .pyup/requirements.txt

# CHECKS ######################################################################

ISORT := pipenv run isort
PYLINT := pipenv run pylint
PYCODESTYLE := pipenv run pycodestyle
PYDOCSTYLE := pipenv run pydocstyle
RSTLINT := pipenv run rst-lint

.PHONY: check
check: isort pylint pycodestyle pydocstyle rstlint ## Run linters and static analysis

.PHONY: isort
isort: install
	$(ISORT) $(PACKAGES) $(CONFIG) --recursive --apply

.PHONY: pylint
pylint: install
	$(PYLINT) $(PACKAGES) $(CONFIG) --rcfile=.pylint.ini

.PHONY: pycodestyle
pycodestyle: install
	$(PYCODESTYLE) $(PACKAGES) $(CONFIG) --config=.pycodestyle.ini

.PHONY: pydocstyle
pydocstyle: install
	$(PYDOCSTYLE) $(PACKAGES) $(CONFIG)

.PHONY: rstlint
rstlint: install
	$(RSTLINT) README.rst CHANGELOG.rst CONTRIBUTING.rst

# TESTS #######################################################################

PYTEST := pipenv run py.test
COVERAGE := pipenv run coverage
COVERAGE_SPACE := pipenv run coverage.space

RANDOM_SEED ?= $(shell date +%s)
FAILURES := .cache/v/cache/lastfailed

PYTEST_OPTIONS := --random --random-seed=$(RANDOM_SEED)
ifdef DISABLE_COVERAGE
PYTEST_OPTIONS += --no-cov --disable-warnings
endif
PYTEST_RERUN_OPTIONS := --last-failed --exitfirst

.PHONY: test
test: test-all ## Run unit and integration tests

.PHONY: test-unit
test-unit: install
	@ ( mv $(FAILURES) $(FAILURES).bak || true ) > /dev/null 2>&1
	$(PYTEST) $(PACKAGE) $(PYTEST_OPTIONS)
	@ ( mv $(FAILURES).bak $(FAILURES) || true ) > /dev/null 2>&1
	$(COVERAGE_SPACE) $(REPOSITORY) unit

.PHONY: test-int
test-int: install
	@ if test -e $(FAILURES); then $(PYTEST) tests $(PYTEST_RERUN_OPTIONS); fi
	@ rm -rf $(FAILURES)
	$(PYTEST) tests $(PYTEST_OPTIONS)
	$(COVERAGE_SPACE) $(REPOSITORY) integration

.PHONY: test-all
test-all: install
	@ if test -e $(FAILURES); then $(PYTEST) $(PACKAGES) $(PYTEST_RERUN_OPTIONS); fi
	@ rm -rf $(FAILURES)
	$(PYTEST) $(PACKAGES) $(PYTEST_OPTIONS)
	$(COVERAGE_SPACE) $(REPOSITORY) overall

.PHONY: read-coverage
read-coverage:
	bin/open htmlcov/index.html

# DOCUMENTATION ###############################################################

.PHONY: docs
docs: install
	mkdir -p docs/about
	ln -sf `realpath README.rst --relative-to=docs` docs/readme.rst
	ln -sf `realpath CHANGELOG.rst --relative-to=docs/about` docs/about/changelog.rst
	ln -sf `realpath CONTRIBUTING.rst --relative-to=docs/about` docs/about/contributing.rst
	ln -sf `realpath LICENSE --relative-to=docs/about` docs/about/license.rst
	cd docs/notebooks && pipenv run find *.ipynb -exec jupyter nbconvert --to rst {} \;
	cd docs && pipenv run make html
	@echo "\033[95m\n\nBuild successful! View the docs homepage at docs/_build/html/index.html.\n\033[0m"
	# && sphinx-apidoc  -o api ../responsibly

.PHONY: show
show: docs
	sleep 5 && open http://localhost:8000/docs/_build/html &
	pipenv run python -m "http.server"

.PHONY: publish
publish: docs
	cd docs && pipenv run sh ./gh-pages.sh

# PYREVERSE := pipenv run pyreverse
# MKDOCS := pipenv run mkdocs
#
# MKDOCS_INDEX := site/index.html
#
# .PHONY: docs
# docs: uml mkdocs ## Generate documentation
#
# .PHONY: uml
# uml: install docs/*.png
# docs/*.png: $(MODULES)
# 	$(PYREVERSE) $(PACKAGE) -p $(PACKAGE) -a 1 -f ALL -o png --ignore tests
# 	- mv -f classes_$(PACKAGE).png docs/classes.png
# 	- mv -f packages_$(PACKAGE).png docs/packages.png
#
# .PHONY: mkdocs
# mkdocs: install $(MKDOCS_INDEX)
# $(MKDOCS_INDEX): mkdocs.yml docs/*.md
# 	ln -sf `realpath README.md --relative-to=docs` docs/index.md
# 	ln -sf `realpath CHANGELOG.md --relative-to=docs/about` docs/about/changelog.md
# 	ln -sf `realpath CONTRIBUTING.md --relative-to=docs/about` docs/about/contributing.md
# 	ln -sf `realpath LICENSE.md --relative-to=docs/about` docs/about/license.md
# 	$(MKDOCS) build --clean --strict
#
# .PHONY: mkdocs-live
# mkdocs-live: mkdocs
# 	eval "sleep 3; bin/open http://127.0.0.1:8000" &
# 	$(MKDOCS) serve

# BUILD #######################################################################

PYINSTALLER := pipenv run pyinstaller
PYINSTALLER_MAKESPEC := pipenv run pyi-makespec

DIST_FILES := dist/*.tar.gz dist/*.whl
EXE_FILES := dist/$(PROJECT).*

.PHONY: build
build: dist

.PHONY: dist
dist: install $(DIST_FILES)
$(DIST_FILES): $(MODULES)
	rm -f $(DIST_FILES)
	$(SETUP) check --strict --metadata --restructuredtext
	$(SETUP) sdist
	$(SETUP) bdist_wheel
	$(TWINE) check dist/*


.PHONY: exe
exe: install $(EXE_FILES)
$(EXE_FILES): $(MODULES) $(PROJECT).spec
	# For framework/shared support: https://github.com/yyuu/pyenv/wiki
	$(PYINSTALLER) $(PROJECT).spec --noconfirm --clean

$(PROJECT).spec:
	$(PYINSTALLER_MAKESPEC) $(PACKAGE)/__main__.py --onefile --windowed --name=$(PROJECT)

# RELEASE #####################################################################

SETUP := pipenv run python setup.py
TWINE := pipenv run twine

.PHONY: upload
upload: dist ## Upload the current version to PyPI
	git diff --name-only --exit-code
	$(TWINE) upload dist/*.*
	bin/open https://pypi.org/project/$(PROJECT)

.PHONY: upload-test
upload-test: dist ## Upload the current version to Test PyPI
	git diff --name-only --exit-code
	$(TWINE) upload --repository-url https://test.pypi.org/legacy/ dist/*.*
	bin/open https://test.pypi.org/project/$(PROJECT)


# CLEANUP #####################################################################

.PHONY: clean
clean: .clean-build .clean-docs .clean-test .clean-install ## Delete all generated and temporary files

.PHONY: clean-all
clean-all: clean
	rm -rf $(VENV)

.PHONY: .clean-install
.clean-install:
	find $(PACKAGES) -name '*.pyc' -delete
	find $(PACKAGES) -name '__pycache__' -delete
	rm -rf *.egg-info

.PHONY: .clean-test
.clean-test:
	rm -rf .cache .pytest .coverage htmlcov xmlreport

.PHONY: .clean-docs
.clean-docs:
	# rm -rf *.rst docs/apidocs *.html docs/*.png site
	cd docs && pipenv run make clean
	cd docs/notebooks && find . ! -name '*.ipynb' -type f -exec rm -rf {} + && rm -rf -- ./*/
	cd docs && rm -f readme.rst
	rm -rf docs/about

.PHONY: .clean-build
.clean-build:
	rm -rf *.spec dist build

# HELP ########################################################################

.PHONY: help
help: all
	@ grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
