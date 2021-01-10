FLAKE8 ?= pipenv run flake8
MYPY ?= pipenv run mypy --pretty

flake8:
	$(FLAKE8) src/image_classifier

mypy:
	$(MYPY) src/image_classifier
