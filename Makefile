FLAKE8 ?= pipenv run flake8 --ignore=E501
MYPY ?= pipenv run mypy --pretty

flake8:
	$(FLAKE8) image_classifier

mypy:
	$(MYPY) image_classifier
