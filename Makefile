# Define shell
SHELL := /bin/zsh

# Makefile targets
.PHONY: install-virtualenv activate-venv install-dependencies deactivate-venv freeze-dependencies

# Install virtualenv
install-virtualenv:
	@pip install virtualenv

# Activate virtual environment
activate-venv:
	@source ./bin/activate

# Install project dependencies
install-dependencies:
	@pip install -r requirements.txt

# Deactivate virtual environment
deactivate-venv:
	@deactivate

# Freeze dependencies into requirements.txt
freeze-dependencies:
	@pip freeze -l > requirements.txt
