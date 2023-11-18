# Makefile for managing Python virtual environment and dependencies

# Rule to create and activate virtual environment
bin/activate: requirements.txt
	python3 -m venv .
	./bin/pip install -r requirements.txt

# Rule to run the application using the virtual environment
run: bin/activate
	./bin/python3 app.py

# Rule to clean up __pycache__ and the virtual environment
clean:
	rm -rf __pycache__
	rm -rf bin lib lib64 include pyvenv.cfg

# Rule to install virtualenv globally (if not already installed)
install-virtualenv:
	pip install virtualenv

# Rule to freeze dependencies into requirements.txt
freeze-dependencies:
	./bin/pip freeze -l > requirements.txt
