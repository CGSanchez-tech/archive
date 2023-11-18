# paws-and-reflect

hihihihihihihii
<img src="https://media.tenor.com/w_xkJNZpzhgAAAAM/goofy.gif" width="40" height="40">

## Usage

To streamline the setup and management of the project, you can use the provided Makefile commands. Ensure you have `make` installed on your system to use these commands.

### Makefile Commands

- **Install Virtualenv**:
  Installs virtualenv globally if it's not already installed.

  ```bash
  make install-virtualenv
  ```

- **Install Dependencies**:
  Creates the virtual environment in the project root and installs required dependencies.

  ```bash
  make bin/activate
  ```

- **Run the Application**:
  Runs the application using the virtual environment.

  ```bash
  make run
  ```

- **Freeze Dependencies**:
  Generates or updates the `requirements.txt` file with the current package installations.

  ```bash
  make freeze-dependencies
  ```

- **Clean Up**:
  Removes virtual environment and `__pycache__` directories.

  ```bash
  make clean
  ```

### Manual Setup

If you prefer not using Makefile commands, follow these steps:

#### Install Virtualenv

```bash
pip install virtualenv
```

#### Create Virtualenv and Activate

```bash
python3 -m venv .
source ./bin/activate
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Run the Application

```bash
python3 app.py
```

#### Freeze Dependencies

```bash
pip freeze -l > requirements.txt
```

#### Exit Virtualenv

```bash
deactivate
```
