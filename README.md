# paws-and-reflect

hihihihihihihii
<img src="https://media.tenor.com/w_xkJNZpzhgAAAAM/goofy.gif" width="40" height="40">

## Usage

To streamline the setup and management of the project, you can use the provided Makefile commands. Ensure you have `make` installed on your system to use these commands.

### Makefile Commands

- **Install Virtualenv**:
  Installs virtualenv if it's not already installed.

  ```bash
  make install-virtualenv
  ```

- **Activate Virtual Environment**:
  Activates the Python virtual environment. Note: This needs to be done directly in the shell for it to persist.

  ```bash
  source ./bin/activate
  ```

- **Install Dependencies**:
  Installs the required dependencies for the project.

  ```bash
  make install-dependencies
  ```

- **Deactivate Virtual Environment**:
  Deactivates the Python virtual environment. This should be run directly in the shell.

  ```bash
  deactivate
  ```

- **Freeze Dependencies**:
  Generates or updates the `requirements.txt` file with the current package installations.

  ```bash
  make freeze-dependencies
  ```

### Manual Setup

If you prefer not using Makefile commands, follow these steps:

#### Install Virtualenv

```bash
pip install virtualenv
```

#### Activate Virtualenv

```bash
source ./bin/activate
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Exit Virtualenv

```bash
deactivate
```

#### Freeze Dependencies

```bash
pip freeze -l > requirements.txt
```
