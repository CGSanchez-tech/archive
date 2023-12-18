<div style="display: flex; justify-content: center;">
  <h1> Paws & Reflect </h1>
  <img src="https://media.tenor.com/w_xkJNZpzhgAAAAM/goofy.gif" width="80" height="80">
</div>

## Usage

### Install Virtualenv

```bash
pip install virtualenv
```

### Create Virtualenv and Activate

```bash
python3.11 -m venv venv
source ./venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** Some libraries, particularly TensorFlow, might require manual installation within the virtual environment due to specific compatibility needs.
If you encounter any installation issues, try installing these libraries separately using pip install.

### Run the Application

```bash
python3 app.py
```

### Freeze Dependencies

To capture the current state of the virtual environment, run the following command:

```bash
pip freeze -l > requirements.txt
```

### Exit Virtualenv

```bash
deactivate
```

### Acknowledgements

[Kaggle Dataset](https://www.kaggle.com/datasets/anshtanwar/pets-facial-expression-dataset)
