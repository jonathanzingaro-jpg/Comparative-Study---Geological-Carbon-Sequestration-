# Comparative Study of Geological Carbon Sequestration

Training and inference scripts for Temporal CNN, FNO, U-FNO, U-Net, and V-Net models for geological carbon sequestration.

---

## Requirements

- Python 3.12+
- pip 24.0.0+

Verify your versions:

```shell
python --version
pip --version
```

To upgrade Python, see the [Python Downloads](https://www.python.org/downloads/) page.  
To upgrade pip:

```shell
pip install --upgrade pip
```

---

## Setup

### Extract the Dataset

The dataset is provided as a compressed `.zip` file in the `Dataset/` directory. Extract it before running any scripts:

```shell
unzip Dataset/<dataset_name>.zip -d Dataset/
```

### Create a Virtual Environment

From the project root:

```shell
python -m venv .venv
```

Then activate it for your OS:

**macOS / Linux**

```shell
source .venv/bin/activate
```

**Windows**

```shell
.venv\Scripts\activate
```

Then install dependencies:

```shell
pip install -r requirements.txt
```

---

## Data Preprocessing

Before training, preprocess the raw dataset by running:

```shell
python RunDataPreprocessing.py
```

---

## Training

Training scripts are located in `Run Files/`. Each script corresponds to a different model architecture. Run the script from the project root:

| Model | Command |
|-------|---------|
| CNN   | `python "Run Files/dP/RunTraining_CNN.py"` |
| FNO   | `python "Run Files/dP/RunTraining_FNO.py"` |
| UFNO  | `python "Run Files/dP/RunTraining_UFNO.py"` |
| UNet  | `python "Run Files/dP/RunTraining_UNet.py"` |
| VNet  | `python "Run Files/dP/RunTraining_VNet.py"` |

Configuration options for each model can be adjusted in the `Config/` directory.

---

## Post-Processing

After training, run the post-processing script to evaluate and export results:

```shell
python RunPostProcessing.py
```

---

## Project Structure

```
comparative-study-geological-carbon-sequestration/
├── Config/                  # Model and training configuration files
├── Dataset/                 # Compressed dataset (.zip)
├── Functions/               # Shared functions
├── Run Files/
│   ├── dP/                  # Pressure build-up model scripts
│   │   ├── RunTraining_CNN.py
│   │   ├── RunTraining_FNO.py
│   │   ├── RunTraining_UFNO.py
│   │   ├── RunTraining_UNet.py
│   │   └── RunTraining_VNet.py
│   └── SG/                  # CO2 saturation model scripts
├── RunDataPreprocessing.py  # Data preprocessing
├── RunPostProcessing.py     # Post-processing
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Authors

Dataset originally developed by [Gege Wen](https://github.com/gegewen) and subsequently modified by the authors.
