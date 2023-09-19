# Development Setup

---
## Setup
### Create Virtual environment
***Optional***

```bash
python -m venv .venv
```

Then, open a new terminal and activate environment

### Install dependencies

```bash
pip install -r requirements.txt
```

### Download dataset

1. Download zip file from [official source](https://www.physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip)
2. Unzip folder and make inside the project
3. Rename and move dataset

```bash
mkdir data
mv "mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0" "data/mitdb"
rmdir "mit-bih-arrhythmia-database-1.0.0"
rmdir "mit-bih-arrhythmia-database-1.0.0.zip"
```

## Deployment

### Preprocess dataset

1. Activate environment **[Given a virtual environment]**

2. Execute script

```bash
python preprocess.py
```

### Train PACE

1. Activate environment **[Given a virtual environment]**

2. Execute script

```bash
python train.py > output.txt
```