# Code to reproduce figures from:  
McHugh SB, Lopes-dos-Santos V, Castelli M, Gava GP, Thompson SE, Tam SKE, Hartwich K, Perry B, Toth R, Denison T, Sharott, A, Dupret D (2024). 
Offline hippocampal reactivation during dentate spikes supports flexible memory. Neuron 112(22): 1-14.
https://doi.org/10.1016/j.neuron.2024.08.022

## What this repo does
This repository contains code to reproduce Figure panels in the published manuscript.

## Quickstart
```bash
git clone https://github.com/yourusername/neuron_2024_repro.git
cd neuron_2024_repro
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

## Full set-up
## 📦 Setup Instructions

### 1️⃣ Install Git LFS (Required for Data Files)

This repository stores large `.npy` files using **Git Large File Storage (LFS)**.

Install Git LFS **before cloning** the repository.

**macOS (Homebrew)**

```bash
brew install git-lfs
```

**Ubuntu / Debian**

```bash
sudo apt update
sudo apt install git-lfs
```

**Conda**

```bash
conda install -c conda-forge git-lfs
```

Initialize Git LFS:

```bash
git lfs install
```

---

### 2️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/neuron_2024_repro.git
cd neuron_2024_repro
```

Download the large data files:

```bash
git lfs pull
```

---

### 3️⃣ Create Python 3.10 Environment

This project requires **Python 3.10**.

#### Using `venv`

```bash
python3.10 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

#### OR using Conda

```bash
conda create -n neuron2024 python=3.10
conda activate neuron2024
pip install -r requirements.txt
```

---

### 4️⃣ Launch Jupyter

```bash
python3.10 -m jupyter notebook
```

Open the notebooks and run the cells.

---

## ⚠️ Troubleshooting

If you encounter an error such as:

```
UnpicklingError: invalid load key, 'v'
```

This indicates that Git LFS is not installed or the data files were not downloaded correctly.

Run:

```bash
git lfs install
git lfs pull
```

and restart the notebook.

---

## 📂 Data Storage

Large `.npy` data files are stored using Git LFS.
Git LFS must be installed to correctly download and use the data files in this repository.

