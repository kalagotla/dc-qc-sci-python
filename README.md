# Scientific Python for Engineers Workshop

A 3-hour hands-on workshop introducing Python for scientific computing, data processing, and deep learning.

## Workshop Overview

| Hour | Topic | Description |
|------|-------|-------------|
| 1 | Python Basics & Environment Setup | Anaconda, venvs, pip, uv, Python fundamentals |
| 2 | Scientific Data Processing | Loading, processing, and visualizing research data |
| 3 | Deep Learning with PyTorch | Introduction to DNNs + ML Challenge invitation |

## Quick Start

### Option 1: Using uv (Recommended - Fastest Setup)

[uv](https://github.com/astral-sh/uv) is a modern Python package manager that can replace Anaconda for most workflows.

**Install uv:**

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Setup the workshop environment:**

```bash
# Clone the repository
git clone https://github.com/kalagotla/dc-qc-sci-python.git
cd dc-qc-sci-python

# Create environment and install dependencies (one command!)
uv sync

# Launch Jupyter Lab
uv run jupyter lab
```

### Option 2: Using Anaconda (Traditional Approach)

**Install Anaconda:**

Download from [anaconda.com/download](https://www.anaconda.com/download)

**Setup the workshop environment:**

```bash
# Clone the repository
git clone https://github.com/kalagotla/dc-qc-sci-python.git
cd dc-qc-sci-python

# Create conda environment
conda create -n sci-python python=3.11 -y
conda activate sci-python

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Lab
jupyter lab
```

### Option 3: Using pip with venv

```bash
# Clone the repository
git clone https://github.com/kalagotla/dc-qc-sci-python.git
cd dc-qc-sci-python

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Lab
jupyter lab
```

## Repository Structure

```
dc-qc-sci-python/
├── README.md                    # This file
├── requirements.txt             # Dependencies for pip
├── pyproject.toml               # Project config (for uv)
│
├── hour1_basics/                # Hour 1: Python Basics
│   ├── 01_environment_setup.md  # Setup guide
│   ├── 02_python_basics.ipynb   # Python fundamentals
│   └── modules/                 # Example module
│       └── my_functions.py
│
├── hour2_scientific/            # Hour 2: Scientific Computing
│   ├── 01_data_loading.ipynb    # Data processing tutorial
│   ├── modules/                 # Helpers for Hour 2
│   │   ├── animation.py         # Flow field animation
│   │   └── helpers.py           # Plot and data utilities
│   └── data/
│       ├── sample_data.csv      # Sample experimental data (Part 1)
│       └── cylinder/            # LES cylinder CFD subset (Part 2)
│           ├── README.md        # Data description
│           ├── cylinder.sp.x   # Grid file
│           └── sol-*.q         # Flow snapshots (e.g. lptlib format)
│
├── hour3_ml/                    # Hour 3: Machine Learning
│   ├── data/
│   │   └── airfoil_self_noise.dat
│   ├── simple_dnns/             # DNN package
│   │   ├── network.py, univariate.py, bivariate.py, simple_data.py, noise_data.py
│   │   └── pretrained_models/  # Saved checkpoints (.tar) and generate.py
│   ├── models/                  # Saved model checkpoints
│   ├── dnns_with_pytorch.ipynb  # PyTorch tutorial
│   └── ml_challenge_info.md     # ML Challenge info
```

## Prerequisites

- Basic familiarity with programming concepts
- A computer with internet access
- Willingness to learn!

## Workshop Agenda

### Hour 1: Python Basics & Environment Setup (60 min)

1. **Environment Setup** (15 min)
   - Installing Python (Anaconda vs uv)
   - Virtual environments explained
   - Package managers: pip, conda, uv

2. **Python Fundamentals** (45 min)
   - Data types and variables
   - Lists, tuples, and dictionaries
   - Control flow (if/else, loops)
   - Functions and modules
   - Importing code from other files

### Hour 2: Scientific Data Processing (60 min)

1. **NumPy Arrays** (20 min)
   - Creating and manipulating arrays
   - Array operations and broadcasting
   - Aggregations and statistics

2. **Data Loading with Pandas** (20 min)
   - Reading CSV, Excel files
   - DataFrames and Series
   - Data exploration and cleaning

3. **Visualization with Matplotlib** (20 min)
   - Line plots, scatter plots
   - Customizing figures
   - Saving publication-ready figures

### Hour 3: Deep Learning with PyTorch (60 min)

1. **Introduction to DNNs** (30 min)
   - What is a neural network?
   - Training process explained
   - Hands-on with simple datasets

2. **Practical Examples** (20 min)
   - Univariate regression
   - Bivariate regression
   - Model validation

3. **ML Challenge & Next Steps** (10 min)
   - Introduction to [Fluids Challenge](https://fluids-challenge.engin.umich.edu/)
   - Upcoming workshop series
   - Resources for continued learning

## Troubleshooting

### Common Issues

**"Command not found" for uv:**
- Restart your terminal after installation
- On Windows, you may need to add uv to your PATH

**Jupyter Lab won't start:**
```bash
# Make sure you're in the right environment
# For uv:
uv run jupyter lab

# For conda:
conda activate sci-python
jupyter lab
```

**Import errors in notebooks:**
- Make sure you've installed all dependencies
- Restart the Jupyter kernel after installing new packages

## Resources

- [Python Documentation](https://docs.python.org/3/)
- [NumPy User Guide](https://numpy.org/doc/stable/user/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

## Contact

**Dilip Kalagotla**
- Email: dilip.kalagotla@gmail.com

**Elijah LaLonde**
- Email elalonde@fsu.edu
## License

This workshop material is provided under the MIT License. See [LICENSE](LICENSE) for details.
