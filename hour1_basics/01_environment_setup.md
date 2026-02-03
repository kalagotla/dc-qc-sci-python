# Python Environment Setup Guide

This guide covers different approaches to setting up a Python environment for scientific computing.

## Table of Contents

1. [Why Virtual Environments?](#why-virtual-environments)
2. [Option 1: Anaconda](#option-1-anaconda)
3. [Option 2: uv (Modern Alternative)](#option-2-uv-modern-alternative)
4. [Option 3: pip + venv](#option-3-pip--venv)
5. [Package Managers Comparison](#package-managers-comparison)
6. [Which Should I Use?](#which-should-i-use)

---

## Why Virtual Environments?

Virtual environments solve the "it works on my machine" problem by:

- **Isolating dependencies**: Each project has its own packages
- **Avoiding version conflicts**: Project A can use numpy 1.24, Project B can use numpy 1.26
- **Reproducibility**: Share exact dependencies with collaborators
- **Clean uninstall**: Delete the environment, everything is gone

### Without Virtual Environments (Bad!)

```
System Python
├── numpy 1.24 (Project A needs this)
├── numpy 1.26 (Project B needs this) ← CONFLICT!
└── matplotlib 3.7
```

### With Virtual Environments (Good!)

```
Project A Environment          Project B Environment
├── numpy 1.24                 ├── numpy 1.26
├── matplotlib 3.7             ├── matplotlib 3.8
└── pandas 2.0                 └── scipy 1.11
```

---

## Option 1: Anaconda

**Anaconda** is a popular Python distribution that includes:
- Python interpreter
- conda package manager
- 1500+ pre-installed scientific packages
- Anaconda Navigator (GUI)

### Installation

1. Download from [anaconda.com/download](https://www.anaconda.com/download)
2. Run the installer
3. Follow the prompts (accept defaults)

### Creating Environments

```bash
# Create a new environment with Python 3.11
conda create -n myproject python=3.11

# Activate the environment
conda activate myproject

# Install packages
conda install numpy pandas matplotlib

# Or use pip within conda
pip install torch torchvision

# Deactivate when done
conda deactivate

# List all environments
conda env list

# Remove an environment
conda env remove -n myproject
```

### Pros and Cons

**Pros:**
- Easy to install
- Includes many scientific packages
- GUI available (Anaconda Navigator)
- Good for beginners

**Cons:**
- Large download (~500MB)
- Can be slow to resolve dependencies
- Sometimes conflicts with pip packages

---

## Option 2: uv (Modern Alternative)

**uv** is a new, extremely fast Python package manager written in Rust. It can replace pip, venv, and conda for most use cases.

### Installation

```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

### Creating Projects

```bash
# Initialize a new project (creates pyproject.toml)
uv init myproject
cd myproject

# Add dependencies
uv add numpy pandas matplotlib torch

# Sync dependencies (creates .venv automatically)
uv sync

# Run Python within the environment
uv run python script.py

# Run Jupyter Lab
uv run jupyter lab

# Remove a dependency
uv remove pandas
```

### Key Commands

| Command | Description |
|---------|-------------|
| `uv init` | Create new project |
| `uv add <pkg>` | Add a dependency |
| `uv remove <pkg>` | Remove a dependency |
| `uv sync` | Install all dependencies |
| `uv run <cmd>` | Run command in environment |
| `uv pip install <pkg>` | Direct pip-like install |

### Pros and Cons

**Pros:**
- Extremely fast (10-100x faster than pip/conda)
- Modern dependency resolution
- Single tool for everything
- Small download size

**Cons:**
- Newer tool (less documentation)
- Some exotic packages may have issues
- Learning curve if used to conda

---

## Option 3: pip + venv

The built-in Python approach using `venv` (virtual environments) and `pip` (package installer).

### Creating Environments

```bash
# Create a virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate (Windows CMD)
.venv\Scripts\activate.bat

# Activate (macOS/Linux)
source .venv/bin/activate

# Install packages
pip install numpy pandas matplotlib

# Install from requirements.txt
pip install -r requirements.txt

# Save current packages to requirements.txt
pip freeze > requirements.txt

# Deactivate
deactivate
```

### Pros and Cons

**Pros:**
- Built into Python (no extra install)
- Simple and well-documented
- Works everywhere

**Cons:**
- Slower than uv
- No dependency resolver (can have conflicts)
- Manual environment activation

---

## Package Managers Comparison

| Feature | Anaconda/conda | uv | pip + venv |
|---------|---------------|-----|------------|
| Speed | Slow | Very Fast | Medium |
| Dependency Resolution | Good | Excellent | Basic |
| Ease of Use | Easy | Medium | Easy |
| Install Size | Large | Small | None |
| Scientific Packages | Excellent | Good | Good |
| GPU Support (CUDA) | Excellent | Good | Manual |

---

## Which Should I Use?

### Use **Anaconda** if:
- You're new to Python
- You want a GUI (Anaconda Navigator)
- You need easy CUDA/GPU setup
- You prefer everything pre-installed

### Use **uv** if:
- You want fast installs
- You're comfortable with command line
- You want modern tooling
- You're starting a new project

### Use **pip + venv** if:
- You need the simplest setup
- You're on a restricted system
- You're already familiar with it
- You can't install extra tools

---

## Workshop Setup

For this workshop, we'll use **uv** (recommended) or **pip**:

### Using uv

```bash
# Clone the repo
git clone https://github.com/kalagotla/dc-qc-sci-python.git
cd dc-qc-sci-python

# Install everything and launch Jupyter
uv sync
uv run jupyter lab
```

### Using pip

```bash
# Clone the repo
git clone https://github.com/kalagotla/dc-qc-sci-python.git
cd dc-qc-sci-python

# Create and activate environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Install and launch
pip install -r requirements.txt
jupyter lab
```

---

## Common Issues

### "conda: command not found"
- Close and reopen your terminal
- Make sure "Add to PATH" was checked during install

### "uv: command not found"
- Restart your terminal after installing
- Windows: Check if `%USERPROFILE%\.cargo\bin` is in PATH

### "pip install fails with permission error"
- Use a virtual environment (don't install globally)
- Never use `sudo pip install` on Linux/Mac

### "Module not found" in Jupyter
- Make sure you're running Jupyter from the activated environment
- Use `uv run jupyter lab` or activate venv first

---

## Next Steps

Now that your environment is set up, open `02_python_basics.ipynb` to start learning Python fundamentals!
