"""
simple_dnns - Simple Deep Neural Networks Package

A package for learning Deep Neural Networks with PyTorch.
Provides easy-to-use classes for different regression problems.

Available Classes:
    - Univariate: 1D input → 1D output (sin(x) + cos(x))
    - Bivariate: 2D input → 1D output (sin(x1) + cos(x2))
    - SimpleData: Noisy 1D data regression
    - NoiseData: Real-world airfoil noise data (requires data file)
    - Net: Configurable feedforward neural network

Pre-trained Models:
    Pre-trained model checkpoints are available in the pretrained_models subpackage:
    
    from simple_dnns.pretrained_models import MODELS_DIR, UNIVARIATE_MODEL
    model.continue_train(filename=str(UNIVARIATE_MODEL))

Example Usage:
    from simple_dnns import Univariate
    
    model = Univariate(epochs=100, learning_rate=0.01)
    model.set_parameters()
    model.train()
    model.validation()
"""

from .univariate import Univariate
from .bivariate import Bivariate
from .simple_data import SimpleData
from .noise_data import NoiseData
from .network import Net
from . import pretrained_models

__version__ = "1.0.2"
__author__ = "Dilip Kalagotla"
