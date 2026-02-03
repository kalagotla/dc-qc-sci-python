"""
Pre-trained model checkpoints for the DNN workshop.

This package contains pre-trained model checkpoints that can be used
to skip long training sessions during lectures.

Available Models:
    - univariate.tar: Pre-trained univariate regression model
    - bivariate.tar: Pre-trained bivariate regression model
    - simpledata.tar: Pre-trained noisy data regression model

Usage:
    from simple_dnns.pretrained_models import MODELS_DIR
    from simple_dnns import Univariate
    
    model = Univariate()
    model.set_parameters()
    model.continue_train(filename=str(MODELS_DIR / 'univariate.tar'))
"""

from pathlib import Path

# Directory containing the pre-trained model files
MODELS_DIR = Path(__file__).parent

# Convenience paths for each model
UNIVARIATE_MODEL = MODELS_DIR / 'univariate.tar'
BIVARIATE_MODEL = MODELS_DIR / 'bivariate.tar'
SIMPLEDATA_MODEL = MODELS_DIR / 'simpledata.tar'

__all__ = ['MODELS_DIR', 'UNIVARIATE_MODEL', 'BIVARIATE_MODEL', 'SIMPLEDATA_MODEL']
