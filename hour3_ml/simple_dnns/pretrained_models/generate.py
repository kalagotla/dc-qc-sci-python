"""
Generate pre-trained model checkpoints for the DNN workshop.

This script trains and saves model checkpoints so that students can
skip long training sessions during the lecture if needed.

Run this script from the hour3_ml directory:
    python -m simple_dnns.pretrained_models.generate

Or from the project root:
    uv run python -m hour3_ml.simple_dnns.pretrained_models.generate
"""

import sys
import os
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()

# Add hour3_ml to path for imports
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))

import torch
from simple_dnns import Univariate, Bivariate, SimpleData


def get_model_path(filename):
    """Get the full path for a model file in the pretrained_models directory."""
    return str(SCRIPT_DIR / filename)


def generate_univariate_model():
    """Train and save a univariate model."""
    print("Training Univariate model...")
    model = Univariate(
        layers=[1, 5, 10, 5, 1],
        epochs=200,
        learning_rate=0.01,
        momentum=0.9
    )
    model.set_parameters()
    
    output_path = get_model_path('univariate.tar')
    model.train(plot=False, save_at=200, filename=output_path)
    
    # Save final model
    torch.save({
        'epoch': 200,
        'model_state_dict': model.net.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
        'layers': model.layers,
    }, output_path)
    print(f"  Saved to {output_path}")


def generate_bivariate_model():
    """Train and save a bivariate model."""
    print("Training Bivariate model...")
    model = Bivariate(
        layers=[2, 20, 20, 1],
        epochs=200,
        learning_rate=0.1,
        momentum=0.7
    )
    model.set_parameters()
    
    output_path = get_model_path('bivariate.tar')
    model.train(plot=False, save_at=200, filename=output_path)
    
    # Save final model
    torch.save({
        'epoch': 200,
        'model_state_dict': model.net.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
        'layers': model.layers,
    }, output_path)
    print(f"  Saved to {output_path}")


def generate_simpledata_model():
    """Train and save a simple data model."""
    print("Training SimpleData model...")
    model = SimpleData(
        layers=[1, 5, 5, 5, 5, 1],
        epochs=200,
        learning_rate=0.01,
        momentum=0.7
    )
    model.set_parameters()
    
    output_path = get_model_path('simpledata.tar')
    model.train(plot=False, save_at=200, filename=output_path)
    
    # Save final model
    torch.save({
        'epoch': 200,
        'model_state_dict': model.net.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
        'layers': model.layers,
    }, output_path)
    print(f"  Saved to {output_path}")


if __name__ == '__main__':
    print("Generating pre-trained model checkpoints...")
    print("=" * 50)
    
    generate_univariate_model()
    generate_bivariate_model()
    generate_simpledata_model()
    
    print("=" * 50)
    print(f"Done! Pre-trained models saved to {SCRIPT_DIR}")
    print("\nTo use a pre-trained model in the notebook:")
    print("  from simple_dnns.pretrained_models import MODELS_DIR")
    print("  model.continue_train(filename=str(MODELS_DIR / 'univariate.tar'))")
