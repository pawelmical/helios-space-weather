#!/usr/bin/env python
"""
HELIOS Training Runner
======================
Run this script to train the Bz prediction model.

This script handles the Python path configuration to avoid the 'code' module
naming conflict between the project's code/ directory and Python's built-in
code module (which is required by PyTorch).

Usage:
    python run_training.py
    python run_training.py --epochs 100 --batch-size 64

Author: HELIOS Team
Date: February 2026
"""

import sys
import os

# CRITICAL: Fix the 'code' module conflict BEFORE any other imports
# Remove the project's 'code' module from cache and path

# Remove 'code' from sys.modules if it was loaded
for key in list(sys.modules.keys()):
    if key == 'code' or key.startswith('code.'):
        del sys.modules[key]

# Remove paths that contain the project's code directory
_script_dir = os.path.dirname(os.path.abspath(__file__))
_clean_path = []
for p in sys.path:
    if p == '' or p == '.':
        # Replace empty/current dir with absolute path but mark for later
        continue
    if p and os.path.exists(os.path.join(p, 'code', '__init__.py')):
        continue
    _clean_path.append(p)

# Add the project directory back (we need it for NeuralNetwork_ML)
_clean_path.insert(0, _script_dir)
sys.path = _clean_path

# Now it's safe to import torch and our modules
if __name__ == '__main__':
    from NeuralNetwork_ML.train import main
    sys.exit(main())
