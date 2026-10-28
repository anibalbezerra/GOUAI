"""Lets the step scripts import gouai_train without installing the package (python scripts/NN_*.py ...)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
