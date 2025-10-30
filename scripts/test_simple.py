#!/usr/bin/env python3
print("Hello from test script!")
import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
except ImportError as e:
    print(f"PyTorch import error: {e}")

try:
    import matplotlib
    print(f"Matplotlib version: {matplotlib.__version__}")
except ImportError as e:
    print(f"Matplotlib import error: {e}")

try:
    import pandas as pd
    print(f"Pandas version: {pd.__version__}")
except ImportError as e:
    print(f"Pandas import error: {e}")

print("Test script completed!")
