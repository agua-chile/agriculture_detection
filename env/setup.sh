#!/usr/bin/env bash
# Simple script to set up a Python virtual environment and install packages from requirements.txt
# to run: chmod +x setup.sh && ./setup.sh

set -euo pipefail

# Create virtual environment with Python
python_version="3.10"  # specify desired Python version here
echo "Creating virtual environment with Python $python_version..."
python$python_version -m venv .venv

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Make sure venv Python is being used
PYTHON=".venv/bin/python"
PIP=".venv/bin/pip"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Error: Virtual environment Python not found"
  exit 1
fi

# Upgrade pip, setuptools, wheel
echo "Upgrading pip, setuptools, wheel..."
"$PYTHON" -m pip install --upgrade pip setuptools wheel

# Install all packages from exported requirements.txt
if [[ ! -f requirements.txt ]]; then
  echo "Error: requirements.txt not found in current directory."
  exit 1
fi

echo "Installing all packages from requirements.txt..."
"$PIP" install -r requirements.txt

# Install ipykernel separately to make notebooks work
echo "Installing ipykernel..."
"$PIP" install -U ipykernel

echo "Setup complete! Virtual environment created and activated using $("$PYTHON" --version)."
