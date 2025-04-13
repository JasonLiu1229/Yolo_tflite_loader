#!/bin/bash

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install Python3 and try again."
    exit 1
fi

# create a virtual environment
if [ ! -d "venv" ]; then
    echo "Creating a virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "pip3 is not installed. Installing pip..."
    python3 -m ensurepip --upgrade
fi

# Install required Python packages
if [ -f "requirements.txt" ]; then
    echo "Installing Python requirements from requirements.txt..."
    pip3 install -r requirements.txt
else
    echo "requirements.txt not found. Please create the file and list your dependencies."
    exit 1
fi

echo "Installation complete."
