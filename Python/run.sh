#!/bin/bash
echo "Starting Object Detection System"
cd ~/segregation_system/
source .venv/bin/activate

cd object_detection/
python pd.py
