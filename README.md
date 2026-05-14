# MediLens - Medical Image Analysis Platform

An end-to-end AI-powered medical imaging platform that processes DICOM scans, detects tumors using deep learning, and visualizes results through an interactive web dashboard.

## What it does
- Loads and processes real DICOM medical scans (CT/MRI)
- Detects tumors and anomalies using a trained UNet model
- Shows results in a dark-themed web dashboard
- Tracks all ML experiments with MLflow
- Interactive slice viewer to scroll through scan series

## Tech Stack
- Medical Imaging: pydicom, MONAI
- Deep Learning: PyTorch, UNet
- Experiment Tracking: MLflow
- Web Backend: Flask
- Dataset: TCIA HCC-TACE-Seg (real clinical data)

## Quick Start
git clone https://github.com/deeksha26052003/medilens.git
cd medilens
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 src/tumor_detection.py
python3 app.py

## Results
- Trained on real HCC liver CT dataset from TCIA
- Loss reduced from 0.55 to 0.31 over 15 epochs
- Detects anomalies with percentage coverage reporting
- Processes 512x512 DICOM slices in real-time

## Author
Deeksha Bankapur - MS Computer Science, Northeastern University
