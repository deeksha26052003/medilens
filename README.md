# 🔬 MediLens — Medical Image Analysis Platform

> AI-powered medical imaging platform for real-time tumor detection and DICOM scan analysis

![Python](https://img.shields.io/badge/Python-3.14-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red) ![MONAI](https://img.shields.io/badge/MONAI-1.5-green) ![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey)

## 🎯 Overview

MediLens is an end-to-end medical AI platform that:

- Loads and processes real **DICOM** medical scans (CT/MRI)
- Detects tumors and anomalies using a trained **UNet** model
- Visualizes results in an interactive **dark-themed web dashboard**
- Tracks all ML experiments with **MLflow**
- Trained on real clinical data from **TCIA HCC-TACE-Seg** dataset

## 🌐 Web Dashboard

- Upload any DICOM file
- Get instant tumor detection results
- View CT scan + detection mask + overlay side by side
- Anomaly coverage percentage
- Patient metadata display

## 🧠 Tech Stack

| Layer                 | Technology                                 |
| --------------------- | ------------------------------------------ |
| Medical Imaging       | pydicom, nibabel                           |
| AI Segmentation       | MONAI, PyTorch UNet                        |
| Experiment Tracking   | MLflow                                     |
| Web Backend           | Flask                                      |
| Dataset               | TCIA HCC-TACE-Seg (real clinical CT scans) |
| Distributed Computing | Databricks (WIP)                           |

## 🏗️ Architecture

DICOM File
↓
dicom_loader.py — load + normalize
↓
segmentation.py — UNet predicts mask
↓
tumor_detection.py — trained on HCC liver CT data
↓
tracking.py — MLflow logs every run
↓
app.py — Flask web dashboard

## 🚀 Quick Start

```bash
git clone https://github.com/deeksha26052003/medilens.git
cd medilens
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 src/tumor_detection.py
python3 app.py
```

Open **http://127.0.0.1:5001**

## 📊 Model Performance

- Dataset: TCIA HCC-TACE-Seg (real liver CT scans)
- Training slices: 50
- Epochs: 15
- Loss: 0.55 → 0.31
- Image size: 512x512

## 🗂️ Project Structure

medilens/
├── src/
│ ├── ingestion/ # DICOM loader
│ ├── segmentation/ # MONAI UNet
│ ├── viewer/ # Interactive viewer
│ ├── train.py # Model training
│ ├── tumor_detection.py # Tumor detection
│ └── tracking.py # MLflow logging
├── templates/ # Web UI
├── models/ # Saved models
├── data/ # DICOM datasets
└── app.py # Flask web app

## 🔮 Future Work

- Databricks distributed training
- Slice selector in web UI
- PDF report download
- Confidence scores
- Fine-tune on larger labeled dataset
- De-identification pipeline for real patient data

## 👩‍💻 Author

**Deeksha Manjunatha Bankapur** — MS Computer Science, Northeastern University (Khoury College)
