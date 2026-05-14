from src.segmentation.segmentation import run_segmentation, get_model
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
import os
import sys
import torch
import numpy as np
import pydicom
import base64
import io
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs('uploads', exist_ok=True)

# Load tumor detection model


def load_tumor_model():
    from monai.networks.nets import UNet
    from monai.networks.layers import Norm
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH
    )
    if os.path.exists('models/tumor_detection.pth'):
        model.load_state_dict(torch.load(
            'models/tumor_detection.pth', map_location='cpu'))
    model.eval()
    return model


tumor_model = load_tumor_model()


def process_dicom(file_path):
    ds = pydicom.dcmread(file_path)
    img = ds.pixel_array.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img, ds


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight',
                facecolor='#0f0f1a', edgecolor='none')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    try:
        img, ds = process_dicom(file_path)

        # Run tumor detection
        img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)
        img_resized = torch.nn.functional.interpolate(
            img_tensor, size=(128, 128))

        with torch.no_grad():
            output = tumor_model(img_resized)
            mask = torch.argmax(output, dim=1).squeeze().numpy()

        # Original scan
        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 6))
        fig1.patch.set_facecolor('#0f0f1a')
        ax1.imshow(img, cmap='gray')
        ax1.set_title('CT Scan', color='white', fontsize=14)
        ax1.axis('off')
        original_b64 = fig_to_base64(fig1)
        plt.close(fig1)

        # Tumor mask
        fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
        fig2.patch.set_facecolor('#0f0f1a')
        ax2.imshow(mask, cmap='jet')
        ax2.set_title('Tumor Detection', color='white', fontsize=14)
        ax2.axis('off')
        mask_b64 = fig_to_base64(fig2)
        plt.close(fig2)

        # Overlay
        fig3, ax3 = plt.subplots(1, 1, figsize=(6, 6))
        fig3.patch.set_facecolor('#0f0f1a')
        ax3.imshow(img, cmap='gray')
        ax3.imshow(mask, cmap='Reds', alpha=0.5)
        ax3.set_title('Overlay', color='white', fontsize=14)
        ax3.axis('off')
        overlay_b64 = fig_to_base64(fig3)
        plt.close(fig3)

        # Metadata
        metadata = {
            'modality': str(getattr(ds, 'Modality', 'N/A')),
            'rows': str(getattr(ds, 'Rows', 'N/A')),
            'columns': str(getattr(ds, 'Columns', 'N/A')),
            'patient_id': str(getattr(ds, 'PatientID', 'N/A')),
            'tumor_pixels': int(np.sum(mask)),
            'total_pixels': int(mask.size),
            'tumor_percentage': round(float(np.sum(mask)) / mask.size * 100, 2)
        }

        return jsonify({
            'original': original_b64,
            'mask': mask_b64,
            'overlay': overlay_b64,
            'metadata': metadata
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)
