from monai.losses import DiceLoss
from monai.networks.layers import Norm
from monai.networks.nets import UNet
import mlflow.pytorch
import mlflow
import matplotlib.pyplot as plt
import os
import sys
import torch
import numpy as np
import pydicom
import matplotlib
matplotlib.use('Agg')


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_hcc_slices(base_folder, max_slices=50):
    """Load CT slices from HCC dataset"""
    all_slices = []
    for series_folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, series_folder)
        if not os.path.isdir(folder_path):
            continue
        files = sorted([f for f in os.listdir(
            folder_path) if f.endswith('.dcm')])
        for f in files[:max_slices]:
            ds = pydicom.dcmread(os.path.join(folder_path, f))
            img = ds.pixel_array.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)
            img_resized = torch.nn.functional.interpolate(
                img_tensor, size=(128, 128))
            all_slices.append(img_resized.squeeze(0))
        break  # just use first series for now
    return all_slices


def create_tumor_labels(slices):
    """
    Create tumor labels using intensity thresholding
    Tumors appear as bright regions in CT scans
    High intensity (>0.7) = potential tumor = class 1
    Everything else = background = class 0
    """
    labels = []
    for img in slices:
        img_np = img.squeeze().numpy()
        label = (img_np > 0.7).astype(np.int64)
        labels.append(torch.tensor(label).unsqueeze(0))
    return labels


def train_tumor_model(base_folder, epochs=15):
    mlflow.set_experiment("medilens")

    with mlflow.start_run(run_name=f"tumor_detection_{epochs}epochs"):

        print("Loading HCC data...")
        slices = load_hcc_slices(base_folder)
        labels = create_tumor_labels(slices)
        print(f"Loaded {len(slices)} slices!")

        model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH
        )

        loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        mlflow.log_param("epochs", epochs)
        mlflow.log_param("dataset", "HCC-TACE-Seg")
        mlflow.log_param("model", "UNet")
        mlflow.log_param("task", "tumor_detection")
        mlflow.log_param("num_slices", len(slices))

        print("Training tumor detection model...")
        model.train()

        for epoch in range(epochs):
            epoch_loss = 0
            for img, label in zip(slices, labels):
                img_input = img.unsqueeze(0)
                label_input = label.unsqueeze(0)
                optimizer.zero_grad()
                output = model(img_input)
                loss = loss_fn(output, label_input)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(slices)
            mlflow.log_metric("loss", avg_loss, step=epoch)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        # Save model
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), 'models/tumor_detection.pth')
        mlflow.pytorch.log_model(model, "tumor_detection_model")
        print("Model saved!")

        # Visualize
        model.eval()
        with torch.no_grad():
            test_img = slices[40].unsqueeze(0)
            output = model(test_img)
            mask = torch.argmax(output, dim=1).squeeze().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        original = slices[40].squeeze().numpy()
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original CT Scan')
        axes[0].axis('off')
        axes[1].imshow(mask, cmap='jet')
        axes[1].set_title('Tumor Detection Mask')
        axes[1].axis('off')
        axes[2].imshow(original, cmap='gray')
        axes[2].imshow(mask, cmap='Reds', alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        plt.savefig('data/sample/tumor_detection_output.png',
                    bbox_inches='tight')
        mlflow.log_artifact('data/sample/tumor_detection_output.png')
        print("Done! Check data/sample/tumor_detection_output.png")


if __name__ == "__main__":
    train_tumor_model('data/hcc', epochs=15)
