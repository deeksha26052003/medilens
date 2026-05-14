from monai.transforms import (
    Compose,
    ScaleIntensity,
    EnsureChannelFirst,
    Resize,
)
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


def prepare_data(folder_path):
    """Load all DICOM slices and prepare as tensors"""
    slices = []
    for f in sorted(os.listdir(folder_path)):
        if f.endswith('.dcm'):
            ds = pydicom.dcmread(os.path.join(folder_path, f))
            img = ds.pixel_array.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            # Resize to 128x128 for faster training
            img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)
            img_resized = torch.nn.functional.interpolate(
                img_tensor, size=(128, 128))
            slices.append(img_resized.squeeze(0))
    return slices


def create_pseudo_labels(slices):
    """Create pseudo labels using intensity thresholding"""
    labels = []
    for img in slices:
        img_np = img.squeeze().numpy()
        # Threshold: bright regions = vessels (class 1), dark = background (class 0)
        label = (img_np > 0.5).astype(np.int64)
        labels.append(torch.tensor(label).unsqueeze(0))
    return labels


def train_model(folder_path, epochs=10):
    """Train UNet on our DICOM data"""

    mlflow.set_experiment("medilens")

    with mlflow.start_run(run_name=f"training_unet_{epochs}epochs"):

        print("Loading data...")
        slices = prepare_data(folder_path)
        labels = create_pseudo_labels(slices)
        print(f"Loaded {len(slices)} slices!")

        model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64),
            strides=(2, 2),
            num_res_units=2,
            norm=Norm.BATCH
        )

        loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", 1e-3)
        mlflow.log_param("model", "UNet")
        mlflow.log_param("num_slices", len(slices))

        print("Training started...")
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
        torch.save(model.state_dict(), 'models/unet_trained.pth')
        mlflow.pytorch.log_model(model, "trained_unet")
        print("Model saved!")

        # Visualize result
        model.eval()
        with torch.no_grad():
            test_img = slices[9].unsqueeze(0)
            output = model(test_img)
            mask = torch.argmax(output, dim=1).squeeze().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(slices[9].squeeze().numpy(), cmap='gray')
        axes[0].set_title('MRI Slice')
        axes[0].axis('off')
        axes[1].imshow(mask, cmap='jet')
        axes[1].set_title('Trained Segmentation')
        axes[1].axis('off')
        plt.savefig('data/sample/trained_output.png', bbox_inches='tight')
        mlflow.log_artifact('data/sample/trained_output.png')
        print("Done! Check data/sample/trained_output.png")


if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    folder = 'data/sample/1.3.6.1.4.1.14519.5.2.1.8700.9668.283183069665765319358456872530'
    train_model(folder, epochs=10)
