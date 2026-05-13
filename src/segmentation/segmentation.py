import numpy as np
import torch
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.transforms import (
    Compose,
    LoadImage,
    ScaleIntensity,
    EnsureChannelFirst,
    Resize,
    ToTensor
)


def get_model():
    """Define MONAI UNet for segmentation"""
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH
    )
    return model


def preprocess_image(pixel_array):
    """Preprocess a DICOM pixel array for model input"""
    img = pixel_array.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return img


def run_segmentation(pixel_array):
    """Run segmentation on a single DICOM slice"""
    model = get_model()
    model.eval()

    with torch.no_grad():
        img_tensor = preprocess_image(pixel_array)
        output = model(img_tensor)
        mask = torch.argmax(output, dim=1).squeeze().numpy()

    return mask


if __name__ == "__main__":
    import pydicom
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ds = pydicom.dcmread(
        'data/sample/1.3.6.1.4.1.14519.5.2.1.8700.9668.283183069665765319358456872530/1-10.dcm')
    mask = run_segmentation(ds.pixel_array)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    img = ds.pixel_array.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original MRI')
    axes[0].axis('off')
    axes[1].imshow(mask, cmap='jet')
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')
    plt.savefig('data/sample/segmentation_output.png', bbox_inches='tight')
    print('Segmentation done!')
