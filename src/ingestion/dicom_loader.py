import pydicom
import numpy as np
import matplotlib.pyplot as plt
import os


def load_dicom(file_path):
    """Load a single DICOM file and return pixel array"""
    ds = pydicom.dcmread(file_path)
    image = ds.pixel_array.astype(np.float32)
    # Normalize to 0-255
    image = (image - image.min()) / (image.max() - image.min()) * 255
    return image, ds


def load_dicom_folder(folder_path):
    """Load all DICOM files from a folder (one scan series)"""
    slices = []
    for f in sorted(os.listdir(folder_path)):
        if f.endswith('.dcm'):
            img, ds = load_dicom(os.path.join(folder_path, f))
            slices.append(img)
    return np.array(slices)


def visualize_dicom(image, title="DICOM Scan"):
    """Quick visualization of a DICOM slice"""
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    print("DICOM loader ready!")
