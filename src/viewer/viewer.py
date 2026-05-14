import sys
import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from segmentation.segmentation import run_segmentation


class DICOMViewer:
    def __init__(self, folder_path):
        self.slices = self.load_all_slices(folder_path)
        self.current_idx = 0
        self.total = len(self.slices)

    def load_all_slices(self, folder_path):
        slices = []
        for f in sorted(os.listdir(folder_path)):
            if f.endswith('.dcm'):
                ds = pydicom.dcmread(os.path.join(folder_path, f))
                slices.append(ds.pixel_array)
        return slices

    def normalize(self, img):
        img = img.astype(np.float32)
        return (img - img.min()) / (img.max() - img.min())

    def show(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        plt.subplots_adjust(bottom=0.2)
        fig.patch.set_facecolor('#1a1a2e')

        def update(val):
            idx = int(slider.val)
            self.current_idx = idx
            img = self.normalize(self.slices[idx])
            mask = run_segmentation(self.slices[idx])

            axes[0].clear()
            axes[1].clear()
            axes[0].imshow(img, cmap='gray')
            axes[0].set_title(f'MRI Scan - Slice {idx+1}/{self.total}', color='white')
            axes[0].axis('off')
            axes[1].imshow(img, cmap='gray')
            axes[1].imshow(mask, cmap='jet', alpha=0.4)
            axes[1].set_title('Segmentation Overlay', color='white')
            axes[1].axis('off')
            fig.canvas.draw_idle()

        ax_slider = plt.axes([0.2, 0.08, 0.6, 0.04])
        slider = widgets.Slider(ax_slider, 'Slice', 0, self.total-1, valinit=0, valstep=1)
        slider.on_changed(update)
        update(0)
        plt.show()


if __name__ == "__main__":
    folder = 'data/sample/1.3.6.1.4.1.14519.5.2.1.8700.9668.283183069665765319358456872530'
    viewer = DICOMViewer(folder)
    viewer.show()
