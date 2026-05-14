import mlflow
import mlflow.pytorch
import torch
import numpy as np
from datetime import datetime


def init_mlflow(experiment_name="medilens"):
    mlflow.set_experiment(experiment_name)
    print(f"MLflow experiment set: {experiment_name}")


def log_segmentation_run(slice_idx, pixel_array, mask, model):
    with mlflow.start_run(run_name=f"slice_{slice_idx}_{datetime.now().strftime('%H%M%S')}"):
        # Log parameters
        mlflow.log_param("slice_index", slice_idx)
        mlflow.log_param("image_shape", str(pixel_array.shape))
        mlflow.log_param("model_type", "UNet")
        mlflow.log_param("spatial_dims", 2)
        mlflow.log_param("in_channels", 1)
        mlflow.log_param("out_channels", 2)

        # Log metrics
        mlflow.log_metric("mask_coverage", float(np.mean(mask)))
        mlflow.log_metric("pixel_mean", float(pixel_array.mean()))
        mlflow.log_metric("pixel_std", float(pixel_array.std()))

        # Log model
        mlflow.pytorch.log_model(model, "unet_model")

        print(f"Logged run for slice {slice_idx}")


if __name__ == "__main__":
    import sys
    import os
    import pydicom
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from segmentation.segmentation import run_segmentation, get_model

    init_mlflow()

    ds = pydicom.dcmread(
        'data/sample/1.3.6.1.4.1.14519.5.2.1.8700.9668.283183069665765319358456872530/1-10.dcm')
    model = get_model()
    mask = run_segmentation(ds.pixel_array)

    log_segmentation_run(10, ds.pixel_array, mask, model)
    print("Done! Run: mlflow ui")
