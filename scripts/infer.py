import argparse
from pathlib import Path

import monai
from trainlib.trainer import SegmentationTrainer
from trainlib.utils import load_config

parser = argparse.ArgumentParser(description="Segmentation inference")
parser.add_argument("--config", type=str, required=True, help="path to the config file")
parser.add_argument("--model", type=str, required=True, help="path to the model weights")
parser.add_argument("--files", type=str, required=True, help="A single file or a folder with files")
parser.add_argument(
    "--input_postfix",
    type=str,
    required=False,
    help=(
        "Image file, to be prediction upon, should end with this postfix."
        " Required if `--files` is a folder. "
    ),
)
parser.add_argument(
    "--output_postfix", type=str, required=True, help="String to be appended to prediction filename"
)
parser.add_argument(
    "--overwrite", action="store_true", required=False, help="Overwrite existing predictions."
)

parser.add_argument("--roi_size", type=int, required=False, default=96, help="Edge length of ROI")
parser.add_argument(
    "--batch_size", type=int, required=False, default=64, help="Batchsize during prediction."
)
parser.add_argument(
    "--overlap", type=float, required=False, default=0.75, help="ROI overlap during prediction"
)

args = parser.parse_args()

image_files = Path(args.files)
if not image_files.exists():
    raise FileNotFoundError(f"{image_files}")
elif image_files.is_file():
    image_files = [image_files]
else:
    image_files = image_files.glob(f"**/*{args.input_postfix}")

if not Path(args.model).exists():
    FileNotFoundError(f"{args.model}")

config = load_config(args.config)
monai.utils.set_determinism(seed=config.seed)

trainer = SegmentationTrainer(
    progress_bar=True,
    early_stopping=True,
    metrics=["MeanDice", "HausdorffDistance", "SurfaceDistance"],
    save_latest_metrics=True,
    config=config,
)

if __name__ == "__main__":
    for fn in image_files:
        if not args.overwrite:
            out_name = fn.name
            suffixes = fn.suffixes
            for suff in suffixes:
                out_name = out_name.replace(suff, "")
            out_name = fn.parent / (
                out_name + "_" + args.output_postfix + ".nii.gz"
            )  # currently preds are saved as NIfTI
            if out_name.exists():
                print(f"Prediction exists. Skipping {fn}")
                continue

        pred = trainer.predict(
            [fn],
            args.model,
            roi_size=(args.roi_size,) * 3,
            sw_batch_size=args.batch_size,
            overlap=args.overlap,
        )

        pred["image_meta_dict"]["affine"] = pred["image_meta_dict"]["affine"].squeeze()
        pred["image_meta_dict"]["original_affine"] = pred["image_meta_dict"][
            "original_affine"
        ].squeeze()
        pred["pred"].affine = pred["pred"].affine.squeeze()

        trainer.save_prediction(pred, output_postfix=args.output_postfix)
