import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

from data import load_tusimple_data
from metrics import bce_dice_loss, dice_coef, iou_coef
from model import build_unet


def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net for TuSimple lane segmentation.")
    parser.add_argument("--dataset-root", required=True, help="Path to TuSimple train_set folder.")
    parser.add_argument(
        "--json",
        nargs="+",
        required=True,
        help="One or more TuSimple label JSON files.",
    )
    parser.add_argument("--limit", type=int, default=200, help="Number of samples to load. Use 0 for all.")
    parser.add_argument("--img-size", type=int, default=256, help="Square training image size.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def plot_history(history, output_dir):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(history.history["dice_coef"], label="train Dice")
    plt.plot(history.history["val_dice_coef"], label="val Dice")
    plt.plot(history.history["iou_coef"], label="train IoU")
    plt.plot(history.history["val_iou_coef"], label="val IoU")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_metrics.png", dpi=160)
    plt.close()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, Y = load_tusimple_data(
        dataset_root=args.dataset_root,
        json_paths=args.json,
        image_size=(args.img_size, args.img_size),
        limit=args.limit,
    )

    print("Images:", X.shape)
    print("Masks:", Y.shape)

    model = build_unet(input_shape=(args.img_size, args.img_size, 3))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss=bce_dice_loss,
        metrics=["accuracy", dice_coef, iou_coef],
    )

    history = model.fit(
        X,
        Y,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    model.save(output_dir / "unet_tusimple.keras")
    plot_history(history, output_dir)

    print(f"Saved model and plots to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
