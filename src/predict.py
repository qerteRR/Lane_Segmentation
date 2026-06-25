import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from metrics import bce_dice_loss, dice_coef, iou_coef


def parse_args():
    parser = argparse.ArgumentParser(description="Run lane segmentation on one image.")
    parser.add_argument("--model", required=True, help="Path to trained .keras model.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--output", default="outputs/prediction.png", help="Path to output visualization.")
    parser.add_argument("--img-size", type=int, default=256, help="Model input image size.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Mask threshold.")
    return parser.parse_args()


def load_image(image_path, img_size):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (img_size, img_size))
    model_input = resized.astype(np.float32) / 255.0
    return image_rgb, model_input[np.newaxis, ...]


def make_overlay(image_rgb, mask, alpha=0.55):
    mask_resized = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]))
    binary = mask_resized > 0

    overlay = image_rgb.copy()
    overlay[binary] = [255, 40, 40]
    return cv2.addWeighted(image_rgb, 1 - alpha, overlay, alpha, 0)


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(
        args.model,
        custom_objects={
            "bce_dice_loss": bce_dice_loss,
            "dice_coef": dice_coef,
            "iou_coef": iou_coef,
        },
    )

    original, model_input = load_image(args.image, args.img_size)
    prediction = model.predict(model_input)[0, ..., 0]
    binary_mask = prediction >= args.threshold
    overlay = make_overlay(original, binary_mask)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(prediction, cmap="gray")
    plt.title("Predicted mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print(f"Saved segmentation result to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
