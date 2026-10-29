import time
import torch
import numpy as np
from PIL import Image
from basicsr.models.archs.RetinexMamba_arch import RetinexMamba


INPUT = "test_images/images.jpg"
OUTPUT = "/home/hp/retinex_results/enhanced_blended.png"
WEIGHTS = "pretrained_weights/LOL_v1.pth"

TILE = 64
STRIDE = 48
BATCH = 2


def load_model():
    print("Loading RetinexMamba...")

    model = RetinexMamba(
        in_channels=3,
        out_channels=3,
        n_feat=40,
        stage=1,
        num_blocks=[1, 2, 2]
    )

    ckpt = torch.load(WEIGHTS, map_location="cpu")
    model.load_state_dict(ckpt["params"], strict=True)
    model.eval()

    print("MODEL LOADED")
    return model


def make_positions(length):
    positions = list(range(0, max(length - TILE + 1, 1), STRIDE))

    last = length - TILE

    if positions[-1] != last:
        positions.append(last)

    return positions


def make_weight():
    # Smooth feathering window to reduce tile boundaries.
    w = np.hanning(TILE).astype(np.float32)

    # Avoid zero weight at the outermost pixels.
    w = np.maximum(w, 0.05)

    weight = np.outer(w, w)

    return weight[:, :, None]


def main():

    model = load_model()

    image = Image.open(INPUT).convert("RGB")
    original_w, original_h = image.size

    print("INPUT SIZE:", image.size)

    arr = np.asarray(image).astype(np.float32) / 255.0

    # Pad to at least one complete tile.
    padded_h = max(original_h, TILE)
    padded_w = max(original_w, TILE)

    if padded_h % 4 != 0:
        padded_h += 4 - padded_h % 4

    if padded_w % 4 != 0:
        padded_w += 4 - padded_w % 4

    pad_h = padded_h - original_h
    pad_w = padded_w - original_w

    if pad_h > 0 or pad_w > 0:
        arr = np.pad(
            arr,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="reflect"
        )

    print("PADDED SIZE:", (padded_w, padded_h))

    ys = make_positions(padded_h)
    xs = make_positions(padded_w)

    positions = []

    for y in ys:
        for x in xs:
            positions.append((y, x))

    total = len(positions)

    print("TILE SIZE:", TILE)
    print("STRIDE:", STRIDE)
    print("OVERLAP:", TILE - STRIDE)
    print("TOTAL TILES:", total)
    print("BATCH SIZE:", BATCH)
    print()
    print("STARTING BLENDED INFERENCE...")

    output = np.zeros(
        (padded_h, padded_w, 3),
        dtype=np.float32
    )

    weights = np.zeros(
        (padded_h, padded_w, 1),
        dtype=np.float32
    )

    feather = make_weight()

    start_time = time.time()

    with torch.no_grad():

        for i in range(0, total, BATCH):

            batch_positions = positions[i:i+BATCH]

            batch = []

            for y, x in batch_positions:

                tile = arr[
                    y:y+TILE,
                    x:x+TILE
                ]

                tensor = torch.from_numpy(
                    tile
                ).permute(2, 0, 1).float()

                batch.append(tensor)

            x = torch.stack(batch, dim=0)

            y = model(x)

            if isinstance(y, (tuple, list)):
                y = y[0]

            y = y.detach().cpu().numpy()

            y = np.transpose(
                y,
                (0, 2, 3, 1)
            )

            y = np.clip(y, 0.0, 1.0)

            for j, (py, px) in enumerate(batch_positions):

                output[
                    py:py+TILE,
                    px:px+TILE
                ] += y[j] * feather

                weights[
                    py:py+TILE,
                    px:px+TILE
                ] += feather

            done = min(i + BATCH, total)

            elapsed = time.time() - start_time

            print(
                f"Processed {done}/{total} "
                f"({done/total*100:.1f}%) "
                f"- {elapsed:.1f}s"
            )

    output = output / np.maximum(weights, 1e-8)

    # Crop back to original image dimensions.
    output = output[:original_h, :original_w]

    output = np.clip(output, 0.0, 1.0)

    output = (
        output * 255.0
    ).round().astype(np.uint8)

    Image.fromarray(output).save(OUTPUT)

    elapsed = time.time() - start_time

    print()
    print("==============================")
    print("BLENDED INFERENCE COMPLETE")
    print("OUTPUT:", OUTPUT)
    print("OUTPUT SIZE:", (original_w, original_h))
    print("TOTAL TIME:", round(elapsed, 2), "seconds")
    print("==============================")


if __name__ == "__main__":
    main()
