import os
import time
import torch
import numpy as np
from PIL import Image
from basicsr.models.archs.RetinexMamba_arch import RetinexMamba


INPUT = "test_images/images.jpg"
OUTPUT = "/home/hp/retinex_results/enhanced_output.png"
WEIGHTS = "pretrained_weights/LOL_v1.pth"

TILE = 64
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


def main():

    model = load_model()

    image = Image.open(INPUT).convert("RGB")
    original_w, original_h = image.size

    print("INPUT SIZE:", image.size)

    arr = np.asarray(image).astype(np.float32) / 255.0

    # Pad to multiples of 64 for tiled processing
    padded_h = ((original_h + TILE - 1) // TILE) * TILE
    padded_w = ((original_w + TILE - 1) // TILE) * TILE

    pad_h = padded_h - original_h
    pad_w = padded_w - original_w

    if pad_h > 0 or pad_w > 0:
        arr = np.pad(
            arr,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="reflect"
        )

    print("PADDED SIZE:", (padded_w, padded_h))
    print("TILE SIZE:", TILE)
    print("BATCH SIZE:", BATCH)

    tiles = []
    positions = []

    for y in range(0, padded_h, TILE):
        for x in range(0, padded_w, TILE):

            tile = arr[y:y+TILE, x:x+TILE]

            tensor = torch.from_numpy(tile).permute(2, 0, 1).float()

            tiles.append(tensor)
            positions.append((y, x))

    total = len(tiles)

    print("TOTAL TILES:", total)
    print("STARTING INFERENCE...")
    print()

    output = np.zeros_like(arr, dtype=np.float32)
    count = np.zeros(
        (padded_h, padded_w, 1),
        dtype=np.float32
    )

    start_time = time.time()

    with torch.no_grad():

        for i in range(0, total, BATCH):

            batch_tiles = tiles[i:i+BATCH]

            x = torch.stack(batch_tiles, dim=0)

            y = model(x)

            if isinstance(y, (tuple, list)):
                y = y[0]

            y = y.detach().cpu().numpy()

            y = np.transpose(y, (0, 2, 3, 1))

            for j in range(len(batch_tiles)):

                tile_y, tile_x = positions[i+j]

                output[
                    tile_y:tile_y+TILE,
                    tile_x:tile_x+TILE
                ] += y[j]

                count[
                    tile_y:tile_y+TILE,
                    tile_x:tile_x+TILE
                ] += 1.0

            done = min(i + BATCH, total)

            elapsed = time.time() - start_time

            print(
                f"Processed {done}/{total} tiles "
                f"({done/total*100:.1f}%) "
                f"- {elapsed:.1f}s"
            )

    output = output / count

    # Remove padding
    output = output[:original_h, :original_w]

    # Valid image range
    output = np.clip(output, 0.0, 1.0)

    output = (output * 255.0).round().astype(np.uint8)

    Image.fromarray(output).save(OUTPUT)

    elapsed = time.time() - start_time

    print()
    print("================================")
    print("INFERENCE COMPLETE")
    print("OUTPUT:", OUTPUT)
    print("OUTPUT SIZE:", (original_w, original_h))
    print("TOTAL TIME:", round(elapsed, 2), "seconds")
    print("================================")


if __name__ == "__main__":
    main()
