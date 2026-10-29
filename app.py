import streamlit as st
import torch
import numpy as np
from PIL import Image
import sys
import os
import time

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

WEIGHTS = os.path.join(
    BASE_DIR, "pretrained_weights", "LOL_v1.pth"
)

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="RetinexMamba - Low Light Enhancement",
    page_icon="🌙",
    layout="wide"
)

# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------
st.markdown("""
<div style="text-align:center; padding:20px 0 10px 0;">
    <h1 style="font-size:42px; margin-bottom:5px;">🌙 RetinexMamba</h1>
    <p style="font-size:20px; color:#666; margin-top:0;">Low-Light Image Enhancement</p>
    <p style="font-size:15px; color:#888;">Enhance dark and low-light photographs using deep learning</p>
</div>
""", unsafe_allow_html=True)

st.divider()

st.info("📌 Upload a low-light image below and click **Enhance Image** to generate the enhanced result.")

# ---------------------------------------------------------
# LOAD CPU FALLBACK
# ---------------------------------------------------------
@st.cache_resource
def load_model():

    from basicsr.models.archs.RetinexMamba_arch import RetinexMamba

    # Import our CPU selective scan implementation
    import selective_scan_cpu

    # Connect CPU implementation to the model
    import basicsr.models.archs.SS2D_arch as ss2d

    ss2d.selective_scan_fn = selective_scan_cpu.selective_scan_fn

    model = RetinexMamba(
        in_channels=3,
        out_channels=3,
        n_feat=40,
        stage=1,
        num_blocks=[1, 2, 2]
    )

    checkpoint = torch.load(
        WEIGHTS,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint["params"],
        strict=True
    )

    model.eval()

    return model


# ---------------------------------------------------------
# IMAGE ENHANCEMENT
# ---------------------------------------------------------
def enhance_image(image, model):

    TILE = 64
    STRIDE = 48
    BATCH = 2

    original_w, original_h = image.size

    img = np.array(image).astype(np.float32) / 255.0

    # RGB -> CHW
    img = torch.from_numpy(
        img.transpose(2, 0, 1)
    ).float()

    # -----------------------------------------------------
    # PAD IMAGE
    # -----------------------------------------------------
    h, w = img.shape[1], img.shape[2]

    pad_h = max(TILE - h, 0)
    pad_w = max(TILE - w, 0)

    if pad_h > 0 or pad_w > 0:
        img = torch.nn.functional.pad(
            img.unsqueeze(0),
            (0, pad_w, 0, pad_h),
            mode="reflect"
        ).squeeze(0)

    h, w = img.shape[1], img.shape[2]

    # Make dimensions multiples of TILE
    extra_h = (TILE - h % TILE) % TILE
    extra_w = (TILE - w % TILE) % TILE

    if extra_h > 0 or extra_w > 0:
        img = torch.nn.functional.pad(
            img.unsqueeze(0),
            (0, extra_w, 0, extra_h),
            mode="reflect"
        ).squeeze(0)

    h, w = img.shape[1], img.shape[2]

    # -----------------------------------------------------
    # TILE POSITIONS
    # -----------------------------------------------------
    def make_positions(length):

        if length <= TILE:
            return [0]

        positions = list(
            range(0, length - TILE + 1, STRIDE)
        )

        last = length - TILE

        if positions[-1] != last:
            positions.append(last)

        return positions

    ys = make_positions(h)
    xs = make_positions(w)

    # -----------------------------------------------------
    # FEATHER WEIGHT
    # -----------------------------------------------------
    window = np.hanning(TILE)

    window = np.maximum(window, 0.05)

    weight = np.outer(window, window)

    weight = torch.from_numpy(
        weight.astype(np.float32)
    )

    output = torch.zeros(
        3, h, w,
        dtype=torch.float32
    )

    weight_sum = torch.zeros(
        1, h, w,
        dtype=torch.float32
    )

    # -----------------------------------------------------
    # PROCESS TILES
    # -----------------------------------------------------
    tiles = []
    locations = []

    for y in ys:
        for x in xs:

            tile = img[
                :,
                y:y + TILE,
                x:x + TILE
            ]

            tiles.append(tile)
            locations.append((y, x))

    total_tiles = len(tiles)

    progress = st.progress(0)
    status = st.empty()

    with torch.no_grad():

        for start in range(0, total_tiles, BATCH):

            batch_tiles = tiles[
                start:start + BATCH
            ]

            batch_locations = locations[
                start:start + BATCH
            ]

            batch = torch.stack(
                batch_tiles
            )

            status.text(
                f"Enhancing tiles "
                f"{min(start + BATCH, total_tiles)}"
                f"/{total_tiles}..."
            )

            result = model(batch)

            result = torch.clamp(
                result,
                0,
                1
            )

            for i, (y, x) in enumerate(
                batch_locations
            ):

                tile_result = result[i]

                output[
                    :,
                    y:y + TILE,
                    x:x + TILE
                ] += (
                    tile_result * weight
                )

                weight_sum[
                    :,
                    y:y + TILE,
                    x:x + TILE
                ] += weight

            progress.progress(
                min(
                    (start + BATCH) / total_tiles,
                    1.0
                )
            )

    status.empty()
    progress.empty()

    # -----------------------------------------------------
    # BLEND
    # -----------------------------------------------------
    output = output / torch.clamp(
        weight_sum,
        min=1e-8
    )

    # Crop to original size
    output = output[
        :,
        :original_h,
        :original_w
    ]

    output = (
        output
        .clamp(0, 1)
        .numpy()
        .transpose(1, 2, 0)
    )

    output = (
        output * 255
    ).round().astype(np.uint8)

    return Image.fromarray(
        output
    )


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:

    st.header("Model Information")

    st.write("**Model:** RetinexMamba")

    st.write("**Task:** Low-Light Enhancement")

    st.write("**Device:** CPU")

    st.write("**Tile Size:** 64 × 64")

    st.write("**Overlap:** 16 pixels")

    st.write("**Offline Mode:** Enabled")


# ---------------------------------------------------------
# IMAGE UPLOAD
# ---------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a low-light image",
    type=["jpg", "jpeg", "png"]
)


# ---------------------------------------------------------
# PROCESS
# ---------------------------------------------------------
if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader("Input Image")

    st.image(
        image,
        caption="Original Low-Light Image",
        use_column_width=True
    )

    if st.button(
        "✨ Enhance Image",
        type="primary"
    ):

        try:

            start_time = time.time()

            with st.spinner(
                "Loading RetinexMamba model..."
            ):

                model = load_model()

            with st.spinner(
                "Enhancing image..."
            ):

                enhanced = enhance_image(
                    image,
                    model
                )

            elapsed = time.time() - start_time

            st.success(
                f"Enhancement completed in "
                f"{elapsed:.2f} seconds."
            )

            # -------------------------------------------------
            # BEFORE / AFTER
            # -------------------------------------------------
            st.subheader(
                "Before vs Enhanced"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.image(
                    image,
                    caption="Before",
                    use_column_width=True
                )

            with col2:

                st.image(
                    enhanced,
                    caption="RetinexMamba Enhanced",
                    use_column_width=True
                )

            # -------------------------------------------------
            # DOWNLOAD
            # -------------------------------------------------
            import io

            buffer = io.BytesIO()

            enhanced.save(
                buffer,
                format="PNG"
            )

            st.download_button(
                label="⬇️ Download Enhanced Image",
                data=buffer.getvalue(),
                file_name="retinexmamba_enhanced.png",
                mime="image/png"
            )

        except Exception as e:

            st.error(
                "An error occurred during enhancement."
            )

            st.exception(e)
