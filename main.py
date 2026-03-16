import os
from io import BytesIO

import uvicorn
import numpy as np
from PIL import Image, ImageFilter
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from rembg import remove, new_session

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Better than creating model repeatedly
session = new_session("isnet-general-use")


def pil_to_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def get_corner_samples(img: Image.Image, sample_size=20):
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]

    corners = [
        arr[0:sample_size, 0:sample_size],                  # top-left
        arr[0:sample_size, w-sample_size:w],               # top-right
        arr[h-sample_size:h, 0:sample_size],               # bottom-left
        arr[h-sample_size:h, w-sample_size:w],             # bottom-right
    ]
    samples = np.concatenate([c.reshape(-1, 3) for c in corners], axis=0)
    return samples


def is_near_solid_background(img: Image.Image, tolerance=28):
    samples = get_corner_samples(img)
    mean_color = samples.mean(axis=0)
    distances = np.linalg.norm(samples - mean_color, axis=1)
    return distances.mean() < tolerance, mean_color


def remove_solid_bg(img: Image.Image, bg_color, threshold=55, feather=1):
    rgba = img.convert("RGBA")
    arr = np.array(rgba).astype(np.uint8)

    rgb = arr[:, :, :3].astype(np.int16)
    bg = np.array(bg_color).astype(np.int16)

    # Distance of every pixel from detected bg color
    dist = np.linalg.norm(rgb - bg, axis=2)

    # Create alpha mask
    alpha = np.where(dist < threshold, 0, 255).astype(np.uint8)

    # Slight feathering for smoother edges
    mask = Image.fromarray(alpha, mode="L")
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))

    arr[:, :, 3] = np.array(mask)
    return Image.fromarray(arr, mode="RGBA")


def clean_alpha(img: Image.Image):
    rgba = img.convert("RGBA")
    arr = np.array(rgba)

    alpha = Image.fromarray(arr[:, :, 3], mode="L")
    alpha = alpha.filter(ImageFilter.MedianFilter(size=3))
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=0.6))

    arr[:, :, 3] = np.array(alpha)
    return Image.fromarray(arr, mode="RGBA")


@app.get("/")
async def root():
    return {"status": "ok"}


@app.post("/remove-bg")
async def remove_bg(file: UploadFile = File(...)):
    try:
        input_bytes = await file.read()
        img = Image.open(BytesIO(input_bytes)).convert("RGBA")

        # 1) Try solid background removal first
        solid_bg, bg_color = is_near_solid_background(img)

        if solid_bg:
            output_img = remove_solid_bg(img, bg_color=bg_color, threshold=60, feather=1)
            output_img = clean_alpha(output_img)
            output_bytes = pil_to_bytes(output_img)
        else:
            # 2) Fallback to AI removal
            output_bytes = remove(
                input_bytes,
                session=session,
                alpha_matting=True,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
                alpha_matting_erode_size=8,
                post_process_mask=True,
            )

        return StreamingResponse(BytesIO(output_bytes), media_type="image/png")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)