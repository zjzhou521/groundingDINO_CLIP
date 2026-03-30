import base64
from io import BytesIO

from PIL import Image


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def image_to_png_bytes(image: Image.Image) -> tuple[bytes, str]:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), "image/png"


def image_bytes_to_data_url(image_bytes: bytes, content_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{content_type};base64,{encoded}"


def clamp_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x_min, y_min, x_max, y_max = box
    left = max(0, min(int(x_min), width - 1))
    top = max(0, min(int(y_min), height - 1))
    right = max(left + 1, min(int(x_max), width))
    bottom = max(top + 1, min(int(y_max), height))
    return left, top, right, bottom
