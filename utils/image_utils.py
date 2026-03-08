from io import BytesIO
from PIL import Image, ImageEnhance


def enhance_image_bytes(img: Image.Image, max_size: int = 1200, quality: int = 85) -> bytes:
    """Enhance and compress image, return bytes suitable for sending to APIs.

    - Resize to max_size (longer side) preserving aspect ratio
    - Increase contrast slightly
    - Save as JPEG bytes
    """
    # Resize
    w, h = img.size
    max_side = max(w, h)
    if max_side > max_size:
        scale = max_size / max_side
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.LANCZOS)

    # Enhance
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.1)

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
