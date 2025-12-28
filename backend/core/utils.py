"""Utility functions for image processing and conversion."""

import base64
import io
import gc
from typing import Optional
from PIL import Image
import torch


def pil_to_base64(pil_image: Image.Image, format: str = "JPEG", quality: int = 95) -> str:
    """
    Convert PIL Image to base64 encoded string.
    
    Args:
        pil_image: PIL Image object
        format: Output format (JPEG, PNG)
        quality: JPEG quality (1-100)
    
    Returns:
        Base64 encoded string
    """
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format, quality=quality)
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')


def base64_to_pil(base64_string: str) -> Image.Image:
    """
    Convert base64 encoded string to PIL Image.
    
    Args:
        base64_string: Base64 encoded image string
    
    Returns:
        PIL Image object
    """
    img_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(img_data))


def validate_image(
    image: Image.Image,
    max_size: Optional[int] = None,
    allowed_formats: Optional[list] = None
) -> tuple[bool, str]:
    """
    Validate image constraints.
    
    Args:
        image: PIL Image object
        max_size: Maximum file size in bytes
        allowed_formats: List of allowed formats
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check format
    if allowed_formats and image.format not in allowed_formats:
        return False, f"Format {image.format} not allowed. Allowed: {', '.join(allowed_formats)}"
    
    # Check size (approximate)
    if max_size:
        buffer = io.BytesIO()
        image.save(buffer, format=image.format or 'JPEG')
        size_bytes = len(buffer.getvalue())
        
        if size_bytes > max_size:
            max_mb = max_size / (1024 * 1024)
            actual_mb = size_bytes / (1024 * 1024)
            return False, f"Image too large: {actual_mb:.2f}MB (max: {max_mb:.2f}MB)"
    
    return True, ""


def flush_memory():
    """Clear GPU and system memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
