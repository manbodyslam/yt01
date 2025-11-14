"""
Image utility functions for quality assessment and processing
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Tuple
from PIL import Image


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load image from file path

    Args:
        image_path: Path to image file

    Returns:
        Image as numpy array in RGB format
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load with OpenCV and convert BGR to RGB
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(image: np.ndarray, output_path: Union[str, Path], quality: int = 95) -> None:
    """
    Save image to file

    Args:
        image: Image as numpy array (RGB format)
        output_path: Output file path
        quality: JPEG quality (1-100)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if output_path.suffix.lower() in ['.jpg', '.jpeg']:
        cv2.imwrite(str(output_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(str(output_path), img_bgr)


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    keep_aspect: bool = True
) -> np.ndarray:
    """
    Resize image to target size

    Args:
        image: Input image
        target_size: (width, height)
        keep_aspect: Keep aspect ratio

    Returns:
        Resized image
    """
    if keep_aspect:
        h, w = image.shape[:2]
        target_w, target_h = target_size

        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)


def calculate_sharpness(image: np.ndarray, region: Tuple[int, int, int, int] = None) -> float:
    """
    Calculate image sharpness using Laplacian variance

    Args:
        image: Input image (RGB or grayscale)
        region: Optional region (x, y, w, h) to analyze

    Returns:
        Sharpness score (higher = sharper)
    """
    # ✅ Check if image is empty or None
    if image is None or image.size == 0:
        return 0.0

    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Crop to region if specified
        if region is not None:
            x, y, w, h = region
            gray = gray[y:y+h, x:x+w]

            # ✅ Check if cropped region is empty
            if gray.size == 0:
                return 0.0

        # ✅ Final check before Laplacian
        if gray.size == 0 or gray.shape[0] == 0 or gray.shape[1] == 0:
            return 0.0

        # Calculate Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()

        return float(sharpness)

    except Exception as e:
        # ✅ Catch any OpenCV errors
        logger.warning(f"Failed to calculate sharpness: {e}")
        return 0.0


def calculate_contrast(image: np.ndarray, region: Tuple[int, int, int, int] = None) -> float:
    """
    Calculate image contrast (RMS contrast)

    Args:
        image: Input image (RGB or grayscale)
        region: Optional region (x, y, w, h) to analyze

    Returns:
        Contrast score (0-255, higher = more contrast)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Crop to region if specified
    if region is not None:
        x, y, w, h = region
        gray = gray[y:y+h, x:x+w]

    # Calculate RMS contrast
    mean = gray.mean()
    std = gray.std()
    contrast = std

    return float(contrast)


def is_blurry(image: np.ndarray, threshold: float = 100.0) -> bool:
    """
    Check if image is blurry

    Args:
        image: Input image
        threshold: Sharpness threshold (below = blurry)

    Returns:
        True if blurry, False otherwise
    """
    sharpness = calculate_sharpness(image)
    return sharpness < threshold


def calculate_brightness(image: np.ndarray, region: Tuple[int, int, int, int] = None) -> float:
    """
    Calculate average brightness of image

    Args:
        image: Input image (RGB)
        region: Optional region (x, y, w, h) to analyze

    Returns:
        Brightness score (0-255)
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Crop to region if specified
    if region is not None:
        x, y, w, h = region
        gray = gray[y:y+h, x:x+w]

    return float(gray.mean())


def get_image_dimensions(image_path: Union[str, Path]) -> Tuple[int, int]:
    """
    Get image dimensions without loading full image

    Args:
        image_path: Path to image file

    Returns:
        (width, height)
    """
    with Image.open(image_path) as img:
        return img.size


def get_short_side(image_path: Union[str, Path]) -> int:
    """
    Get length of shortest side

    Args:
        image_path: Path to image file

    Returns:
        Length of short side in pixels
    """
    w, h = get_image_dimensions(image_path)
    return min(w, h)
