"""
Logo/Watermark removal using OpenCV Inpainting
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from loguru import logger


class LogoRemover:
    """Remove logos/watermarks from images using OpenCV inpainting"""

    def __init__(
        self,
        logo_regions: Optional[List[Tuple[int, int, int, int]]] = None,
        inpaint_radius: int = 5,
        method: str = "telea"
    ):
        """
        Initialize Logo Remover

        Args:
            logo_regions: List of (x, y, width, height) tuples defining logo positions
                          If None, logo removal is disabled
            inpaint_radius: Radius of circular neighborhood (3-10 recommended)
            method: Inpainting method - "telea" (fast) or "ns" (slower, better quality)
        """
        self.logo_regions = logo_regions or []
        self.inpaint_radius = inpaint_radius

        # Select inpainting method
        if method.lower() == "ns":
            self.inpaint_method = cv2.INPAINT_NS  # Navier-Stokes based method
        else:
            self.inpaint_method = cv2.INPAINT_TELEA  # Fast marching method (default)

        if self.logo_regions:
            logger.info(
                f"LogoRemover initialized with {len(self.logo_regions)} regions, "
                f"radius={inpaint_radius}, method={method}"
            )
        else:
            logger.info("LogoRemover initialized but disabled (no regions defined)")

    def remove_logos(self, image: np.ndarray) -> np.ndarray:
        """
        Remove logos from image

        Args:
            image: Input image (BGR or RGB)

        Returns:
            Image with logos removed
        """
        if not self.logo_regions:
            return image  # No regions defined, return original

        # Create mask for all logo regions
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for x, y, w, h in self.logo_regions:
            # Ensure region is within image bounds
            x = max(0, min(x, image.shape[1] - 1))
            y = max(0, min(y, image.shape[0] - 1))
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)

            if w > 0 and h > 0:
                mask[y:y+h, x:x+w] = 255  # White = area to inpaint

        # Apply inpainting
        try:
            result = cv2.inpaint(image, mask, self.inpaint_radius, self.inpaint_method)
            logger.debug(f"Removed {len(self.logo_regions)} logo regions")
            return result
        except Exception as e:
            logger.error(f"Inpainting failed: {e}")
            return image  # Return original on error

    def add_region(self, x: int, y: int, width: int, height: int):
        """Add a new logo region"""
        self.logo_regions.append((x, y, width, height))
        logger.debug(f"Added logo region: ({x}, {y}, {width}, {height})")

    def clear_regions(self):
        """Clear all logo regions"""
        self.logo_regions = []
        logger.debug("Cleared all logo regions")

    def set_regions_from_corners(
        self,
        corner: str,
        width: int,
        height: int,
        image_width: int,
        image_height: int,
        margin: int = 10
    ):
        """
        Set logo region based on corner position

        Args:
            corner: "top-left", "top-right", "bottom-left", "bottom-right", "center"
            width: Logo width in pixels
            height: Logo height in pixels
            image_width: Full image width
            image_height: Full image height
            margin: Margin from edge in pixels
        """
        self.clear_regions()

        if corner == "top-left":
            x, y = margin, margin
        elif corner == "top-right":
            x, y = image_width - width - margin, margin
        elif corner == "bottom-left":
            x, y = margin, image_height - height - margin
        elif corner == "bottom-right":
            x, y = image_width - width - margin, image_height - height - margin
        elif corner == "center":
            x = (image_width - width) // 2
            y = (image_height - height) // 2
        else:
            logger.error(f"Unknown corner: {corner}")
            return

        self.add_region(x, y, width, height)
        logger.info(f"Set logo region at {corner}: ({x}, {y}, {width}, {height})")


def remove_logo_from_image(
    image: np.ndarray,
    regions: List[Tuple[int, int, int, int]],
    inpaint_radius: int = 5,
    method: str = "telea"
) -> np.ndarray:
    """
    Convenience function to remove logo from a single image

    Args:
        image: Input image
        regions: List of (x, y, width, height) logo regions
        inpaint_radius: Inpainting radius
        method: "telea" or "ns"

    Returns:
        Image with logos removed
    """
    remover = LogoRemover(regions, inpaint_radius, method)
    return remover.remove_logos(image)
