"""
Palette & Mood Module - Extract color palette and select text/background colors
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from loguru import logger
from sklearn.cluster import KMeans
from PIL import Image
import colorsys

from config import settings
from utils.image_utils import load_image


class PaletteExtractor:
    """
    Extracts color palette from images and selects appropriate colors for text/background
    """

    def __init__(self, num_colors: int = None):
        """
        Initialize Palette Extractor

        Args:
            num_colors: Number of dominant colors to extract
        """
        self.num_colors = num_colors or settings.PALETTE_COLORS
        logger.info(f"PaletteExtractor initialized with {self.num_colors} colors")

    def extract_palette(self, image_path: Path) -> List[Tuple[int, int, int]]:
        """
        Extract dominant colors from image using K-Means clustering

        Args:
            image_path: Path to image

        Returns:
            List of RGB color tuples
        """
        try:
            # Load image
            image = load_image(image_path)

            # Resize for faster processing
            h, w = image.shape[:2]
            max_size = 400
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                image = np.array(Image.fromarray(image).resize((new_w, new_h)))

            # Reshape to list of pixels
            pixels = image.reshape(-1, 3)

            # Remove very dark and very bright pixels (often not representative)
            brightness = pixels.mean(axis=1)
            mask = (brightness > 20) & (brightness < 235)
            filtered_pixels = pixels[mask]

            if len(filtered_pixels) == 0:
                filtered_pixels = pixels

            # K-Means clustering
            kmeans = KMeans(n_clusters=self.num_colors, random_state=42, n_init=10)
            kmeans.fit(filtered_pixels)

            # Get cluster centers (dominant colors)
            colors = kmeans.cluster_centers_.astype(int)

            # Sort by frequency
            labels = kmeans.labels_
            counts = np.bincount(labels)
            sorted_indices = np.argsort(-counts)

            palette = [tuple(colors[i]) for i in sorted_indices]

            logger.debug(f"Extracted palette from {image_path.name}: {palette}")

            return palette

        except Exception as e:
            logger.error(f"Failed to extract palette from {image_path}: {e}")
            # Return default palette
            return [(255, 255, 255)] * self.num_colors

    def get_complementary_color(self, rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Get complementary color

        Args:
            rgb: RGB color tuple

        Returns:
            Complementary RGB color
        """
        r, g, b = [x / 255.0 for x in rgb]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)

        # Rotate hue by 180 degrees
        h_comp = (h + 0.5) % 1.0

        r_comp, g_comp, b_comp = colorsys.hsv_to_rgb(h_comp, s, v)

        return (int(r_comp * 255), int(g_comp * 255), int(b_comp * 255))

    def get_contrast_color(self, rgb: Tuple[int, int, int], light_threshold: int = 128) -> Tuple[int, int, int]:
        """
        Get high-contrast color (black or white)

        Args:
            rgb: Background RGB color
            light_threshold: Threshold for determining if color is light

        Returns:
            Black or white RGB tuple
        """
        # Calculate perceived brightness
        r, g, b = rgb
        brightness = (0.299 * r + 0.587 * g + 0.114 * b)

        if brightness > light_threshold:
            return (0, 0, 0)  # Black for light backgrounds
        else:
            return (255, 255, 255)  # White for dark backgrounds

    def select_text_colors(self, palette: List[Tuple[int, int, int]]) -> Dict[str, Tuple[int, int, int]]:
        """
        Select appropriate colors for text elements

        Args:
            palette: Extracted color palette

        Returns:
            Dictionary with color assignments
        """
        if not palette:
            return {
                'title': (255, 255, 255),
                'title_stroke': (0, 0, 0),
                'subtitle': (255, 255, 255),
                'shadow': (0, 0, 0),
            }

        # Get dominant color (most frequent)
        dominant = palette[0]

        # Get second most frequent
        secondary = palette[1] if len(palette) > 1 else dominant

        # Title: Use white or black for contrast
        title_color = self.get_contrast_color(dominant)

        # Title stroke: Use dominant or complementary
        # For white text, use darker stroke; for black text, use lighter stroke
        if title_color == (255, 255, 255):
            # White text - use dark stroke
            title_stroke = self._darken_color(dominant, 0.5)
        else:
            # Black text - use light stroke
            title_stroke = self._lighten_color(dominant, 0.5)

        # Subtitle: Use secondary color or slightly transparent version of title
        subtitle_color = title_color

        # Shadow: Always dark but not pure black
        shadow_color = (30, 30, 30)

        colors = {
            'title': title_color,
            'title_stroke': title_stroke,
            'subtitle': subtitle_color,
            'shadow': shadow_color,
            'dominant': dominant,
            'secondary': secondary,
        }

        logger.info(f"Selected text colors: {colors}")

        return colors

    def _darken_color(self, rgb: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
        """
        Darken a color by factor

        Args:
            rgb: RGB color
            factor: Darkening factor (0-1)

        Returns:
            Darkened RGB color
        """
        r, g, b = rgb
        return (
            int(r * (1 - factor)),
            int(g * (1 - factor)),
            int(b * (1 - factor))
        )

    def _lighten_color(self, rgb: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
        """
        Lighten a color by factor

        Args:
            rgb: RGB color
            factor: Lightening factor (0-1)

        Returns:
            Lightened RGB color
        """
        r, g, b = rgb
        return (
            int(r + (255 - r) * factor),
            int(g + (255 - g) * factor),
            int(b + (255 - b) * factor)
        )

    def analyze_mood(self, palette: List[Tuple[int, int, int]]) -> str:
        """
        Analyze mood/tone from palette

        Args:
            palette: Color palette

        Returns:
            Mood description (e.g., "warm", "cool", "vibrant", "muted")
        """
        if not palette:
            return "neutral"

        # Calculate average HSV values
        hsv_values = []
        for rgb in palette:
            r, g, b = [x / 255.0 for x in rgb]
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            hsv_values.append((h, s, v))

        avg_h = np.mean([h for h, s, v in hsv_values])
        avg_s = np.mean([s for h, s, v in hsv_values])
        avg_v = np.mean([v for h, s, v in hsv_values])

        # Determine mood
        moods = []

        # Warm vs Cool
        if 0 <= avg_h < 0.15 or avg_h > 0.85:  # Red-ish
            moods.append("warm")
        elif 0.15 <= avg_h < 0.35:  # Yellow-Green
            moods.append("fresh")
        elif 0.45 <= avg_h < 0.65:  # Blue-ish
            moods.append("cool")
        elif 0.65 <= avg_h < 0.85:  # Purple-Pink
            moods.append("dreamy")

        # Vibrant vs Muted
        if avg_s > 0.6:
            moods.append("vibrant")
        elif avg_s < 0.3:
            moods.append("muted")

        # Bright vs Dark
        if avg_v > 0.7:
            moods.append("bright")
        elif avg_v < 0.3:
            moods.append("dark")

        mood = "_".join(moods) if moods else "neutral"

        logger.debug(f"Analyzed mood: {mood}")

        return mood
