"""
Scorer Module - Score and select best images for background and faces
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger

from utils.image_utils import (
    load_image,
    calculate_sharpness,
    calculate_contrast,
    calculate_brightness,
    is_blurry,
)


class Scorer:
    """
    Scores images and selects best background and face images
    """

    def __init__(self):
        """
        Initialize Scorer
        """
        logger.info("Scorer initialized")

    def score_image(self, image_metadata: Dict, check_blur: bool = True) -> float:
        """
        Score an image based on quality metrics

        Args:
            image_metadata: Image metadata dictionary
            check_blur: Whether to penalize blurry images

        Returns:
            Image quality score (0-1)
        """
        try:
            image_path = image_metadata['path']
            image = load_image(image_path)

            # Calculate metrics
            sharpness = calculate_sharpness(image)
            contrast = calculate_contrast(image)
            brightness = calculate_brightness(image)

            # Check blur
            blurry = is_blurry(image) if check_blur else False

            # Normalize sharpness (typical range 0-500)
            sharpness_norm = min(sharpness / 500.0, 1.0)

            # Normalize contrast (typical range 0-100)
            contrast_norm = min(contrast / 100.0, 1.0)

            # Normalize brightness (ideal range 80-180)
            if 80 <= brightness <= 180:
                brightness_norm = 1.0
            elif brightness < 80:
                brightness_norm = brightness / 80.0
            else:
                brightness_norm = max(0, 1 - (brightness - 180) / 75.0)

            # Composite score
            score = (
                0.4 * sharpness_norm +
                0.3 * contrast_norm +
                0.3 * brightness_norm
            )

            # Penalize if blurry
            if blurry:
                score *= 0.5

            logger.debug(
                f"{image_path.name}: score={score:.3f} "
                f"(sharpness={sharpness:.1f}, contrast={contrast:.1f}, "
                f"brightness={brightness:.1f}, blurry={blurry})"
            )

            return float(score)

        except Exception as e:
            logger.error(f"Failed to score image {image_metadata.get('path')}: {e}")
            return 0.0

    def score_all_images(self, image_metadata_list: List[Dict]) -> List[Dict]:
        """
        Score all images

        Args:
            image_metadata_list: List of image metadata

        Returns:
            List of image metadata with added 'score' field
        """
        logger.info(f"Scoring {len(image_metadata_list)} images...")

        scored_images = []

        for img_meta in image_metadata_list:
            score = self.score_image(img_meta)
            img_meta['score'] = score
            scored_images.append(img_meta)

        # Sort by score (descending)
        scored_images.sort(key=lambda x: x['score'], reverse=True)

        logger.info(f"Scored all images. Best score: {scored_images[0]['score']:.3f}")

        return scored_images

    def select_background_image(
        self,
        image_metadata_list: List[Dict],
        exclude_paths: Optional[List[Path]] = None
    ) -> Optional[Dict]:
        """
        Select best background image

        Args:
            image_metadata_list: List of image metadata
            exclude_paths: Optional list of paths to exclude

        Returns:
            Best background image metadata
        """
        exclude_paths = exclude_paths or []

        # Filter out excluded images
        candidates = [
            img for img in image_metadata_list
            if img['path'] not in exclude_paths
        ]

        if not candidates:
            logger.warning("No background image candidates available")
            return None

        # Score images
        scored = self.score_all_images(candidates)

        # Select best
        best_bg = scored[0]

        logger.info(f"Selected background image: {best_bg['filename']} (score={best_bg['score']:.3f})")

        return best_bg

    def select_face_images(
        self,
        characters: Dict[str, Dict],
        prioritize_different: bool = True
    ) -> Dict[str, Dict]:
        """
        Select best face image for each character

        Args:
            characters: Character data from FaceService
            prioritize_different: Try to use different images for different characters

        Returns:
            Updated character data with selected face images
        """
        used_images = set()

        for role, char_data in characters.items():
            face_data = char_data['face_data']

            # Check if we should look for alternative if image already used
            if prioritize_different and face_data['image_path'] in used_images:
                # Try to find alternative face from same cluster
                cluster_faces = char_data['cluster_stats']['face_indices']

                # Find faces from different images
                alternatives = []
                for face_idx in cluster_faces:
                    # Access from parent scope (FaceService.face_db)
                    # This is tricky - we need access to face_db
                    # For now, keep the best face even if duplicate
                    pass

            used_images.add(face_data['image_path'])

            logger.info(f"{role}: {face_data['face_id']} (composite={face_data['composite_score']:.3f})")

        return characters

    def evaluate_composition(self, image_path: Path, face_bboxes: List) -> float:
        """
        Evaluate composition quality of an image

        Args:
            image_path: Path to image
            face_bboxes: List of face bounding boxes

        Returns:
            Composition score (0-1)
        """
        try:
            image = load_image(image_path)
            h, w = image.shape[:2]

            scores = []

            # 1. Rule of thirds
            # Check if faces are positioned along rule of thirds lines
            third_h = h / 3
            third_w = w / 3

            for bbox in face_bboxes:
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # Distance to nearest third line
                dist_x = min(
                    abs(center_x - third_w),
                    abs(center_x - 2 * third_w)
                )
                dist_y = min(
                    abs(center_y - third_h),
                    abs(center_y - 2 * third_h)
                )

                # Normalize (closer to third = better)
                dist_norm_x = 1 - min(dist_x / (w / 6), 1)
                dist_norm_y = 1 - min(dist_y / (h / 6), 1)

                composition_score = (dist_norm_x + dist_norm_y) / 2
                scores.append(composition_score)

            # 2. Check if faces are not too close to edges
            edge_margin = 0.1  # 10% margin
            edge_scores = []

            for bbox in face_bboxes:
                x1, y1, x2, y2 = bbox

                # Check distances from edges
                dist_left = x1 / w
                dist_right = (w - x2) / w
                dist_top = y1 / h
                dist_bottom = (h - y2) / h

                min_dist = min(dist_left, dist_right, dist_top, dist_bottom)

                if min_dist >= edge_margin:
                    edge_scores.append(1.0)
                else:
                    edge_scores.append(min_dist / edge_margin)

            scores.extend(edge_scores)

            # Average all scores
            final_score = np.mean(scores) if scores else 0.5

            return float(final_score)

        except Exception as e:
            logger.error(f"Failed to evaluate composition: {e}")
            return 0.5
