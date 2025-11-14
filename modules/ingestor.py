"""
Ingestor Module - Load and filter images from local folder or extract from videos
"""

from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger
import shutil

from config import settings
from utils.image_utils import get_short_side, get_image_dimensions


class Ingestor:
    """
    Loads images from local folder and filters by extension and size
    """

    def __init__(self, source_dir: Path = None):
        """
        Initialize Ingestor

        Args:
            source_dir: Source directory for images (default: workspace/raw)
        """
        self.source_dir = source_dir or settings.RAW_DIR
        self.allowed_extensions = settings.ALLOWED_EXTENSIONS
        self.min_size = settings.MIN_IMAGE_SIZE

        logger.info(f"Ingestor initialized with source: {self.source_dir}")

    def discover_images(self) -> List[Path]:
        """
        Discover all image files in source directory

        Returns:
            List of image file paths
        """
        if not self.source_dir.exists():
            logger.error(f"Source directory does not exist: {self.source_dir}")
            return []

        all_files = []
        for ext in self.allowed_extensions:
            # Handle both .jpg and .JPG
            all_files.extend(self.source_dir.rglob(f"*{ext}"))
            all_files.extend(self.source_dir.rglob(f"*{ext.upper()}"))

        logger.info(f"Discovered {len(all_files)} image files")
        return all_files

    def filter_by_size(self, image_paths: List[Path]) -> List[Path]:
        """
        Filter images by minimum size (short side)

        Args:
            image_paths: List of image paths

        Returns:
            Filtered list of image paths
        """
        filtered = []

        for img_path in image_paths:
            try:
                short_side = get_short_side(img_path)

                if short_side >= self.min_size:
                    filtered.append(img_path)
                else:
                    logger.debug(
                        f"Skipped {img_path.name}: short side {short_side}px < {self.min_size}px"
                    )

            except Exception as e:
                logger.warning(f"Failed to check size for {img_path.name}: {e}")
                continue

        logger.info(
            f"Filtered {len(filtered)}/{len(image_paths)} images by size (min {self.min_size}px)"
        )
        return filtered

    def get_image_metadata(self, image_path: Path) -> Dict:
        """
        Get metadata for an image

        Args:
            image_path: Path to image

        Returns:
            Dictionary with image metadata
        """
        try:
            width, height = get_image_dimensions(image_path)

            return {
                "path": image_path,
                "filename": image_path.name,
                "width": width,
                "height": height,
                "short_side": min(width, height),
                "long_side": max(width, height),
                "aspect_ratio": width / height,
                "file_size": image_path.stat().st_size,
            }

        except Exception as e:
            logger.error(f"Failed to get metadata for {image_path}: {e}")
            return None

    def ingest(self) -> List[Dict]:
        """
        Main ingestion pipeline: discover, filter, and return metadata

        Returns:
            List of dictionaries containing image metadata
        """
        logger.info("Starting image ingestion...")

        # 1. Discover images
        image_paths = self.discover_images()

        if not image_paths:
            logger.warning("No images found to ingest")
            return []

        # 2. Filter by size
        filtered_paths = self.filter_by_size(image_paths)

        if not filtered_paths:
            logger.warning("No images passed size filter")
            return []

        # 3. Get metadata for each image
        metadata_list = []
        for img_path in filtered_paths:
            metadata = self.get_image_metadata(img_path)
            if metadata:
                metadata_list.append(metadata)

        logger.info(f"Successfully ingested {len(metadata_list)} images")

        return metadata_list

    def copy_to_workspace(self, source_paths: List[Path]) -> List[Path]:
        """
        Copy images to workspace/raw directory

        Args:
            source_paths: List of source image paths

        Returns:
            List of copied image paths in workspace
        """
        settings.RAW_DIR.mkdir(parents=True, exist_ok=True)

        copied_paths = []
        for src_path in source_paths:
            try:
                dst_path = settings.RAW_DIR / src_path.name

                # Handle duplicate names
                counter = 1
                while dst_path.exists():
                    stem = src_path.stem
                    suffix = src_path.suffix
                    dst_path = settings.RAW_DIR / f"{stem}_{counter}{suffix}"
                    counter += 1

                shutil.copy2(src_path, dst_path)
                copied_paths.append(dst_path)
                logger.debug(f"Copied {src_path.name} to workspace")

            except Exception as e:
                logger.error(f"Failed to copy {src_path}: {e}")
                continue

        logger.info(f"Copied {len(copied_paths)} images to workspace")
        return copied_paths

    def clear_workspace(self) -> None:
        """
        Clear all images from workspace directories
        """
        for directory in [settings.RAW_DIR, settings.TEMP_DIR]:
            if directory.exists():
                for file in directory.iterdir():
                    if file.is_file():
                        file.unlink()
                logger.info(f"Cleared {directory}")

    def discover_videos(self) -> List[Path]:
        """
        Discover all video files in source directory

        Returns:
            List of video file paths
        """
        if not self.source_dir.exists():
            logger.error(f"Source directory does not exist: {self.source_dir}")
            return []

        all_files = []
        for ext in settings.VIDEO_FORMATS:
            all_files.extend(self.source_dir.rglob(f"*{ext}"))
            all_files.extend(self.source_dir.rglob(f"*{ext.upper()}"))

        logger.info(f"Discovered {len(all_files)} video files")
        return all_files

    def ingest_from_videos(
        self,
        extract_key_frames: bool = True,
        num_frames_per_video: int = 20
    ) -> List[Dict]:
        """
        Extract frames from videos and ingest them

        Args:
            extract_key_frames: Use key frame extraction (scene detection)
            num_frames_per_video: Number of frames to extract per video

        Returns:
            List of image metadata from extracted frames
        """
        from modules.video_extractor import VideoExtractor

        logger.info("Starting video frame extraction...")

        # Discover videos
        video_paths = self.discover_videos()

        if not video_paths:
            logger.warning("No videos found to process")
            return []

        # Extract frames
        extractor = VideoExtractor(output_dir=settings.RAW_DIR)
        all_frames = []

        for video_path in video_paths:
            if extract_key_frames:
                frames = extractor.extract_key_frames(video_path, num_frames_per_video)
            else:
                frames = extractor.extract_from_video(video_path)

            all_frames.extend(frames)

        logger.info(f"Extracted {len(all_frames)} frames from {len(video_paths)} videos")

        # Get metadata for extracted frames
        metadata_list = []
        for frame_path in all_frames:
            metadata = self.get_image_metadata(frame_path)
            if metadata:
                metadata_list.append(metadata)

        return metadata_list
