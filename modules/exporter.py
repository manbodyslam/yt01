"""
Exporter Module - Save thumbnails with proper naming convention
"""

from pathlib import Path
from datetime import datetime
from typing import Optional
from loguru import logger
import re

from config import settings


class Exporter:
    """
    Exports thumbnails with consistent naming and optional versioning
    """

    def __init__(self, output_dir: Path = None):
        """
        Initialize Exporter

        Args:
            output_dir: Output directory (default: workspace/out)
        """
        self.output_dir = output_dir or settings.OUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporter initialized with output dir: {self.output_dir}")

    def generate_filename(
        self,
        title: str,
        version: Optional[int] = None,
        date: Optional[str] = None
    ) -> str:
        """
        Generate filename following pattern: TITLE__YYYYMMDD_HHMMSS__vXXX.jpg
        เพิ่ม timestamp เพื่อป้องกัน browser cache

        Args:
            title: Thumbnail title
            version: Version number (auto-increment if None)
            date: Date string (use today if None)

        Returns:
            Filename string
        """
        # Clean title (remove special chars, replace spaces with underscores)
        clean_title = self._clean_title(title)

        # Get date + time (เพิ่ม timestamp เพื่อป้องกัน browser cache)
        now = datetime.now()
        if date is None:
            date = now.strftime("%Y%m%d")

        # เพิ่ม timestamp (HH:MM:SS) เพื่อให้แต่ละครั้งที่สร้างได้ไฟล์ใหม่
        timestamp = now.strftime("%H%M%S")
        datetime_str = f"{date}_{timestamp}"

        # Get version
        if version is None:
            version = self._get_next_version(clean_title, date)

        # Format version with leading zeros
        version_str = f"v{version:03d}"

        # Construct filename (เพิ่ม timestamp เข้าไป)
        filename = f"{clean_title}__{datetime_str}__{version_str}.{settings.OUTPUT_FORMAT}"

        return filename

    def _clean_title(self, title: str) -> str:
        """
        Clean title for use in filename

        Args:
            title: Original title

        Returns:
            Cleaned title
        """
        # Remove Thai tones and special characters, keep only Thai, English, numbers, spaces
        # Replace spaces with underscores
        title = title.strip()

        # Remove special characters except Thai, English, numbers, spaces
        title = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9\s_-]', '', title)

        # Replace spaces and multiple underscores with single underscore
        title = re.sub(r'[\s_-]+', '_', title)

        # Limit length
        max_length = 50
        if len(title) > max_length:
            title = title[:max_length]

        return title

    def _get_next_version(self, clean_title: str, date: str) -> int:
        """
        Get next available version number

        Args:
            clean_title: Cleaned title
            date: Date string

        Returns:
            Next version number
        """
        # Find existing files with same title and date
        pattern = f"{clean_title}__{date}__v*.{settings.OUTPUT_FORMAT}"

        existing_files = list(self.output_dir.glob(pattern))

        if not existing_files:
            return 1

        # Extract version numbers
        versions = []
        for file in existing_files:
            match = re.search(r'v(\d+)', file.stem)
            if match:
                versions.append(int(match.group(1)))

        if versions:
            return max(versions) + 1
        else:
            return 1

    def save(
        self,
        thumbnail_path: Path,
        title: str,
        version: Optional[int] = None,
        metadata: Optional[dict] = None
    ) -> Path:
        """
        Save thumbnail with proper naming

        Args:
            thumbnail_path: Path to thumbnail image
            title: Thumbnail title
            version: Optional version number
            metadata: Optional metadata to save alongside

        Returns:
            Path to saved file
        """
        # Generate filename
        filename = self.generate_filename(title, version)
        output_path = self.output_dir / filename

        # Copy file
        if thumbnail_path.exists():
            import shutil
            shutil.copy2(thumbnail_path, output_path)
            logger.info(f"Saved thumbnail: {output_path}")
        else:
            logger.error(f"Thumbnail not found: {thumbnail_path}")
            return None

        # Save metadata if provided
        if metadata:
            self._save_metadata(output_path, metadata)

        return output_path

    def _save_metadata(self, thumbnail_path: Path, metadata: dict) -> None:
        """
        Save metadata JSON alongside thumbnail

        Args:
            thumbnail_path: Path to thumbnail
            metadata: Metadata dictionary
        """
        import json

        metadata_path = thumbnail_path.with_suffix('.json')

        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            logger.debug(f"Saved metadata: {metadata_path}")

        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def list_thumbnails(self) -> list[Path]:
        """
        List all thumbnails in output directory

        Returns:
            List of thumbnail paths
        """
        pattern = f"*.{settings.OUTPUT_FORMAT}"
        thumbnails = sorted(self.output_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

        logger.info(f"Found {len(thumbnails)} thumbnails in {self.output_dir}")

        return thumbnails

    def get_latest_version(self, title: str, date: Optional[str] = None) -> Optional[Path]:
        """
        Get latest version of a thumbnail

        Args:
            title: Thumbnail title
            date: Optional date (use today if None)

        Returns:
            Path to latest version or None
        """
        clean_title = self._clean_title(title)

        if date is None:
            date = datetime.now().strftime("%Y%m%d")

        pattern = f"{clean_title}__{date}__v*.{settings.OUTPUT_FORMAT}"
        matching = list(self.output_dir.glob(pattern))

        if not matching:
            return None

        # Sort by version number
        matching.sort(key=lambda p: int(re.search(r'v(\d+)', p.stem).group(1)), reverse=True)

        return matching[0]

    def cleanup_old_versions(self, title: str, keep_versions: int = 3) -> int:
        """
        Cleanup old versions, keeping only the N most recent

        Args:
            title: Thumbnail title
            keep_versions: Number of versions to keep

        Returns:
            Number of files deleted
        """
        clean_title = self._clean_title(title)
        pattern = f"{clean_title}__*__v*.{settings.OUTPUT_FORMAT}"

        matching = list(self.output_dir.glob(pattern))

        if len(matching) <= keep_versions:
            return 0

        # Sort by modification time
        matching.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Delete old versions
        deleted = 0
        for old_file in matching[keep_versions:]:
            try:
                old_file.unlink()
                logger.debug(f"Deleted old version: {old_file}")

                # Also delete metadata if exists
                metadata_file = old_file.with_suffix('.json')
                if metadata_file.exists():
                    metadata_file.unlink()

                deleted += 1

            except Exception as e:
                logger.error(f"Failed to delete {old_file}: {e}")

        logger.info(f"Cleaned up {deleted} old versions")

        return deleted
