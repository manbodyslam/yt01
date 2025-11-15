"""
YouTube Thumbnail Generator - FastAPI Application
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Dict
from pathlib import Path
from loguru import logger
import sys
import shutil
import httpx
import base64
from urllib.parse import urlparse, parse_qs
import mimetypes
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import os
import re
from datetime import datetime
from uuid import uuid4
import asyncio
import traceback
import time
import gc
from multiprocessing import Process

from config import settings
from utils.task_storage import get_storage
from modules import (
    Ingestor,
    FaceService,
    Scorer,
    PaletteExtractor,
    LayoutEngine,
    Renderer,
    Exporter,
    VideoExtractor,
    AIAnalyzer,
)
from modules.thumbnail_validator import ThumbnailValidator
import json

# ============================================================================
# Progress Tracking System (Real-time updates via SSE)
# ============================================================================
class ProgressTracker:
    """Track progress of video processing jobs and stream updates via SSE"""

    def __init__(self):
        self.jobs: Dict[str, Dict] = {}  # {job_id: {progress, status, message, step}}

    def create_job(self, job_id: str) -> None:
        """Create a new job"""
        self.jobs[job_id] = {
            "progress": 0,
            "status": "started",
            "message": "เริ่มต้นประมวลผล...",
            "step": "init",
            "timestamp": time.time()
        }

    def update(self, job_id: str, progress: int, message: str, step: str = None) -> None:
        """Update job progress"""
        if job_id in self.jobs:
            self.jobs[job_id].update({
                "progress": progress,
                "message": message,
                "step": step or self.jobs[job_id].get("step", "processing"),
                "timestamp": time.time()
            })

    def complete(self, job_id: str, success: bool = True, message: str = None) -> None:
        """Mark job as completed"""
        if job_id in self.jobs:
            self.jobs[job_id].update({
                "progress": 100 if success else self.jobs[job_id]["progress"],
                "status": "completed" if success else "failed",
                "message": message or ("เสร็จสมบูรณ์!" if success else "เกิดข้อผิดพลาด"),
                "timestamp": time.time()
            })

    def get_job(self, job_id: str) -> Optional[Dict]:
        """Get job status"""
        return self.jobs.get(job_id)

    def cleanup_old_jobs(self, max_age_seconds: int = 600) -> None:
        """Remove jobs older than max_age_seconds (default: 10 minutes)"""
        current_time = time.time()
        to_remove = [
            job_id for job_id, job in self.jobs.items()
            if current_time - job.get("timestamp", 0) > max_age_seconds
        ]
        for job_id in to_remove:
            del self.jobs[job_id]

# Global progress tracker
progress_tracker = ProgressTracker()

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/app.log", rotation="500 MB", retention="10 days", level="DEBUG")

# Create logs directory
Path("logs").mkdir(exist_ok=True)

# ============================================================================
# Preset System
# ============================================================================
def load_presets() -> dict:
    """
    Load preset configurations from presets.json

    Returns:
        Dictionary of presets with crop point mappings
    """
    try:
        preset_file = Path(__file__).parent / "presets.json"
        with open(preset_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("presets.json not found, using default presets")
        return {
            "1": {
                "id": "1",
                "name": "แบบครึ่งตัว (หัว-เอว)",
                "description": "แสดงตัวละครตั้งแต่หัวถึงเอว",
                "num_characters": 3,
                "crop_point": "waist",
                "layout_type": "tri_hero"
            }
        }
    except Exception as e:
        logger.error(f"Error loading presets: {e}")
        return {}

# Crop point to multiplier mapping
CROP_MULTIPLIERS = {
    "waist": 3.5,       # แบบครึ่งตัว (หัว-เอว) - ค่าเดิม
    "full_body": 6.5,   # แบบเต็มตัว (หัว-เท้า) - แบบใหม่
}

# Load presets on startup
PRESETS = load_presets()
logger.info(f"Loaded {len(PRESETS)} preset(s): {list(PRESETS.keys())}")

# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Automated YouTube Thumbnail Generator with Face Detection and Clustering"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ============================================================================
# ASYNC TASK STORAGE (for long-running operations)
# ============================================================================
# File-based task storage for multi-process communication
task_storage = get_storage()


# ============================================================================
# Google Drive Helper Functions
# ============================================================================
def extract_google_drive_id(url: str) -> Optional[str]:
    """
    Extract file ID from Google Drive URL

    Supports these formats:
    - https://drive.google.com/file/d/FILE_ID/view
    - https://drive.google.com/file/d/FILE_ID/view?usp=sharing
    - https://drive.google.com/open?id=FILE_ID
    - https://drive.google.com/uc?id=FILE_ID

    Args:
        url: Google Drive share URL

    Returns:
        File ID or None if not found
    """
    try:
        # Pattern 1: /file/d/FILE_ID/
        match = re.search(r'/file/d/([a-zA-Z0-9_-]+)', url)
        if match:
            return match.group(1)

        # Pattern 2: ?id=FILE_ID or &id=FILE_ID
        match = re.search(r'[?&]id=([a-zA-Z0-9_-]+)', url)
        if match:
            return match.group(1)

        # Pattern 3: uc?id=FILE_ID (direct download URL)
        parsed = urlparse(url)
        if 'id' in parse_qs(parsed.query):
            return parse_qs(parsed.query)['id'][0]

        logger.error(f"Could not extract file ID from URL: {url}")
        return None

    except Exception as e:
        logger.error(f"Error extracting Google Drive ID: {e}")
        return None


def get_google_drive_direct_url(file_id: str) -> str:
    """
    Convert Google Drive file ID to direct download URL

    Args:
        file_id: Google Drive file ID

    Returns:
        Direct download URL
    """
    # Method 1: Simple direct download (works for files < 100MB without virus scan)
    return f"https://drive.google.com/uc?export=download&id={file_id}"


async def download_from_google_drive_direct(
    url: str,
    destination: Path,
    timeout: int = 600
) -> tuple[bool, Optional[str]]:
    """
    Download file from Google Drive using direct download URL

    This method:
    - Doesn't require API key or authentication
    - Works for public files (Anyone with the link)
    - Handles virus scan warning for large files (>100MB)

    Args:
        url: Google Drive share URL
        destination: Local path to save file
        timeout: Download timeout in seconds (default: 600 = 10 minutes)

    Returns:
        (success: bool, error_message: Optional[str])
    """
    try:
        # Extract file ID
        file_id = extract_google_drive_id(url)
        if not file_id:
            return False, "Invalid Google Drive URL - could not extract file ID"

        logger.info(f"📥 Downloading from Google Drive: file_id={file_id}")

        # Get direct download URL
        download_url = get_google_drive_direct_url(file_id)

        # Create HTTP client with longer timeout for large files
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=5)
        ) as client:
            # First request - may get virus scan warning for large files
            response = await client.get(download_url)

            # Check if we got a virus scan warning page
            if 'virus scan warning' in response.text.lower() or 'confirm=' in response.text:
                logger.info("⚠️  Large file detected, bypassing virus scan warning...")

                # Extract confirm token and uuid from the HTML form
                confirm_match = re.search(r'name="confirm" value="([^"]+)"', response.text)
                uuid_match = re.search(r'name="uuid" value="([^"]+)"', response.text)

                if confirm_match and uuid_match:
                    confirm_token = confirm_match.group(1)
                    uuid_token = uuid_match.group(1)

                    # Use drive.usercontent.google.com (new domain for large files)
                    download_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm={confirm_token}&uuid={uuid_token}"

                    logger.info(f"✓ Got confirmation tokens (confirm={confirm_token}, uuid={uuid_token[:8]}...), downloading...")
                    response = await client.get(download_url)
                else:
                    # Fallback: try old method
                    logger.warning("Could not extract new tokens, trying old method...")
                    match = re.search(r'confirm=([0-9A-Za-z_-]+)', response.text)
                    if match:
                        confirm_token = match.group(1)
                        download_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={confirm_token}"
                        logger.info("✓ Using old confirmation method...")
                        response = await client.get(download_url)

            # Check response
            if response.status_code != 200:
                error_msg = f"Download failed with status {response.status_code}"
                logger.error(error_msg)
                return False, error_msg

            # Check if we got HTML instead of video (means file is not public)
            content_type = response.headers.get('content-type', '')
            if 'text/html' in content_type:
                error_msg = (
                    "ไม่สามารถดาวน์โหลดได้ - ไฟล์อาจไม่เป็น public\n"
                    "กรุณาตั้งค่าไฟล์เป็น 'Anyone with the link' ใน Google Drive"
                )
                logger.error(error_msg)
                return False, error_msg

            # Save to file
            destination.parent.mkdir(parents=True, exist_ok=True)

            total_size = int(response.headers.get('content-length', 0))
            logger.info(f"📦 File size: {total_size / (1024*1024):.2f} MB")

            with open(destination, 'wb') as f:
                f.write(response.content)

            logger.info(f"✅ Downloaded successfully: {destination}")
            return True, None

    except httpx.TimeoutException:
        error_msg = f"Download timeout after {timeout} seconds"
        logger.error(error_msg)
        return False, error_msg
    except httpx.HTTPError as e:
        error_msg = f"HTTP error downloading from Google Drive: {e}"
        logger.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Error downloading from Google Drive: {e}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg


# Helper Functions
def _clear_workspace_files():
    """
    Clear workspace directories (raw frames and output thumbnails)
    Called automatically when uploading new video (internal helper)
    """
    try:
        # Clear raw frames
        raw_dir = settings.RAW_DIR
        if raw_dir.exists():
            for file in raw_dir.iterdir():
                if file.is_file() and file.name != '.gitkeep':
                    file.unlink()
            logger.info(f"✓ Cleared {raw_dir}")

        # ⚠️ DO NOT clear OUT_DIR - keep thumbnails for history
        # User wants to keep all generated thumbnails

        # Clear videos directory
        video_dir = settings.WORKSPACE_DIR / "videos"
        if video_dir.exists():
            for file in video_dir.iterdir():
                if file.is_file() and file.name != '.gitkeep':
                    file.unlink()
            logger.info(f"✓ Cleared {video_dir}")

        logger.info("🧹 Workspace cleared successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to clear workspace: {e}")
        return False


# Request/Response Models
class GenerateRequest(BaseModel):
    """Request model for thumbnail generation"""
    title: str = Field(..., description="Thumbnail title text")
    subtitle: Optional[str] = Field(None, description="Optional subtitle text")
    num_characters: int = Field(3, ge=1, le=4, description="Number of characters to include (1-4)")
    source_folder: Optional[str] = Field(None, description="Source folder path (uses workspace/raw if not specified)")
    text_style: str = Field("style1", description="Text style preset: style1 (แดง-เหลือง), style2 (ขาว-เหลือง), style3 (ขาว-ส้ม), auto")


class GenerateResponse(BaseModel):
    """Response model for thumbnail generation"""
    success: bool
    thumbnail_path: Optional[str] = None
    filename: Optional[str] = None
    metadata: Optional[dict] = None
    error: Optional[str] = None
    job_id: Optional[str] = None  # For real-time progress tracking via SSE


class StatusResponse(BaseModel):
    """Status response model"""
    status: str
    message: str


class VideoUploadResponse(BaseModel):
    """Response model for video upload"""
    success: bool
    video_path: Optional[str] = None
    filename: Optional[str] = None
    extracted_frames: Optional[int] = None
    error: Optional[str] = None


class N8nGenerateRequest(BaseModel):
    """Request model for n8n integration - generate from video URL"""
    video_url: str = Field(..., description="URL of the video to download")
    title: str = Field(..., description="Thumbnail title")
    subtitle: Optional[str] = Field(None, description="Optional subtitle")
    num_characters: int = Field(3, ge=3, le=3, description="Number of characters (fixed at 3)")
    num_frames: int = Field(200, ge=5, le=200, description="Number of frames to extract (5-200)")
    text_style: str = Field("style1", description="Text style: style1, style2, style3, auto")
    return_base64: bool = Field(False, description="Return thumbnail as base64 instead of URL")


class N8nGenerateResponse(BaseModel):
    """Response model for n8n integration"""
    success: bool
    filename: Optional[str] = None
    thumbnail_url: Optional[str] = None
    thumbnail_base64: Optional[str] = None
    metadata: Optional[Dict] = None
    error: Optional[str] = None


class GoogleDriveRequest(BaseModel):
    """Request model for Google Drive integration"""
    file_id: str = Field(..., description="Google Drive file ID")
    access_token: Optional[str] = Field(None, description="OAuth 2.0 access token (if not using service account)")
    title: str = Field(..., description="Thumbnail title")
    subtitle: Optional[str] = Field(None, description="Optional subtitle")
    num_characters: int = Field(3, ge=3, le=3, description="Number of characters (fixed at 3)")
    num_frames: int = Field(200, ge=5, le=200, description="Number of frames to extract (5-200)")
    text_style: str = Field("style1", description="Text style: style1, style2, style3, auto")
    return_base64: bool = Field(False, description="Return thumbnail as base64 instead of URL")


class BatchGenerateResponse(BaseModel):
    """Response model for batch generation"""
    success: bool
    thumbnails: List[Dict] = []  # List of {filename, url, metadata}
    total_generated: int = 0
    failed: int = 0
    errors: List[str] = []
    message: Optional[str] = None


# Pipeline class
class ThumbnailPipeline:
    """
    Complete thumbnail generation pipeline
    """

    def __init__(self):
        """Initialize pipeline components"""
        self.ingestor = Ingestor()
        self.face_service = FaceService()
        self.scorer = Scorer()
        self.palette_extractor = PaletteExtractor()
        self.layout_engine = LayoutEngine()
        self.renderer = Renderer()
        self.exporter = Exporter()
        self.ai_analyzer = AIAnalyzer()
        self.validator = ThumbnailValidator()

        logger.info("Thumbnail Pipeline initialized")

    def generate(
        self,
        title: str,
        subtitle: Optional[str] = None,
        num_characters: int = 3,
        source_folder: Optional[Path] = None,
        text_style: str = "style1",
        layout_type: Optional[str] = None,
        custom_positions: Optional[List[Dict]] = None,
        allow_duplicates: bool = False,
        randomize: bool = False,
        vertical_align: str = "top"
    ) -> dict:
        """
        Run complete thumbnail generation pipeline

        Args:
            title: Title text
            subtitle: Subtitle text
            num_characters: Number of characters to include
            source_folder: Source image folder
            text_style: Text style preset (style1, style2, style3, auto)
            allow_duplicates: If True, allow selecting same person multiple times (last resort fallback)

        Returns:
            Result dictionary
        """
        # Start timing
        start_time = time.time()
        start_datetime = datetime.now()

        try:
            logger.info(f"Starting thumbnail generation: '{title}'")

            # AI DISABLED - Use fixed defaults
            ai_suggestions = {
                'layout_type': 'tri_hero',  # ใช้ tri_hero เป็นค่าเริ่มต้น
                'text_style': 'style1',
                'mood': 'neutral'
            }
            logger.info(f"🎨 Using fixed layout: tri_hero (AI disabled)")

            # Use default text style
            if text_style == "auto":
                text_style = "style1"
                logger.info(f"   Using default text style: {text_style}")

            # 2. Ingest images
            if source_folder:
                self.ingestor.source_dir = Path(source_folder)

            image_metadata_list = self.ingestor.ingest()

            if not image_metadata_list:
                raise ValueError("No valid images found in source folder")

            logger.info(f"Ingested {len(image_metadata_list)} images")

            # 2. Analyze faces with Multi-Pass Adaptive System
            # 🎯 รับประกันเจอหน้าครบ 100%!
            num_clusters_found = self.face_service.analyze_all_images_adaptive(
                image_metadata_list,
                required_characters=num_characters
            )

            if not self.face_service.face_db:
                raise ValueError("No faces detected in any images")

            # เช็คว่าเจอคนครบหรือเปล่า
            if num_clusters_found < num_characters:
                # ถ้า Multi-Pass ลองทุกวิธีแล้วยังไม่ครบ → throw error
                from utils.exceptions import InsufficientCharactersError
                raise InsufficientCharactersError(
                    found=num_clusters_found,
                    required=num_characters,
                    message=f"Found only {num_clusters_found}/{num_characters} different people. "
                            f"Multi-pass adaptive system tried all thresholds but still insufficient."
                )

            # 3. Select characters
            characters = self.face_service.select_characters(num_characters, allow_duplicates=allow_duplicates, randomize=randomize)

            if not characters:
                raise ValueError("Could not select characters (no clusters found)")

            logger.info(f"Selected {len(characters)} characters")

            # 5. Score images and select background
            # Exclude ALL images with faces (not just selected characters)
            # เลือกเฉพาะเฟรมที่ไม่มีหน้าคนเลย สำหรับพื้นหลังที่เบลอ
            all_face_image_paths = {face['image_path'] for face in self.face_service.get_all_faces()}
            background_image = self.scorer.select_background_image(
                image_metadata_list,
                exclude_paths=list(all_face_image_paths)
            )

            if not background_image:
                # Fall back to any image
                background_image = image_metadata_list[0]

            logger.info(f"Selected background: {background_image['filename']}")

            # 6. Extract color palette
            palette = self.palette_extractor.extract_palette(background_image['path'])
            colors = self.palette_extractor.select_text_colors(palette)
            mood = self.palette_extractor.analyze_mood(palette)

            logger.info(f"Extracted palette: {palette[:3]}... (mood: {mood})")

            # 7. Determine final layout (user override > AI suggestion > fallback)
            actual_char_count = len(characters)

            # If user specified layout, use it; otherwise use AI suggestion
            if layout_type:
                suggested_layout = layout_type
                logger.info(f"🎨 Using user-specified layout: {layout_type}")
            else:
                suggested_layout = ai_suggestions['layout_type']
                logger.info(f"🤖 Using AI-suggested layout: {suggested_layout}")

            # Layout requirements mapping
            layout_requirements = {
                'solo_focus': 1,
                'duo_focus': 2,
                'duo_diagonal': 2,
                'tri_hero': 3,
                'tri_pyramid': 3,
                'tri_staggered': 3,
                'quad_lineup': 4
            }

            # Validate layout matches character count
            if suggested_layout in layout_requirements:
                required_count = layout_requirements[suggested_layout]
                if actual_char_count != required_count:
                    # Fallback to appropriate layout based on actual count
                    fallback_layouts = {
                        1: 'solo_focus',
                        2: 'duo_focus',
                        3: 'tri_hero',
                        4: 'quad_lineup'
                    }
                    final_layout = fallback_layouts.get(actual_char_count, 'solo_focus')
                    logger.warning(
                        f"Layout mismatch: '{suggested_layout}' (needs {required_count} chars) "
                        f"but only have {actual_char_count} chars. Falling back to '{final_layout}'"
                    )
                else:
                    final_layout = suggested_layout
            else:
                # Unknown layout, use safe fallback
                fallback_layouts = {
                    1: 'solo_focus',
                    2: 'duo_focus',
                    3: 'tri_hero',
                    4: 'quad_lineup'
                }
                final_layout = fallback_layouts.get(actual_char_count, 'solo_focus')
                logger.warning(f"Unknown layout '{suggested_layout}', falling back to '{final_layout}'")

            # Create layout with validated layout type and custom positions
            layout = self.layout_engine.create_layout(
                characters=characters,
                layout_type=final_layout,
                custom_positions=custom_positions,
                vertical_align=vertical_align,
                title=title,
                subtitle=subtitle
            )

            if final_layout == suggested_layout:
                logger.info(f"Created layout: {layout['type']} (AI suggested)")
            else:
                logger.info(f"Created layout: {layout['type']} (fallback from AI suggestion '{suggested_layout}')")

            # 8. Render thumbnail
            thumbnail = self.renderer.create_thumbnail(
                background_image_path=background_image['path'],
                characters=characters,
                layout=layout,
                colors=colors,
                title=title,
                subtitle=subtitle,
                text_style=text_style
            )

            # 9. Save to temp
            temp_path = settings.TEMP_DIR / "current_thumbnail.jpg"
            self.renderer.save_thumbnail(thumbnail, temp_path)

            # 10. Export with proper naming
            final_path = self.exporter.save(temp_path, title)

            # Prepare metadata
            metadata = {
                'title': title,
                'subtitle': subtitle,
                'num_characters': len(characters),
                'layout_type': layout['type'],
                'text_style': text_style,
                'background_image': str(background_image['filename']),
                'characters': {
                    role: {
                        'cluster_id': data['cluster_id'],
                        'face_count': data['cluster_stats']['face_count'],
                        'source_image': data['face_data']['image_path'].name,
                    }
                    for role, data in characters.items()
                },
                'palette': palette,
                'mood': mood,
                'ai_suggestions': ai_suggestions,
            }

            logger.info(f"✓ Thumbnail generated successfully: {final_path}")

            # 11. AI Validation (Option 4: Hybrid Validation)
            # ⚠️ OPTIMIZATION: Skip validation ตอนสร้าง เพื่อลดค่าใช้จ่าย
            # จะ validate แค่ top 10 thumbnails ตอนท้ายแทน
            validation_result = None
            skip_inline_validation = True  # ✅ ประหยัดค่า Gemini API

            if settings.ENABLE_AI_VALIDATION and self.validator.model and not skip_inline_validation:
                logger.info("")
                logger.info("=" * 60)
                logger.info("🤖 AI VALIDATION (Gemini 2.5 Pro)")
                logger.info("=" * 60)

                validation_result = self.validator.validate(final_path, title=title)

                status_emoji = "✅" if validation_result.passed else "⚠️"
                logger.info(f"{status_emoji} Overall Score: {validation_result.score}/10")
                logger.info(f"📝 Threshold: {settings.VALIDATION_THRESHOLD}/10")

                if validation_result.detailed_scores:
                    logger.info("📊 Detailed Scores:")
                    for key, value in validation_result.detailed_scores.items():
                        logger.info(f"   • {key.replace('_', ' ').title()}: {value}")

                logger.info(f"💬 AI Feedback: {validation_result.feedback}")
                logger.info("=" * 60)
                logger.info("")

                if not validation_result.passed:
                    logger.warning(
                        f"⚠️  Thumbnail scored {validation_result.score}/10 "
                        f"(below threshold {settings.VALIDATION_THRESHOLD})"
                    )
                    logger.info(
                        "💡 Consider regenerating with different parameters or "
                        "adjusting text/layout based on AI feedback"
                    )
                else:
                    logger.success(f"✅ Thumbnail passed AI validation!")

            # Add validation results to metadata
            if validation_result:
                metadata['validation'] = {
                    'enabled': True,
                    'model': settings.GEMINI_MODEL,
                    'score': validation_result.score,
                    'passed': validation_result.passed,
                    'threshold': settings.VALIDATION_THRESHOLD,
                    'feedback': validation_result.feedback,
                    'detailed_scores': validation_result.detailed_scores,
                }
            else:
                metadata['validation'] = {
                    'enabled': False,
                    'reason': 'AI validation disabled or Gemini not configured'
                }

            # Convert numpy types to Python native types for JSON serialization
            def convert_numpy(obj):
                """Recursively convert numpy types to Python native types"""
                import numpy as np
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj

            metadata = convert_numpy(metadata)

            # Calculate processing time
            end_time = time.time()
            end_datetime = datetime.now()
            total_seconds = end_time - start_time

            # Format time nicely
            minutes = int(total_seconds // 60)
            seconds = total_seconds % 60

            # Add timing information to metadata
            metadata['processing_time'] = {
                'total_seconds': round(total_seconds, 2),
                'formatted': f"{minutes} นาที {seconds:.2f} วินาที" if minutes > 0 else f"{seconds:.2f} วินาที",
                'start_time': start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            }

            logger.success(f"⏱️  Total processing time: {metadata['processing_time']['formatted']}")

            return {
                'success': True,
                'thumbnail_path': str(final_path),
                'filename': final_path.name,
                'metadata': metadata,
            }

        except Exception as e:
            # ⚠️ IMPORTANT: Let InsufficientCharactersError propagate to generate_from_video
            from utils.exceptions import InsufficientCharactersError
            if isinstance(e, InsufficientCharactersError):
                # Re-raise ให้ generate_from_video จัดการ (Smart Fallback)
                raise

            # For other exceptions, return error dict
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"Pipeline failed: {e}")
            logger.error(f"Full traceback:\n{error_trace}")
            return {
                'success': False,
                'error': str(e),
                'traceback': error_trace if settings.API_HOST == "0.0.0.0" else None,
            }


# Global pipeline instance
pipeline = ThumbnailPipeline()


# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve Web UI"""
    index_path = Path(__file__).parent / "static" / "index.html"

    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return HTMLResponse(
            content="<h1>YouTube Thumbnail Generator API</h1><p>Web UI not found. Please check static files.</p>",
            status_code=200
        )


@app.get("/health", response_model=StatusResponse)
async def health():
    """Health check endpoint"""
    return StatusResponse(
        status="healthy",
        message="All systems operational"
    )


@app.get("/health/detailed")
async def health_detailed():
    """Detailed health check with dependency status"""
    import sys
    import platform
    from datetime import datetime

    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "YouTube Thumbnail Generator",
        "version": "2.0.0",
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "dependencies": {}
    }

    # Check OpenAI
    try:
        import openai
        openai_key = os.getenv("OPENAI_API_KEY")
        health_status["dependencies"]["openai"] = {
            "available": True,
            "api_key_configured": bool(openai_key),
            "version": openai.__version__
        }
    except Exception as e:
        health_status["dependencies"]["openai"] = {
            "available": False,
            "error": str(e)
        }

    # Check Gemini
    try:
        import google.generativeai as genai
        gemini_key = os.getenv("GEMINI_API_KEY")
        health_status["dependencies"]["gemini"] = {
            "available": True,
            "api_key_configured": bool(gemini_key)
        }
    except Exception as e:
        health_status["dependencies"]["gemini"] = {
            "available": False,
            "error": str(e)
        }

    # Check InsightFace
    try:
        import insightface
        health_status["dependencies"]["insightface"] = {
            "available": True,
            "version": insightface.__version__
        }
    except Exception as e:
        health_status["dependencies"]["insightface"] = {
            "available": False,
            "error": str(e)
        }

    # Check OpenCV
    try:
        import cv2
        health_status["dependencies"]["opencv"] = {
            "available": True,
            "version": cv2.__version__
        }
    except Exception as e:
        health_status["dependencies"]["opencv"] = {
            "available": False,
            "error": str(e)
        }

    # Check ffmpeg
    import subprocess
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
        health_status["dependencies"]["ffmpeg"] = {
            "available": result.returncode == 0,
            "version": result.stdout.split('\n')[0] if result.returncode == 0 else None
        }
    except Exception as e:
        health_status["dependencies"]["ffmpeg"] = {
            "available": False,
            "error": str(e)
        }

    return health_status


@app.get("/progress/{job_id}")
async def stream_progress(job_id: str):
    """
    Stream real-time progress updates via Server-Sent Events (SSE)

    Usage:
        const eventSource = new EventSource(`/progress/${jobId}`);
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log(data.progress, data.message);
        };
    """
    async def event_generator():
        """Generate SSE events"""
        # Send initial status
        job = progress_tracker.get_job(job_id)
        if not job:
            yield {
                "event": "error",
                "data": json.dumps({"error": "Job not found"})
            }
            return

        # Stream progress updates
        last_progress = -1
        last_heartbeat = time.time()  # 🆕 Track last heartbeat time
        while True:
            job = progress_tracker.get_job(job_id)

            if not job:
                yield {
                    "event": "error",
                    "data": json.dumps({"error": "Job not found"})
                }
                break

            # Send update if progress changed
            if job["progress"] != last_progress:
                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "progress": job["progress"],
                        "message": job["message"],
                        "step": job["step"],
                        "status": job.get("status", "processing")
                    })
                }
                last_progress = job["progress"]
                last_heartbeat = time.time()  # Reset heartbeat on progress update

            # 🆕 Send heartbeat every 30 seconds to prevent timeout
            current_time = time.time()
            if current_time - last_heartbeat >= 30:
                yield {
                    "event": "heartbeat",
                    "data": json.dumps({"timestamp": current_time})
                }
                last_heartbeat = current_time

            # Check if completed
            if job.get("status") in ["completed", "failed"]:
                yield {
                    "event": "complete",
                    "data": json.dumps({
                        "progress": job["progress"],
                        "message": job["message"],
                        "status": job["status"]
                    })
                }
                break

            # Wait before next update
            await asyncio.sleep(0.5)  # Check every 0.5 seconds

    return EventSourceResponse(event_generator())


@app.post("/generate", response_model=GenerateResponse)
async def generate_thumbnail(request: GenerateRequest):
    """
    Generate thumbnail

    Args:
        request: Generation request

    Returns:
        Generation result
    """
    result = pipeline.generate(
        title=request.title,
        subtitle=request.subtitle,
        num_characters=request.num_characters,
        source_folder=Path(request.source_folder) if request.source_folder else None,
        text_style=request.text_style
    )

    if result['success']:
        return GenerateResponse(**result)
    else:
        raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))


@app.get("/thumbnail/{filename}")
async def get_thumbnail(filename: str):
    """
    Get thumbnail file

    Args:
        filename: Thumbnail filename

    Returns:
        Image file
    """
    file_path = settings.OUT_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    return FileResponse(
        file_path,
        media_type="image/jpeg",
        filename=filename
    )


@app.get("/thumbnails", response_model=List[str])
async def list_thumbnails():
    """
    List all generated thumbnails

    Returns:
        List of thumbnail filenames
    """
    exporter = Exporter()
    thumbnails = exporter.list_thumbnails()

    return [thumb.name for thumb in thumbnails]


@app.get("/presets")
async def get_presets():
    """
    Get available thumbnail presets

    Returns list of preset configurations with:
    - id: Preset identifier
    - name: Display name (Thai)
    - description: Preset description
    - num_characters: Number of characters
    - crop_point: Crop type (waist, full_body)
    - layout_type: Layout type (tri_hero, etc.)

    Example response:
    {
        "1": {
            "id": "1",
            "name": "แบบครึ่งตัว (หัว-เอว)",
            "description": "แสดงตัวละครตั้งแต่หัวถึงเอว - เหมาะสำหรับ YouTube",
            "num_characters": 3,
            "crop_point": "waist",
            "layout_type": "tri_hero"
        },
        "2": {
            "id": "2",
            "name": "แบบเต็มตัว (หัว-เท้า)",
            "description": "แสดงตัวละครเต็มตัวตั้งแต่หัวถึงเท้า",
            "num_characters": 3,
            "crop_point": "full_body",
            "layout_type": "tri_hero"
        }
    }
    """
    return PRESETS


@app.get("/analysis-report")
async def get_analysis_report():
    """
    Get comprehensive analysis report with all frame scores
    สำหรับให้ระบบอื่นมาช่วยวิเคราะห์เพิ่มเติม

    Returns:
        JSON report with all face analysis data
    """
    if pipeline.face_service is None:
        raise HTTPException(
            status_code=400,
            detail="No analysis data available. Please generate a thumbnail first."
        )

    report = pipeline.face_service.export_analysis_report()

    return report


@app.delete("/thumbnail/{filename}", response_model=StatusResponse)
async def delete_thumbnail(filename: str):
    """
    Delete thumbnail

    Args:
        filename: Thumbnail filename

    Returns:
        Status response
    """
    file_path = settings.OUT_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    try:
        file_path.unlink()

        # Also delete metadata if exists
        metadata_path = file_path.with_suffix('.json')
        if metadata_path.exists():
            metadata_path.unlink()

        return StatusResponse(
            status="success",
            message=f"Deleted {filename}"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {str(e)}")


@app.post("/clear-workspace", response_model=StatusResponse)
async def clear_workspace():
    """
    Clear workspace directories (raw, temp folders only)

    Returns:
        Status response
    """
    try:
        ingestor = Ingestor()
        ingestor.clear_workspace()

        return StatusResponse(
            status="success",
            message="Workspace cleared (raw and temp folders)"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear workspace: {str(e)}")


@app.post("/clear-videos", response_model=StatusResponse)
async def clear_videos():
    """
    Delete all uploaded videos to free up space

    Returns:
        Status response
    """
    try:
        video_dir = settings.WORKSPACE_DIR / "videos"

        if not video_dir.exists():
            return StatusResponse(
                status="success",
                message="No videos to delete"
            )

        # Count and delete video files
        video_files = list(video_dir.glob("*"))
        deleted_count = 0
        total_size = 0

        for video_file in video_files:
            if video_file.is_file():
                total_size += video_file.stat().st_size
                video_file.unlink()
                deleted_count += 1

        # Convert size to human readable
        size_mb = total_size / (1024 * 1024)
        size_gb = size_mb / 1024

        if size_gb >= 1:
            size_str = f"{size_gb:.2f} GB"
        else:
            size_str = f"{size_mb:.2f} MB"

        return StatusResponse(
            status="success",
            message=f"Deleted {deleted_count} video(s), freed {size_str}"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear videos: {str(e)}")


@app.get("/storage-info")
async def get_storage_info():
    """
    Get storage information for workspace folders

    Returns:
        Storage information
    """
    try:
        import os

        def get_dir_size(path: Path) -> int:
            """Calculate directory size in bytes"""
            total = 0
            if path.exists():
                for entry in path.rglob("*"):
                    if entry.is_file():
                        total += entry.stat().st_size
            return total

        def format_size(size_bytes: int) -> str:
            """Format size in human readable format"""
            if size_bytes >= 1024**3:
                return f"{size_bytes / (1024**3):.2f} GB"
            elif size_bytes >= 1024**2:
                return f"{size_bytes / (1024**2):.2f} MB"
            elif size_bytes >= 1024:
                return f"{size_bytes / 1024:.2f} KB"
            else:
                return f"{size_bytes} B"

        video_dir = settings.WORKSPACE_DIR / "videos"
        raw_dir = settings.RAW_DIR
        out_dir = settings.OUT_DIR
        temp_dir = settings.TEMP_DIR

        video_size = get_dir_size(video_dir)
        raw_size = get_dir_size(raw_dir)
        out_size = get_dir_size(out_dir)
        temp_size = get_dir_size(temp_dir)
        total_size = video_size + raw_size + out_size + temp_size

        # Count files
        video_count = len(list(video_dir.glob("*"))) if video_dir.exists() else 0
        raw_count = len(list(raw_dir.glob("*"))) if raw_dir.exists() else 0
        out_count = len(list(out_dir.glob("*"))) if out_dir.exists() else 0

        return {
            "total_size": format_size(total_size),
            "total_size_bytes": total_size,
            "folders": {
                "videos": {
                    "size": format_size(video_size),
                    "size_bytes": video_size,
                    "file_count": video_count
                },
                "raw_frames": {
                    "size": format_size(raw_size),
                    "size_bytes": raw_size,
                    "file_count": raw_count
                },
                "thumbnails": {
                    "size": format_size(out_size),
                    "size_bytes": out_size,
                    "file_count": out_count
                },
                "temp": {
                    "size": format_size(temp_size),
                    "size_bytes": temp_size,
                    "file_count": 0
                }
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get storage info: {str(e)}")


@app.post("/upload-video", response_model=VideoUploadResponse)
async def upload_video(
    file: UploadFile = File(...),
    extract_frames: bool = Form(True),
    num_frames: int = Form(200)  # 🎯 เพิ่มเป็น 200 frames - เก็บภาพทั้งหมด 200 ภาพ!
):
    """
    Upload video file and extract frames

    Args:
        file: Video file
        extract_frames: Whether to extract frames immediately
        num_frames: Number of frames to extract

    Returns:
        Upload result with frame count
    """
    try:
        # Clear workspace before uploading new video
        _clear_workspace_files()

        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in settings.VIDEO_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported video format: {file_ext}. Supported: {settings.VIDEO_FORMATS}"
            )

        # Save uploaded video
        video_dir = settings.WORKSPACE_DIR / "videos"
        video_dir.mkdir(exist_ok=True)

        video_path = video_dir / file.filename

        with video_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Uploaded video: {file.filename}")

        # Extract frames if requested
        extracted_count = 0
        if extract_frames:
            extractor = VideoExtractor(output_dir=settings.RAW_DIR)
            frames = extractor.extract_key_frames(video_path, num_frames)
            extracted_count = len(frames)

            logger.info(f"Extracted {extracted_count} frames from {file.filename}")

        return VideoUploadResponse(
            success=True,
            video_path=str(video_path),
            filename=file.filename,
            extracted_frames=extracted_count
        )

    except Exception as e:
        logger.error(f"Video upload failed: {e}", exc_info=True)
        return VideoUploadResponse(
            success=False,
            error=str(e)
        )


@app.post("/generate-from-video", response_model=GenerateResponse)
async def generate_from_video(
    video: UploadFile = File(...),
    title: str = Form(...),
    subtitle: Optional[str] = Form(None),
    num_characters: int = Form(3),
    num_frames: int = Form(150),  # 🚀 เพิ่มเป็น 150 frames - มีตัวเลือกเยอะมาก! หลากหลายขึ้น
    text_style: str = Form("style1"),
    layout_type: Optional[str] = Form(None),
    custom_positions: Optional[str] = Form(None),
    preset_id: Optional[str] = Form("1")  # 🎨 พรีเซ็ต (1=ครึ่งตัว, 2=เต็มตัว) - ค่าเริ่มต้น: 1
):
    """
    Generate thumbnail directly from uploaded video

    Args:
        video: Video file
        title: Thumbnail title
        subtitle: Optional subtitle
        num_characters: Number of characters (0=auto-detect, 1-4=manual)
        num_frames: Number of frames to extract from video
        text_style: Text style preset (style1, style2, style3, auto)
        layout_type: Optional layout override (solo_focus, duo_focus, tri_hero, etc.)
        custom_positions: Optional JSON string of custom positions [{"x": 100, "y": 50, "scale": 2.0}, ...]

    Returns:
        Generation result
    """
    # สร้าง job_id สำหรับติดตาม progress
    job_id = str(uuid4())
    progress_tracker.create_job(job_id)

    try:
        logger.info(f"Received generate-from-video request: title={title}, num_characters={num_characters}, num_frames={num_frames}, preset_id={preset_id}")
        logger.debug(f"Video file: {video.filename}, content_type={video.content_type}")

        # อัปเดต progress: เริ่มต้น
        progress_tracker.update(job_id, 5, "กำลังอัปโหลดวิดีโอ...", "upload")

        # 🎨 Apply preset configuration
        vertical_align = "top"  # Default
        if preset_id and preset_id in PRESETS:
            preset = PRESETS[preset_id]
            crop_point = preset.get("crop_point", "waist")
            crop_multiplier = CROP_MULTIPLIERS.get(crop_point, 3.5)
            vertical_align = preset.get("vertical_align", "top")

            # Temporarily modify crop settings
            original_crop_multiplier = settings.CHARACTER_CROP_HEIGHT_MULTIPLIER
            settings.CHARACTER_CROP_HEIGHT_MULTIPLIER = crop_multiplier

            logger.info(f"🎨 Applied preset '{preset['name']}' (ID: {preset_id})")
            logger.info(f"   └─ Crop: {crop_point} (multiplier: {crop_multiplier})")
            logger.info(f"   └─ Vertical align: {vertical_align}")
        else:
            # Use default if preset not found
            original_crop_multiplier = settings.CHARACTER_CROP_HEIGHT_MULTIPLIER
            if preset_id and preset_id not in PRESETS:
                logger.warning(f"⚠️  Preset ID '{preset_id}' not found, using default")

        # Upload video WITHOUT extracting frames (we'll do Smart Frame Extraction later)
        upload_result = await upload_video(video, extract_frames=False, num_frames=num_frames)

        if not upload_result.success:
            progress_tracker.complete(job_id, False, f"อัปโหลดล้มเหลว: {upload_result.error}")
            raise HTTPException(status_code=500, detail=upload_result.error)

        logger.info(f"Extracted {upload_result.extracted_frames} frames, generating thumbnail...")

        # อัปเดต progress: อัปโหลดเสร็จ เริ่มดึงเฟรม
        progress_tracker.update(job_id, 15, "กำลังดึงเฟรมจากวิดีโอ...", "extract_frames")

        # Parse custom_positions JSON if provided
        parsed_positions = None
        if custom_positions:
            try:
                import json
                parsed_positions = json.loads(custom_positions)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse custom_positions: {e}")

        # Import custom exception
        from utils.exceptions import InsufficientCharactersError

        # 🎯 STRICT 3 CHARACTERS ONLY: บังคับ 3 คนเท่านั้น!
        REQUIRED_CHARACTERS = 3

        # 🎯 TRI LAYOUTS: ใช้ได้เฉพาะ layout 3 คน
        TRI_LAYOUTS = ["tri_hero", "tri_pyramid", "tri_staggered"]

        logger.info(f"🎬 STRICT MODE: ต้องการนักแสดง {REQUIRED_CHARACTERS} คนเท่านั้น!")

        # 🚀 SMART FRAME EXTRACTION: Extract up to 5000 high-quality frames with Smart Frame Selection
        video_path = Path(upload_result.video_path)

        logger.info(f"📹 Starting Smart Frame Extraction (up to {settings.VIDEO_MAX_FRAMES} frames)...")
        extractor = VideoExtractor(
            output_dir=settings.RAW_DIR,
            max_frames=settings.VIDEO_MAX_FRAMES  # 5000 frames
        )

        # Extract frames using Smart Frame Selection (eyes open, frontal, sharp)
        extracted_frames = extractor.extract_from_video(video_path)
        logger.info(f"✅ Extracted {len(extracted_frames)} high-quality frames")

        # อัปเดต progress: ดึงเฟรมเสร็จ
        progress_tracker.update(job_id, 40, f"ดึงเฟรมเสร็จ {len(extracted_frames)} ภาพ", "ingest")

        # Ingest all extracted frames
        logger.info(f"🔄 Ingesting {len(extracted_frames)} frames...")
        pipeline.ingestor.images = []
        pipeline.ingestor.ingest()
        logger.info(f"✅ Total frames ingested: {len(pipeline.ingestor.images)}")

        # อัปเดต progress: เริ่มวิเคราะห์ใบหน้า
        progress_tracker.update(job_id, 50, "กำลังวิเคราะห์ใบหน้า...", "analyze_faces")

        # 🔄 RETRY LOGIC: ถ้าหาคนไม่ครบ 3 คน → retry อีก 1 ครั้ง
        max_retries = 1  # Retry 1 ครั้งถ้าหาคนไม่ครบ
        retry_count = 0

        while retry_count <= max_retries:
            try:
                # 🎯 บังคับ layout เป็น tri ถ้าไม่ได้ระบุ
                if not layout_type or layout_type not in TRI_LAYOUTS:
                    layout_type = "tri_hero"  # ค่าเริ่มต้น
                    logger.info(f"🎨 Using default tri layout: {layout_type}")

                # อัปเดต progress: เริ่มสร้าง thumbnail
                progress_tracker.update(job_id, 80, "กำลังสร้าง thumbnail...", "generate")

                # Generate thumbnail from extracted frames (บังคับ 3 คน)
                result = pipeline.generate(
                    title=title,
                    subtitle=subtitle,
                    num_characters=REQUIRED_CHARACTERS,  # บังคับ 3 คน!
                    source_folder=None,  # Use workspace/raw
                    text_style=text_style,
                    layout_type=layout_type,
                    custom_positions=parsed_positions,
                    vertical_align=vertical_align
                )

                if result['success']:
                    # อัปเดต progress: เสร็จสมบูรณ์!
                    progress_tracker.complete(job_id, True, "สร้าง thumbnail เสร็จแล้ว!")
                    result['job_id'] = job_id  # เพิ่ม job_id ใน response

                    # 🧹 Garbage Collection เพื่อคืน RAM
                    gc.collect()

                    return GenerateResponse(**result)
                else:
                    progress_tracker.complete(job_id, False, result.get('error', 'เกิดข้อผิดพลาด'))
                    raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))

            except InsufficientCharactersError as e:
                retry_count += 1

                if retry_count > max_retries:
                    # 🔥 STRICT 3 CHARACTERS ONLY: บังคับ 3 คนเท่านั้น ไม่มี fallback!
                    found_people = e.found

                    error_msg = (
                        f"❌ ต้องการนักแสดง {REQUIRED_CHARACTERS} คน แต่พบเพียง {found_people} คนเท่านั้น!\n"
                        f"📹 กรุณาอัพโหลดวิดีโอที่มีนักแสดงชัดเจนอย่างน้อย {REQUIRED_CHARACTERS} คน\n"
                        f"💡 Tips:\n"
                        f"   - ใบหน้าทั้ง {REQUIRED_CHARACTERS} คนต้องชัดเจน ไม่เบลอ\n"
                        f"   - หันหน้ามาทางกล้อง\n"
                        f"   - มีแสงสว่างเพียงพอ\n"
                        f"   - ปรากฏในเฟรมเดียวกันหรือหลายเฟรม\n"
                        f"📊 พบนักแสดง: {found_people}/{REQUIRED_CHARACTERS} คน (ไม่เพียงพอ)"
                    )
                    logger.error(error_msg)
                    raise HTTPException(status_code=400, detail=error_msg)


                # 🔄 RETRY: Extract more frames from different parts of video
                logger.warning(
                    f"⚠️  Only found {e.found}/{e.required} different people. "
                    f"Extracting more frames (attempt {retry_count}/{max_retries})..."
                )

                # Extract additional frames with Smart Frame Selection (2000 more frames)
                additional_target = 2000
                logger.info(f"📹 Extracting up to {additional_target} additional frames...")

                # Calculate new max_frames = current + additional
                current_frame_count = len(list(settings.RAW_DIR.glob("*.jpg")))
                new_max = current_frame_count + additional_target

                extractor = VideoExtractor(
                    output_dir=settings.RAW_DIR,
                    max_frames=new_max  # Extract more frames
                )
                new_frames = extractor.extract_from_video(video_path)

                logger.info(f"✓ Extracted {len(new_frames)} total frames (retry {retry_count}). Re-analyzing...")

                # Re-ingest all frames (รวมเก่าและใหม่)
                pipeline.ingestor.images = []  # Clear old images
                pipeline.ingestor.ingest()     # Re-ingest from workspace/raw

                # Loop back to retry generate

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generation from video failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 🔄 Restore original crop multiplier
        if 'original_crop_multiplier' in locals():
            settings.CHARACTER_CROP_HEIGHT_MULTIPLIER = original_crop_multiplier
            logger.debug(f"🔄 Restored crop multiplier to {original_crop_multiplier}")


@app.post("/api/generate", response_model=GenerateResponse, tags=["api"])
async def api_generate(
    video: Optional[UploadFile] = File(None),
    google_drive_url: Optional[str] = Form(None),
    title: str = Form(...),
    subtitle: Optional[str] = Form(None),
    num_characters: int = Form(3),
    num_frames: int = Form(150),
    text_style: str = Form("style1"),
    layout_type: Optional[str] = Form(None),
    custom_positions: Optional[str] = Form(None),
    preset_id: Optional[str] = Form("1")  # 🎨 Preset: 1=ครึ่งตัว(top), 2=เต็มตัว(bottom)
):
    """
    🎬 Universal API: Generate thumbnail from video file OR Google Drive URL

    This endpoint accepts:
    1. Video file upload (multipart/form-data)
    2. Google Drive share URL (direct download without API)

    Usage Examples:

    ## Method 1: Upload video file
    ```bash
    curl -X POST "http://localhost:8000/api/generate" \
      -F "video=@myvideo.mp4" \
      -F "title=หัวข้อบน" \
      -F "subtitle=หัวข้อล่าง"
    ```

    ## Method 2: Google Drive URL
    ```bash
    curl -X POST "http://localhost:8000/api/generate" \
      -F "google_drive_url=https://drive.google.com/file/d/YOUR_FILE_ID/view" \
      -F "title=หัวข้อบน" \
      -F "subtitle=หัวข้อล่าง"
    ```

    Args:
        video: Video file (optional if google_drive_url is provided)
        google_drive_url: Google Drive share URL (optional if video is provided)
        title: Thumbnail title
        subtitle: Optional subtitle
        num_characters: Number of characters (fixed at 3)
        num_frames: Number of frames to extract
        text_style: Text style (style1, style2, style3, auto)
        layout_type: Layout type (tri_hero, tri_pyramid, tri_staggered)
        custom_positions: Custom positions JSON

    Returns:
        GenerateResponse with thumbnail path and metadata

    Note:
        - Must provide either 'video' file OR 'google_drive_url' (not both, not neither)
        - Google Drive file must be public ("Anyone with the link")
    """
    try:
        logger.info(f"📝 API Generate request: title={title}")

        # Validate: must provide either video or google_drive_url (not both, not neither)
        if not video and not google_drive_url:
            raise HTTPException(
                status_code=400,
                detail="ต้องระบุ 'video' (file upload) หรือ 'google_drive_url' อย่างใดอย่างหนึ่ง"
            )

        if video and google_drive_url:
            raise HTTPException(
                status_code=400,
                detail="ระบุได้แค่ 'video' หรือ 'google_drive_url' อย่างใดอย่างหนึ่งเท่านั้น (ไม่ใช่ทั้งคู่)"
            )

        video_path = None

        # Case 1: Google Drive URL provided
        if google_drive_url:
            logger.info(f"📥 Downloading from Google Drive: {google_drive_url}")

            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"gdrive_{timestamp}.mp4"
            video_path = settings.WORKSPACE_DIR / "videos" / video_filename

            # Download from Google Drive
            success, error = await download_from_google_drive_direct(
                url=google_drive_url,
                destination=video_path,
                timeout=600  # 10 minutes
            )

            if not success:
                raise HTTPException(status_code=400, detail=error)

            logger.info(f"✅ Downloaded from Google Drive: {video_path.name}")

        # Case 2: Video file uploaded
        elif video:
            logger.info(f"📤 Uploading video file: {video.filename}")

            # Upload video using existing upload_video function
            upload_result = await upload_video(video, extract_frames=False, num_frames=num_frames)

            if not upload_result.success:
                raise HTTPException(status_code=500, detail=upload_result.error)

            video_path = Path(upload_result.video_path)
            logger.info(f"✅ Video uploaded: {video_path.name}")

        # Now we have video_path from either method
        # Extract frames and generate thumbnail
        logger.info(f"🎬 Starting frame extraction from: {video_path}")

        # Extract frames using VideoExtractor
        extractor = VideoExtractor(
            output_dir=settings.RAW_DIR,
            max_frames=settings.VIDEO_MAX_FRAMES
        )

        extracted_frames = extractor.extract_from_video(video_path)
        logger.info(f"✅ Extracted {len(extracted_frames)} high-quality frames")

        # Ingest all extracted frames
        logger.info(f"🔄 Ingesting {len(extracted_frames)} frames...")
        pipeline.ingestor.images = []
        pipeline.ingestor.ingest()
        logger.info(f"✅ Total frames ingested: {len(pipeline.ingestor.images)}")

        # Parse custom_positions JSON if provided
        parsed_positions = None
        if custom_positions:
            try:
                import json
                parsed_positions = json.loads(custom_positions)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse custom_positions: {e}")

        # Import custom exception
        from utils.exceptions import InsufficientCharactersError

        # 🎯 STRICT 3 CHARACTERS ONLY
        REQUIRED_CHARACTERS = 3
        TRI_LAYOUTS = ["tri_hero", "tri_pyramid", "tri_staggered"]

        logger.info(f"🎬 STRICT MODE: ต้องการนักแสดง {REQUIRED_CHARACTERS} คนเท่านั้น!")

        # 🔄 RETRY LOGIC: ถ้าหาคนไม่ครบ 3 คน → retry อีก 1 ครั้ง
        max_retries = 1
        retry_count = 0

        # 🎨 Apply preset configuration
        vertical_align = "top"  # Default
        if preset_id and preset_id in PRESETS:
            preset = PRESETS[preset_id]
            crop_point = preset.get("crop_point", "waist")
            crop_multiplier = CROP_MULTIPLIERS.get(crop_point, 3.5)
            vertical_align = preset.get("vertical_align", "top")

            # Temporarily modify crop settings
            original_crop_multiplier = settings.CHARACTER_CROP_HEIGHT_MULTIPLIER
            settings.CHARACTER_CROP_HEIGHT_MULTIPLIER = crop_multiplier

            logger.info(f"🎨 Applied preset '{preset['name']}' (ID: {preset_id})")
            logger.info(f"   └─ Crop: {crop_point} (multiplier: {crop_multiplier})")
            logger.info(f"   └─ Vertical align: {vertical_align}")
        else:
            # Use default if preset not found
            original_crop_multiplier = settings.CHARACTER_CROP_HEIGHT_MULTIPLIER
            if preset_id and preset_id not in PRESETS:
                logger.warning(f"⚠️  Preset ID '{preset_id}' not found, using default")

        while retry_count <= max_retries:
            try:
                # 🎯 บังคับ layout เป็น tri ถ้าไม่ได้ระบุ
                if not layout_type or layout_type not in TRI_LAYOUTS:
                    layout_type = "tri_hero"
                    logger.info(f"🎨 Using default tri layout: {layout_type}")

                # Generate thumbnail
                result = pipeline.generate(
                    title=title,
                    subtitle=subtitle,
                    num_characters=REQUIRED_CHARACTERS,
                    source_folder=None,
                    text_style=text_style,
                    layout_type=layout_type,
                    custom_positions=parsed_positions,
                    vertical_align=vertical_align  # 🎨 Use preset vertical alignment
                )

                if result['success']:
                    logger.info(f"✅ API Generate success: {result['filename']}")
                    return GenerateResponse(**result)
                else:
                    raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))

            except InsufficientCharactersError as e:
                retry_count += 1

                if retry_count > max_retries:
                    # 🔥 STRICT 3 CHARACTERS ONLY: บังคับ 3 คนเท่านั้น
                    found_people = e.found

                    error_msg = (
                        f"❌ ต้องการนักแสดง {REQUIRED_CHARACTERS} คน แต่พบเพียง {found_people} คนเท่านั้น!\n"
                        f"📹 กรุณาอัพโหลดวิดีโอที่มีนักแสดงชัดเจนอย่างน้อย {REQUIRED_CHARACTERS} คน\n"
                        f"💡 Tips:\n"
                        f"   - ใบหน้าทั้ง {REQUIRED_CHARACTERS} คนต้องชัดเจน ไม่เบลอ\n"
                        f"   - หันหน้ามาทางกล้อง\n"
                        f"   - มีแสงสว่างเพียงพอ\n"
                        f"   - ปรากฏในเฟรมเดียวกันหรือหลายเฟรม\n"
                        f"📊 พบนักแสดง: {found_people}/{REQUIRED_CHARACTERS} คน (ไม่เพียงพอ)"
                    )
                    logger.error(error_msg)
                    raise HTTPException(status_code=400, detail=error_msg)

                # 🔄 RETRY: Extract more frames
                logger.warning(
                    f"⚠️  Only found {e.found}/{e.required} different people. "
                    f"Extracting more frames (attempt {retry_count}/{max_retries})..."
                )

                # Extract additional frames
                additional_target = 2000
                logger.info(f"📹 Extracting up to {additional_target} additional frames...")

                current_frame_count = len(list(settings.RAW_DIR.glob("*.jpg")))
                new_max = current_frame_count + additional_target

                extractor = VideoExtractor(
                    output_dir=settings.RAW_DIR,
                    max_frames=new_max
                )
                new_frames = extractor.extract_from_video(video_path)

                logger.info(f"✓ Extracted {len(new_frames)} total frames (retry {retry_count}). Re-analyzing...")

                # Re-ingest all frames
                pipeline.ingestor.images = []
                pipeline.ingestor.ingest()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API Generate failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 🔄 Restore original crop multiplier
        if 'original_crop_multiplier' in locals():
            settings.CHARACTER_CROP_HEIGHT_MULTIPLIER = original_crop_multiplier
            logger.debug(f"🔄 Restored crop multiplier to {original_crop_multiplier}")


# ============================================================================
# ASYNC API ENDPOINTS (to prevent Cloudflare timeout)
# ============================================================================

def cleanup_workspace(task_id: str, video_path: Optional[Path] = None):
    """
    🧹 Clean up workspace after task completion

    Deletes:
    - All frames in workspace/raw/
    - Video file (if provided)
    - Prevents next task from using old images
    """
    import shutil

    try:
        # Delete all frames in workspace/raw
        raw_dir = settings.RAW_DIR
        if raw_dir.exists():
            deleted_frames = 0
            for frame_file in raw_dir.glob("*.jpg"):
                frame_file.unlink()
                deleted_frames += 1
            logger.info(f"🧹 [Task {task_id}] Deleted {deleted_frames} frames from workspace/raw")

        # Delete video file if provided
        if video_path and Path(video_path).exists():
            Path(video_path).unlink()
            logger.info(f"🧹 [Task {task_id}] Deleted video: {video_path.name}")

        # Clear pipeline ingestor cache
        pipeline.ingestor.images = []

        logger.info(f"✅ [Task {task_id}] Workspace cleaned up successfully")

    except Exception as e:
        logger.warning(f"⚠️  [Task {task_id}] Cleanup warning: {e}")


async def worker_process_async(
    task_id: str,
    video_path: Optional[Path],
    google_drive_url: Optional[str],
    title: str,
    subtitle: Optional[str],
    num_characters: int,
    num_frames: int,
    text_style: str,
    layout_type: Optional[str],
    custom_positions: Optional[str],
    preset_id: Optional[str] = "1"  # 🎨 Add preset support
):
    """
    Async background task to process video and generate thumbnail

    This function runs in the background and updates task status in tasks_storage
    """
    try:
        logger.info(f"🔄 [Task {task_id}] Starting background processing...")

        # 🧹 Clean up workspace before starting (prevent using old images)
        cleanup_workspace(task_id, None)

        # Update status: downloading (if from Google Drive)
        if google_drive_url:
            task_storage.update(task_id, {
                "status": "downloading",
                "progress": 10,
                "message": f"Downloading video from Google Drive...",
                "created_at": datetime.now().isoformat()
            })

            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"gdrive_{timestamp}.mp4"
            video_path = settings.WORKSPACE_DIR / "videos" / video_filename

            # Download from Google Drive
            success, error = await download_from_google_drive_direct(
                url=google_drive_url,
                destination=video_path,
                timeout=600  # 10 minutes
            )

            if not success:
                task_storage.update(task_id, {
                    "status": "failed",
                    "progress": 0,
                    "error": error,
                    "created_at": task_storage.get(task_id).get("created_at", datetime.now().isoformat()),
                    "completed_at": datetime.now().isoformat()
                })
                return

            logger.info(f"✅ [Task {task_id}] Downloaded: {video_path.name}")

        # Update status: extracting frames
        task_storage.update(task_id, {
            "status": "extracting_frames",
            "progress": 30,
            "message": f"Extracting frames from video (target: {num_frames} frames)...",
            "created_at": task_storage.get(task_id).get("created_at", datetime.now().isoformat())
        })

        # Extract frames
        extractor = VideoExtractor(
            output_dir=settings.RAW_DIR,
            max_frames=settings.VIDEO_MAX_FRAMES
        )
        extracted_frames = extractor.extract_from_video(video_path)
        logger.info(f"✅ [Task {task_id}] Extracted {len(extracted_frames)} frames")

        # Update status: detecting faces
        task_storage.update(task_id, {
            "status": "detecting_faces",
            "progress": 50,
            "message": f"Detecting faces in {len(extracted_frames)} frames...",
            "created_at": task_storage.get(task_id).get("created_at", datetime.now().isoformat())
        })

        # Ingest frames
        pipeline.ingestor.images = []
        pipeline.ingestor.ingest()
        logger.info(f"✅ [Task {task_id}] Ingested {len(pipeline.ingestor.images)} frames")

        # Parse custom_positions
        parsed_positions = None
        if custom_positions:
            try:
                import json
                parsed_positions = json.loads(custom_positions)
            except json.JSONDecodeError as e:
                logger.warning(f"[Task {task_id}] Failed to parse custom_positions: {e}")

        # Update status: clustering characters
        task_storage.update(task_id, {
            "status": "clustering",
            "progress": 70,
            "message": f"Clustering faces and selecting best {num_characters} characters...",
            "created_at": task_storage.get(task_id).get("created_at", datetime.now().isoformat())
        })

        # Import custom exception
        from utils.exceptions import InsufficientCharactersError

        # Force tri layout
        REQUIRED_CHARACTERS = 3
        TRI_LAYOUTS = ["tri_hero", "tri_pyramid", "tri_staggered"]

        if not layout_type or layout_type not in TRI_LAYOUTS:
            layout_type = "tri_hero"

        # 🎨 Apply preset configuration
        vertical_align = "top"  # Default
        if preset_id and preset_id in PRESETS:
            preset = PRESETS[preset_id]
            crop_point = preset.get("crop_point", "waist")
            crop_multiplier = CROP_MULTIPLIERS.get(crop_point, 3.5)
            vertical_align = preset.get("vertical_align", "top")

            # Temporarily modify crop settings
            original_crop_multiplier = settings.CHARACTER_CROP_HEIGHT_MULTIPLIER
            settings.CHARACTER_CROP_HEIGHT_MULTIPLIER = crop_multiplier

            logger.info(f"🎨 [Task {task_id}] Applied preset '{preset['name']}' (ID: {preset_id})")
            logger.info(f"   └─ Crop: {crop_point} (multiplier: {crop_multiplier})")
            logger.info(f"   └─ Vertical align: {vertical_align}")
        else:
            # Use default if preset not found
            original_crop_multiplier = settings.CHARACTER_CROP_HEIGHT_MULTIPLIER
            if preset_id and preset_id not in PRESETS:
                logger.warning(f"⚠️  [Task {task_id}] Preset ID '{preset_id}' not found, using default")

        # Retry logic (เพิ่ม 50 frames ถ้าหาคนไม่ครบ)
        max_retries = 1
        retry_count = 0
        current_max_frames = settings.VIDEO_MAX_FRAMES  # 150

        while retry_count <= max_retries:
            try:
                # Update status: generating thumbnail
                task_storage.update(task_id, {
                    "status": "generating",
                    "progress": 85,
                    "message": f"Generating thumbnail with layout '{layout_type}'...",
                    "created_at": task_storage.get(task_id).get("created_at", datetime.now().isoformat())
                })

                # Generate thumbnail
                result = pipeline.generate(
                    title=title,
                    subtitle=subtitle,
                    num_characters=REQUIRED_CHARACTERS,
                    source_folder=None,
                    text_style=text_style,
                    layout_type=layout_type,
                    custom_positions=parsed_positions,
                    vertical_align=vertical_align  # 🎨 Use preset vertical alignment
                )

                if result['success']:
                    # Success!
                    task_storage.update(task_id, {
                        "status": "completed",
                        "progress": 100,
                        "message": "Thumbnail generated successfully!",
                        "result": {
                            "success": True,
                            "thumbnail_path": result['thumbnail_path'],
                            "filename": result['filename'],
                            "metadata": result.get('metadata', {})
                        },
                        "created_at": task_storage.get(task_id).get("created_at", datetime.now().isoformat()),
                        "completed_at": datetime.now().isoformat()
                    })
                    logger.info(f"✅ [Task {task_id}] Completed: {result['filename']}")

                    # 🧹 Clean up workspace after success
                    cleanup_workspace(task_id, video_path)

                    return
                else:
                    raise Exception(result.get('error', 'Unknown error'))

            except InsufficientCharactersError as e:
                retry_count += 1

                if retry_count > max_retries:
                    # หาคนไม่ครบหลัง retry → ใช้จำนวนที่หาได้แทน (ไม่ fail)
                    found_people = e.found

                    if found_people == 0:
                        # ถ้าไม่เจอเลย → fail
                        error_msg = "❌ ไม่พบนักแสดงในวิดีโอ กรุณาอัพโหลดวิดีโอที่มีคนชัดเจน"
                        task_storage.update(task_id, {
                            "status": "failed",
                            "progress": 0,
                            "error": error_msg,
                            "created_at": task_storage.get(task_id).get("created_at", datetime.now().isoformat()),
                            "completed_at": datetime.now().isoformat()
                        })
                        logger.error(f"❌ [Task {task_id}] {error_msg}")
                        cleanup_workspace(task_id, video_path)
                        return

                    # มีคน 1-2 คน → ใช้จำนวนที่หาได้ ไม่ fail
                    logger.warning(f"⚠️  [Task {task_id}] หาได้เพียง {found_people}/{REQUIRED_CHARACTERS} คน → ใช้ {found_people} คนไปก่อน")

                    # ปรับ layout ตามจำนวนคน
                    if found_people == 1:
                        layout_type = "solo_focus"
                    elif found_people == 2:
                        layout_type = "duo_focus"
                    # ถ้า 3 คนใช้ layout เดิม

                    logger.info(f"✅ [Task {task_id}] ปรับเป็น {found_people} คน, layout: {layout_type}")

                    # Generate ทันทีด้วยจำนวนคนที่หาได้
                    task_storage.update(task_id, {
                        "status": "generating",
                        "progress": 85,
                        "message": f"Generating thumbnail with {found_people} characters...",
                        "created_at": task_storage.get(task_id).get("created_at", datetime.now().isoformat())
                    })

                    result = pipeline.generate(
                        title=title,
                        subtitle=subtitle,
                        num_characters=found_people,  # ใช้จำนวนที่หาได้
                        source_folder=None,
                        text_style=text_style,
                        layout_type=layout_type,
                        custom_positions=parsed_positions,
                        vertical_align=vertical_align
                    )

                    if result['success']:
                        task_storage.update(task_id, {
                            "status": "completed",
                            "progress": 100,
                            "message": "Thumbnail generated successfully!",
                            "result": {
                                "success": True,
                                "thumbnail_path": result['thumbnail_path'],
                                "filename": result['filename'],
                                "metadata": result.get('metadata', {})
                            },
                            "created_at": task_storage.get(task_id).get("created_at", datetime.now().isoformat()),
                            "completed_at": datetime.now().isoformat()
                        })
                        logger.info(f"✅ [Task {task_id}] Completed with {found_people} characters: {result['filename']}")
                        cleanup_workspace(task_id, video_path)
                        return
                    else:
                        raise Exception(result.get('error', 'Unknown error'))

                # Retry: เพิ่ม 50 frames
                current_max_frames += 50
                logger.warning(f"⚠️  [Task {task_id}] Retry {retry_count}/{max_retries}: extracting {current_max_frames} frames (เพิ่ม +50)...")
                task_storage.update(task_id, {
                    "status": "extracting_frames",
                    "progress": 35,
                    "message": f"Retrying: extracting {current_max_frames} frames (เพิ่ม +50)...",
                    "created_at": task_storage.get(task_id).get("created_at", datetime.now().isoformat())
                })

                # Clear old frames first
                cleanup_workspace(task_id, None)

                # สร้าง extractor ใหม่ด้วย max_frames ที่เพิ่มขึ้น
                extractor_retry = VideoExtractor(
                    output_dir=settings.RAW_DIR,
                    max_frames=current_max_frames
                )
                additional_frames = extractor_retry.extract_from_video(video_path)
                pipeline.ingestor.ingest()
                logger.info(f"🔄 [Task {task_id}] Retry: extracted {len(additional_frames)} frames (target: {current_max_frames})")

    except Exception as e:
        # Unexpected error
        error_msg = f"Error processing video: {str(e)}"
        logger.error(f"❌ [Task {task_id}] {error_msg}")
        logger.error(traceback.format_exc())

        task_storage.update(task_id, {
            "status": "failed",
            "progress": 0,
            "error": error_msg,
            "created_at": task_storage.get(task_id).get("created_at", datetime.now().isoformat()),
            "completed_at": datetime.now().isoformat()
        })

        # 🧹 Clean up workspace after error
        cleanup_workspace(task_id, video_path)


def worker_process(
    task_id: str,
    video_path: Optional[str],
    google_drive_url: Optional[str],
    title: str,
    subtitle: Optional[str],
    num_characters: int,
    num_frames: int,
    text_style: str,
    layout_type: Optional[str],
    custom_positions: Optional[str],
    preset_id: Optional[str] = "1"
):
    """
    Sync wrapper for worker_process_async (for multiprocessing.Process)
    """
    import asyncio

    # Convert video_path string back to Path
    video_path_obj = Path(video_path) if video_path else None

    # Run async function in new event loop
    asyncio.run(worker_process_async(
        task_id=task_id,
        video_path=video_path_obj,
        google_drive_url=google_drive_url,
        title=title,
        subtitle=subtitle,
        num_characters=num_characters,
        num_frames=num_frames,
        text_style=text_style,
        layout_type=layout_type,
        custom_positions=custom_positions,
        preset_id=preset_id
    ))


@app.post("/api/generate-async", tags=["api"])
async def api_generate_async(
    background_tasks: BackgroundTasks,
    video: Optional[UploadFile] = File(None),
    google_drive_url: Optional[str] = Form(None),
    title: str = Form(...),
    subtitle: Optional[str] = Form(None),
    num_characters: int = Form(3),
    num_frames: int = Form(150),
    text_style: str = Form("style1"),
    layout_type: Optional[str] = Form(None),
    custom_positions: Optional[str] = Form(None),
    preset_id: Optional[str] = Form("1")  # 🎨 Preset: 1=ครึ่งตัว(top), 2=เต็มตัว(bottom)
):
    """
    🚀 ASYNC API: Generate thumbnail without timeout (for Cloudflare, n8n, etc.)

    This endpoint immediately returns a task_id and processes in the background.
    Use GET /api/task-status/{task_id} to check progress.

    ## Why use this?
    - Cloudflare Free Plan has 100s timeout
    - Video processing takes 6-15 minutes
    - This API returns immediately, processing in background

    ## Usage Flow:

    ### Step 1: Start generation (returns immediately)
    ```bash
    curl -X POST "http://localhost:8000/api/generate-async" \
      -F "google_drive_url=https://drive.google.com/file/d/FILE_ID/view" \
      -F "title=หัวข้อบน" \
      -F "subtitle=หัวข้อล่าง"

    Response:
    {
      "success": true,
      "task_id": "abc-123-def",
      "status_url": "/api/task-status/abc-123-def",
      "message": "Task started. Poll status_url every 5 seconds."
    }
    ```

    ### Step 2: Check status (poll every 5 seconds)
    ```bash
    curl "http://localhost:8000/api/task-status/abc-123-def"

    Response (in progress):
    {
      "task_id": "abc-123-def",
      "status": "detecting_faces",
      "progress": 50,
      "message": "Detecting faces in 325 frames..."
    }

    Response (completed):
    {
      "task_id": "abc-123-def",
      "status": "completed",
      "progress": 100,
      "message": "Thumbnail generated successfully!",
      "result": {
        "success": true,
        "thumbnail_path": "/path/to/thumbnail.jpg",
        "filename": "thumbnail_20231113_123456.jpg"
      }
    }
    ```

    ## Status Values:
    - `pending`: Task queued
    - `downloading`: Downloading video from Google Drive
    - `extracting_frames`: Extracting frames from video
    - `detecting_faces`: Detecting faces with AI
    - `clustering`: Clustering and selecting characters
    - `generating`: Generating thumbnail layout
    - `completed`: ✅ Success! Check `result` field
    - `failed`: ❌ Error. Check `error` field

    Args:
        video: Video file (optional if google_drive_url provided)
        google_drive_url: Google Drive URL (optional if video provided)
        title: Thumbnail title
        subtitle: Optional subtitle
        num_characters: Number of characters (3 only)
        num_frames: Target frames to extract
        text_style: Text style (style1, style2, style3, auto)
        layout_type: Layout (tri_hero, tri_pyramid, tri_staggered)
        custom_positions: Custom positions JSON

    Returns:
        Task ID and status URL for polling
    """
    try:
        # Validate input
        if not video and not google_drive_url:
            raise HTTPException(
                status_code=400,
                detail="ต้องระบุ 'video' หรือ 'google_drive_url' อย่างใดอย่างหนึ่ง"
            )

        if video and google_drive_url:
            raise HTTPException(
                status_code=400,
                detail="ระบุได้แค่ 'video' หรือ 'google_drive_url' อย่างใดอย่างหนึ่งเท่านั้น"
            )

        # Generate task ID
        task_id = str(uuid4())

        # Initialize task status
        task_storage.create(task_id, {
            "status": "pending",
            "progress": 0,
            "message": "Task queued, starting soon..."
        })

        video_path = None

        # Case 1: Video file uploaded
        if video:
            logger.info(f"📤 [Task {task_id}] Uploading video: {video.filename}")

            # Upload video
            upload_result = await upload_video(video, extract_frames=False, num_frames=num_frames)

            if not upload_result.success:
                task_storage.fail(task_id, upload_result.error)
                raise HTTPException(status_code=500, detail=upload_result.error)

            video_path = Path(upload_result.video_path)
            logger.info(f"✅ [Task {task_id}] Video uploaded: {video_path.name}")

        # Start worker process (separate from API server)
        worker = Process(
            target=worker_process,
            args=(
                task_id,
                str(video_path) if video_path else None,
                google_drive_url,
                title,
                subtitle,
                num_characters,
                num_frames,
                text_style,
                layout_type,
                custom_positions,
                preset_id
            ),
            daemon=False  # 🔧 Must be False to allow child processes (Pool)
        )
        worker.start()

        logger.info(f"🚀 [Task {task_id}] Background task started")

        return JSONResponse({
            "success": True,
            "task_id": task_id,
            "status_url": f"/api/task-status/{task_id}",
            "message": "Task started. Poll status_url every 5 seconds to check progress."
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start async task: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/task-status/{task_id}", tags=["api"])
async def get_task_status(task_id: str):
    """
    📊 Get task status for async generation

    Poll this endpoint every 5 seconds to check progress.

    Args:
        task_id: Task ID from /api/generate-async

    Returns:
        Task status with progress, message, and result (if completed)

    Example:
        ```bash
        curl "http://localhost:8000/api/task-status/abc-123-def"
        ```
    """
    task_info = task_storage.get(task_id)

    if task_info is None:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task_id}' not found. It may have expired or never existed."
        )

    return JSONResponse({
        "task_id": task_id,
        **task_info
    })


# ============================================================
# n8n Integration Endpoints
# ============================================================

async def download_video_from_url(url: str, destination: Path) -> tuple[bool, Optional[str]]:
    """
    Download video from URL

    Args:
        url: Video URL
        destination: Destination file path

    Returns:
        (success, error_message)
    """
    try:
        logger.info(f"Downloading video from URL: {url}")

        async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()

            # Write to file
            destination.parent.mkdir(parents=True, exist_ok=True)
            with open(destination, 'wb') as f:
                f.write(response.content)

            logger.info(f"Downloaded video successfully: {destination}")
            return True, None

    except httpx.HTTPError as e:
        error_msg = f"HTTP error downloading video: {e}"
        logger.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Error downloading video: {e}"
        logger.error(error_msg)
        return False, error_msg


@app.post("/api/n8n/generate-from-url", response_model=N8nGenerateResponse, tags=["n8n"])
async def n8n_generate_from_url(request: N8nGenerateRequest):
    """
    n8n Integration: Generate thumbnail from video URL

    This endpoint allows n8n to:
    1. Send a video URL (from Google Drive, Dropbox, etc.)
    2. System downloads the video automatically
    3. Generates thumbnail
    4. Returns thumbnail URL or base64 data

    Example n8n workflow:
    - Google Drive node: Get video file → Get download URL
    - HTTP Request node: POST to this endpoint with video_url
    - Receives thumbnail URL or base64
    - Upload to Google Drive or save to Google Sheets
    """
    try:
        logger.info(f"n8n request: video_url={request.video_url}, title={request.title}")

        # Clear workspace
        await _clear_workspace_files()

        # Extract filename from URL or generate one
        parsed_url = urlparse(request.video_url)
        url_filename = Path(parsed_url.path).name
        if not url_filename or '.' not in url_filename:
            url_filename = f"video_{hash(request.video_url)}.mp4"

        # Download video
        video_path = settings.VIDEOS_DIR / url_filename
        success, error = await download_video_from_url(request.video_url, video_path)

        if not success:
            return N8nGenerateResponse(success=False, error=error)

        # Extract frames using VideoExtractor
        video_extractor = VideoExtractor(
            output_dir=settings.RAW_DIR,
            frame_interval=30,
            max_frames=50,
            min_sharpness=100.0
        )

        frames = video_extractor.extract_key_frames(
            video_path=video_path,
            num_frames=request.num_frames
        )

        if not frames or len(frames) == 0:
            return N8nGenerateResponse(
                success=False,
                error="No frames could be extracted from video"
            )

        logger.info(f"Extracted {len(frames)} frames from video")

        # Generate thumbnail
        result = pipeline.generate(
            title=request.title,
            subtitle=request.subtitle,
            num_characters=request.num_characters,
            source_folder=None,  # Use workspace/raw
            text_style=request.text_style
        )

        if not result['success']:
            return N8nGenerateResponse(success=False, error=result.get('error'))

        # Prepare response
        filename = result['filename']
        thumbnail_path = settings.OUTPUT_DIR / filename

        # Build full URL
        thumbnail_url = f"{settings.API_HOST}:{settings.API_PORT}/thumbnail/{filename}"
        if not thumbnail_url.startswith('http'):
            thumbnail_url = f"http://{thumbnail_url}"

        response_data = {
            "success": True,
            "filename": filename,
            "thumbnail_url": thumbnail_url,
            "metadata": result.get('metadata')
        }

        # If base64 requested, encode the thumbnail
        if request.return_base64:
            with open(thumbnail_path, 'rb') as f:
                image_data = f.read()
                base64_data = base64.b64encode(image_data).decode('utf-8')
                response_data["thumbnail_base64"] = base64_data

        logger.info(f"✅ n8n generation successful: {filename}")
        return N8nGenerateResponse(**response_data)

    except Exception as e:
        logger.error(f"n8n generation failed: {e}", exc_info=True)
        return N8nGenerateResponse(success=False, error=str(e))


async def download_from_google_drive(
    file_id: str,
    destination: Path,
    access_token: Optional[str] = None
) -> tuple[bool, Optional[str]]:
    """
    Download file from Google Drive using OAuth token or Service Account

    Args:
        file_id: Google Drive file ID
        destination: Local path to save the file
        access_token: Optional OAuth 2.0 access token (if not provided, uses service account)

    Returns:
        (success: bool, error_message: Optional[str])
    """
    try:
        logger.info(f"Downloading from Google Drive: file_id={file_id}")

        # Choose authentication method
        if access_token:
            # Method 1: OAuth 2.0 with access token (from n8n)
            logger.info("Using OAuth 2.0 access token authentication")
            creds = Credentials(token=access_token)
        else:
            # Method 3: Service Account (more secure)
            logger.info("Using Service Account authentication")
            service_account_file = os.getenv('GOOGLE_SERVICE_ACCOUNT_FILE', 'service-account.json')

            if not os.path.exists(service_account_file):
                error_msg = f"Service account file not found: {service_account_file}"
                logger.error(error_msg)
                return False, error_msg

            creds = service_account.Credentials.from_service_account_file(
                service_account_file,
                scopes=['https://www.googleapis.com/auth/drive.readonly']
            )

        # Build Google Drive API service
        service = build('drive', 'v3', credentials=creds)

        # Get file metadata first (to check file exists and get name)
        file_metadata = service.files().get(fileId=file_id, fields='name,mimeType,size').execute()
        logger.info(f"File metadata: {file_metadata}")

        # Download file
        request = service.files().get_media(fileId=file_id)

        destination.parent.mkdir(parents=True, exist_ok=True)

        with open(destination, 'wb') as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    progress = int(status.progress() * 100)
                    logger.info(f"Download progress: {progress}%")

        logger.info(f"Downloaded successfully: {destination}")
        return True, None

    except Exception as e:
        error_msg = f"Error downloading from Google Drive: {e}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg


@app.post("/api/n8n/generate-from-google-drive", response_model=N8nGenerateResponse, tags=["n8n"])
async def n8n_generate_from_google_drive(request: GoogleDriveRequest):
    """
    n8n Integration: Generate thumbnail from Google Drive video

    Authentication Methods:
    1. OAuth 2.0: Provide 'access_token' in request (from n8n OAuth2 credential)
    2. Service Account: Leave 'access_token' empty, system uses service account file

    Required Environment Variables (for Service Account method):
    - GOOGLE_SERVICE_ACCOUNT_FILE: Path to service account JSON file (default: service-account.json)

    Flow:
    1. n8n sends file_id + access_token (or uses service account)
    2. System downloads video from Google Drive
    3. Extracts frames and generates thumbnail
    4. Returns thumbnail URL or base64

    Example from n8n:
    ```json
    {
        "file_id": "1a2b3c4d5e6f7g8h9i0j",
        "access_token": "ya29.a0AfH6SMB...",
        "title": "Amazing Video",
        "subtitle": "Best moments",
        "num_characters": 3,
        "num_frames": 20,
        "text_style": "style1",
        "return_base64": false
    }
    ```
    """
    try:
        logger.info(f"🎬 n8n Google Drive generation request: {request.title}")

        # Generate unique filename for video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"gdrive_{timestamp}.mp4"
        video_path = VIDEOS_DIR / video_filename

        # Download video from Google Drive
        logger.info(f"Downloading from Google Drive: file_id={request.file_id}")
        success, error = await download_from_google_drive(
            file_id=request.file_id,
            destination=video_path,
            access_token=request.access_token
        )

        if not success:
            return N8nGenerateResponse(success=False, error=error)

        # Extract frames
        logger.info(f"Extracting {request.num_frames} frames from video...")
        extractor = FrameExtractor()
        frames = extractor.extract_smart_frames(
            str(video_path),
            num_frames=request.num_frames
        )

        if not frames:
            return N8nGenerateResponse(success=False, error="ไม่สามารถดึง frames จากวิดีโอได้")

        logger.info(f"Extracted {len(frames)} frames")

        # Process text style
        if request.text_style == "auto":
            text_style = random.choice(["style1", "style2", "style3"])
        else:
            text_style = request.text_style

        # Generate thumbnail
        logger.info("Generating thumbnail...")
        generator = ThumbnailGenerator()
        result = generator.generate_from_frames(
            frames=frames,
            title=request.title,
            subtitle=request.subtitle,
            text_style=text_style,
            num_characters=request.num_characters
        )

        if not result['success']:
            return N8nGenerateResponse(success=False, error=result.get('error'))

        # Get thumbnail info
        filename = result.get('filename')
        thumbnail_path = THUMBNAILS_DIR / filename

        # Build full URL
        thumbnail_url = f"{settings.API_HOST}:{settings.API_PORT}/thumbnail/{filename}"
        if not thumbnail_url.startswith('http'):
            thumbnail_url = f"http://{thumbnail_url}"

        response_data = {
            "success": True,
            "filename": filename,
            "thumbnail_url": thumbnail_url,
            "metadata": result.get('metadata')
        }

        # If base64 requested, encode the thumbnail
        if request.return_base64:
            with open(thumbnail_path, 'rb') as f:
                image_data = f.read()
                base64_data = base64.b64encode(image_data).decode('utf-8')
                response_data["thumbnail_base64"] = base64_data

        logger.info(f"✅ n8n Google Drive generation successful: {filename}")
        return N8nGenerateResponse(**response_data)

    except Exception as e:
        logger.error(f"n8n Google Drive generation failed: {e}", exc_info=True)
        return N8nGenerateResponse(success=False, error=str(e))


# Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
@app.post("/generate-batch-from-video", response_model=BatchGenerateResponse)
async def generate_batch_from_video(
    video: UploadFile = File(...),
    title: str = Form(...),
    subtitle: Optional[str] = Form(None),
    num_variants: int = Form(3),  # จำนวน thumbnails ที่ต้องการ (3-5)
    num_characters: int = Form(3),
    text_style: str = Form("auto"),
):
    """
    Batch Generate: สร้าง thumbnail หลายแบบพร้อมกัน (3-5 options)

    - แต่ละแบบจะมี layout, ตำแหน่ง, และ faces ที่แตกต่างกัน
    - ใช้ randomization ที่มีอยู่แล้วเพื่อสร้างความหลากหลาย
    """
    try:
        logger.info(f"🎨 Batch generation request: {num_variants} variants")

        # Upload and extract frames (ครั้งเดียว)
        upload_result = await upload_video(video, extract_frames=True, num_frames=100)

        if not upload_result.success:
            return BatchGenerateResponse(
                success=False,
                errors=[upload_result.error]
            )

        thumbnails = []
        errors = []

        # 🎨 กำหนด layouts ที่แตกต่างกันสำหรับแต่ละ variant
        available_layouts = ['tri_hero', 'side_by_side', 'spotlight', 'asymmetric', 'scattered']
        text_styles = ['style1', 'style2', 'style3', 'auto']

        # สร้าง thumbnails หลายแบบ
        for i in range(num_variants):
            try:
                logger.info(f"🎨 Generating variant {i+1}/{num_variants}...")

                # เลือก layout และ text style ที่แตกต่างกันในแต่ละ variant
                variant_layout = available_layouts[i % len(available_layouts)]
                variant_text_style = text_style if text_style != 'auto' else text_styles[i % len(text_styles)]

                # เพิ่ม random seed ที่แตกต่างกัน
                import random
                import time
                random.seed(int(time.time() * 1000) + i)

                logger.info(f"   → Layout: {variant_layout}, Text Style: {variant_text_style}")

                # Generate with different settings
                result = pipeline.generate(
                    title=title,
                    subtitle=subtitle,
                    num_characters=num_characters,
                    source_folder=None,
                    text_style=variant_text_style,
                    layout_type=variant_layout,  # บังคับใช้ layout ที่แตกต่างกัน
                    randomize=True  # สุ่ม characters ให้แตกต่างกัน
                )

                if result['success']:
                    thumbnails.append({
                        'filename': result['thumbnail_path'].split('/')[-1],
                        'url': f"/thumbnail/{result['thumbnail_path'].split('/')[-1]}",
                        'metadata': result.get('metadata', {})
                    })
                else:
                    errors.append(f"Variant {i+1}: {result.get('error', 'Unknown error')}")

            except Exception as e:
                logger.error(f"Error generating variant {i+1}: {e}")
                errors.append(f"Variant {i+1}: {str(e)}")

        success = len(thumbnails) > 0

        # 🤖 AI Validation แค่ top 10 thumbnails (ประหยัดค่าใช้จ่าย)
        if success and settings.ENABLE_AI_VALIDATION and pipeline.validator.model:
            logger.info("")
            logger.info("=" * 80)
            logger.info("🤖 AI VALIDATION - Top 10 Thumbnails Only (Cost Optimization)")
            logger.info("=" * 80)

            # เรียง thumbnails ตาม internal score (ถ้ามี) หรือ ใช้ 10 รูปแรก
            thumbnails_to_validate = thumbnails[:10]

            # Validate แต่ละรูป
            validated_thumbnails = []
            for idx, thumb in enumerate(thumbnails_to_validate):
                logger.info(f"\n🔍 Validating {idx+1}/10: {thumb['filename']}")

                # สร้าง full path
                thumb_path = settings.OUT_DIR / thumb['filename']

                # Validate
                validation = pipeline.validator.validate(thumb_path, title=title)

                # เพิ่ม validation score ลง metadata
                thumb['metadata']['validation'] = {
                    'score': validation.score,
                    'passed': validation.passed,
                    'feedback': validation.feedback,
                    'detailed_scores': validation.detailed_scores,
                }

                validated_thumbnails.append(thumb)

                logger.info(f"   Score: {validation.score}/10 {'✅' if validation.passed else '⚠️'}")

            # เรียงใหม่ตาม Gemini score (สูงสุดก่อน)
            validated_thumbnails.sort(key=lambda x: x['metadata']['validation']['score'], reverse=True)

            # เปลี่ยน thumbnails list เป็น validated + ที่เหลือ
            remaining_thumbnails = thumbnails[10:]
            thumbnails = validated_thumbnails + remaining_thumbnails

            logger.info(f"\n✅ Top 10 validated and re-ranked by AI score")
            logger.info("=" * 80)

        return BatchGenerateResponse(
            success=success,
            thumbnails=thumbnails,
            total_generated=len(thumbnails),
            failed=len(errors),
            errors=errors,
            message=f"Generated {len(thumbnails)}/{num_variants} thumbnails (Top 10 validated by AI)"
        )

    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        return BatchGenerateResponse(
            success=False,
            errors=[str(e)]
        )


@app.get("/analytics")
async def get_analytics():
    """
    Analytics: ดูสถิติการใช้งาน
    """
    analytics_file = Path("logs/analytics.json")

    if analytics_file.exists():
        import json
        with open(analytics_file, 'r') as f:
            data = json.load(f)
        return JSONResponse(content=data)
    else:
        return JSONResponse(content={
            "total_generated": 0,
            "success_count": 0,
            "error_count": 0,
            "message": "No analytics data yet"
        })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )
