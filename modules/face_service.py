"""
Face Service Module - Face detection, clustering, and character selection
"""

import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from loguru import logger
from sklearn.cluster import DBSCAN
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

from config import settings
from utils.image_utils import load_image, calculate_sharpness, calculate_contrast, calculate_brightness
from utils.face_utils import (
    calculate_face_quality,
    get_face_embedding,
    compute_embedding_distance,
    check_eyes_open,
    calculate_face_frontality,
)
from modules.advanced_quality_scorer import AdvancedQualityScorer


class FaceService:
    """
    Detects faces, performs clustering, and selects main/secondary characters
    """

    def __init__(self):
        """
        Initialize Face Service with InsightFace model
        """
        self.model = None
        self.face_db = []  # List of all detected faces with metadata
        self.clusters = {}  # Cluster ID -> list of face indices

        # Initialize Advanced Quality Scorer (if enabled)
        self.advanced_scorer = None
        if settings.ENABLE_ADVANCED_SCORING:
            self.advanced_scorer = AdvancedQualityScorer()
            logger.info("✅ Advanced Quality Scorer enabled")
        else:
            logger.info("⚠️  Advanced Quality Scorer disabled")

        logger.info("Initializing Face Service...")
        self._load_model()

    def _load_model(self):
        """
        🆕 Load InsightFace model with GPU support and ONNX optimization
        """
        try:
            import insightface
            from insightface.app import FaceAnalysis
            import onnxruntime as ort

            # 🆕 Check available providers (GPU/CPU)
            available_providers = ort.get_available_providers()
            logger.info(f"Available ONNX Runtime providers: {available_providers}")

            # 🆕 Select best provider (GPU first, fallback to CPU)
            providers = []
            if 'CUDAExecutionProvider' in available_providers:
                providers.append('CUDAExecutionProvider')
                logger.info("🎮 GPU (CUDA) detected - Using GPU acceleration!")
            elif 'TensorrtExecutionProvider' in available_providers:
                providers.append('TensorrtExecutionProvider')
                logger.info("🎮 TensorRT detected - Using TensorRT acceleration!")

            # Always add CPU as fallback
            providers.append('CPUExecutionProvider')
            logger.info(f"Selected providers: {providers}")

            # 🆕 Configure ONNX Runtime session options
            if settings.ONNX_ENABLE_OPTIMIZATION:
                import os
                # Set thread count via environment variable (InsightFace uses ONNX Runtime internally)
                os.environ['OMP_NUM_THREADS'] = str(settings.ONNX_NUM_THREADS)
                os.environ['MKL_NUM_THREADS'] = str(settings.ONNX_NUM_THREADS)
                logger.info(
                    f"⚡ ONNX Runtime optimization enabled: "
                    f"threads={settings.ONNX_NUM_THREADS}, "
                    f"mode={settings.ONNX_EXECUTION_MODE}"
                )

            # Load model with optimized providers
            self.model = FaceAnalysis(
                name=settings.FACE_DETECTION_MODEL,
                providers=providers
            )

            # 🚀 เพิ่ม detection size เป็น 1280x1280 เพื่อครอบคลุมวิดีโอ 1080x1920 ได้เต็มที่!
            self.model.prepare(ctx_id=0, det_size=(1280, 1280), det_thresh=settings.FACE_CONFIDENCE_THRESHOLD)

            logger.info(
                f"✅ InsightFace model '{settings.FACE_DETECTION_MODEL}' loaded successfully "
                f"(Provider: {providers[0]})"
            )

        except Exception as e:
            logger.error(f"Failed to load InsightFace model: {e}")
            raise

    @staticmethod
    def parse_video_frame_metadata(filename: str) -> Tuple[float, int]:
        """
        Parse timestamp and scene_id from video frame filename

        Expected format: {video_name}_frame_{frame_number:06d}_t{timestamp:.1f}_s{scene_id}.jpg
        Example: video_frame_000120_t15.2_s3.jpg

        Args:
            filename: Frame filename

        Returns:
            Tuple of (timestamp, scene_id). Returns (0.0, 0) if parsing fails.
        """
        try:
            # Extract timestamp using regex: _t{number}
            timestamp_match = re.search(r'_t([\d.]+)', filename)
            timestamp = float(timestamp_match.group(1)) if timestamp_match else 0.0

            # Extract scene_id using regex: _s{number}
            scene_match = re.search(r'_s(\d+)', filename)
            scene_id = int(scene_match.group(1)) if scene_match else 0

            return timestamp, scene_id
        except Exception as e:
            logger.warning(f"Failed to parse metadata from filename '{filename}': {e}")
            return 0.0, 0

    def detect_faces(self, image_path: Path) -> List[Dict]:
        """
        Detect faces in an image

        Args:
            image_path: Path to image

        Returns:
            List of face detection dictionaries
        """
        try:
            # Load image
            image = load_image(image_path)

            # Detect faces
            faces = self.model.get(image)

            # Filter by confidence
            filtered_faces = []
            for face in faces:
                if face.det_score >= settings.FACE_CONFIDENCE_THRESHOLD:
                    # Check minimum face size
                    bbox = face.bbox.astype(int)
                    face_width = bbox[2] - bbox[0]
                    face_height = bbox[3] - bbox[1]

                    if min(face_width, face_height) >= settings.MIN_FACE_SIZE:
                        filtered_faces.append(face)

            logger.debug(f"Detected {len(filtered_faces)} valid faces in {image_path.name}")

            # 🔍 DEBUG MODE: บันทึกใบหน้าที่ตรวจจับได้ทันทีลง temp folder
            if settings.DEBUG_SAVE_DETECTED_FACES and filtered_faces:
                import cv2
                debug_dir = settings.TEMP_DIR / "detected_faces"
                debug_dir.mkdir(parents=True, exist_ok=True)

                for idx, face in enumerate(filtered_faces):
                    try:
                        bbox = face.bbox.astype(int)
                        x1, y1, x2, y2 = bbox

                        # Crop face region
                        face_crop = image[y1:y2, x1:x2]

                        if face_crop.size > 0:
                            # Save to temp folder
                            debug_filename = f"{image_path.stem}_face_{idx}_conf_{face.det_score:.3f}.jpg"
                            debug_path = debug_dir / debug_filename
                            cv2.imwrite(str(debug_path), cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
                            logger.debug(f"💾 Saved: {debug_filename}")
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to save debug face {idx}: {e}")

                logger.info(f"🔍 DEBUG: Saved {len(filtered_faces)} faces from {image_path.name}")

            return filtered_faces

        except Exception as e:
            logger.error(f"Failed to detect faces in {image_path}: {e}")
            return []

    def analyze_image(self, image_metadata: Dict) -> List[Dict]:
        """
        Analyze an image and extract face data with quality metrics

        Args:
            image_metadata: Image metadata from Ingestor

        Returns:
            List of face data dictionaries
        """
        image_path = image_metadata['path']

        # Detect faces
        faces = self.detect_faces(image_path)

        if not faces:
            return []

        # Load image for quality assessment
        image = load_image(image_path)
        image_shape = image.shape

        # บันทึกจำนวนหน้าในภาพนี้ (เพื่อกรอง: เลือกเฉพาะภาพที่มี 1 หน้าเท่านั้น)
        num_faces_in_image = len(faces)

        face_data_list = []

        # 🎨 Parse timestamp and scene_id from filename (for Scene Diversity & Temporal Spacing)
        timestamp, scene_id = self.parse_video_frame_metadata(image_path.name)

        for idx, face in enumerate(faces):
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            # Extract face data
            face_data = {
                'image_path': image_path,
                'image_metadata': image_metadata,
                'bbox': bbox,
                'kps': face.kps if hasattr(face, 'kps') else None,
                'det_score': float(face.det_score),
                'embedding': face.embedding,
                'face_id': f"{image_path.stem}_face_{idx}",
                'num_faces_in_image': num_faces_in_image,  # จำนวนหน้าในภาพนี้
                'timestamp': timestamp,  # 🕒 ใช้สำหรับ Temporal Spacing (เว้นระยะห่างตามเวลา)
                'scene_id': scene_id,  # 🎬 ใช้สำหรับ Scene Diversity (เลือกจากหลายฉาก)
            }

            # Calculate quality metrics
            face_region = (x1, y1, x2 - x1, y2 - y1)

            sharpness = calculate_sharpness(image, face_region)
            contrast = calculate_contrast(image, face_region)
            brightness = calculate_brightness(image, face_region)

            face_data['sharpness'] = sharpness
            face_data['contrast'] = contrast
            face_data['brightness'] = brightness

            # Check if eyes are open (ต้องไม่หลับตา!)
            eyes_open_score = check_eyes_open(face_data, image)
            face_data['eyes_open_score'] = eyes_open_score

            # เช็คตาหลับและกรองออก (ถ้าเปิด FILTER_CLOSED_EYES)
            # 🎯 Threshold 0.40 = ผ่อนปรนมากที่สุด (รองรับนักแสดงใส่แว่น! 👓)
            # แว่นตาจะลด variance/contrast ของดวงตา → score ต่ำลง
            if eyes_open_score < 0.40:
                face_data['eyes_closed'] = True

                # 🚫 HARD FILTER: กรองตาหลับออกเด็ดขาด! (ถ้าเปิดใช้งาน)
                if settings.FILTER_CLOSED_EYES:
                    logger.info(
                        f"🚫 Skipping face {face_data['face_id']}: EYES CLOSED "
                        f"(score={eyes_open_score:.2f} < 0.40) - FILTERED OUT!"
                    )
                    continue  # ข้ามหน้านี้ไป ไม่เก็บเลย!
                else:
                    logger.info(
                        f"⚠️  Face {face_data['face_id']} has CLOSED EYES "
                        f"(score={eyes_open_score:.2f}) - marked but not skipped"
                    )
            else:
                face_data['eyes_closed'] = False

            # 👄 Check if mouth is open (ต้องไม่อ้าปาก!)
            from utils.face_utils import check_mouth_open
            is_mouth_closed = check_mouth_open(face_data, image)
            face_data['mouth_open'] = not is_mouth_closed  # True = ปากเปิด, False = ปากปิด

            if face_data['mouth_open']:
                logger.info(f"⚠️  Face {face_data['face_id']} has OPEN MOUTH - marked but not skipped")

            # 🎯 Check face frontality (ต้องไม่หันหลัง!)
            # ต้องมี landmarks (kps) ถึงจะเช็คได้
            if face_data.get('kps') is not None and len(face_data['kps']) >= 5:
                frontality_score = calculate_face_frontality(face_data['kps'])  # ส่งแค่ landmarks
                face_data['frontality_score'] = frontality_score

                # 🚫 HARD FILTER: กรองเฉพาะหน้าหันหลัง/หันข้างมากๆ เท่านั้น
                # 🎯 Threshold 0.20 = ผ่อนปรนมากๆ - ยอมรับหน้าเอียง/เบี่ยง (ถ้ามีองค์ประกอบหน้าครบ)
                # กรองเฉพาะหน้าหันหลัง profile ชัดๆ เท่านั้น
                if frontality_score < 0.20:
                    logger.info(
                        f"🚫 Skipping face {face_data['face_id']}: PROFILE/BACK "
                        f"(score={frontality_score:.2f} < 0.20) - FILTERED OUT!"
                    )
                    continue  # ข้ามหน้านี้ไป ไม่เก็บเลย!

                face_data['bad_frontality'] = False  # ผ่าน frontality check
            else:
                # ไม่มี landmarks → ไม่สามารถเช็ค frontality ได้ → skip
                logger.warning(f"⚠️  Skipping face {face_data['face_id']}: No landmarks (kps) available")
                continue

            # Analyze emotion (DeepFace AI) - ใช้เป็น scoring weight ไม่กรองทิ้ง
            from utils.face_utils import analyze_emotion
            emotion, emotion_score, emotion_details = analyze_emotion(face_data, image)
            face_data['emotion'] = emotion
            face_data['emotion_score'] = emotion_score
            face_data['emotion_details'] = emotion_details

            # เช็คอารมณ์แย่และกรองออก (ถ้าเปิด FILTER_NEGATIVE_EMOTION)
            negative_emotions = ['angry', 'fear', 'disgust', 'sad']
            if emotion in negative_emotions:
                face_data['negative_emotion'] = True

                # 🚫 HARD FILTER: กรองอารมณ์แย่ออกเด็ดขาด! (ถ้าเปิดใช้งาน)
                if settings.FILTER_NEGATIVE_EMOTION:
                    logger.info(
                        f"🚫 Skipping face {face_data['face_id']}: NEGATIVE EMOTION "
                        f"({emotion}, score={emotion_score:.2f}) - FILTERED OUT!"
                    )
                    continue  # ข้ามหน้านี้ไป ไม่เก็บเลย!
                else:
                    logger.debug(
                        f"Face {face_data['face_id']}: negative emotion detected "
                        f"({emotion}, score={emotion_score:.2f}) - marked but not skipped"
                    )
            else:
                face_data['negative_emotion'] = False

            logger.debug(
                f"Face {face_data['face_id']}: emotion={emotion} "
                f"(score={emotion_score:.2f})"
            )

            # Extract hair and clothing colors for better duplicate detection
            from utils.face_utils import extract_hair_color, extract_clothing_color
            hair_color = extract_hair_color(face_data, image)
            clothing_color = extract_clothing_color(face_data, image)
            face_data['hair_color'] = hair_color
            face_data['clothing_color'] = clothing_color
            logger.debug(
                f"Face {face_data['face_id']}: hair=RGB{hair_color}, "
                f"clothing=RGB{clothing_color}"
            )

            # Calculate overall face quality (excluding face size)
            face_quality = calculate_face_quality(face_data, image_shape)
            face_data['quality_score'] = face_quality

            # Calculate face size score separately (for independent weighting)
            from utils.face_utils import calculate_face_size_score
            face_size = calculate_face_size_score(face_data, image_shape)
            face_data['face_size_score'] = face_size

            # 🎯 Calculate Advanced Quality Scores (if enabled)
            if self.advanced_scorer is not None:
                try:
                    advanced_scores = self.advanced_scorer.calculate_advanced_score(
                        image=image,
                        face_bbox=bbox,
                        landmarks=face_data.get('kps')
                    )
                    # เก็บ advanced scores ไว้ใน face_data
                    face_data['advanced_scores'] = advanced_scores
                    logger.debug(
                        f"Face {face_data['face_id']}: Advanced Score = {advanced_scores['advanced_composite']:.3f}"
                    )
                except Exception as e:
                    logger.warning(f"Advanced scoring failed for {face_data['face_id']}: {e}")
                    face_data['advanced_scores'] = None

            # Calculate composite score
            composite_score = self._calculate_composite_score(face_data)
            face_data['composite_score'] = composite_score

            face_data_list.append(face_data)

        return face_data_list

    def _calculate_composite_score(self, face_data: Dict) -> float:
        """
        Calculate composite quality score based on multiple factors

        Args:
            face_data: Face data dictionary

        Returns:
            Composite score (0-1)
        """
        # Normalize sharpness (typical range 0-500)
        sharpness_norm = min(face_data['sharpness'] / 500.0, 1.0)

        # Normalize contrast (typical range 0-100)
        contrast_norm = min(face_data['contrast'] / 100.0, 1.0)

        # Normalize brightness (ideal range 80-180)
        brightness = face_data['brightness']
        if 80 <= brightness <= 180:
            brightness_norm = 1.0
        elif brightness < 80:
            brightness_norm = brightness / 80.0
        else:
            brightness_norm = max(0, 1 - (brightness - 180) / 75.0)

        # Face quality from face_utils (excluding face size)
        quality_norm = face_data['quality_score']

        # Face size score (0-1, already normalized)
        face_size_norm = face_data.get('face_size_score', 0.5)

        # Eyes open score (0-1, already normalized)
        eyes_open_norm = face_data.get('eyes_open_score', 0.5)

        # Emotion score (0-1, already normalized from DeepFace)
        emotion_norm = face_data.get('emotion_score', 0.5)

        # Weighted composite score (เน้นหน้าใหญ่ชัดเจน)
        composite = (
            settings.WEIGHT_SHARPNESS * sharpness_norm +
            settings.WEIGHT_FACE_SIZE * face_size_norm +  # ขนาดหน้าสำคัญ! ไม่เลือกหน้าเล็ก
            settings.WEIGHT_CONTRAST * contrast_norm +
            settings.WEIGHT_LIGHTING * brightness_norm +
            settings.WEIGHT_FACE_QUALITY * quality_norm +
            settings.WEIGHT_EYES_OPEN * eyes_open_norm +  # น้ำหนักสำหรับตาเปิด
            settings.WEIGHT_EMOTION * emotion_norm  # น้ำหนักสำหรับอารมณ์ (happy/neutral)
        )

        # 🎯 PENALTY: ลดคะแนนสำหรับภาพที่มีปัญหา (แต่ไม่กรองออก - เก็บไว้ทั้งหมด 200 ภาพ)
        # หมายเหตุ: หน้าหันหลังถูกกรองออกแล้วตั้งแต่ต้น (hard filter) ไม่มีใน penalty
        penalty = 1.0

        # ลดคะแนน 30% ถ้าตาหลับ
        if face_data.get('eyes_closed', False):
            penalty *= 0.70

        # ลดคะแนน 20% ถ้าปากอ้าง
        if face_data.get('mouth_open', False):
            penalty *= 0.80

        # ลดคะแนน 15% ถ้าอารมณ์ negative
        if face_data.get('negative_emotion', False):
            penalty *= 0.85

        composite *= penalty

        # 🎯 BLEND WITH ADVANCED SCORE (if available)
        # รวมคะแนนจาก Advanced Quality Scorer (40% advanced + 60% original)
        if settings.ENABLE_ADVANCED_SCORING and face_data.get('advanced_scores') is not None:
            advanced_composite = face_data['advanced_scores']['advanced_composite']

            # Weighted blend: ADVANCED_SCORE_WEIGHT = 0.4 (40% advanced, 60% original)
            blended_score = (
                settings.ADVANCED_SCORE_WEIGHT * advanced_composite +
                (1 - settings.ADVANCED_SCORE_WEIGHT) * composite
            )

            logger.debug(
                f"Blended score: original={composite:.3f}, advanced={advanced_composite:.3f}, "
                f"final={blended_score:.3f} (weight={settings.ADVANCED_SCORE_WEIGHT})"
            )

            return float(blended_score)

        return float(composite)

    def analyze_all_images(self, image_metadata_list: List[Dict]) -> None:
        """
        Analyze all images and build face database

        Args:
            image_metadata_list: List of image metadata from Ingestor
        """
        logger.info(f"Analyzing {len(image_metadata_list)} images for faces...")

        self.face_db = []

        for img_meta in image_metadata_list:
            faces = self.analyze_image(img_meta)
            self.face_db.extend(faces)

        logger.info(f"Found {len(self.face_db)} faces across all images")

    def analyze_all_images_adaptive(
        self,
        image_metadata_list: List[Dict],
        required_characters: int
    ) -> int:
        """
        🎯 Multi-Pass Adaptive Face Analysis - รับประกันเจอหน้าครบ 100%!

        ลองวิเคราะห์หลายรอบด้วย threshold ที่แตกต่างกัน:
        - Pass 1: Very Strict (คุณภาพสูงสุด)
        - Pass 2: Strict (ปกติ)
        - Pass 3: Moderate (ผ่อนปรน - รับประกันครบ)

        Args:
            image_metadata_list: List of image metadata
            required_characters: จำนวนตัวละครที่ต้องการ

        Returns:
            จำนวน clusters ที่เจอได้
        """
        if not settings.ENABLE_MULTIPASS:
            # ถ้าปิด multipass ใช้แบบเดิม
            self.analyze_all_images(image_metadata_list)
            self.cluster_faces()
            return len(self.clusters)

        # กำหนด threshold สำหรับแต่ละ pass
        pass_configs = [
            {
                "name": "Pass 1 (Very Strict)",
                "frontality": settings.PASS1_FRONTALITY,
                "eyes": settings.PASS1_EYES_THRESHOLD,
                "sharpness": settings.PASS1_MIN_SHARPNESS
            },
            {
                "name": "Pass 2 (Strict)",
                "frontality": settings.PASS2_FRONTALITY,
                "eyes": settings.PASS2_EYES_THRESHOLD,
                "sharpness": settings.PASS2_MIN_SHARPNESS
            },
            {
                "name": "Pass 3 (Moderate)",
                "frontality": settings.PASS3_FRONTALITY,
                "eyes": settings.PASS3_EYES_THRESHOLD,
                "sharpness": settings.PASS3_MIN_SHARPNESS
            },
            {
                "name": "Pass 4 (Very Lenient)",
                "frontality": settings.PASS4_FRONTALITY,
                "eyes": settings.PASS4_EYES_THRESHOLD,
                "sharpness": settings.PASS4_MIN_SHARPNESS
            }
        ]

        # ลองแต่ละ pass จนกว่าจะได้หน้าครบ
        for pass_idx, pass_config in enumerate(pass_configs, 1):
            logger.info(f"")
            logger.info(f"{'='*80}")
            logger.info(f"🔍 {pass_config['name']}: Analyzing with thresholds:")
            logger.info(f"   - Frontality ≥ {pass_config['frontality']}")
            logger.info(f"   - Eyes Open ≥ {pass_config['eyes']}")
            logger.info(f"   - Sharpness ≥ {pass_config['sharpness']}")
            logger.info(f"{'='*80}")

            # ชั่วคราวแทนที่ settings เพื่อใช้ threshold ของ pass นี้
            original_frontality = 0.80  # hardcoded in analyze_image
            original_eyes = 0.72
            original_sharpness = settings.VIDEO_MIN_SHARPNESS

            # วิเคราะห์ภาพทั้งหมด (with Early Stopping 🚀)
            self.face_db = []
            for img_idx, img_meta in enumerate(image_metadata_list, 1):
                # ใช้ threshold แบบ dynamic ไม่ได้ เพราะ hardcode ใน analyze_image
                # ดังนั้นต้องเช็คภายหลัง
                faces = self.analyze_image(img_meta)

                # กรองหน้าตาม threshold ของ pass นี้
                filtered_faces = []
                for face in faces:
                    # เช็ค frontality
                    if face.get('frontality_score', 0) < pass_config['frontality']:
                        continue
                    # เช็ค eyes
                    if face.get('eyes_open_score', 0) < pass_config['eyes']:
                        continue
                    # เช็ค sharpness
                    if face.get('sharpness', 0) < pass_config['sharpness']:
                        continue

                    filtered_faces.append(face)

                self.face_db.extend(filtered_faces)

                # 🚀 EARLY STOPPING: เช็คทุกๆ N ภาพว่าหาคนครบหรือยัง
                if settings.ENABLE_EARLY_STOP and img_idx % settings.EARLY_STOP_CHECK_INTERVAL == 0:
                    if len(self.face_db) >= required_characters * 10:  # ต้องมีหน้าอย่างน้อย 10 หน้า/คน
                        # Cluster ชั่วคราวเพื่อเช็ค
                        self.cluster_faces()
                        num_clusters = len(self.clusters)

                        if num_clusters >= required_characters:
                            logger.info(f"🚀 EARLY STOP at image {img_idx}/{len(image_metadata_list)}: Found {num_clusters} people!")
                            break  # หยุดวิเคราะห์ภาพที่เหลือ

            logger.info(f"✅ {pass_config['name']}: Found {len(self.face_db)} qualifying faces")

            # Cluster หน้าที่เจอได้
            if len(self.face_db) > 0:
                self.cluster_faces()
                num_clusters = len(self.clusters)

                logger.info(f"✅ {pass_config['name']}: Clustered into {num_clusters} different people")

                # ถ้าเจอครบแล้ว หยุดเลย!
                if num_clusters >= required_characters:
                    logger.info(f"")
                    logger.info(f"{'='*80}")
                    logger.info(f"🎉 SUCCESS! Found {num_clusters}/{required_characters} people")
                    logger.info(f"   Using {pass_config['name']} threshold")
                    logger.info(f"{'='*80}")
                    return num_clusters
                else:
                    logger.warning(f"⚠️  {pass_config['name']}: Only {num_clusters}/{required_characters} people found")
                    logger.warning(f"   Trying next pass with more lenient thresholds...")
            else:
                logger.warning(f"⚠️  {pass_config['name']}: No faces found")
                logger.warning(f"   Trying next pass with more lenient thresholds...")

        # ถ้าลองทุก pass แล้วยังไม่ครบ
        num_clusters = len(self.clusters)
        logger.error(f"")
        logger.error(f"{'='*80}")
        logger.error(f"❌ All passes failed!")
        logger.error(f"   Found only {num_clusters}/{required_characters} different people")
        logger.error(f"   Tried {len(pass_configs)} passes with progressively lenient thresholds")
        logger.error(f"{'='*80}")

        return num_clusters

    def cluster_faces(self) -> Dict[int, List[int]]:
        """
        Cluster faces by embedding similarity

        Returns:
            Dictionary mapping cluster_id -> list of face indices
        """
        if not self.face_db:
            logger.warning("No faces to cluster")
            return {}

        # Extract embeddings
        embeddings = np.array([face['embedding'] for face in self.face_db])

        logger.info(f"Clustering {len(embeddings)} face embeddings...")

        # Perform clustering
        if settings.CLUSTERING_ALGORITHM == "HDBSCAN" and HDBSCAN_AVAILABLE:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=settings.HDBSCAN_MIN_CLUSTER_SIZE,
                metric='euclidean'
            )
            labels = clusterer.fit_predict(embeddings)
        else:
            # DBSCAN
            clusterer = DBSCAN(
                eps=settings.DBSCAN_EPS,
                min_samples=settings.DBSCAN_MIN_SAMPLES,
                metric='euclidean'
            )
            labels = clusterer.fit_predict(embeddings)

        # Organize faces by cluster
        clusters = {}
        noise_count = 0
        for idx, label in enumerate(labels):
            if label == -1:  # Noise points
                noise_count += 1
                continue

            if label not in clusters:
                clusters[label] = []

            clusters[label].append(idx)

        logger.info(f"Found {len(clusters)} face clusters (excluding {noise_count} noise points)")

        # 🔍 DEBUG: Show cluster contents BEFORE merging
        logger.info(f"=" * 80)
        logger.info(f"🔍 CLUSTER CONTENTS BEFORE MERGING:")
        for cluster_id, face_indices in sorted(clusters.items()):
            logger.info(f"   Cluster {cluster_id}: {len(face_indices)} faces")
            for idx in face_indices[:5]:  # Show first 5 faces
                face = self.face_db[idx]
                logger.info(
                    f"      - face_id={face.get('face_id')}, "
                    f"image={face.get('image_path').name if face.get('image_path') else 'unknown'}, "
                    f"score={face.get('composite_score', 0):.3f}"
                )
            if len(face_indices) > 5:
                logger.info(f"      ... and {len(face_indices) - 5} more faces")
        logger.info(f"=" * 80)

        # LAYER 2: Post-clustering merge (รวม clusters ที่ใกล้กันมาก → คนเดียวกัน)
        if len(clusters) > 1:
            clusters = self._merge_similar_clusters(clusters, embeddings)
            logger.info(f"After merging similar clusters: {len(clusters)} clusters remaining")

        self.clusters = clusters
        return clusters

    def rank_clusters(self) -> List[Tuple[int, Dict]]:
        """
        Rank clusters by frequency and quality

        Returns:
            List of tuples (cluster_id, cluster_stats) sorted by importance
        """
        cluster_rankings = []

        for cluster_id, face_indices in self.clusters.items():
            # Calculate cluster statistics
            face_count = len(face_indices)

            # Average quality scores
            quality_scores = [self.face_db[idx]['quality_score'] for idx in face_indices]
            composite_scores = [self.face_db[idx]['composite_score'] for idx in face_indices]

            avg_quality = np.mean(quality_scores)
            avg_composite = np.mean(composite_scores)
            max_composite = np.max(composite_scores)

            # Cluster importance = frequency * quality
            importance = face_count * avg_composite

            # 🎯 CRITICAL FIX: Penalize group shots heavily!
            # เช็คว่า cluster นี้มีหน้าจาก group shots หรือไม่
            has_group_shot = any(
                self.face_db[idx].get('num_faces_in_image', 1) > 1
                for idx in face_indices
            )

            # ลดคะแนน group shot ตาม config
            if has_group_shot:
                importance *= settings.GROUP_SHOT_PENALTY
                logger.debug(f"Cluster {cluster_id}: GROUP SHOT detected → importance reduced to {importance:.2f}")
            else:
                logger.debug(f"Cluster {cluster_id}: SOLO SHOT → importance kept at {importance:.2f}")

            cluster_stats = {
                'cluster_id': cluster_id,
                'face_count': face_count,
                'avg_quality': avg_quality,
                'avg_composite': avg_composite,
                'max_composite': max_composite,
                'importance': importance,
                'face_indices': face_indices,
                'has_group_shot': has_group_shot,  # บันทึกไว้ด้วย
            }

            cluster_rankings.append((cluster_id, cluster_stats))

        # Sort by importance (descending)
        cluster_rankings.sort(key=lambda x: x[1]['importance'], reverse=True)

        logger.info(f"Ranked {len(cluster_rankings)} clusters")

        return cluster_rankings

    def select_characters(self, num_characters: int = 3, allow_duplicates: bool = False, randomize: bool = False) -> Dict[str, Dict]:
        """
        Select main and secondary characters

        Args:
            num_characters: Number of characters to select (2-4)
            allow_duplicates: If True, allow selecting same person multiple times (last resort fallback)

        Returns:
            Dictionary with character roles and their best face data
        """
        if not self.clusters:
            logger.warning("No clusters available, using fallback: selecting best faces by quality")
            # Fallback: Select best N faces directly by quality score
            if not self.face_db:
                logger.error("No faces in database")
                return {}

            sorted_faces = sorted(self.face_db, key=lambda x: x['composite_score'], reverse=True)
            selected_characters = {}
            num_to_select = min(num_characters, len(sorted_faces))

            for i in range(num_to_select):
                role = "main" if i == 0 else f"secondary_{i}"
                face = sorted_faces[i]
                selected_characters[role] = {
                    'cluster_id': -1,  # No cluster
                    'cluster_stats': {
                        'face_count': 1,
                        'importance': face['composite_score'],
                        'avg_quality': face['quality_score'],
                        'avg_composite': face['composite_score'],
                    },
                    'face_data': face
                }
                logger.info(
                    f"Selected {role} (fallback): {face['face_id']} "
                    f"(score={face['composite_score']:.3f})"
                )

            return selected_characters

        # Rank clusters
        ranked_clusters = self.rank_clusters()

        # 🎲 RANDOMIZATION: Shuffle clusters if requested (for batch generation variety)
        if randomize:
            import random
            random.shuffle(ranked_clusters)
            logger.info(f"🎲 Shuffled clusters for variety")

        # 🔍 DEBUG: Show all clusters with detailed info
        logger.info(f"=" * 80)
        logger.info(f"📊 CLUSTERING RESULTS: {len(ranked_clusters)} total clusters")
        logger.info(f"=" * 80)
        for idx, (cid, stats) in enumerate(ranked_clusters):
            logger.info(
                f"  Cluster #{idx+1} (ID={cid}): "
                f"faces={stats['face_count']}, "
                f"importance={stats['importance']:.2f}, "
                f"{'SOLO' if not stats.get('has_group_shot', False) else 'GROUP'}"
            )
        logger.info(f"=" * 80)

        # LAYER 4: Filter out group shots if enough solo shots available
        if settings.BLOCK_GROUP_SHOTS_IF_SOLO_AVAILABLE:
            solo_clusters = [
                (cid, stats) for cid, stats in ranked_clusters
                if not stats.get('has_group_shot', False)
            ]
            group_clusters = [
                (cid, stats) for cid, stats in ranked_clusters
                if stats.get('has_group_shot', False)
            ]

            logger.info(f"Found {len(solo_clusters)} solo shot clusters, {len(group_clusters)} group shot clusters")

            # ถ้ามี solo shots พอ → บล็อก group shots ทั้งหมด
            if len(solo_clusters) >= num_characters:
                ranked_clusters = solo_clusters
                logger.info(f"✓ BLOCKING all group shots - using only {len(solo_clusters)} solo shot clusters")
            else:
                # ไม่มี solo shots พอ → ใช้ทั้ง solo + group (solo ก่อน)
                ranked_clusters = solo_clusters + group_clusters
                logger.warning(
                    f"⚠️  Not enough solo shots ({len(solo_clusters)}/{num_characters}) - "
                    f"will use some group shots"
                )

        # LAYER 3: Dynamic selection with IRON RULE - ห้ามเลือกคนซ้ำ
        # ตรวจสอบ diversity ระหว่างทาง แล้ว skip duplicates ไปหา cluster ถัดไป
        from utils.face_utils import compute_embedding_distance, compute_color_similarity, compute_landmarks_similarity

        selected_characters = {}
        used_faces = []  # เก็บ face data ที่เลือกไปแล้ว (embedding, landmarks, hair, clothing, image_path)
        used_image_paths = set()  # เก็บ image_path ที่ใช้ไปแล้ว (ห้ามซ้ำ!)
        cluster_timestamps = {}  # 🕒 เก็บ timestamps ที่ใช้ไปแล้วของแต่ละ cluster (เพื่อ Temporal Spacing)

        selected_count = 0
        cluster_idx = 0

        while selected_count < num_characters and cluster_idx < len(ranked_clusters):
            cluster_id, cluster_stats = ranked_clusters[cluster_idx]

            logger.info(f"\n🔍 Evaluating cluster {cluster_id} (#{cluster_idx+1}/{len(ranked_clusters)}) for selection #{selected_count+1}")

            # 🚀 MULTIPLE FACES SELECTION: เลือก 15 รูปดีที่สุดต่อคน
            all_faces = self._select_multiple_faces_from_cluster(
                cluster_stats['face_indices'],
                num_faces=settings.FACES_PER_CHARACTER,
                solo_shots_only=True  # บังคับ solo shots เท่านั้น!
            )

            # ถ้าไม่เจอหน้าเลย (ทุกรูปใน cluster ถูกกรองออก หรือเป็น group shots)
            if not all_faces:
                logger.warning(f"⚠️  SKIP cluster {cluster_id}: no valid solo shots available")
                cluster_idx += 1
                continue

            # ใช้รูปดีที่สุด (รูปแรก) สำหรับ thumbnail
            best_face = all_faces[0]

            logger.info(f"   📸 Selected {len(all_faces)} faces from cluster {cluster_id} (using best for thumbnail)")


            face_embedding = best_face['embedding']
            face_landmarks = best_face.get('kps')  # เพิ่ม landmarks
            face_image_path = best_face.get('image_path')  # เพิ่ม image_path
            clothing_color = best_face.get('clothing_color', (0, 0, 0))

            # 🔍 DEBUG: Show selected face details
            logger.info(
                f"   Candidate face: "
                f"image={face_image_path.name if face_image_path else 'unknown'}, "
                f"face_id={best_face.get('face_id')}, "
                f"score={best_face.get('composite_score', 0):.3f}, "
                f"clothing=RGB{clothing_color}"
            )

            # 🚨 IRON RULE: ตรวจสอบว่าเป็นคนซ้ำหรือไม่ (Weighted Score - Face + Landmarks + Clothing ONLY)
            is_duplicate = False

            # Skip duplicate checking if allow_duplicates=True (last resort fallback)
            if allow_duplicates:
                logger.info(f"   ⚠️  ALLOW_DUPLICATES=True - skipping duplicate check (fallback mode)")
            else:
                # 🎯 OPTIMIZATION: Check if cluster_id already selected
                # ถ้า cluster นี้ถูกเลือกไปแล้ว (เลือกหน้าหลายหน้าจาก cluster เดียวกัน)
                # → ให้เช็ค duplicate
                # ถ้า cluster นี้ยังไม่เคยเลือก → ไว้ใจ clustering ว่าแยกคนถูกแล้ว
                selected_cluster_ids = [used_face['cluster_id'] for used_face in used_faces]

                if cluster_id in selected_cluster_ids:
                    logger.info(f"   ⚠️  Cluster {cluster_id} already selected before - checking for intra-cluster duplicates...")
                    # เช็ค duplicate เฉพาะกับหน้าจาก cluster เดียวกัน
                    for idx, used_face in enumerate(used_faces):
                        if used_face['cluster_id'] != cluster_id:
                            continue  # ข้าม - คนละ cluster = คนละคน (trust clustering)

                        # 1. Check face similarity (distance: 0 = same, 1 = different)
                        face_dist = compute_embedding_distance(face_embedding, used_face['embedding'])

                        # 2. Check face geometry similarity (landmarks: 0 = different, 1 = same)
                        landmarks_sim = compute_landmarks_similarity(face_landmarks, used_face['landmarks'])

                        # 3. Check clothing color similarity (similarity: 0 = different, 1 = same)
                        clothing_sim = compute_color_similarity(clothing_color, used_face['clothing_color'])

                        # Weighted Duplicate Score (0-1, lower = more similar)
                        # Weights: Face 70%, Landmarks 15%, Clothing 15% (ตัด Hair ออก - ไม่น่าเชื่อถือ)
                        duplicate_score = (
                            face_dist * 0.70 +                  # Face embedding distance (lower = same person)
                            (1 - landmarks_sim) * 0.15 +        # Landmarks dissimilarity (รูปร่างหน้า)
                            (1 - clothing_sim) * 0.15           # Clothing dissimilarity (เพิ่มจาก 7.5% → 15%)
                        )

                        # 🔍 DEBUG: Show comparison details
                        logger.info(
                            f"      vs selected#{idx+1} (same cluster): "
                            f"duplicate_score={duplicate_score:.3f} "
                            f"(face={face_dist:.3f}, landmarks={landmarks_sim:.2f}, "
                            f"clothing={clothing_sim:.2f}) "
                            f"→ {'DUPLICATE!' if duplicate_score < 0.20 else 'OK'}"
                        )

                        # คนเดียวกัน ถ้า duplicate_score < 0.20
                        if duplicate_score < 0.20:
                            logger.info(
                                f"⚠️  SKIP cluster {cluster_id}: duplicate face within same cluster "
                                f"(duplicate_score={duplicate_score:.3f}, face_dist={face_dist:.3f}, "
                                f"landmarks_sim={landmarks_sim:.2f}, clothing_sim={clothing_sim:.2f}) "
                                f"→ looking for next cluster..."
                            )
                            is_duplicate = True
                            break
                else:
                    # ✅ Cluster นี้ยังไม่เคยเลือก → ไว้ใจ clustering
                    logger.info(f"   ✅ Cluster {cluster_id} is new - trusting clustering (skip duplicate check)")


            if not is_duplicate:
                # ไม่ซ้ำ → เลือก cluster นี้
                # Determine role
                if selected_count == 0:
                    role = "main"
                else:
                    role = f"secondary_{selected_count}"

                selected_characters[role] = {
                    'cluster_id': cluster_id,
                    'cluster_stats': cluster_stats,
                    'face_data': best_face,  # รูปดีที่สุดสำหรับ thumbnail
                    'all_faces': all_faces,  # 🚀 เก็บทุกรูป (15 รูป) ไว้ด้วย!
                }

                # เก็บข้อมูล face + landmarks + clothing + cluster_id ที่เลือกแล้ว (ไม่เก็บ hair_color)
                used_faces.append({
                    'embedding': face_embedding,
                    'landmarks': face_landmarks,
                    'clothing_color': clothing_color,
                    'cluster_id': cluster_id  # เพิ่ม cluster_id เพื่อให้ duplicate check ใช้งานได้
                })

                # 🚀 เก็บ image_path ทั้งหมด 15 รูป ที่ใช้แล้ว (ห้ามใช้รูปเดิม!)
                for face in all_faces:
                    used_image_paths.add(face.get('image_path'))

                # 🕒 เก็บ timestamp ทั้งหมดที่ใช้แล้วสำหรับ cluster นี้ (Temporal Spacing)
                if cluster_id not in cluster_timestamps:
                    cluster_timestamps[cluster_id] = []
                for face in all_faces:
                    cluster_timestamps[cluster_id].append(face.get('timestamp', 0.0))
                logger.debug(f"   📝 Recorded {len(all_faces)} timestamps for cluster {cluster_id}")

                selected_count += 1

                # แสดงข้อมูลรายละเอียด
                shot_type = "GROUP SHOT" if cluster_stats.get('has_group_shot', False) else "SOLO SHOT"
                num_faces = best_face.get('num_faces_in_image', 1)
                logger.info(
                    f"✅ Selected {role}: cluster {cluster_id} "
                    f"(count={cluster_stats['face_count']}, "
                    f"importance={cluster_stats['importance']:.2f}, "
                    f"{shot_type}, faces_in_frame={num_faces}, "
                    f"clothing=RGB{clothing_color})"
                )

            # ไปหา cluster ถัดไป
            cluster_idx += 1

        # ✅ ADAPTIVE: Return ตามจำนวนที่หาได้ (ไม่ raise error)
        if selected_count < num_characters:
            logger.warning(
                f"⚠️  Found only {selected_count}/{num_characters} different people. "
                f"Returning {selected_count} character(s) (adaptive strategy)"
            )
        else:
            logger.info(f"✅ Successfully selected {selected_count} different characters (required: {num_characters})")

        return selected_characters

    def _select_best_face_from_cluster(
        self,
        face_indices: List[int],
        exclude_image_paths: Optional[set] = None,
        exclude_closed_eyes: bool = True,
        exclude_timestamps: Optional[List[float]] = None
    ) -> Optional[Dict]:
        """
        Select the best face from a cluster based on composite score

        **Preference:**
        1. Solo shots (1 face in image) are strongly preferred
        2. If no solo shots available, select from group shots
        3. NEVER select from images that were already used (exclude_image_paths)
        4. Skip faces with closed eyes if exclude_closed_eyes=True
        5. 🕒 Skip faces too close in time to already-selected faces (Temporal Spacing)

        Args:
            face_indices: List of face indices in cluster
            exclude_image_paths: Set of image paths to exclude (already used)
            exclude_closed_eyes: If True, skip faces with closed eyes (default: True)
            exclude_timestamps: List of timestamps already selected for this cluster

        Returns:
            Best face data dictionary, or None if no valid face available
        """
        if exclude_image_paths is None:
            exclude_image_paths = set()
        if exclude_timestamps is None:
            exclude_timestamps = []

        # 🔍 DEBUG: Show all faces in this cluster
        logger.debug(f"   Cluster has {len(face_indices)} faces total (exclude_closed_eyes={exclude_closed_eyes}):")
        for idx in face_indices:
            face = self.face_db[idx]
            face_image_path = face.get('image_path')
            logger.debug(
                f"      - face_id={face.get('face_id')}, "
                f"image={face_image_path.name if face_image_path else 'unknown'}, "
                f"excluded={face_image_path in exclude_image_paths}, "
                f"eyes_closed={face.get('eyes_closed', False)}, "
                f"mouth_open={face.get('mouth_open', False)}, "
                f"frontality={face.get('frontality_score', 0):.2f}, "
                f"score={face.get('composite_score', 0):.3f}"
            )

        # แยกหน้าเป็น solo shots (1 หน้า) และ group shots (หลายหน้า)
        # และกรองรูปที่ใช้ไปแล้วออก + กรองหน้าหลับตา (ถ้าต้องการ)
        solo_faces = []
        group_faces = []
        excluded_count = 0
        eyes_closed_count = 0
        mouth_open_count = 0
        temporal_filtered_count = 0

        for idx in face_indices:
            face = self.face_db[idx]
            face_image_path = face.get('image_path')

            # ข้ามรูปที่ใช้ไปแล้ว
            if face_image_path in exclude_image_paths:
                excluded_count += 1
                continue

            # ข้ามหน้าหลับตา (ถ้าต้องการ)
            if exclude_closed_eyes and face.get('eyes_closed', False):
                eyes_closed_count += 1
                continue

            # 👄 ข้ามหน้าอ้าปาก (ถ้าต้องการ)
            if exclude_closed_eyes and face.get('mouth_open', False):
                mouth_open_count += 1
                continue

            # 🕒 TEMPORAL SPACING: ข้ามหน้าที่อยู่ใกล้กันเกินไปตามเวลา (ป้องกันภาพซ้ำซาก)
            if settings.ENABLE_SCENE_DIVERSITY and exclude_timestamps:
                face_timestamp = face.get('timestamp', 0.0)
                # เช็คว่าหน้านี้ใกล้กับหน้าที่เลือกไปแล้วหรือไม่ (ภายใน MIN_TEMPORAL_GAP วินาที)
                is_too_close = any(
                    abs(face_timestamp - used_timestamp) < settings.MIN_TEMPORAL_GAP
                    for used_timestamp in exclude_timestamps
                )
                if is_too_close:
                    temporal_filtered_count += 1
                    logger.debug(f"      ⏭️  Skipped face (timestamp={face_timestamp:.1f}s - too close to used timestamps)")
                    continue

            # หมายเหตุ: หน้าหันหลังถูกกรองออกแล้วตั้งแต่ต้น (hard filter) ไม่ต้องเช็คที่นี่

            if face.get('num_faces_in_image', 1) == 1:
                solo_faces.append(face)
            else:
                group_faces.append(face)

        logger.debug(f"   After filtering: {len(solo_faces)} solo, {len(group_faces)} group, {excluded_count} excluded, {eyes_closed_count} eyes_closed, {mouth_open_count} mouth_open, {temporal_filtered_count} temporal_filtered")

        # 🚨 STRICT SOLO SHOTS ONLY: บังคับเลือกเฉพาะ solo shots (คนเดียวในเฟรม) เท่านั้น!
        # ห้ามเลือก group shots (หลายคนในเฟรม) เพราะจะได้หน้าใกล้ที่ไม่สวย
        candidates = solo_faces  # ไม่เอา group_faces เลย!

        if not candidates:
            # ไม่มีรูปที่ใช้ได้เลย
            if group_faces:
                logger.warning(f"⚠️  No solo shots available, only {len(group_faces)} group shots (BLOCKED!)")
            logger.debug(f"No unused solo shot images available in cluster ({len(exclude_image_paths)} images excluded)")
            return None

        # เลือกภาพที่ดีที่สุดจาก candidates
        best_face = max(candidates, key=lambda x: x['composite_score'])

        logger.debug(
            f"Selected {'solo' if best_face in solo_faces else 'group'} shot "
            f"(score={best_face['composite_score']:.3f}, "
            f"num_faces={best_face.get('num_faces_in_image', 1)}, "
            f"image={best_face.get('image_path').name if best_face.get('image_path') else 'unknown'})"
        )

        return best_face

    def _select_multiple_faces_from_cluster(
        self,
        face_indices: List[int],
        num_faces: int,
        solo_shots_only: bool = True
    ) -> List[Dict]:
        """
        Select multiple best faces from a single cluster

        Args:
            face_indices: List of face indices in cluster
            num_faces: Number of faces to select
            solo_shots_only: Only select solo shots (default: True)

        Returns:
            List of face data dictionaries sorted by quality
        """
        # Get all faces from this cluster
        all_faces = [self.face_db[idx] for idx in face_indices]

        # Filter faces
        valid_faces = []
        for face in all_faces:
            # Skip closed eyes
            if face.get('eyes_closed', False):
                continue
            # Skip mouth open
            if face.get('mouth_open', False):
                continue
            # Skip group shots if solo_shots_only=True
            if solo_shots_only and face.get('num_faces_in_image', 1) > 1:
                continue
            valid_faces.append(face)

        logger.info(f"   Cluster filtering: {len(all_faces)} total → {len(valid_faces)} valid (solo_shots_only={solo_shots_only})")

        if not valid_faces:
            logger.warning(f"   ⚠️  No valid faces in cluster (all filtered out)")
            return []

        # Sort by composite score (best first)
        valid_faces.sort(key=lambda x: x['composite_score'], reverse=True)

        # Select top N faces with temporal diversity
        selected = []
        used_timestamps = []

        for face in valid_faces:
            if len(selected) >= num_faces:
                break

            # Check temporal spacing (ห้ามเลือกภาพที่ใกล้กันเกินไป)
            if settings.ENABLE_SCENE_DIVERSITY and used_timestamps:
                face_timestamp = face.get('timestamp', 0.0)
                is_too_close = any(
                    abs(face_timestamp - used_ts) < settings.MIN_TEMPORAL_GAP
                    for used_ts in used_timestamps
                )
                if is_too_close:
                    continue

            selected.append(face)
            used_timestamps.append(face.get('timestamp', 0.0))

        logger.info(
            f"   Selected {len(selected)}/{num_faces} faces from cluster "
            f"(from {len(valid_faces)} valid faces, {len(all_faces)} total)"
        )

        return selected

    def _merge_similar_clusters(self, clusters: Dict[int, List[int]], embeddings: np.ndarray) -> Dict[int, List[int]]:
        """
        LAYER 2: Merge clusters that are too similar (likely same person)

        Args:
            clusters: Dictionary of cluster_id -> list of face indices
            embeddings: Face embeddings array

        Returns:
            Merged clusters dictionary
        """
        from utils.face_utils import compute_embedding_distance

        # Calculate centroid for each cluster
        cluster_centroids = {}
        for cluster_id, face_indices in clusters.items():
            cluster_embeddings = embeddings[face_indices]
            centroid = np.mean(cluster_embeddings, axis=0)
            cluster_centroids[cluster_id] = centroid

        # Find clusters to merge (distance < threshold)
        merge_groups = []  # List of lists: [[c1, c2, c3], [c4, c5], ...]
        merged_ids = set()

        cluster_ids = list(clusters.keys())
        for i, id1 in enumerate(cluster_ids):
            if id1 in merged_ids:
                continue

            group = [id1]
            for id2 in cluster_ids[i+1:]:
                if id2 in merged_ids:
                    continue

                # Calculate distance between centroids
                dist = compute_embedding_distance(cluster_centroids[id1], cluster_centroids[id2])

                # 🔍 DEBUG: Show centroid comparison
                logger.debug(
                    f"   Comparing cluster {id1} vs {id2}: "
                    f"distance={dist:.3f}, threshold={settings.CLUSTER_MERGE_THRESHOLD}, "
                    f"will_merge={dist < settings.CLUSTER_MERGE_THRESHOLD}"
                )

                if dist < settings.CLUSTER_MERGE_THRESHOLD:
                    logger.info(
                        f"⚠️  MERGING cluster {id2} into cluster {id1} "
                        f"(distance={dist:.3f} < {settings.CLUSTER_MERGE_THRESHOLD})"
                    )
                    group.append(id2)
                    merged_ids.add(id2)

            merge_groups.append(group)

        # Rebuild clusters with merged groups
        new_clusters = {}
        for new_id, group in enumerate(merge_groups):
            # Combine all face indices from this group
            combined_faces = []
            for old_id in group:
                combined_faces.extend(clusters[old_id])

            new_clusters[new_id] = combined_faces

            if len(group) > 1:
                logger.debug(f"Merged clusters {group} into new cluster {new_id} ({len(combined_faces)} faces)")

        return new_clusters

    def _validate_character_diversity(self, selected_characters: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        LAYER 3: Validate that selected characters are different people

        Args:
            selected_characters: Dictionary of role -> character data

        Returns:
            Validated characters (duplicates removed)
        """
        from utils.face_utils import compute_embedding_distance

        validated = {}
        used_embeddings = []

        for role, char_data in selected_characters.items():
            face_embedding = char_data['face_data']['embedding']

            # Check if this embedding is too similar to already selected characters
            is_duplicate = False
            for used_emb in used_embeddings:
                dist = compute_embedding_distance(face_embedding, used_emb)
                if dist < settings.MIN_CHARACTER_DISTANCE:
                    logger.warning(
                        f"Skipping {role}: too similar to already selected character "
                        f"(distance={dist:.3f} < {settings.MIN_CHARACTER_DISTANCE})"
                    )
                    is_duplicate = True
                    break

            if not is_duplicate:
                validated[role] = char_data
                used_embeddings.append(face_embedding)

        return validated

    def get_all_faces(self) -> List[Dict]:
        """
        Get all faces in database

        Returns:
            List of face data dictionaries
        """
        return self.face_db

    def export_analysis_report(self) -> Dict:
        """
        Export comprehensive analysis report with all frame scores
        สำหรับให้ระบบอื่นมาช่วยวิเคราะห์เพิ่มเติม

        Returns:
            Dictionary with analysis data
        """
        report = {
            "total_faces": len(self.face_db),
            "total_clusters": len(self.clusters),
            "faces": [],
            "clusters": {},
            "summary": {
                "avg_composite_score": 0,
                "avg_sharpness": 0,
                "avg_frontality": 0,
                "faces_passed_frontality": 0,
                "faces_with_eyes_open": 0,
                "faces_solo_shots": 0
            }
        }

        # คำนวณค่าเฉลี่ย
        if len(self.face_db) > 0:
            total_composite = 0
            total_sharpness = 0
            total_frontality = 0
            frontality_count = 0
            eyes_open_count = 0
            solo_shots_count = 0

            for face_data in self.face_db:
                total_composite += face_data.get('composite_score', 0)
                total_sharpness += face_data.get('sharpness', 0)

                frontality = face_data.get('frontality_score')
                if frontality is not None:
                    total_frontality += frontality
                    frontality_count += 1
                    if frontality >= 0.80:
                        report["summary"]["faces_passed_frontality"] += 1

                if not face_data.get('eyes_closed', True):
                    eyes_open_count += 1

                if face_data.get('num_faces_in_frame', 0) == 1:
                    solo_shots_count += 1

                # เพิ่มข้อมูลแต่ละหน้า (ไม่รวม embedding ที่เป็น binary)
                face_info = {
                    "face_id": face_data.get('face_id'),
                    "image_path": face_data.get('image_path'),
                    "composite_score": round(face_data.get('composite_score', 0), 4),
                    "sharpness": round(face_data.get('sharpness', 0), 2),
                    "contrast": round(face_data.get('contrast', 0), 2),
                    "brightness": round(face_data.get('brightness', 0), 2),
                    "quality_score": round(face_data.get('quality_score', 0), 4),
                    "face_size_score": round(face_data.get('face_size_score', 0), 4),
                    "frontality_score": round(frontality, 4) if frontality is not None else None,
                    "eyes_closed": face_data.get('eyes_closed', False),
                    "mouth_open": face_data.get('mouth_open', False),
                    "negative_emotion": face_data.get('negative_emotion', False),
                    "num_faces_in_frame": face_data.get('num_faces_in_frame', 0),
                    "is_solo_shot": face_data.get('num_faces_in_frame', 0) == 1,
                    "bbox": face_data.get('bbox', []),
                    "cluster_id": None  # จะอัพเดทภายหลัง
                }

                report["faces"].append(face_info)

            # คำนวณค่าเฉลี่ย
            report["summary"]["avg_composite_score"] = round(total_composite / len(self.face_db), 4)
            report["summary"]["avg_sharpness"] = round(total_sharpness / len(self.face_db), 2)
            if frontality_count > 0:
                report["summary"]["avg_frontality"] = round(total_frontality / frontality_count, 4)
            report["summary"]["faces_with_eyes_open"] = eyes_open_count
            report["summary"]["faces_solo_shots"] = solo_shots_count

        # เพิ่มข้อมูล clusters
        for cluster_id, face_indices in self.clusters.items():
            cluster_info = {
                "size": len(face_indices),
                "face_indices": face_indices,
                "avg_score": 0,
                "best_score": 0,
                "face_ids": []
            }

            if len(face_indices) > 0:
                scores = []
                for idx in face_indices:
                    if idx < len(self.face_db):
                        face = self.face_db[idx]
                        scores.append(face.get('composite_score', 0))
                        cluster_info["face_ids"].append(face.get('face_id'))

                        # อัพเดท cluster_id ในข้อมูลหน้า
                        if idx < len(report["faces"]):
                            report["faces"][idx]["cluster_id"] = cluster_id

                if scores:
                    cluster_info["avg_score"] = round(sum(scores) / len(scores), 4)
                    cluster_info["best_score"] = round(max(scores), 4)

            report["clusters"][str(cluster_id)] = cluster_info

        logger.info(f"📊 Exported analysis report: {len(self.face_db)} faces, {len(self.clusters)} clusters")
        return report
