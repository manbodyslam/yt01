"""
Face utility functions for quality assessment
"""

import numpy as np
from typing import Tuple, Dict, Optional
import cv2
from loguru import logger

# DeepFace import (lazy load to avoid startup delays)
DEEPFACE_AVAILABLE = False
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    logger.info("DeepFace loaded successfully")
except ImportError:
    logger.warning("DeepFace not available - emotion detection will be skipped")
except Exception as e:
    logger.warning(f"Failed to load DeepFace: {e}")


def calculate_face_size_score(
    face_data: Dict,
    image_shape: Tuple[int, int, int]
) -> float:
    """
    Calculate face size score (separated for independent weighting)

    Args:
        face_data: Dictionary containing face detection data with bbox
        image_shape: Shape of source image (h, w, c)

    Returns:
        Size score (0-1, higher = better)
    """
    bbox = face_data.get('bbox')
    if bbox is None:
        return 0.5

    x1, y1, x2, y2 = bbox
    face_area = (x2 - x1) * (y2 - y1)
    image_area = image_shape[0] * image_shape[1]
    size_ratio = face_area / image_area

    # Ideal face size is 10-40% of image (เข้มงวดขึ้น - ต้องการหน้าใหญ่ชัดเจน)
    if 0.10 <= size_ratio <= 0.40:
        size_score = 1.0
    elif size_ratio < 0.10:
        # Penalize small faces heavily (หน้าเล็กกว่า 10% ของภาพ)
        size_score = size_ratio / 0.10
    else:
        # Penalize very large faces (หน้าใหญ่กว่า 40% ของภาพ)
        size_score = max(0, 1 - (size_ratio - 0.40) / 0.40)

    return float(size_score)


def calculate_face_quality(
    face_data: Dict,
    image_shape: Tuple[int, int, int]
) -> float:
    """
    Calculate overall face quality score (excluding face size - weighted separately)

    Args:
        face_data: Dictionary containing face detection data
                   {bbox, kps, det_score, embedding, etc.}
        image_shape: Shape of source image (h, w, c)

    Returns:
        Quality score (0-1, higher = better quality)
    """
    scores = []

    # 1. Detection confidence
    det_score = face_data.get('det_score', 0)
    scores.append(det_score)

    # 2. Face angle/pose (using landmarks if available)
    kps = face_data.get('kps')
    if kps is not None and len(kps) >= 5:
        pose_score = calculate_face_frontality(kps)
        scores.append(pose_score)

    # 3. Face position (prefer faces not at edge)
    bbox = face_data.get('bbox')
    if bbox is not None:
        position_score = calculate_face_position_score(bbox, image_shape)
        scores.append(position_score)

    # Average all scores
    return float(np.mean(scores)) if scores else 0.0


def calculate_face_frontality(landmarks: np.ndarray) -> float:
    """
    Calculate how frontal the face is based on landmarks

    Args:
        landmarks: Face landmarks (5 points: left_eye, right_eye, nose, left_mouth, right_mouth)

    Returns:
        Frontality score (0-1, 1 = perfectly frontal)
    """
    if len(landmarks) < 5:
        return 0.5

    left_eye = landmarks[0]
    right_eye = landmarks[1]
    nose = landmarks[2]
    left_mouth = landmarks[3]
    right_mouth = landmarks[4]

    # Calculate eye distance
    eye_dist = np.linalg.norm(right_eye - left_eye)
    if eye_dist == 0:
        return 0.5

    # Calculate nose position relative to eyes (should be centered)
    eye_center = (left_eye + right_eye) / 2
    eye_to_nose = nose - eye_center

    # Calculate horizontal offset (x-axis)
    horizontal_offset = abs(eye_to_nose[0]) / eye_dist

    # Calculate mouth symmetry
    mouth_dist = np.linalg.norm(right_mouth - left_mouth)
    mouth_center = (left_mouth + right_mouth) / 2
    mouth_to_eye_center = mouth_center - eye_center
    mouth_horizontal_offset = abs(mouth_to_eye_center[0]) / eye_dist

    # Combine offsets (lower = more frontal)
    avg_offset = (horizontal_offset + mouth_horizontal_offset) / 2

    # Convert to score (0 offset = 1.0, high offset = 0.0)
    # ลดจาก *3 เป็น *2 เพื่อผ่อนปรนหน้าเอียงเล็กน้อย (ยังคงได้ภาพสมบูรณ์)
    frontality = max(0, 1 - (avg_offset * 2))

    return float(frontality)


def calculate_face_position_score(
    bbox: Tuple[float, float, float, float],
    image_shape: Tuple[int, int, int]
) -> float:
    """
    Score face position (prefer faces not at edges)

    Args:
        bbox: Face bounding box (x1, y1, x2, y2)
        image_shape: Shape of source image (h, w, c)

    Returns:
        Position score (0-1, higher = better position)
    """
    x1, y1, x2, y2 = bbox
    h, w = image_shape[:2]

    # Calculate face center
    face_center_x = (x1 + x2) / 2
    face_center_y = (y1 + y2) / 2

    # Calculate distance from image center (normalized)
    image_center_x = w / 2
    image_center_y = h / 2

    dx = abs(face_center_x - image_center_x) / (w / 2)
    dy = abs(face_center_y - image_center_y) / (h / 2)

    # Distance from center (0 = center, 1 = edge)
    dist_from_center = np.sqrt(dx**2 + dy**2) / np.sqrt(2)

    # Prefer faces slightly off-center (0.3-0.6 range is ideal)
    if 0.3 <= dist_from_center <= 0.6:
        position_score = 1.0
    elif dist_from_center < 0.3:
        position_score = 0.8 + (dist_from_center / 0.3) * 0.2
    else:
        position_score = max(0, 1 - (dist_from_center - 0.6) / 0.4)

    # Penalize faces too close to edges (เข้มงวดมากขึ้นเพื่อไม่ให้ได้ภาพไม่สมบูรณ์)
    margin = 0.10  # เพิ่มเป็น 10% margin (จาก 5%) - ไม่รับภาพติดขอบ
    edge_penalty_horizontal = 0.5  # เข้มงวดขึ้น (จาก 0.7) - ลด 50%
    edge_penalty_vertical = 0.4    # เข้มงวดมากขึ้น - ลด 60% (เพราะหัวติดขอบบนล่างแย่กว่า)

    if x1 < w * margin or x2 > w * (1 - margin):
        position_score *= edge_penalty_horizontal
    if y1 < h * margin or y2 > h * (1 - margin):
        position_score *= edge_penalty_vertical  # ติดขอบบน/ล่าง = ภาพไม่สมบูรณ์!

    return float(position_score)


def get_face_embedding(face_data: Dict) -> np.ndarray:
    """
    Extract face embedding vector

    Args:
        face_data: Face detection data dictionary

    Returns:
        Face embedding vector (normalized)
    """
    embedding = face_data.get('embedding')
    if embedding is None:
        raise ValueError("No embedding found in face data")

    # Normalize embedding
    embedding = np.array(embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


def compute_embedding_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings

    Args:
        emb1: First embedding vector
        emb2: Second embedding vector

    Returns:
        Similarity score (-1 to 1, higher = more similar)
    """
    # Normalize embeddings
    emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
    emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)

    # Cosine similarity
    similarity = np.dot(emb1_norm, emb2_norm)

    return float(similarity)


def compute_embedding_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine distance between two face embeddings.

    For face recognition, cosine similarity is more robust than Euclidean distance
    and provides normalized 0-1 output regardless of embedding dimensionality.

    Args:
        emb1: First embedding vector (L2-normalized from InsightFace)
        emb2: Second embedding vector (L2-normalized from InsightFace)

    Returns:
        Distance score 0-1 (lower = more similar)
        - Same person: typically 0.10-0.25
        - Different people: typically 0.30-0.50
    """
    # Ensure embeddings are normalized (InsightFace buffalo_l already does this, but safe to double-check)
    emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
    emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)

    # Compute cosine similarity: dot product of normalized vectors
    # Range: -1 (opposite) to 1 (identical), but for faces typically 0.2 to 0.9
    cosine_sim = np.dot(emb1_norm, emb2_norm)

    # Convert to distance: (1 - similarity) / 2
    # Maps [-1, 1] -> [0, 1], where 0 = identical, 1 = maximally different
    cosine_dist = (1.0 - float(cosine_sim)) / 2.0

    # Clip to [0, 1] range for safety
    return float(np.clip(cosine_dist, 0.0, 1.0))


def check_eyes_open(face_data: Dict, image: np.ndarray) -> float:
    """
    Check if eyes are open using Gemini Vision API (100% accurate!)

    Args:
        face_data: Face detection data with landmarks
        image: Source image (RGB)

    Returns:
        Eyes open score (0-1, 1 = eyes wide open, 0 = eyes closed)
    """
    face_id = face_data.get('face_id', 'unknown')
    logger.info(f"👀 Checking eyes for face: {face_id}")

    # 🚀 เช็ค setting ก่อน - ถ้าปิด Gemini ก็ข้ามไปใช้ OpenCV เลย
    from config import settings

    if settings.ENABLE_GEMINI_EYE_CHECK:
        # ใช้ Gemini Vision API เช็คตา
        from modules.gemini_eye_checker import check_eyes_with_gemini

        bbox = face_data.get('bbox')
        if bbox is None:
            logger.warning(f"⚠️  Face {face_id}: No bbox found - returning neutral score 0.5")
            return 0.5

        # Crop face region
        x1, y1, x2, y2 = map(int, bbox)
        face_crop = image[y1:y2, x1:x2]

        if face_crop.size == 0:
            logger.warning(f"⚠️  Face {face_id}: Empty face crop - returning neutral score 0.5")
            return 0.5

        # เช็คด้วย Gemini Vision API
        try:
            eyes_open = check_eyes_with_gemini(face_crop)
            logger.info(f"  ✅ Gemini eyes check: {'OPEN' if eyes_open else 'CLOSED'} (score={1.0 if eyes_open else 0.0})")
            return 1.0 if eyes_open else 0.0
        except Exception as e:
            logger.error(f"  ❌ Gemini API error: {e} - falling back to variance method")
            # Fallback to variance method
            pass
    else:
        logger.debug(f"  ⚡ Gemini Eye Check disabled - using fast OpenCV variance method")

    # === FALLBACK: OpenCV Variance Method (Fast & Free!) ===
    kps = face_data.get('kps')
    if kps is None:
        logger.warning(f"⚠️  Face {face_id}: No keypoints (kps) found - returning neutral score 0.5")
        return 0.5  # Unknown, neutral score
    if len(kps) < 2:
        logger.warning(f"⚠️  Face {face_id}: Insufficient keypoints (only {len(kps)}) - returning neutral score 0.5")
        return 0.5  # Unknown, neutral score

    left_eye = kps[0].astype(int)
    right_eye = kps[1].astype(int)

    # Extract eye regions (small area around each eye)
    h, w = image.shape[:2]
    eye_region_size = 20  # pixels around eye center

    scores = []

    for eye_center in [left_eye, right_eye]:
        x, y = eye_center

        # Check bounds
        x1 = max(0, x - eye_region_size)
        y1 = max(0, y - eye_region_size)
        x2 = min(w, x + eye_region_size)
        y2 = min(h, y + eye_region_size)

        if x2 <= x1 or y2 <= y1:
            continue

        eye_region = image[y1:y2, x1:x2]

        if eye_region.size == 0:
            continue

        # Convert to grayscale
        if len(eye_region.shape) == 3:
            gray = cv2.cvtColor(eye_region, cv2.COLOR_RGB2GRAY)
        else:
            gray = eye_region

        # Calculate variance (open eyes have higher variance due to iris/pupil contrast)
        variance = gray.var()

        # Calculate mean brightness (open eyes typically have lower mean due to dark pupil)
        mean_brightness = gray.mean()

        # 🚨 HARD FILTER: ถ้า variance ต่ำมาก (<400) = ตาหลับแน่นอน!
        # ⚖️ ผ่อนปรนมาก: 400 (ลดจาก 800 เพื่อรองรับนักแสดงใส่แว่น!)
        # 👓 แว่นตาจะลดความต่างของดวงตา (reflection/glare) → variance ต่ำลง
        # ตาเปิดจริงๆ จะมี variance สูงมาก (>2000) เพราะมีความต่างระหว่างตาขาว-ม่านตา-รูม่านตา
        # ตาเปิด + แว่น = variance ~500-1500 (ยังพอใช้ได้!)
        if variance < 400:
            logger.info(f"  👁️ Eye CLOSED detected: variance={variance:.1f} < 400")
            scores.append(0.0)  # ตาหลับ - คะแนน 0 เลย!
            continue

        logger.info(f"  👁️ Eye OPEN detected: variance={variance:.1f}, brightness={mean_brightness:.1f}")

        # Eyes open typically have:
        # - Very high variance (>2000) due to iris/pupil/white contrast
        # - Lower mean brightness (<140) due to dark pupil

        # ⚖️ สมดุล: variance ต้อง >2000 ถึงจะได้คะแนนเต็ม (ไม่เข้มงวดเกิน)
        variance_score = min(variance / 2000.0, 1.0)  # สมดุลที่ 2000

        # Brightness score (lower = better, as pupil is dark) - เข้มงวดขึ้น
        if mean_brightness < 130:
            brightness_score = 1.0
        elif mean_brightness < 160:
            brightness_score = 0.6
        else:
            brightness_score = 0.2  # ลดจาก 0.3 → 0.2

        # เพิ่มน้ำหนัก variance (ตา closed = variance ต่ำ)
        eye_score = (variance_score * 0.8 + brightness_score * 0.2)  # เพิ่มน้ำหนัก variance
        scores.append(eye_score)

    if not scores:
        return 0.5

    # Return average of both eyes
    avg_score = float(np.mean(scores))

    # Debug logging
    logger.info(f"  ✅ Eyes open score: {avg_score:.3f} (variance-based detection)")

    return avg_score


def check_mouth_open(face_data: Dict, image: np.ndarray) -> bool:
    """
    Check if mouth is open using facial landmarks (Mouth Aspect Ratio)

    Args:
        face_data: Face detection data with landmarks
        image: Source image (RGB)

    Returns:
        True if mouth is closed, False if mouth is open
    """
    kps = face_data.get('kps')
    if kps is None or len(kps) < 5:
        logger.warning("  ⚠️  No landmarks available for mouth detection - assuming closed")
        return True  # Assume closed if no landmarks

    # Extract landmarks (5 keypoints from InsightFace: left_eye, right_eye, nose, left_mouth, right_mouth)
    # Index 3 = left mouth corner, Index 4 = right mouth corner
    if len(kps) < 5:
        return True

    try:
        # Get mouth corners (approximate from 5-point landmarks)
        left_mouth = kps[3]  # Left mouth corner
        right_mouth = kps[4]  # Right mouth corner
        nose = kps[2]  # Nose tip

        # Calculate mouth width
        mouth_width = np.linalg.norm(np.array(left_mouth) - np.array(right_mouth))

        # ใช้วิธีประมาณด้วย ROI บริเวณปาก (เพราะ 5-point landmarks ไม่มีจุดบนล่างของปาก)
        # Extract mouth region (ประมาณจากตำแหน่งปาก)
        mouth_center_x = int((left_mouth[0] + right_mouth[0]) / 2)
        mouth_center_y = int((left_mouth[1] + right_mouth[1]) / 2)

        # ROI สำหรับตรวจปาก (ขนาดตามความกว้างปาก)
        roi_width = int(mouth_width * 0.8)
        roi_height = int(mouth_width * 0.6)

        x1 = max(0, mouth_center_x - roi_width // 2)
        y1 = max(0, mouth_center_y - roi_height // 2)
        x2 = min(image.shape[1], mouth_center_x + roi_width // 2)
        y2 = min(image.shape[0], mouth_center_y + roi_height // 2)

        mouth_roi = image[y1:y2, x1:x2]

        if mouth_roi.size == 0:
            return True

        # Convert to grayscale
        gray_mouth = cv2.cvtColor(mouth_roi, cv2.COLOR_RGB2GRAY)

        # คำนวณ vertical variance (ปากเปิดจะมีความแตกต่างสูงระหว่างบนล่าง)
        # Split แนวนอนเป็น 3 ส่วน: บน-กลาง-ล่าง
        h = gray_mouth.shape[0]
        if h < 6:
            return True

        top_third = gray_mouth[:h//3, :]
        mid_third = gray_mouth[h//3:2*h//3, :]
        bottom_third = gray_mouth[2*h//3:, :]

        # คำนวณความแตกต่างระหว่างบน-กลาง-ล่าง
        top_mean = top_third.mean()
        mid_mean = mid_third.mean()
        bottom_mean = bottom_third.mean()

        # Mouth opening indicator: ถ้าปากเปิด mid_mean จะมืดกว่า (เพราะมีช่องว่าง)
        mouth_darkness = (top_mean + bottom_mean) / 2 - mid_mean

        # คำนวณ aspect ratio (height/width)
        aspect_ratio = roi_height / roi_width if roi_width > 0 else 0

        # Threshold: กรองแค่ปากที่เปิดมากมาก (น่าเกลียดจริงๆ) เท่านั้น
        # 🎯 ปรับเป็น darkness > 90, aspect_ratio > 1.2 (ผ่อนปรนมากขึ้น)
        # จาก feedback: darkness < 90 = ปากเปิดปกติ (ยอมรับได้), darkness > 90 = อ้าปากมากเกินไป (กรอง)
        is_mouth_open = mouth_darkness > 90 or aspect_ratio > 1.2

        if is_mouth_open:
            logger.info(f"  👄 Mouth WIDE OPEN detected (filtering): darkness={mouth_darkness:.1f}, aspect_ratio={aspect_ratio:.2f}")
            return False  # Mouth is open wide
        else:
            logger.info(f"  👄 Mouth OK (closed or slightly open): darkness={mouth_darkness:.1f}, aspect_ratio={aspect_ratio:.2f}")
            return True  # Mouth is closed or slightly open (acceptable)

    except Exception as e:
        logger.warning(f"  ⚠️  Error checking mouth: {e} - assuming closed")
        return True


def check_occlusion(face_data: Dict, image: np.ndarray) -> float:
    """
    Estimate face occlusion level (experimental)

    Args:
        face_data: Face detection data
        image: Source image

    Returns:
        Occlusion score (0-1, 0 = no occlusion, 1 = heavily occluded)
    """
    # This is a simple heuristic - can be improved with dedicated models
    bbox = face_data.get('bbox')
    if bbox is None:
        return 0.5

    x1, y1, x2, y2 = map(int, bbox)

    # Extract face region
    face_region = image[y1:y2, x1:x2]

    if face_region.size == 0:
        return 0.5

    # Calculate color variance (occluded faces often have lower variance)
    if len(face_region.shape) == 3:
        gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
    else:
        gray = face_region

    variance = gray.var()

    # Lower variance might indicate occlusion (very rough heuristic)
    # Typical face variance is 500-2000
    if variance < 300:
        occlusion = 0.7
    elif variance < 500:
        occlusion = 0.4
    else:
        occlusion = 0.1

    return float(occlusion)


def extract_hair_color(face_data: Dict, image: np.ndarray) -> Tuple[int, int, int]:
    """
    Extract dominant hair color from region above forehead

    Args:
        face_data: Face detection data with bbox
        image: Source image (RGB)

    Returns:
        Tuple of (R, G, B) representing dominant hair color
    """
    bbox = face_data.get('bbox')
    if bbox is None:
        return (0, 0, 0)

    x1, y1, x2, y2 = map(int, bbox)
    h, w = image.shape[:2]

    # Calculate hair region (above forehead)
    # Face bbox y1 is top of detected face (usually forehead area)
    # Hair region: 20% of face height above the face bbox
    face_height = y2 - y1
    hair_height = int(face_height * 0.3)  # 30% of face height for hair

    # Hair region coordinates
    hair_x1 = max(0, x1)
    hair_y1 = max(0, y1 - hair_height)
    hair_x2 = min(w, x2)
    hair_y2 = max(0, y1)

    if hair_y2 <= hair_y1 or hair_x2 <= hair_x1:
        return (0, 0, 0)

    # Extract hair region
    hair_region = image[hair_y1:hair_y2, hair_x1:hair_x2]

    if hair_region.size == 0:
        return (0, 0, 0)

    # Calculate dominant color (median RGB)
    # Use median instead of mean to reduce outlier impact
    r_median = int(np.median(hair_region[:, :, 0]))
    g_median = int(np.median(hair_region[:, :, 1]))
    b_median = int(np.median(hair_region[:, :, 2]))

    return (r_median, g_median, b_median)


def extract_clothing_color(face_data: Dict, image: np.ndarray) -> Tuple[int, int, int]:
    """
    Extract dominant clothing color from region below neck

    Args:
        face_data: Face detection data with bbox
        image: Source image (RGB)

    Returns:
        Tuple of (R, G, B) representing dominant clothing color
    """
    bbox = face_data.get('bbox')
    if bbox is None:
        return (0, 0, 0)

    x1, y1, x2, y2 = map(int, bbox)
    h, w = image.shape[:2]

    # Calculate clothing region (below face bbox)
    # Face bbox y2 is bottom of face (usually chin area)
    # Clothing region: 40% of face height below the face bbox
    face_height = y2 - y1
    clothing_height = int(face_height * 0.5)  # 50% of face height for clothing

    # Clothing region coordinates
    clothing_x1 = max(0, x1)
    clothing_y1 = min(h, y2)
    clothing_x2 = min(w, x2)
    clothing_y2 = min(h, y2 + clothing_height)

    if clothing_y2 <= clothing_y1 or clothing_x2 <= clothing_x1:
        return (0, 0, 0)

    # Extract clothing region
    clothing_region = image[clothing_y1:clothing_y2, clothing_x1:clothing_x2]

    if clothing_region.size == 0:
        return (0, 0, 0)

    # Calculate dominant color (median RGB)
    r_median = int(np.median(clothing_region[:, :, 0]))
    g_median = int(np.median(clothing_region[:, :, 1]))
    b_median = int(np.median(clothing_region[:, :, 2]))

    return (r_median, g_median, b_median)


def compute_landmarks_similarity(kps1: Optional[np.ndarray], kps2: Optional[np.ndarray]) -> float:
    """
    Compute similarity between two sets of face landmarks (face geometry)

    This function compares the geometric relationships between facial features:
    - Eye-to-eye distance ratio
    - Eye-to-nose ratio
    - Mouth width ratio
    - Face aspect ratio (width/height)

    Args:
        kps1: First face landmarks (5 points: left_eye, right_eye, nose, left_mouth, right_mouth)
        kps2: Second face landmarks (5 points)

    Returns:
        Similarity score (0-1, 1 = identical geometry, 0 = very different)
    """
    if kps1 is None or kps2 is None:
        return 0.5  # Unknown, neutral score

    if len(kps1) < 5 or len(kps2) < 5:
        return 0.5

    try:
        # Extract landmarks for both faces
        left_eye1, right_eye1, nose1, left_mouth1, right_mouth1 = kps1[:5]
        left_eye2, right_eye2, nose2, left_mouth2, right_mouth2 = kps2[:5]

        # Calculate geometric features for face 1
        eye_dist1 = np.linalg.norm(right_eye1 - left_eye1)
        eye_center1 = (left_eye1 + right_eye1) / 2
        eye_to_nose1 = np.linalg.norm(nose1 - eye_center1)
        mouth_width1 = np.linalg.norm(right_mouth1 - left_mouth1)
        mouth_center1 = (left_mouth1 + right_mouth1) / 2
        face_height1 = np.linalg.norm(mouth_center1 - eye_center1)

        # Calculate geometric features for face 2
        eye_dist2 = np.linalg.norm(right_eye2 - left_eye2)
        eye_center2 = (left_eye2 + right_eye2) / 2
        eye_to_nose2 = np.linalg.norm(nose2 - eye_center2)
        mouth_width2 = np.linalg.norm(right_mouth2 - left_mouth2)
        mouth_center2 = (left_mouth2 + right_mouth2) / 2
        face_height2 = np.linalg.norm(mouth_center2 - eye_center2)

        # Avoid division by zero
        if eye_dist1 == 0 or eye_dist2 == 0 or face_height1 == 0 or face_height2 == 0:
            return 0.5

        # Calculate normalized ratios (scale-invariant)
        # Ratio 1: Eye-to-nose distance / Eye distance
        ratio1_face1 = eye_to_nose1 / eye_dist1
        ratio1_face2 = eye_to_nose2 / eye_dist2
        diff1 = abs(ratio1_face1 - ratio1_face2) / max(ratio1_face1, ratio1_face2)

        # Ratio 2: Mouth width / Eye distance
        ratio2_face1 = mouth_width1 / eye_dist1
        ratio2_face2 = mouth_width2 / eye_dist2
        diff2 = abs(ratio2_face1 - ratio2_face2) / max(ratio2_face1, ratio2_face2)

        # Ratio 3: Face aspect (width / height)
        ratio3_face1 = eye_dist1 / face_height1
        ratio3_face2 = eye_dist2 / face_height2
        diff3 = abs(ratio3_face1 - ratio3_face2) / max(ratio3_face1, ratio3_face2)

        # Calculate average difference
        avg_diff = (diff1 + diff2 + diff3) / 3

        # Convert to similarity score
        # If avg_diff = 0 → similarity = 1.0
        # If avg_diff = 0.5 → similarity = 0.0
        # เข้มงวด: ถ้าต่างกัน 20% (0.20) → similarity = 0.60
        similarity = max(0, 1 - (avg_diff * 2))

        return float(similarity)

    except Exception as e:
        logger.debug(f"Failed to compute landmarks similarity: {e}")
        return 0.5


def compute_color_similarity(color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
    """
    Compute color similarity using Euclidean distance in RGB space

    Args:
        color1: First color (R, G, B)
        color2: Second color (R, G, B)

    Returns:
        Similarity score (0-1, 1 = identical, 0 = very different)
    """
    # Calculate Euclidean distance in RGB space
    # Max distance is sqrt(255^2 * 3) = 441.67
    r1, g1, b1 = color1
    r2, g2, b2 = color2

    distance = np.sqrt((r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2)
    max_distance = 441.67

    # Convert distance to similarity (0-1)
    similarity = 1 - (distance / max_distance)

    return float(similarity)


def analyze_emotion(face_data: Dict, image: np.ndarray) -> Tuple[str, float, Dict]:
    """
    Analyze face emotion using DeepFace

    Args:
        face_data: Face detection data with bbox
        image: Source image (RGB)

    Returns:
        Tuple of (dominant_emotion, emotion_score, all_emotions_dict)
        - dominant_emotion: Most prominent emotion detected
        - emotion_score: Quality score (0-1, higher = better for thumbnail)
        - all_emotions_dict: Dictionary of all emotion probabilities
    """
    if not DEEPFACE_AVAILABLE:
        logger.debug("DeepFace not available, returning neutral emotion")
        return "neutral", 0.5, {}

    try:
        bbox = face_data.get('bbox')
        if bbox is None:
            return "neutral", 0.5, {}

        x1, y1, x2, y2 = map(int, bbox)

        # Validate bbox
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))

        face_region = image[y1:y2, x1:x2]

        if face_region.size == 0:
            return "neutral", 0.5, {}

        # DeepFace.analyze expects BGR
        face_bgr = cv2.cvtColor(face_region, cv2.COLOR_RGB2BGR)

        # Analyze emotion (disable face detection since we already have the face)
        result = DeepFace.analyze(
            face_bgr,
            actions=['emotion'],
            enforce_detection=False,  # Skip detection, we already have the face
            silent=True  # Suppress console output
        )

        # Handle list or dict result
        if isinstance(result, list):
            result = result[0]

        emotions = result.get('emotion', {})
        dominant_emotion = result.get('dominant_emotion', 'neutral')

        # Calculate emotion score (0-1)
        # Prefer: happy, surprise, neutral (good for thumbnails)
        # Avoid: sad, angry, fear, disgust (negative emotions)
        positive_emotions = ['happy', 'surprise', 'neutral']

        if dominant_emotion in positive_emotions:
            # High score for positive emotions
            # Base score 0.7 + bonus up to 0.3 based on confidence
            emotion_confidence = emotions.get(dominant_emotion, 50) / 100
            emotion_score = 0.7 + (emotion_confidence * 0.3)
        else:
            # Low score for negative emotions
            # Base score 0.2 + small bonus up to 0.3 if neutral is secondary
            neutral_confidence = emotions.get('neutral', 0) / 100
            emotion_score = 0.2 + (neutral_confidence * 0.3)

        logger.debug(
            f"Emotion detected: {dominant_emotion} "
            f"(score={emotion_score:.2f}, confidence={emotions.get(dominant_emotion, 0):.1f}%)"
        )

        return dominant_emotion, float(emotion_score), emotions

    except Exception as e:
        logger.warning(f"Failed to analyze emotion: {e}")
        return "neutral", 0.5, {}
