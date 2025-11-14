"""
Thumbnail Validator Module - AI-powered quality validation using Gemini
Option 4: Hybrid Validation - ใช้ Gemini 1.5 Pro ประเมินคุณภาพ thumbnail
"""

from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger
import google.generativeai as genai
from PIL import Image
import re

from config import settings


class ValidationResult:
    """ผลการประเมิน thumbnail"""

    def __init__(
        self,
        score: float,
        passed: bool,
        feedback: str,
        detailed_scores: Optional[Dict[str, float]] = None
    ):
        self.score = score  # คะแนนรวม (0-10)
        self.passed = passed  # ผ่านหรือไม่ (>= threshold)
        self.feedback = feedback  # คำแนะนำจาก AI
        self.detailed_scores = detailed_scores or {}  # คะแนนแยกตามหัวข้อ

    def __repr__(self):
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        return f"ValidationResult({status}, score={self.score}/10)"


class ThumbnailValidator:
    """
    ตัวตรวจสอบคุณภาพ thumbnail ด้วย Gemini 1.5 Pro

    ใช้สำหรับ Option 4: Hybrid Validation
    - สร้าง thumbnail ด้วย rule-based system
    - ให้ AI วิเคราะห์คุณภาพ
    - ถ้าคะแนนต่ำ → regenerate ใหม่
    """

    def __init__(self):
        """Initialize Gemini validator"""
        self.model_name = settings.GEMINI_MODEL
        self.threshold = settings.VALIDATION_THRESHOLD

        # Configure Gemini
        if not settings.GEMINI_API_KEY:
            logger.warning("⚠️  GEMINI_API_KEY not set - validation disabled")
            self.model = None
            return

        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"✅ ThumbnailValidator initialized with {self.model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini: {e}")
            self.model = None

    def validate(self, thumbnail_path: Path, title: str = "") -> ValidationResult:
        """
        ประเมินคุณภาพ thumbnail

        Args:
            thumbnail_path: Path to thumbnail image
            title: Optional title for context

        Returns:
            ValidationResult with score and feedback
        """
        if not self.model:
            logger.warning("Validation skipped - Gemini not configured")
            return ValidationResult(
                score=7.0,  # Default pass
                passed=True,
                feedback="Validation disabled (no Gemini API key)"
            )

        if not thumbnail_path.exists():
            logger.error(f"Thumbnail not found: {thumbnail_path}")
            return ValidationResult(
                score=0.0,
                passed=False,
                feedback=f"File not found: {thumbnail_path}"
            )

        try:
            # Load image
            img = Image.open(thumbnail_path)

            # Create prompt
            prompt = self._create_validation_prompt(title)

            # Ask Gemini
            logger.info(f"🔍 Validating thumbnail with {self.model_name}...")
            response = self.model.generate_content([prompt, img])

            # Parse response
            result = self._parse_response(response.text)

            # Log result
            status = "✅" if result.passed else "❌"
            logger.info(f"{status} Score: {result.score}/10 - {result.feedback}")

            return result

        except Exception as e:
            logger.error(f"❌ Validation error: {e}")
            return ValidationResult(
                score=0.0,
                passed=False,
                feedback=f"Validation error: {str(e)}"
            )

    def _create_validation_prompt(self, title: str = "") -> str:
        """สร้าง prompt สำหรับ Gemini"""

        context = f"Title: \"{title}\"\n\n" if title else ""

        prompt = f"""วิเคราะห์ YouTube thumbnail นี้และให้คะแนน 0-10:

{context}เกณฑ์การประเมิน (คะแนนเต็มแต่ละหัวข้อ):
1. **Composition (2 คะแนน)**: Visual balance, จัดวางองค์ประกอบ, ตัวละครไม่ซ้อนทับกัน
2. **Face Quality (3 คะแนน)**: ความชัดของหน้า, หน้าตรง, ตาเปิด, ไม่เบลอ
3. **Text Readability (2 คะแนน)**: อ่านข้อความง่าย, ขนาดตัวหนังสือเหมาะสม, สีตัดกันชัด
4. **Color Harmony (2 คะแนน)**: สีกลมกลืน, สะดุดตา, เหมาะกับ YouTube
5. **Click-worthiness (1 คะแนน)**: ดึงดูดใจ, น่าคลิก, professional

**ตอบกลับในรูปแบบนี้เท่านั้น (ต้องตรงตามรูปแบบ!):**

SCORE: [0-10]
COMPOSITION: [0-2]
FACE_QUALITY: [0-3]
TEXT_READABILITY: [0-2]
COLOR_HARMONY: [0-2]
CLICK_WORTHINESS: [0-1]
FEEDBACK: [คำแนะนำสั้นๆ 1-2 ประโยค ว่าควรปรับปรุงอะไร หรือดีอยู่แล้ว]

ตัวอย่าง:
SCORE: 8.5
COMPOSITION: 2.0
FACE_QUALITY: 2.5
TEXT_READABILITY: 1.8
COLOR_HARMONY: 1.5
CLICK_WORTHINESS: 0.7
FEEDBACK: ภาพรวมดี แต่ตัวหนังสือควรใหญ่ขึ้นเล็กน้อยเพื่อให้อ่านง่ายขึ้น สีพื้นหลังอาจจะสดใสไปนิด
"""
        return prompt

    def _parse_response(self, response_text: str) -> ValidationResult:
        """แปลง response จาก Gemini เป็น ValidationResult"""

        try:
            # Extract scores using regex
            score_match = re.search(r'SCORE:\s*([0-9.]+)', response_text)
            comp_match = re.search(r'COMPOSITION:\s*([0-9.]+)', response_text)
            face_match = re.search(r'FACE_QUALITY:\s*([0-9.]+)', response_text)
            text_match = re.search(r'TEXT_READABILITY:\s*([0-9.]+)', response_text)
            color_match = re.search(r'COLOR_HARMONY:\s*([0-9.]+)', response_text)
            click_match = re.search(r'CLICK_WORTHINESS:\s*([0-9.]+)', response_text)
            feedback_match = re.search(r'FEEDBACK:\s*(.+?)(?:\n|$)', response_text, re.DOTALL)

            # Parse overall score
            score = float(score_match.group(1)) if score_match else 7.0

            # Parse detailed scores
            detailed_scores = {}
            if comp_match:
                detailed_scores['composition'] = float(comp_match.group(1))
            if face_match:
                detailed_scores['face_quality'] = float(face_match.group(1))
            if text_match:
                detailed_scores['text_readability'] = float(text_match.group(1))
            if color_match:
                detailed_scores['color_harmony'] = float(color_match.group(1))
            if click_match:
                detailed_scores['click_worthiness'] = float(click_match.group(1))

            # Parse feedback
            feedback = feedback_match.group(1).strip() if feedback_match else "No specific feedback"

            # Determine pass/fail
            passed = score >= self.threshold

            return ValidationResult(
                score=score,
                passed=passed,
                feedback=feedback,
                detailed_scores=detailed_scores
            )

        except Exception as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            logger.debug(f"Response text: {response_text}")

            # Fallback: try to extract just the score
            numbers = re.findall(r'\b([0-9.]+)\b', response_text)
            score = float(numbers[0]) if numbers else 5.0

            return ValidationResult(
                score=score,
                passed=score >= self.threshold,
                feedback=response_text[:200] if response_text else "Parse error"
            )

    def validate_with_retry(
        self,
        thumbnail_path: Path,
        title: str = "",
        max_attempts: int = 3
    ) -> ValidationResult:
        """
        ประเมินพร้อม retry กรณี API error

        Args:
            thumbnail_path: Path to thumbnail
            title: Optional title
            max_attempts: Max retry attempts

        Returns:
            ValidationResult
        """
        for attempt in range(max_attempts):
            try:
                result = self.validate(thumbnail_path, title)
                return result
            except Exception as e:
                if attempt < max_attempts - 1:
                    logger.warning(f"Validation attempt {attempt + 1} failed: {e}, retrying...")
                else:
                    logger.error(f"All validation attempts failed: {e}")
                    return ValidationResult(
                        score=0.0,
                        passed=False,
                        feedback=f"Validation failed after {max_attempts} attempts"
                    )
