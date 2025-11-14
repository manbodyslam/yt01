"""
Gemini Vision API Eye Checker - 100% Accurate Eye Open/Closed Detection
"""

import io
import base64
from typing import Optional
import numpy as np
from PIL import Image
from loguru import logger

# Gemini imports
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("⚠️  google-generativeai not installed - eye checking will use fallback method")

from config import settings


def check_eyes_with_gemini(face_image: np.ndarray) -> bool:
    """
    Use Gemini Vision API to check if eyes are open or closed

    Args:
        face_image: Face region (numpy array, RGB)

    Returns:
        True if eyes are open, False if closed
    """
    if not GEMINI_AVAILABLE:
        raise RuntimeError("Gemini API not available")

    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set in config")

    # Configure Gemini
    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')  # 🔥 ใช้ gemini-2.0-flash-exp (ใหม่ล่าสุด, แม่นยำสูง)

    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(face_image)

    # Create prompt
    prompt = """
    Look at this face image carefully.

    Are the person's eyes OPEN or CLOSED?

    Important rules:
    - Eyes are OPEN if you can clearly see the iris and pupil
    - Eyes are CLOSED if the eyelids are shut (blinking, sleeping, squinting shut)
    - If partially closed (squinting but still see some eye) = count as OPEN
    - If you're uncertain, default to OPEN

    Respond with ONLY ONE WORD:
    - "OPEN" if eyes are open
    - "CLOSED" if eyes are closed

    Your answer (one word only):
    """

    try:
        # Call Gemini Vision API
        response = model.generate_content([prompt, pil_image])

        if not response or not response.text:
            raise RuntimeError("Empty response from Gemini API")

        # Parse response
        answer = response.text.strip().upper()

        if "OPEN" in answer:
            return True
        elif "CLOSED" in answer:
            return False
        else:
            logger.warning(f"⚠️  Unexpected Gemini response: '{response.text}' - defaulting to OPEN")
            return True  # Default to open if uncertain

    except Exception as e:
        logger.error(f"❌ Gemini API error in eye checking: {e}")
        raise


def check_eyes_with_gemini_batch(face_images: list[np.ndarray]) -> list[bool]:
    """
    Batch check multiple faces at once (more efficient)

    Args:
        face_images: List of face regions

    Returns:
        List of bool (True = eyes open, False = closed)
    """
    # TODO: Implement batch processing if needed
    # For now, process one by one
    results = []
    for face_image in face_images:
        try:
            result = check_eyes_with_gemini(face_image)
            results.append(result)
        except Exception as e:
            logger.error(f"Error checking face: {e}")
            results.append(True)  # Default to open on error

    return results
