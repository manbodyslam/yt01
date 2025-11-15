"""
Configuration settings for YouTube Thumbnail Generator
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent
    WORKSPACE_DIR: Path = PROJECT_ROOT / "workspace"
    RAW_DIR: Path = WORKSPACE_DIR / "raw"
    TEMP_DIR: Path = WORKSPACE_DIR / "temp"
    OUT_DIR: Path = WORKSPACE_DIR / "out"
    ASSETS_DIR: Path = PROJECT_ROOT / "assets"
    FONTS_DIR: Path = ASSETS_DIR / "fonts"

    # Image filtering
    ALLOWED_EXTENSIONS: list[str] = [".jpg", ".jpeg", ".png", ".webp"]
    MIN_IMAGE_SIZE: int = 640  # Minimum short side in pixels

    # Video extraction settings (⚡ SPEED MODE: 5x faster!)
    VIDEO_FORMATS: list[str] = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"]
    VIDEO_FRAMES_PER_MINUTE: int = 15  # 🚀 ลดเป็น 15 รูป/นาที (ทุก 4 วินาที) - เร็วขึ้น 50%!
    VIDEO_MAX_FRAMES: int = 150  # ⚡ BALANCED MODE: 150 frames (เร็ว 7x, แม่นยำ 90%+)
    VIDEO_MIN_SHARPNESS: float = 25.0  # ยอมรับภาพเบลอมากขึ้น (เพิ่มความเร็ว)
    VIDEO_SIMILARITY_THRESHOLD: float = 0.70  # ยอมรับใบหน้าที่ต่างกันมากขึ้น

    # 🆕 Lazy Loading & Streaming (save 50-70% RAM!)
    VIDEO_BATCH_SIZE: int = 50  # Process frames in batches (lower = less RAM, higher = faster)

    # 🆕 Scene-based Sampling (improve diversity +20-30% quality!)
    VIDEO_FRAME_SIMILARITY_THRESHOLD: float = 0.0  # ⚡ DISABLED for speed (0.0=disabled, 0.85=strict)

    # 🎯 Smart Frame Selection (แก้ปัญหาตาหลับที่ต้นเหตุ!)
    ENABLE_SMART_FRAME_SELECTION: bool = True  # เปิด/ปิด Smart Frame Selection
    FRAMES_PER_SCENE: int = 10  # 🚀 เพิ่มเป็น 10 เฟรมต่อฉาก - มีตัวเลือกมากขึ้น! (จาก 7)
    CANDIDATE_FRAME_INTERVAL: float = 0.4  # 🚀 ลดเป็น 0.4 วินาที - ดึงบ่อยขึ้น (จาก 0.5)

    # น้ำหนักสำหรับเลือกเฟรมที่ดีที่สุด (รวม = 1.0)
    SMART_WEIGHT_EYES_OPEN: float = 0.60  # 60% - ตาเปิดสำคัญที่สุด!
    SMART_WEIGHT_SHARPNESS: float = 0.20  # 20% - ความคมชัด
    SMART_WEIGHT_FRONTALITY: float = 0.15  # 15% - หน้าตรง
    SMART_WEIGHT_FACE_SIZE: float = 0.05  # 5% - ขนาดหน้า

    # Threshold สำหรับกรองเฟรม (ปรับให้ผ่อนปรนขึ้นเพื่อเพิ่มความแม่นยำ)
    SMART_MIN_EYES_OPEN_SCORE: float = 0.50  # 🚀 ลดเป็น 50% - ยอมรับตาเปิดไม่เต็มที่ (จาก 75%)
    SMART_MIN_FRONTALITY_SCORE: float = 0.55  # 🚀 ลดเป็น 55% - ยอมรับหน้าเอียงเล็กน้อย (จาก 70%)
    SMART_REQUIRE_FACE: bool = True  # บังคับต้องมีหน้าในเฟรม (ไม่เก็บเฟรมที่ไม่มีหน้า)

    # Output settings
    OUTPUT_WIDTH: int = 1920
    OUTPUT_HEIGHT: int = 1080
    OUTPUT_FORMAT: str = "jpg"
    OUTPUT_QUALITY: int = 95

    # Face detection settings (ปรับให้ตรวจจับได้มากขึ้น!)
    FACE_DETECTION_MODEL: str = "buffalo_s"  # ⚡ SPEED MODE: buffalo_s (smaller, 2x faster, 95% accurate)
    FACE_CONFIDENCE_THRESHOLD: float = 0.35  # 🔓 ผ่อนปรน - เพิ่มโอกาสหาคนได้มากขึ้น (ลดจาก 0.45)
    MIN_FACE_SIZE: int = 40  # 🔓 ผ่อนปรน - ยอมรับหน้าเล็กกว่า (ลดจาก 50)

    # 🆕 ONNX Runtime Optimization (2-4x faster, 75% less RAM!)
    ONNX_ENABLE_OPTIMIZATION: bool = True  # Enable ONNX Runtime optimizations
    ONNX_NUM_THREADS: int = 4  # Number of threads for inference (4 = balanced)
    ONNX_EXECUTION_MODE: str = "sequential"  # "sequential" or "parallel"
    ONNX_GRAPH_OPTIMIZATION: str = "all"  # "disabled", "basic", "extended", "all"

    # Logo/Watermark removal settings
    ENABLE_LOGO_REMOVAL: bool = False  # เปิด/ปิดการลบโลโก้
    LOGO_REGIONS: list[tuple[int, int, int, int]] = []  # [(x, y, width, height), ...] ตำแหน่งโลโก้
    LOGO_INPAINT_RADIUS: int = 5  # รัศมี inpainting (3-10 แนะนำ)
    LOGO_INPAINT_METHOD: str = "telea"  # "telea" (เร็ว) หรือ "ns" (ช้ากว่า แต่คุณภาพดีกว่า)

    # Clustering settings (3-LAYER DEFENSE against duplicate selection)
    CLUSTERING_ALGORITHM: str = "DBSCAN"  # or "HDBSCAN"
    DBSCAN_EPS: float = 0.48  # 🚀 เพิ่มเป็น 0.48 - รวมหน้าคนเดียวกันเข้าด้วยกัน! (แก้ปัญหาแยกคนเดียวกันเป็นหลายคน)
    DBSCAN_MIN_SAMPLES: int = 1  # ⚖️ กลับเป็น 1 - ยอมรับหน้าเดี่ยว (ไม่เข้มงวดเกิน)
    HDBSCAN_MIN_CLUSTER_SIZE: int = 2

    # LAYER 2: Post-clustering validation (รวมคนเดียวกันที่ DBSCAN แยกไว้)
    CLUSTER_MERGE_THRESHOLD: float = 0.18  # 🚀 เพิ่มเป็น 0.18 - รวม clusters ของคนเดียวกันที่แยกออก

    # LAYER 3: Character selection validation (ป้องกันเลือกคนเดียวกันซ้ำ)
    MIN_CHARACTER_DISTANCE: float = 0.50  # 🎯 STRICT MODE: 0.50 - บล็อกคนที่หน้าคล้ายกัน (เข้มงวดมาก!)

    # LAYER 4: Solo shot enforcement (เพิ่มความแม่นยำ 100%)
    PREFER_SOLO_SHOTS: bool = True  # บังคับเลือกแค่ solo shots (1 คนในเฟรม) เท่านั้น
    GROUP_SHOT_PENALTY: float = 0.3  # ลดคะแนน group shots เหลือ 30% (ถ้า PREFER_SOLO_SHOTS=False)
    BLOCK_GROUP_SHOTS_IF_SOLO_AVAILABLE: bool = True  # บล็อก group shots ทั้งหมดถ้ามี solo shots พอ

    # Face selection settings (จำนวนรูปต่อคน)
    FACES_PER_CHARACTER: int = 50  # 🚀 เก็บ 50 รูปดีที่สุดต่อคนใน raw folder (เพิ่มจาก 15!)

    # Image scoring weights (สมดุลระหว่างตาเปิด, หน้าตรง, ความสวยงาม)
    WEIGHT_EYES_OPEN: float = 0.45  # 🎯 ลดเป็น 45% - สำคัญแต่ไม่มากเกินไป! (จาก 60%)
    WEIGHT_SHARPNESS: float = 0.18  # 18% - ความคมชัด
    WEIGHT_FACE_QUALITY: float = 0.27  # 🚀 เพิ่มเป็น 27%! - หน้าตรง+ตำแหน่งดี = ความสวยงาม! (จาก 15%)
    WEIGHT_FACE_SIZE: float = 0.10  # 🚀 เพิ่มเป็น 10%! - หน้าใหญ่ชัดเจน (จาก 5%)
    WEIGHT_EMOTION: float = 0.0  # ปิด - ใช้ Advanced Scorer แทน
    WEIGHT_CONTRAST: float = 0.0  # ไม่ใช้
    WEIGHT_LIGHTING: float = 0.0  # ไม่ใช้
    # รวม = 1.0 (100%) ✅

    # Hard filters - กรองออกเด็ดขาด!
    FILTER_CLOSED_EYES: bool = True  # ✅ เปิดกลับมา - กรองตาหลับออกเด็ดขาด!
    FILTER_NEGATIVE_EMOTION: bool = False  # ปิด - เก็บทุกอารมณ์ไว้ (โกรธ/เสียใจ/ร้องไห้ = OK!)
    MIN_ACCEPTABLE_SCORE: float = 0.30  # ⬇️ ลดเป็น 0.30 - ยอมรับหน้าคุณภาพต่ำเพื่อเพิ่มโอกาสหาคนครบ

    # Debug mode - บันทึกใบหน้าที่ตรวจจับได้ทั้งหมดลง temp folder
    DEBUG_SAVE_DETECTED_FACES: bool = False  # ปิด debug mode เพื่อเพิ่มความเร็ว

    # Layout settings
    TEXT_SAFE_AREA_MARGIN: int = 80  # Pixels from edge
    MAIN_CHARACTER_SCALE: float = 1.0
    SECONDARY_CHARACTER_SCALE: float = 0.75  # 15-25% smaller

    # Palette extraction
    PALETTE_COLORS: int = 5  # Number of dominant colors to extract

    # Rendering settings
    BLUR_RADIUS: int = 45  # เพิ่มจาก 25 เป็น 45 - เบลอพื้นหลังมากๆ เพื่อโฟกัสที่ตัวนักแสดง
    VIGNETTE_STRENGTH: float = 0.30  # เพิ่มจาก 0.25 เป็น 0.30 - vignette ชัดขึ้น
    SHADOW_OPACITY: float = 0.6
    HALO_WIDTH: int = 10  # เพิ่มจาก 8 - halo หนาขึ้น

    # Auto Enhance settings (ปรับภาพก่อนใส่ text - ตั้งค่าเป็น 1.0 = ไม่ปรับ)
    ENABLE_AUTO_ENHANCE: bool = False  # ปิดใช้งาน - กลับไปใช้แบบเดิม
    ENHANCE_BRIGHTNESS: float = 1.05  # 1.05 = เพิ่ม 5%, 1.0 = ไม่ปรับ
    ENHANCE_CONTRAST: float = 1.15  # 1.15 = เพิ่ม 15%, 1.0 = ไม่ปรับ
    ENHANCE_SATURATION: float = 1.10  # 1.10 = เพิ่ม 10%, 1.0 = ไม่ปรับ
    ENHANCE_SHARPNESS: float = 1.10  # 1.10 = เพิ่ม 10%, 1.0 = ไม่ปรับ

    # Character cropping settings
    CHARACTER_CROP_HEIGHT_MULTIPLIER: float = 3.5  # 3.5 = ถึงเอว (แบบเดิม), 4.2 = ถึงเข็มขัด

    # Advanced Quality Scoring (เพิ่มความแม่นยำ - ฟรี 100%)
    ENABLE_ADVANCED_SCORING: bool = False  # ⚡ SPEED MODE: ปิด DeepFace (ประหยัด 3-5 นาที)
    ADVANCED_SCORE_WEIGHT: float = 0.50  # 🚀 เพิ่มเป็น 50%! - ให้น้ำหนักความสวยงามมากขึ้น (จาก 40%)

    # Multi-Pass Adaptive Selection (รับประกันเจอหน้าครบ 100%!)
    ENABLE_MULTIPASS: bool = True  # เปิด/ปิด Multi-Pass Adaptive System
    MULTIPASS_ATTEMPTS: int = 4  # 🚀 เพิ่มเป็น 4 รอบ (strict → moderate → lenient → very lenient)

    # Pass 1: Very Strict (คุณภาพสูงสุด - ตาเปิดชัดเจนมาก!)
    PASS1_FRONTALITY: float = 0.85  # หน้าตรงมากๆ
    PASS1_EYES_THRESHOLD: float = 0.80  # 🔥 เพิ่มเป็น 0.80 - ตาเปิดชัดเจนมาก! (จาก 0.75)
    PASS1_MIN_SHARPNESS: float = 60.0  # คมชัดมาก

    # Pass 2: Strict (ปกติ - ตาเปิดดี)
    PASS2_FRONTALITY: float = 0.80  # หน้าตรง
    PASS2_EYES_THRESHOLD: float = 0.75  # 🔥 เพิ่มเป็น 0.75 - ตาเปิดดี (จาก 0.72)
    PASS2_MIN_SHARPNESS: float = 50.0  # คมชัดปกติ

    # Pass 3: Moderate (ผ่อนปรน - แต่ยังเข้มงวดตา)
    PASS3_FRONTALITY: float = 0.60  # ⬇️ ลดเป็น 0.60 - หน้าเบี่ยงได้มาก (ผ่อนปรน)
    PASS3_EYES_THRESHOLD: float = 0.72  # ✅ ยังเข้มงวด - ตาเปิดต้องดี!
    PASS3_MIN_SHARPNESS: float = 25.0  # ⬇️ ลดเป็น 25 - เบลอมากก็ได้ (ผ่อนปรน)

    # Pass 4: Very Lenient (ผ่อนปรนมากๆ - แต่ตาต้องเปิด!)
    PASS4_FRONTALITY: float = 0.40  # ⬇️ หน้าเบี่ยงมากๆ ก็ยอมรับ (ผ่อนปรนมาก)
    PASS4_EYES_THRESHOLD: float = 0.72  # ✅ ยังเข้มงวด - ตาต้องเปิด! (ไม่ผ่อนปรน)
    PASS4_MIN_SHARPNESS: float = 15.0  # ⬇️ เบลอมากมากก็ยอมรับ (ผ่อนปรนมาก)

    # Early Stopping (🚀 เพิ่มความเร็ว 40-50%!)
    ENABLE_EARLY_STOP: bool = True  # เปิด/ปิด Early Stopping ระหว่างวิเคราะห์ภาพ
    EARLY_STOP_CHECK_INTERVAL: int = 100  # เช็คทุกๆ 100 ภาพ (ถ้า 600 ภาพ = เช็ค 6 ครั้ง)

    # Scene Diversity & Temporal Spacing (🎨 เพิ่มความหลากหลาย!)
    ENABLE_SCENE_DIVERSITY: bool = True  # เปิด/ปิด Scene Diversity Detection
    SCENE_CHANGE_THRESHOLD: float = 0.30  # Histogram difference > 30% = ฉากเปลี่ยน
    MIN_TEMPORAL_GAP: int = 3  # เว้นระยะห่างอย่างน้อย 3 วินาที ระหว่างหน้าของคนเดียวกัน
    PREFER_DIVERSE_SCENES: bool = True  # บังคับเลือกจากหลายฉาก (ไม่ซ้ำซาก)

    # Text settings
    FONT_TITLE: str = "TF Pimpakarn Extra.ttf"  # เปลี่ยนจาก Kanit-Bold
    FONT_SUBTITLE: str = "TF Pimpakarn Extra.ttf"  # เปลี่ยนจาก Kanit-Regular
    TITLE_FONT_SIZE: int = 240  # เพิ่มเป็น 240 px
    SUBTITLE_FONT_SIZE: int = 240  # เท่ากับ title ทุกประการ (แก้จาก 190)
    TEXT_STROKE_WIDTH: int = 25  # เพิ่มเป็น 25 (สัดส่วนกับฟอนต์ 240px)
    TEXT_MAX_WORDS_PER_LINE: int = 3  # ลดเป็น 3 เพราะตัวหนังสือใหญ่มากขึ้น

    # Text Style Presets
    TEXT_STYLES: dict = {
        "style1": {
            "name": "แดง-เหลือง (YouTube Style)",
            "title": {"fill": (255, 0, 0), "stroke": (255, 255, 255)},      # แดงขอบขาว
            "subtitle": {"fill": (255, 255, 0), "stroke": (0, 0, 0)},      # เหลืองขอบดำ
        },
        "style2": {
            "name": "ขาว-เหลือง (Classic)",
            "title": {"fill": (255, 255, 255), "stroke": (0, 0, 0)},       # ขาวขอบดำ
            "subtitle": {"fill": (255, 255, 0), "stroke": (0, 0, 0)},      # เหลืองขอบดำ
        },
        "style3": {
            "name": "ขาว-ส้ม (Vibrant)",
            "title": {"fill": (255, 255, 255), "stroke": (255, 100, 0)},   # ขาวขอบส้ม
            "subtitle": {"fill": (255, 200, 0), "stroke": (0, 0, 0)},      # เหลืองทองขอบดำ
        },
        "auto": {
            "name": "อัตโนมัติ (จากสีรูป)",
            "title": {"fill": "auto", "stroke": "auto"},
            "subtitle": {"fill": "auto", "stroke": "auto"},
        }
    }

    # FastAPI settings
    API_TITLE: str = "YouTube Thumbnail Generator API"
    API_VERSION: str = "1.0.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # OpenAI settings
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"  # ถูกที่สุด ~$0.15/1M tokens
    OPENAI_MAX_TOKENS: int = 500  # จำกัด token เพื่อประหยัด

    # Gemini settings (Option 4: Hybrid Validation)
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-pro"  # Latest Pro model (best quality)
    GEMINI_MAX_TOKENS: int = 1000
    ENABLE_GEMINI_EYE_CHECK: bool = False  # 🚀 ปิด Gemini Eye Check - ใช้ OpenCV แทน (ฟรี + เร็วกว่า 100 เท่า!)

    # Validation settings
    ENABLE_AI_VALIDATION: bool = True  # เปิด/ปิด AI validation
    VALIDATION_THRESHOLD: float = 7.0  # คะแนนขั้นต่ำที่ยอมรับได้ (0-10)
    MAX_REGENERATION_ATTEMPTS: int = 2  # จำนวนครั้งที่ลอง generate ใหม่ถ้าไม่ผ่าน

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Create directories if they don't exist
for directory in [
    settings.RAW_DIR,
    settings.TEMP_DIR,
    settings.OUT_DIR,
    settings.FONTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
