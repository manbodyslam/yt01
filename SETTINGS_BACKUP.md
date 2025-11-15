# 📝 บันทึกค่าการตั้งค่าทั้งหมด - YouTube Thumbnail Generator

**วันที่บันทึก:** 2025-11-15
**Commit ปัจจุบัน:** `f912707` - "📨 ปรับปรุง error message สำหรับ n8n"
**สถานะ:** ✅ ใช้งานได้ปกติ (ก่อนแก้ไขที่มีปัญหา)

---

## 🎯 Preset Configuration

### Preset 1: แบบครึ่งตัว (หัว-เอว) - YouTube
```json
{
  "id": "1",
  "name": "แบบครึ่งตัว (หัว-เอว)",
  "description": "แสดงตัวละครตั้งแต่หัวถึงเอว - เหมาะสำหรับ YouTube",
  "num_characters": 3,
  "crop_point": "waist",
  "layout_type": "tri_hero",
  "vertical_align": "top"
}
```

**ค่าที่ใช้:**
- `CHARACTER_CROP_HEIGHT_MULTIPLIER`: 3.5
- `crop_width`: face_h * 2.0
- `crop_offset`: 0.25 (face อยู่ที่ 25% จากบน)
- `scale`: 1.15 (115% ของ canvas height)
- `vertical_align`: "top"
- `top_margin`: 85px

**ผลลัพธ์:**
- เห็นหัว→เอว
- หัวห่างจากขอบบน 85px
- ตัวละครสูง ~1242px (1080 * 1.15)

---

### Preset 2: แบบเต็มตัว (หัว-เท้า) - YouTube Shorts
```json
{
  "id": "2",
  "name": "แบบเต็มตัว (หัว-เท้า)",
  "description": "แสดงตัวละครเต็มตัวตั้งแต่หัวถึงเท้า ชิดขอบล่าง - เหมาะสำหรับ YouTube Shorts",
  "num_characters": 3,
  "crop_point": "full_body",
  "layout_type": "tri_hero",
  "vertical_align": "bottom"
}
```

**ค่าที่ใช้ (ในโค้ดปัจจุบัน):**
- `CHARACTER_CROP_HEIGHT_MULTIPLIER`: 6.5 (ใน CROP_MULTIPLIERS)
- `crop_width`: face_h * 2.0
- `crop_offset`: 0.25
- `scale`: 1.15
- `vertical_align`: "bottom"

**ผลลัพธ์:**
- เห็นหัว→เท้า
- เท้าชิดขอบล่าง
- ตัวละครสูง ~1242px

**⚠️ ปัญหาที่พบ:** multiplier 6.5 ทำให้ตัวเล็ก

---

## 📐 Layout Configuration (tri_hero)

**ตำแหน่งตัวละคร 3 คน:**

```python
# ตัวซ้าย
position: x = 20% (384px), y = 0
scale: 1.15
z_index: 9

# ตัวกลาง
position: x = 50% (960px), y = 0
scale: 1.15
z_index: 10 (อยู่หน้าสุด)

# ตัวขวา
position: x = 80% (1536px), y = 0
scale: 1.15
z_index: 9
```

---

## 🎨 Character Rendering Settings

### Crop Settings (ตอน crop ตัวละคร):
```python
# ค่าพื้นฐาน
CHARACTER_CROP_HEIGHT_MULTIPLIER = 3.5  # หัว→เอว

# ตัวกลาง vs ตัวข้าง (ในโค้ดเดิม):
if is_center:
    crop_width = face_h * 1.8  # แคบ
else:
    crop_width = face_h * 2.4  # กว้าง

# Crop offset
crop_y1 = face_center_y - int(crop_height * 0.38)  # หน้าอยู่ 38% จากบน
```

**ตัวอย่างการคำนวณ (face_h = 150px):**
```
crop_height = 150 * 3.5 = 525px
crop_width (center) = 150 * 1.8 = 270px
crop_width (side) = 150 * 2.4 = 360px
crop_y1 = face_center_y - (525 * 0.38) = face_center_y - 199.5px
```

### Resize Settings:
```python
# คำนวณ target size
target_h = canvas_height * placement.scale  # 1080 * 1.15 = 1242px
aspect_ratio = character_img.width / character_img.height
target_w = target_h * aspect_ratio

# ตัวอย่าง (crop 270x525):
aspect_ratio = 270/525 = 0.514
target_w = 1242 * 0.514 = 638px
```

### Position Settings:
```python
# แนวนอน (X): ตรงกลาง
paste_x = placement.position.x - new_w // 2

# แนวตั้ง (Y):
if vertical_align == "bottom":
    paste_y = canvas.height - new_h  # ชิดล่าง
else:  # "top"
    top_margin = 85
    paste_y = placement.position.y - head_top_scaled + top_margin
```

---

## 🔢 Face Detection Settings

```python
# Model
FACE_DETECTION_MODEL = "buffalo_s"  # เร็ว, แม่นยำ 95%

# Thresholds (ผ่อนปรนเพื่อหาคนได้มากขึ้น)
FACE_CONFIDENCE_THRESHOLD = 0.35  # ลดจาก 0.45
MIN_FACE_SIZE = 40  # ลดจาก 50

# ONNX Optimization
ONNX_ENABLE_OPTIMIZATION = True
ONNX_NUM_THREADS = 4
ONNX_EXECUTION_MODE = "sequential"
```

---

## 🎯 Character Selection Settings

```python
# Force 3 Characters
desired_counts = [3]  # บังคับ 3 คนเสมอ
allow_duplicates = True  # ยอมรับซ้ำได้

# Rejection Rule
if num_found < 2:
    return None  # ปฏิเสธ! ต้องมีอย่างน้อย 2 คน

# Duplication
if num_found == 2:
    # Duplicate คนที่ดีที่สุดเป็นคนที่ 3
    duplicate_key = f"{best_char_key}_dup"
    chars[duplicate_key] = chars[best_char_key].copy()
```

---

## 📊 Output Settings

```python
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
OUTPUT_FORMAT = "jpg"
OUTPUT_QUALITY = 95
```

---

## 🎬 Video Processing Settings

```python
# Frame Extraction (SPEED MODE)
VIDEO_FRAMES_PER_MINUTE = 15  # ทุก 4 วินาที
VIDEO_MAX_FRAMES = 150  # สูงสุด 150 frames
VIDEO_BATCH_SIZE = 50  # Process เป็น batch

# Smart Frame Selection
ENABLE_SMART_FRAME_SELECTION = True
FRAMES_PER_SCENE = 10
CANDIDATE_FRAME_INTERVAL = 0.4  # วินาที

# Quality Weights
SMART_WEIGHT_EYES_OPEN = 0.60  # 60%
SMART_WEIGHT_SHARPNESS = 0.20  # 20%
SMART_WEIGHT_FRONTALITY = 0.15  # 15%
SMART_WEIGHT_FACE_SIZE = 0.05  # 5%

# Thresholds
SMART_MIN_EYES_OPEN_SCORE = 0.50
SMART_MIN_FRONTALITY_SCORE = 0.55
```

---

## 🔄 Clustering Settings

```python
CLUSTERING_ALGORITHM = "DBSCAN"
DBSCAN_EPS = 0.48  # รวมหน้าคนเดียวกันเข้าด้วยกัน
DBSCAN_MIN_SAMPLES = 1  # ยอมรับหน้าเดี่ยว
VIDEO_SIMILARITY_THRESHOLD = 0.70  # ยอมรับใบหน้าต่างกันมากขึ้น
```

---

## 📝 ประวัติการแก้ไขที่สำคัญ

### ✅ Commits ที่ใช้งานได้ (ปัจจุบัน):
- `f912707` - error message n8n
- `a150aa0` - FORCE 3 CHARACTERS
- `9f0d349` - parameter fixes
- `095ee2c` - COMPLETE REFACTOR

### ❌ Commits ที่ถูกลบออก (มีปัญหา):
- `334651b` - Preset 2 ใช้ค่าเดียวกับ Preset 1 (คำนวณผิด)
- `7c87aaf` - FIX COMPLETE (ตำแหน่งผิด)
- `f25235e` - Face-Size Based Scaling (ซับซ้อนเกินไป)
- `f1140bb` - แก้หัวขาด (crop_offset 0.38→0.25)
- `837cf7c` - เพิ่มขนาดตัวละคร (scale 1.15→1.50)
- `817919e` - Normalize character crop
- `b4fb87a` - FIX Preset 2

---

## 🚨 ปัญหาที่พบและวิธีแก้

### ปัญหา 1: หัวขาด
**สาเหตุ:** crop_offset = 0.38 → พื้นที่ด้านบนน้อยเกินไป
**วิธีแก้ที่ลอง:** ลดเป็น 0.25 (ใน commit ที่ถูกลบ)
**สถานะ:** ยังไม่ได้แก้ในโค้ดปัจจุบัน (ใช้ 0.38 อยู่)

### ปัญหา 2: ตัวละครขนาดไม่เท่ากัน
**สาเหตุ:** crop_width ต่างกัน (center: 1.8x, side: 2.4x)
**วิธีแก้ที่ลอง:** ใช้ normalized_face_h + crop_width เท่ากัน
**สถานะ:** ยังไม่ได้แก้ในโค้ดปัจจุบัน

### ปัญหา 3: Preset 2 ตัวเล็ก
**สาเหตุ:** multiplier = 6.5 ทำให้ crop area ใหญ่เกินไป
**วิธีแก้ที่ต้องการ:** ใช้ค่าเดียวกับ Preset 1 (3.5) + vertical_align = "bottom"
**สถานะ:** ยังไม่ได้แก้ในโค้ดปัจจุบัน

---

## 💡 แนวทางการแก้ไขในอนาคต

### สำหรับ Preset 2:
1. เปลี่ยน `CROP_MULTIPLIERS["full_body"]` จาก 6.5 → 3.5
2. ใช้ `vertical_align = "bottom"` (มีอยู่แล้ว)
3. ผลลัพธ์: เห็นหัว→เอว เหมือน Preset 1, เอวลงไปหลุดจอเอง

### สำหรับการ normalize ขนาดตัวละคร:
1. ใช้ `normalized_face_h` สำหรับ crop ทุกคน
2. ตั้ง `crop_width` เท่ากันทุกคน (ไม่แยก center/side)
3. Resize ตาม placement.scale โดยตรง

### สำหรับแก้หัวขาด:
1. ลด `crop_offset` จาก 0.38 → 0.20-0.25
2. หรือ เพิ่ม `top_margin` จาก 85 → 50-60px

---

## 📞 API Endpoints

### Health Check:
```
GET /health
```

### Process Async (n8n):
```
POST /process-async
Body: {
  "video_url": "https://drive.google.com/...",
  "webhook_url": "https://...",
  "preset_id": "1" หรือ "2"
}
```

### Task Status:
```
GET /tasks/{task_id}
```

---

**🔒 ไฟล์นี้เก็บไว้สำหรับอ้างอิง - อย่าลบ!**
