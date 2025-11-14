# Google Drive Integration - คู่มือการใช้งาน

## ภาพรวม

ระบบรองรับการสร้าง thumbnail จาก Google Drive โดยตรง ไม่ต้อง download วิดีโอมาก่อน

### ฟีเจอร์หลัก:
- ดาวน์โหลดวิดีโอจาก Google Drive โดยอัตโนมัติ
- ไม่ต้องใช้ Google API key หรือ authentication
- รองรับไฟล์ใหญ่ (จัดการ virus scan warning อัตโนมัติ)
- ใช้งานง่าย แค่ส่ง share link มาได้เลย

---

## วิธีการใช้งาน

### Endpoint ใหม่: `/api/generate`

Endpoint นี้รับทั้ง:
1. **Video file upload** (แบบเดิม)
2. **Google Drive URL** (แบบใหม่)

---

## 1. ส่ง Google Drive URL

### ตัวอย่าง Basic:

```bash
curl -X POST "http://localhost:8000/api/generate" \
  -F "google_drive_url=https://drive.google.com/file/d/YOUR_FILE_ID/view" \
  -F "title=หัวข้อบน" \
  -F "subtitle=หัวข้อล่าง"
```

### ตัวอย่างแบบเต็ม (พร้อม parameters):

```bash
curl -X POST "http://localhost:8000/api/generate" \
  -F "google_drive_url=https://drive.google.com/file/d/1abc123xyz/view" \
  -F "title=หัวข้อบน" \
  -F "subtitle=หัวข้อล่าง" \
  -F "num_characters=3" \
  -F "num_frames=150" \
  -F "text_style=style1" \
  -F "layout_type=tri_hero"
```

### ผ่าน Cloudflare Tunnel:

```bash
curl -X POST "https://workshops-mime-donna-seeks.trycloudflare.com/api/generate" \
  -F "google_drive_url=https://drive.google.com/file/d/YOUR_FILE_ID/view" \
  -F "title=หัวข้อบน" \
  -F "subtitle=หัวข้อล่าง"
```

---

## 2. Upload Video File (แบบเดิม)

```bash
curl -X POST "http://localhost:8000/api/generate" \
  -F "video=@/path/to/video.mp4" \
  -F "title=หัวข้อบน" \
  -F "subtitle=หัวข้อล่าง"
```

---

## Parameters ที่รองรับ

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `google_drive_url` | string | ใช่* | - | Google Drive share URL |
| `video` | file | ใช่* | - | Video file upload |
| `title` | string | ใช่ | - | หัวข้อบน |
| `subtitle` | string | ไม่ | null | หัวข้อล่าง |
| `num_characters` | int | ไม่ | 3 | จำนวนตัวละคร (บังคับ 3) |
| `num_frames` | int | ไม่ | 150 | จำนวนเฟรมที่ดึง |
| `text_style` | string | ไม่ | style1 | style1, style2, style3, auto |
| `layout_type` | string | ไม่ | tri_hero | tri_hero, tri_pyramid, tri_staggered |

**หมายเหตุ:** ต้องระบุ `google_drive_url` **หรือ** `video` อย่างใดอย่างหนึ่ง (ไม่ใช่ทั้งคู่)

---

## ข้อกำหนดสำหรับ Google Drive

### 1. ไฟล์ต้อง Share เป็น Public

**ขั้นตอน:**
1. เปิดไฟล์ใน Google Drive
2. คลิก "Share" หรือ "แชร์"
3. เลือก "Anyone with the link" หรือ "ทุกคนที่มีลิงก์"
4. ตั้งค่าเป็น "Viewer" หรือ "ผู้ดู"
5. คัดลอก link

### 2. URL Formats ที่รองรับ

```
✅ https://drive.google.com/file/d/FILE_ID/view
✅ https://drive.google.com/file/d/FILE_ID/view?usp=sharing
✅ https://drive.google.com/open?id=FILE_ID
✅ https://drive.google.com/uc?id=FILE_ID
```

### 3. ข้อจำกัด

- ไฟล์ต้องเป็น **public** ("Anyone with the link")
- ไฟล์ขนาดใหญ่อาจใช้เวลาดาวน์โหลดนานขึ้น (timeout: 10 นาที)
- รองรับไฟล์ทุกขนาด (จัดการ virus scan warning อัตโนมัติ)

---

## Response Format

### สำเร็จ:

```json
{
  "success": true,
  "thumbnail_path": "/Users/winai/Desktop/code_x/yt01/workspace/out/thumbnail_20231113_123456.jpg",
  "filename": "thumbnail_20231113_123456.jpg",
  "metadata": {
    "title": "หัวข้อบน",
    "subtitle": "หัวข้อล่าง",
    "num_characters": 3,
    "layout_type": "tri_hero",
    "text_style": "style1",
    ...
  }
}
```

### ผิดพลาด:

```json
{
  "success": false,
  "error": "ไม่สามารถดาวน์โหลดได้ - ไฟล์อาจไม่เป็น public\nกรุณาตั้งค่าไฟล์เป็น 'Anyone with the link' ใน Google Drive"
}
```

---

## ตัวอย่างการใช้งานกับ n8n

### Workflow:

1. **Google Drive Trigger** → เมื่อมีไฟล์วิดีโอใหม่
2. **Get Share Link** → ดึง share link ของไฟล์
3. **HTTP Request** → ส่ง request ไปที่ `/api/generate`
4. **รับ Thumbnail กลับมา** → เก็บหรืออัปโหลดต่อ

### n8n HTTP Request Node:

```json
{
  "method": "POST",
  "url": "https://workshops-mime-donna-seeks.trycloudflare.com/api/generate",
  "bodyParameters": {
    "parameters": [
      {
        "name": "google_drive_url",
        "value": "={{ $json.webViewLink }}"
      },
      {
        "name": "title",
        "value": "หัวข้อบน"
      },
      {
        "name": "subtitle",
        "value": "หัวข้อล่าง"
      }
    ]
  }
}
```

---

## Technical Details

### การทำงานภายใน:

1. **Extract File ID** จาก Google Drive URL
2. **Convert** เป็น direct download URL: `https://drive.google.com/uc?export=download&id=FILE_ID`
3. **Download** ไฟล์ผ่าน httpx (timeout: 10 minutes)
4. **จัดการ Virus Scan Warning** สำหรับไฟล์ใหญ่ (>100MB) อัตโนมัติ
5. **Extract frames** จากวิดีโอที่ดาวน์โหลดมา
6. **Generate thumbnail** ตามปกติ

### Security:

- ไม่เก็บ credentials ใดๆ
- ไม่ใช้ Google API key
- ดาวน์โหลดเฉพาะไฟล์ที่ share เป็น public
- Timeout protection (10 minutes)

---

## Troubleshooting

### ปัญหา: "ไม่สามารถดาวน์โหลดได้"

**สาเหตุ:** ไฟล์ไม่ได้ share เป็น public

**วิธีแก้:**
1. เปิดไฟล์ใน Google Drive
2. คลิก "Share"
3. เปลี่ยนเป็น "Anyone with the link"

### ปัญหา: "Invalid Google Drive URL"

**สาเหตุ:** URL ไม่ถูกต้องหรือไม่ใช่ Google Drive link

**วิธีแก้:**
- ตรวจสอบว่า URL เป็นรูปแบบ `https://drive.google.com/file/d/.../view`
- ลอง copy link ใหม่จาก Google Drive

### ปัญหา: "Download timeout"

**สาเหตุ:** ไฟล์ใหญ่เกินไป หรือ internet ช้า

**วิธีแก้:**
- ลองใหม่อีกครั้ง
- อัปโหลดไฟล์ที่เล็กกว่า
- ตรวจสอบความเร็ว internet

---

## Examples

### ตัวอย่างที่ 1: Basic Usage

```bash
curl -X POST "http://localhost:8000/api/generate" \
  -F "google_drive_url=https://drive.google.com/file/d/1abc123xyz/view" \
  -F "title=EP.1 ตอนแรก" \
  -F "subtitle=เนื้อหาสุดพิเศษ"
```

### ตัวอย่างที่ 2: Custom Parameters

```bash
curl -X POST "http://localhost:8000/api/generate" \
  -F "google_drive_url=https://drive.google.com/file/d/1abc123xyz/view" \
  -F "title=EP.2 ตอนที่สอง" \
  -F "subtitle=ดูกันให้จุใจ" \
  -F "text_style=style2" \
  -F "layout_type=tri_pyramid"
```

### ตัวอย่างที่ 3: ดึง Thumbnail ที่สร้างมา

```bash
# 1. สร้าง thumbnail
RESPONSE=$(curl -X POST "http://localhost:8000/api/generate" \
  -F "google_drive_url=https://drive.google.com/file/d/1abc123xyz/view" \
  -F "title=EP.3" \
  -F "subtitle=ตอนสุดท้าย")

# 2. ดึงชื่อไฟล์
FILENAME=$(echo $RESPONSE | jq -r '.filename')

# 3. ดาวน์โหลด thumbnail
curl "http://localhost:8000/thumbnail/$FILENAME" -o output.jpg
```

---

## เปรียบเทียบ 2 วิธี

| Feature | Video Upload | Google Drive URL |
|---------|--------------|------------------|
| ความเร็ว | เร็ว (ถ้าไฟล์เล็ก) | ช้ากว่า (ต้องดาวน์โหลดก่อน) |
| ขนาดไฟล์ | จำกัดโดย server | ไม่จำกัด (timeout 10 min) |
| ความสะดวก | ต้องมีไฟล์ในเครื่อง | แค่ copy link |
| Automation | ยากกว่า | ง่ายกว่า (ใช้กับ n8n) |
| Authentication | ไม่ต้อง | ไม่ต้อง (ถ้า public) |

---

## สรุป

### ข้อดีของ Google Drive Integration:

1. ไม่ต้อง download วิดีโอมาก่อน
2. ใช้งานง่าย แค่ copy link
3. เหมาะกับ automation (n8n, Zapier)
4. ไม่ต้อง setup Google API
5. รองรับไฟล์ใหญ่

### Use Cases:

- **n8n Automation:** ดึงวิดีโอจาก Google Drive → สร้าง thumbnail อัตโนมัติ
- **Workflow:** แชร์วิดีโอบน Drive → ระบบสร้าง thumbnail ให้อัตโนมัติ
- **Team Collaboration:** ทีมอัปโหลด Drive → admin สร้าง thumbnail

---

**เอกสารนี้อัปเดตล่าสุด:** 13 พฤศจิกายน 2025
