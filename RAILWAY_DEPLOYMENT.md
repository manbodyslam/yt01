# 🚀 Railway Deployment Guide

คู่มือการ deploy YouTube Thumbnail Generator บน Railway.app

---

## 📋 สิ่งที่ต้องเตรียม

1. บัญชี Railway (สมัครฟรีที่ https://railway.app)
2. GitHub account (เชื่อมกับ Railway)
3. Git repository ของโปรเจคนี้

---

## 🎯 ขั้นตอนการ Deploy

### 1. เตรียม Repository

```bash
# ตรวจสอบว่ามีไฟล์ที่จำเป็นครบหรือไม่
ls -la Procfile nixpacks.toml railway.json requirements.txt

# Commit ไฟล์ทั้งหมด
git add .
git commit -m "Prepare for Railway deployment"
git push origin main
```

### 2. สร้าง Project บน Railway

1. ไปที่ https://railway.app
2. คลิก **"New Project"**
3. เลือก **"Deploy from GitHub repo"**
4. เลือก repository ของคุณ
5. Railway จะเริ่ม build และ deploy อัตโนมัติ

### 3. ตั้งค่า Environment Variables (ถ้าจำเป็น)

ไปที่ Project Settings → Variables และเพิ่ม:

```
# ไม่ต้องตั้ง PORT (Railway จะตั้งให้อัตโนมัติ)

# Optional: ถ้าใช้ OpenAI
OPENAI_API_KEY=sk-...

# Optional: ถ้าใช้ Google Gemini
GEMINI_API_KEY=...
```

### 4. รอการ Deploy เสร็จสมบูรณ์

- Build time: ~5-10 นาที (ครั้งแรก)
- Deploy time: ~2-3 นาที
- Total: ~7-13 นาที

### 5. ตรวจสอบว่า Deploy สำเร็จ

```bash
# เช็ค health endpoint
curl https://your-app.railway.app/health

# Expected response:
{
  "status": "healthy",
  "service": "YouTube Thumbnail Generator",
  "timestamp": "..."
}
```

---

## 🔧 การใช้งาน

### ทดสอบ API

```bash
# 1. ทดสอบ Async API (แนะนำ - ไม่มี timeout)
curl -X POST "https://your-app.railway.app/api/generate-async" \
  -F "video=@test.mp4" \
  -F "title=Test Title" \
  -F "subtitle=Test Subtitle"

# Response:
{
  "success": true,
  "task_id": "abc-123-xyz",
  "status_url": "/api/task-status/abc-123-xyz",
  "message": "Task started. Poll status_url every 5 seconds to check progress."
}

# 2. เช็ค status
curl "https://your-app.railway.app/api/task-status/abc-123-xyz"

# Response (processing):
{
  "task_id": "abc-123-xyz",
  "status": "detecting_faces",
  "progress": 50,
  "message": "Detecting faces..."
}

# Response (completed):
{
  "task_id": "abc-123-xyz",
  "status": "completed",
  "progress": 100,
  "result": {
    "success": true,
    "thumbnail_path": "/path/to/thumbnail.jpg",
    ...
  }
}
```

### ใช้กับ Google Drive

```bash
curl -X POST "https://your-app.railway.app/api/generate-async" \
  -F "google_drive_url=https://drive.google.com/file/d/YOUR_FILE_ID/view" \
  -F "title=EP.1 ตอนแรก" \
  -F "subtitle=เนื้อหาสุดพิเศษ"
```

---

## ⚙️ การตั้งค่าเพิ่มเติม

### เพิ่ม Custom Domain

1. ไปที่ Project Settings → Domains
2. คลิก **"Generate Domain"** (ได้ฟรี .railway.app subdomain)
3. หรือเพิ่ม Custom Domain ของคุณเอง

### ตั้งค่า Health Check

Railway จะ ping `/health` ทุก 5 นาทีอัตโนมัติ (ตั้งค่าใน `railway.json` แล้ว)

### ดู Logs

```bash
# ผ่าน Railway CLI
railway logs

# หรือดูผ่าน Web Dashboard
# Project → Deployments → View Logs
```

---

## 💰 การประหยัดต้นทุน

### 1. Scale to Zero (เปิดอยู่แล้ว)

Railway จะหยุด container อัตโนมัติเมื่อไม่มี request:
- Idle time: 5-10 นาที
- Cold start: ~20-30 วินาที
- **จ่ายเฉพาะเวลาที่ทำงาน**

### 2. Monitor Usage

```bash
# ดู usage ผ่าน CLI
railway variables

# หรือดูผ่าน Dashboard
# Project → Usage
```

### 3. ตั้ง Budget Limit

1. Project Settings → Usage
2. ตั้ง **Monthly Budget** (แนะนำ: $10-20/เดือน)
3. Railway จะแจ้งเตือนเมื่อใกล้ถึง limit

---

## 🐛 Troubleshooting

### ปัญหา: Build Failed

**สาเหตุ:** Dependencies ไม่สามารถติดตั้งได้

**แก้ไข:**
```bash
# เช็ค build logs
railway logs

# แก้ไข requirements.txt ถ้าจำเป็น
# แล้ว push ใหม่
git add requirements.txt
git commit -m "Fix dependencies"
git push
```

### ปัญหา: ffmpeg not found

**สาเหตุ:** System dependencies ไม่ได้ติดตั้ง

**แก้ไข:**
ตรวจสอบว่า `nixpacks.toml` มี:
```toml
[phases.setup]
nixPkgs = ["ffmpeg", "libGL"]
```

### ปัญหา: Timeout Error (500s)

**แก้ไข:**
ใช้ `/api/generate-async` แทน `/api/generate`:
- Return task_id ทันที (ไม่มี timeout)
- Poll `/api/task-status/{task_id}` เพื่อเช็ค progress

### ปัญหา: Memory Error (OOM)

**สาเหตุ:** ใช้ RAM เกิน 8GB

**แก้ไข:**
1. ตรวจสอบว่าไม่มี concurrent requests มากเกินไป
2. ลด `num_frames` จาก 325 → 150
3. Upgrade เป็น Railway Pro plan (32GB RAM)

### ปัญหา: Files Lost After Restart

**สาเหตุ:** Railway ใช้ ephemeral storage

**แก้ไข 1:** Return thumbnail เป็น Base64
```python
import base64

with open(thumbnail_path, "rb") as f:
    thumbnail_base64 = base64.b64encode(f.read()).decode()

return {"thumbnail_base64": thumbnail_base64}
```

**แก้ไข 2:** ใช้ Cloud Storage (Cloudflare R2 / AWS S3)

---

## 📊 Resource Limits

| Spec | Railway Hobby | Railway Pro |
|------|--------------|-------------|
| RAM | 8GB | 32GB |
| CPU | Shared | Dedicated |
| Timeout | 500s | Unlimited |
| Storage | Ephemeral | Ephemeral + Volume |
| Price | $5/GB-month | $10/GB-month |

**สำหรับแอปนี้:**
- RAM usage: ~3GB per task
- CPU: 100% single core (12-15 นาที)
- **Hobby plan เพียงพอ** สำหรับ 10 tasks/วัน

---

## 🔄 การ Update โปรเจค

```bash
# 1. แก้ไขโค้ด
# 2. Commit & Push
git add .
git commit -m "Update feature"
git push

# 3. Railway จะ auto-deploy ทันที!
```

---

## 📈 Monitoring & Analytics

### ดู Metrics

1. Railway Dashboard → Metrics
2. ดูได้:
   - CPU usage
   - Memory usage
   - Network traffic
   - Request count
   - Error rate

### Set up Alerts

1. Project Settings → Notifications
2. เพิ่ม Discord/Slack webhook
3. ตั้งค่า alerts:
   - Deploy success/failure
   - High memory usage
   - Error threshold

---

## 🎯 Best Practices

### 1. ใช้ Async API เสมอ
```bash
# ✅ ดี - ไม่มี timeout
POST /api/generate-async

# ❌ ไม่ดี - มี timeout 500s
POST /api/generate
```

### 2. Cleanup Temporary Files
```python
import os
import atexit

def cleanup():
    # ลบไฟล์ temp หลังเสร็จ
    for file in temp_files:
        os.remove(file)

atexit.register(cleanup)
```

### 3. Monitor Usage
- เช็ค usage ทุกสัปดาห์
- ตั้ง budget alert
- Optimize ถ้าใช้เกิน budget

### 4. Use CDN สำหรับ Static Files
- ใช้ Cloudflare CDN สำหรับ `/static`
- ลด bandwidth cost

---

## 💡 สรุป

### ข้อดีของ Railway:
✅ Scale to Zero → จ่ายตามใช้
✅ Auto-deploy on git push
✅ ง่ายที่สุด ไม่ต้องจัดการ server
✅ Built-in HTTPS + Domain
✅ ราคาถูก (~400 บาท/เดือน สำหรับ 10 tasks/วัน)

### ข้อควรระวัง:
⚠️ Ephemeral storage (ต้องใช้ S3/R2 หรือ return base64)
⚠️ Cold start ~20-30s (ไม่เป็นปัญหาถ้าใช้ async API)
⚠️ 500s timeout (แก้ได้ด้วย async API)

---

## 🆘 ต้องการความช่วยเหลือ?

- Railway Docs: https://docs.railway.app
- Railway Discord: https://discord.gg/railway
- Project Issues: [GitHub Issues]

---

**เอกสารนี้อัปเดตล่าสุด:** 14 พฤศจิกายน 2025
