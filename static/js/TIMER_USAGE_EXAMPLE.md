# 📝 Timer Usage Example

## วิธีใช้ Timer ใน app.js

### 1. เมื่อกดปุ่ม "สร้าง Thumbnail" (Video Tab)

```javascript
// ใน app.js - event listener สำหรับปุ่ม generate-from-video-btn

document.getElementById('generate-from-video-btn').addEventListener('click', async () => {
    // 1. เริ่ม Timer
    videoTimer.start();

    // แสดง progress bar
    document.getElementById('video-progress').classList.add('show');

    try {
        // 2. ขั้นตอนที่ 1: อัปโหลดวิดีโอ (0-20%)
        videoTimer.updateStep(
            'อัปโหลดวิดีโอ',
            10,
            'กำลังอัปโหลดไฟล์วิดีโอไปยัง server...'
        );

        // อัปโหลดวิดีโอ
        const uploadResponse = await fetch('/api/upload-video', {
            method: 'POST',
            body: formData
        });

        // 3. ขั้นตอนที่ 2: ดึงเฟรม (20-50%)
        videoTimer.updateStep(
            'ดึงเฟรมจากวิดีโอ',
            30,
            'กำลังวิเคราะห์วิดีโอและดึงเฟรมที่สำคัญ (325 frames)...'
        );

        // รอ API ดึงเฟรม
        await pollForFrameExtraction();

        // 4. ขั้นตอนที่ 3: ตรวจจับใบหน้า (50-70%)
        videoTimer.updateStep(
            'ตรวจจับใบหน้า',
            60,
            'กำลังใช้ AI ตรวจจับใบหน้าและประเมินคุณภาพ...'
        );

        // รอ API ตรวจจับใบหน้า
        await pollForFaceDetection();

        // 5. ขั้นตอนที่ 4: จัดกลุ่มตัวละคร (70-85%)
        videoTimer.updateStep(
            'จัดกลุ่มตัวละคร',
            75,
            'กำลังจัดกลุ่มใบหน้าเป็นตัวละครแต่ละคน...'
        );

        // รอ API clustering
        await pollForClustering();

        // 6. ขั้นตอนที่ 5: สร้าง Thumbnail (85-95%)
        videoTimer.updateStep(
            'สร้าง Thumbnail',
            90,
            'กำลังสร้าง layout และประกอบ thumbnail...'
        );

        // เรียก API สร้าง thumbnail
        const result = await generateThumbnail();

        // 7. เสร็จสมบูรณ์ (100%)
        videoTimer.updateStep(
            'เสร็จสมบูรณ์!',
            100,
            'สร้าง thumbnail สำเร็จ!'
        );

        // หยุด timer พร้อมแสดงสรุป
        videoTimer.finish(true);

        // แสดงผลลัพธ์
        showResult(result);

    } catch (error) {
        // กรณีเกิด error
        videoTimer.finish(false);
        showError(error);
    }
});
```

### 2. สำหรับ Batch Generation

```javascript
// ใน app.js - event listener สำหรับปุ่ม generate-batch-btn

document.getElementById('generate-batch-btn').addEventListener('click', async () => {
    // 1. เริ่ม Timer
    batchTimer.start();

    document.getElementById('batch-progress').classList.add('show');

    try {
        // ขั้นตอนต่างๆ คล้ายกับ video แต่มีหลาย thumbnails

        batchTimer.updateStep(
            'ดึงเฟรมจากวิดีโอ',
            20,
            'กำลังดึงเฟรม 347 frames จากวิดีโอ...'
        );

        // ... polling for progress ...

        batchTimer.updateStep(
            'สร้าง Thumbnail 1/5',
            50,
            'กำลังสร้าง thumbnail แบบที่ 1 จาก 5...'
        );

        // อัปเดตแบบ realtime สำหรับแต่ละ thumbnail
        for (let i = 1; i <= 5; i++) {
            const percentage = 50 + (i * 10); // 50, 60, 70, 80, 90
            batchTimer.updateStep(
                `สร้าง Thumbnail ${i}/5`,
                percentage,
                `กำลังสร้าง thumbnail แบบที่ ${i} จาก 5...`
            );

            await generateThumbnailVariant(i);
        }

        batchTimer.updateStep('เสร็จสมบูรณ์!', 100, 'สร้าง 5 thumbnails สำเร็จ!');
        batchTimer.finish(true);

        showBatchResults(results);

    } catch (error) {
        batchTimer.finish(false);
        showError(error);
    }
});
```

### 3. ขั้นตอนแนะนำสำหรับแต่ละ Phase

#### Video Generation (Typical Flow)

| ขั้นตอน | % | คำอธิบาย | เวลาโดยประมาณ |
|---------|---|----------|--------------|
| อัปโหลดวิดีโอ | 0-10% | อัปโหลดไฟล์ไปยัง server | 5-30s |
| ดึงเฟรม | 10-40% | PySceneDetect + Frame extraction | 2-5m |
| ตรวจจับใบหน้า | 40-65% | InsightFace face detection | 1-3m |
| จัดกลุ่มตัวละคร | 65-80% | DBSCAN clustering | 10-30s |
| เลือกใบหน้าที่ดีที่สุด | 80-90% | Face scoring & selection | 10-20s |
| สร้าง Thumbnail | 90-100% | Layout + Rendering | 5-10s |

#### Batch Generation

| ขั้นตอน | % | คำอธิบาย |
|---------|---|----------|
| ดึงเฟรม | 0-30% | เหมือน video (ทำครั้งเดียว) |
| ตรวจจับใบหน้า | 30-50% | เหมือน video (ทำครั้งเดียว) |
| สร้าง Thumbnail 1 | 50-60% | สร้าง variant แรก |
| สร้าง Thumbnail 2 | 60-70% | สร้าง variant ที่ 2 |
| สร้าง Thumbnail 3 | 70-80% | สร้าง variant ที่ 3 |
| สร้าง Thumbnail 4 | 80-90% | สร้าง variant ที่ 4 |
| สร้าง Thumbnail 5 | 90-100% | สร้าง variant ที่ 5 |

### 4. ฟังก์ชัน Polling (สำหรับ Backend ที่ทำงานแบบ async)

```javascript
// ตัวอย่าง polling function ถ้า backend ส่ง progress มา

async function pollForProgress(endpoint) {
    while (true) {
        const response = await fetch(endpoint);
        const data = await response.json();

        if (data.status === 'completed') {
            return data.result;
        }

        if (data.status === 'error') {
            throw new Error(data.error);
        }

        // อัปเดต progress จาก backend
        if (data.progress) {
            videoTimer.updateStep(
                data.current_step,
                data.progress_percentage,
                data.description
            );
        }

        // รอ 2 วินาที แล้ว poll อีกครั้ง
        await new Promise(resolve => setTimeout(resolve, 2000));
    }
}
```

### 5. Timer API Reference

```javascript
// สร้าง timer instance
const myTimer = new ProcessTimer('prefix'); // 'video' หรือ 'batch'

// เริ่ม timer
myTimer.start();

// อัปเดตขั้นตอน (พร้อม % และคำอธิบาย)
myTimer.updateStep(
    'ชื่อขั้นตอน',      // string
    50,                 // number (0-100) หรือ null
    'คำอธิบายเพิ่มเติม' // string หรือ null
);

// อัปเดตแค่ progress bar (ไม่เปลี่ยนขั้นตอน)
myTimer.updateProgressBar(75);

// หยุด timer
myTimer.stop();

// เสร็จสิ้น (แสดงสรุป)
myTimer.finish(true);  // true = สำเร็จ, false = ผิดพลาด

// Reset timer
myTimer.reset();
```

### 6. Testing Timer

ใส่โค้ดนี้ใน console เพื่อทดสอบ timer:

```javascript
// ทดสอบ video timer
videoTimer.start();
setTimeout(() => videoTimer.updateStep('Test Step 1', 25, 'Testing...'), 1000);
setTimeout(() => videoTimer.updateStep('Test Step 2', 50, 'Still testing...'), 3000);
setTimeout(() => videoTimer.updateStep('Test Step 3', 75, 'Almost done...'), 5000);
setTimeout(() => {
    videoTimer.updateStep('Complete!', 100, 'Test finished!');
    videoTimer.finish(true);
}, 7000);
```

---

## 📝 หมายเหตุ

- Timer จะอัปเดตทุก 1 วินาทีโดยอัตโนมัติ
- Progress bar และเวลาจะ sync กันเอง
- เมื่อเสร็จจะแสดงสรุปเวลาแต่ละขั้นตอน
- Timer จะซ่อนอัตโนมัติหลัง 30 วินาที (ถ้าสำเร็จ)
- สามารถใช้ timer หลาย instance พร้อมกันได้ (video + batch)
