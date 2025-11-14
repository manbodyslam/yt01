# Character Position Settings

## การตั้งค่าปัจจุบัน (Final Settings)

### ✅ แบบที่เลือกใช้: แบบที่ 6
- **top_margin = 85 pixels**
- หัวห่างจากขอบบนของเฟรม 85 pixels
- นับจาก: หัวจริงๆ (ไม่ใช่ขอบรูป)
- อนุญาตให้หัวถูกตัดออกได้ แต่ต้องมีอย่างน้อย 30% ของตัวละครอยู่ในเฟรม

### ไฟล์ที่แก้ไข
**ไฟล์:** `/Users/winai/Desktop/code_x/yt01/modules/renderer.py`

**บรรทัด:** 342-344

```python
# กำหนดระยะห่างของหัวจากขอบบน (นับจากหัวจริงๆ)
top_margin = 85  # แบบที่ 6: หัวห่างจากขอบบน 85px
paste_y = placement.position.y - head_top_scaled + top_margin
```

**บรรทัด:** 356-361 (Boundary Check)

```python
# 2. ป้องกันรูปหลุดออกนอกเฟรม (Canvas Boundary Check)
#    นับจากหัวจริงๆ และอนุญาตให้หัวถูกตัดออกได้ (paste_y ติดลบได้)
paste_x = max(0, min(paste_x, canvas.width - new_w))
# อนุญาตให้หัวถูกตัดออก แต่ต้องมีอย่างน้อย 30% ของตัวละครในเฟรม
min_visible_height = int(new_h * 0.3)  # ต้องเห็นอย่างน้อย 30% ของตัวละคร
paste_y = max(-new_h + min_visible_height, min(paste_y, canvas.height - min_visible_height))
```

---

## ประวัติการทดลอง

### แบบที่ 1: top_margin = 0px ❌
- หัวชนขอบบนพอดี
- **ปัญหา:** ชิดเกินไป หัวไปกินตัวอักษร

### แบบที่ 2: top_margin = 10px ❌
- หัวห่างจากขอบบน 10 pixels
- **ปัญหา:** ยังไม่พอ ยังชิดเกินไป

### แบบที่ 3: top_margin = 25px ❌
- หัวห่างจากขอบบน 25 pixels
- **ปัญหา:** ยังไม่พอ

### แบบที่ 4: top_margin = 45px ❌
- หัวห่างจากขอบบน 45 pixels
- **ปัญหา:** ยังไม่พอ

### แบบที่ 5: top_margin = 65px ❌
- หัวห่างจากขอบบน 65 pixels
- **ปัญหา:** ยังไม่พอ

### แบบที่ 6: top_margin = 85px ✅ **เลือกใช้**
- หัวห่างจากขอบบน 85 pixels
- **ผลลัพธ์:** พอดี ไม่ไปกินตัวอักษร

---

## วิธีปรับแก้

ถ้าต้องการเปลี่ยนระยะห่าง:

1. เปิดไฟล์ `/Users/winai/Desktop/code_x/yt01/modules/renderer.py`
2. ไปที่บรรทัด 343
3. เปลี่ยนค่า `top_margin = 85` เป็นค่าที่ต้องการ (หน่วยเป็น pixels)
4. บันทึกไฟล์
5. เซิร์ฟเวอร์จะ reload อัตโนมัติ (ถ้าใช้ `--reload` flag)

**ตัวอย่าง:**
```python
top_margin = 100  # เพิ่มระยะห่างเป็น 100px
```

---

## หมายเหตุ

### Canvas Size
- ความกว้าง: 1920 pixels
- ความสูง: 1080 pixels

### การนับระยะ
- **นับจากหัวจริงๆ (head position)** ไม่ใช่ขอบบนของรูป
- `paste_y = placement.position.y - head_top_scaled + top_margin`
- ยิ่ง top_margin มาก = หัวยิ่งอยู่ต่ำลง (ห่างจากขอบบนมากขึ้น)

### Boundary Check
- อนุญาตให้ `paste_y` เป็นค่าลบได้ (หัวถูกตัดออก)
- แต่ต้องมีอย่างน้อย 30% ของตัวละครอยู่ในเฟรม
- ป้องกันไม่ให้ตัวละครหลุดล่างเกิน

---

## วันที่บันทึก
2025-11-11

## ผู้บันทึก
Claude Code

## Status
✅ Active - ใช้งานปัจจุบัน
