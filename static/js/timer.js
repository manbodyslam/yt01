// ⏱️ Timer Utility for Thumbnail Generator
// ติดตามเวลาที่ใช้ในการสร้าง thumbnail และแสดงผลแบบ realtime

class ProcessTimer {
    constructor(prefix) {
        this.prefix = prefix; // 'video' หรือ 'batch'
        this.startTime = null;
        this.intervalId = null;
        this.stepTimings = [];
        this.currentStep = null;
    }

    // เริ่ม timer
    start() {
        this.startTime = Date.now();
        this.stepTimings = [];
        this.currentStep = null;

        // แสดง timer display
        const timerDisplay = document.getElementById(`${this.prefix}-timer-display`);
        if (timerDisplay) {
            timerDisplay.style.display = 'block';
        }

        // เริ่ม update ทุก 1 วินาที
        this.intervalId = setInterval(() => this.updateDisplay(), 1000);

        this.updateStep('กำลังเริ่มต้น...');
    }

    // หยุด timer
    stop() {
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
    }

    // อัปเดตขั้นตอนปัจจุบัน (พร้อม % และคำอธิบาย)
    updateStep(stepName, percentage = null, description = null) {
        const now = Date.now();

        // บันทึกเวลาของขั้นตอนก่อนหน้า
        if (this.currentStep) {
            const elapsed = ((now - this.currentStep.startTime) / 1000).toFixed(1);
            this.stepTimings.push({
                name: this.currentStep.name,
                duration: elapsed
            });
        }

        // เริ่มขั้นตอนใหม่
        this.currentStep = {
            name: stepName,
            startTime: now,
            description: description || stepName
        };

        // อัปเดต Progress Bar
        if (percentage !== null) {
            this.updateProgressBar(percentage);
        }

        // อัปเดต UI
        this.updateDisplay();
    }

    // อัปเดต Progress Bar
    updateProgressBar(percentage) {
        const progressFill = document.getElementById(`${this.prefix}-progress-fill`);
        if (progressFill) {
            progressFill.style.width = `${percentage}%`;
            progressFill.textContent = `${percentage}%`;
        }
    }

    // อัปเดตการแสดงผล
    updateDisplay() {
        if (!this.startTime) return;

        const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        const timeStr = `${minutes}:${seconds.toString().padStart(2, '0')}`;

        // อัปเดตเวลารวม
        const elapsedTimeEl = document.getElementById(`${this.prefix}-elapsed-time`);
        if (elapsedTimeEl) {
            elapsedTimeEl.textContent = timeStr;
        }

        // อัปเดตขั้นตอนปัจจุบัน
        const stepInfoEl = document.getElementById(`${this.prefix}-step-info`);
        if (stepInfoEl && this.currentStep) {
            const stepElapsed = Math.floor((Date.now() - this.currentStep.startTime) / 1000);
            stepInfoEl.textContent = `${this.currentStep.name} (${stepElapsed}s)`;
        }

        // อัปเดตรายละเอียดขั้นตอนที่ผ่านมา
        this.updateTimingDetails();
    }

    // อัปเดตรายละเอียดเวลาแต่ละขั้นตอน
    updateTimingDetails() {
        const detailsEl = document.getElementById(`${this.prefix}-timing-details`);
        if (!detailsEl || this.stepTimings.length === 0) return;

        const html = this.stepTimings
            .map(step => `✓ ${step.name}: ${step.duration}s`)
            .join('<br>');
        detailsEl.innerHTML = html;
    }

    // เสร็จสิ้น - แสดงสรุป
    finish(success = true) {
        this.stop();

        // บันทึกขั้นตอนสุดท้าย
        if (this.currentStep) {
            const elapsed = ((Date.now() - this.currentStep.startTime) / 1000).toFixed(1);
            this.stepTimings.push({
                name: this.currentStep.name,
                duration: elapsed
            });
        }

        const totalElapsed = ((Date.now() - this.startTime) / 1000).toFixed(1);

        // อัปเดตข้อความสรุป
        const stepInfoEl = document.getElementById(`${this.prefix}-step-info`);
        if (stepInfoEl) {
            const icon = success ? '✅' : '❌';
            stepInfoEl.textContent = `${icon} ${success ? 'เสร็จสมบูรณ์' : 'เกิดข้อผิดพลาด'} - รวม ${totalElapsed}s`;
        }

        // แสดงรายละเอียดทั้งหมด
        this.updateTimingDetails();

        // ซ่อนหลัง 30 วินาที (ถ้าสำเร็จ)
        if (success) {
            setTimeout(() => {
                const timerDisplay = document.getElementById(`${this.prefix}-timer-display`);
                if (timerDisplay) {
                    timerDisplay.style.display = 'none';
                }
            }, 30000);
        }
    }

    // Reset timer
    reset() {
        this.stop();
        this.startTime = null;
        this.stepTimings = [];
        this.currentStep = null;

        const timerDisplay = document.getElementById(`${this.prefix}-timer-display`);
        if (timerDisplay) {
            timerDisplay.style.display = 'none';
        }
    }
}

// สร้าง timer instances สำหรับ video และ batch
const videoTimer = new ProcessTimer('video');
const batchTimer = new ProcessTimer('batch');
