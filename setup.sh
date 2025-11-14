#!/bin/bash

echo "================================================"
echo "YouTube Thumbnail Generator - Setup Script"
echo "================================================"
echo ""

# Check Python
echo "1. ตรวจสอบ Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 ไม่ได้ติดตั้ง"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "✓ พบ $PYTHON_VERSION"
echo ""

# Create virtual environment (optional but recommended)
echo "2. สร้าง virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ สร้าง venv สำเร็จ"
else
    echo "✓ venv มีอยู่แล้ว"
fi
echo ""

# Activate virtual environment
echo "3. Activate virtual environment..."
source venv/bin/activate
echo "✓ Activated"
echo ""

# Upgrade pip
echo "4. อัปเกรด pip..."
pip install --upgrade pip
echo ""

# Install requirements
echo "5. ติดตั้ง dependencies..."
echo "   (อาจใช้เวลาสักครู่...)"
pip install -r requirements.txt

echo ""
echo "================================================"
echo "✅ ติดตั้งเสร็จสมบูรณ์!"
echo "================================================"
echo ""
echo "วิธีรัน:"
echo "  1. Activate venv:  source venv/bin/activate"
echo "  2. รัน server:     python main.py"
echo "  3. เปิดเบราว์เซอร์: http://localhost:8000"
echo ""
echo "หรือใช้:"
echo "  ./run.sh"
echo ""
