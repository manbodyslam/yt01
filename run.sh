#!/bin/bash

echo "Starting YouTube Thumbnail Generator..."
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "กรุณารัน: ./setup.sh ก่อน"
    exit 1
fi

# Activate venv
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "❌ Dependencies ไม่ครบ!"
    echo "กรุณารัน: ./setup.sh ก่อน"
    exit 1
fi

# Run server
echo "✓ Starting server..."
echo "✓ Web UI: http://localhost:8000"
echo "✓ API Docs: http://localhost:8000/docs"
echo ""
echo "กด Ctrl+C เพื่อหยุด server"
echo ""

python main.py
