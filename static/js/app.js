// YouTube Thumbnail Generator - Frontend JavaScript

const API_BASE = window.location.origin;

// Load presets from API
async function loadPresets() {
    try {
        const response = await fetch(`${API_BASE}/presets`);
        const presets = await response.json();

        const select = document.getElementById('video-preset');
        select.innerHTML = ''; // Clear existing options

        // Populate dropdown with presets
        for (const [id, preset] of Object.entries(presets)) {
            const option = document.createElement('option');
            option.value = id;
            option.textContent = `${preset.name} - ${preset.description}`;

            // Set preset 1 as default
            if (id === '1') {
                option.selected = true;
            }

            select.appendChild(option);
        }

        console.log('✓ Loaded presets:', Object.keys(presets).length);
    } catch (error) {
        console.error('Failed to load presets:', error);
    }
}

// Load presets when page loads
document.addEventListener('DOMContentLoaded', () => {
    loadPresets();
});

// Tab switching
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        const tabName = tab.dataset.tab;

        // Update active tab
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');

        // Update active content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tabName}-tab`).classList.add('active');

        // Load gallery if gallery tab
        if (tabName === 'gallery') {
            loadGallery();
        }

        // Load storage info if storage tab
        if (tabName === 'storage') {
            loadStorageInfo();
        }

        // Load analytics if analytics tab
        if (tabName === 'analytics') {
            loadAnalytics();
        }
    });
});

// File upload handlers
setupFileUpload('video-upload-area', 'video-file', 'video-file-info', 'video-filename', false);
setupFileUpload('batch-upload-area', 'batch-file', 'batch-file-info', 'batch-filename', false);
setupFileUpload('images-upload-area', 'images-files', 'images-file-info', 'images-count', true);

function setupFileUpload(areaId, inputId, infoId, displayId, multiple) {
    const area = document.getElementById(areaId);
    const input = document.getElementById(inputId);
    const info = document.getElementById(infoId);
    const display = document.getElementById(displayId);

    // Click to upload
    area.addEventListener('click', () => {
        input.click();
    });

    // Drag and drop
    area.addEventListener('dragover', (e) => {
        e.preventDefault();
        area.classList.add('drag-over');
    });

    area.addEventListener('dragleave', () => {
        area.classList.remove('drag-over');
    });

    area.addEventListener('drop', (e) => {
        e.preventDefault();
        area.classList.remove('drag-over');

        if (e.dataTransfer.files.length > 0) {
            input.files = e.dataTransfer.files;
            updateFileInfo(input.files, info, display, multiple);
        }
    });

    // File select
    input.addEventListener('change', () => {
        if (input.files.length > 0) {
            updateFileInfo(input.files, info, display, multiple);
        }
    });
}

function updateFileInfo(files, infoElement, displayElement, multiple) {
    if (multiple) {
        displayElement.textContent = files.length;
    } else {
        displayElement.textContent = files[0].name;
    }
    infoElement.classList.add('show');
}

// Generate from video
document.getElementById('generate-from-video-btn').addEventListener('click', async () => {
    const fileInput = document.getElementById('video-file');
    const title = document.getElementById('video-title').value;
    const subtitle = document.getElementById('video-subtitle').value;
    const numCharacters = document.getElementById('video-num-characters').value;
    const numFrames = 50;  // Fixed at 50 frames

    const successAlert = document.getElementById('video-success-alert');
    const errorAlert = document.getElementById('video-error-alert');
    const spinner = document.getElementById('video-spinner');
    const result = document.getElementById('video-result');

    // Reset
    successAlert.classList.remove('show');
    errorAlert.classList.remove('show');
    result.classList.remove('show');

    // Validate
    if (!fileInput.files[0]) {
        showError(errorAlert, 'กรุณาเลือกไฟล์วิดีโอ');
        return;
    }

    if (!title) {
        showError(errorAlert, 'กรุณากรอกชื่อ Thumbnail');
        return;
    }

    // Prepare form data
    const formData = new FormData();
    const textStyle = document.getElementById('video-text-style').value;
    const presetId = document.getElementById('video-preset').value;
    formData.append('video', fileInput.files[0]);
    formData.append('title', title);
    if (subtitle && subtitle.trim() !== '') {
        formData.append('subtitle', subtitle);
    }
    formData.append('num_characters', numCharacters);
    formData.append('num_frames', numFrames);
    formData.append('text_style', textStyle);
    formData.append('preset_id', presetId);

    // Show spinner and progress bar
    spinner.classList.add('show');
    const progressBar = document.getElementById('video-progress');
    progressBar.classList.add('show');

    // 🚀 เริ่ม timer
    console.log('🚀 Starting video timer...');
    videoTimer.start();
    videoTimer.updateStep('อัปโหลดวิดีโอ', 10, 'กำลังอัปโหลดไฟล์วิดีโอไปยัง server...');
    console.log('✅ Timer started');

    let eventSource; // สำหรับ SSE connection

    try {
        const response = await fetch(`${API_BASE}/generate-from-video`, {
            method: 'POST',
            body: formData
        });

        // Handle non-200 responses
        if (!response.ok) {
            let errorMessage = 'เกิดข้อผิดพลาด';
            try {
                const errorData = await response.json();
                errorMessage = errorData.detail || errorData.error || errorMessage;
            } catch (e) {
                errorMessage = `HTTP ${response.status}: ${response.statusText}`;
            }
            videoTimer.finish(false);
            showError(errorAlert, errorMessage);
            return;
        }

        const data = await response.json();

        // 🚀 เชื่อมต่อ SSE เพื่อรับ real-time progress
        if (data.job_id) {
            eventSource = new EventSource(`${API_BASE}/progress/${data.job_id}`);

            eventSource.addEventListener('progress', (event) => {
                const progressData = JSON.parse(event.data);
                console.log('Progress update:', progressData);

                // อัปเดต progress bar
                videoTimer.updateStep(
                    progressData.message,
                    progressData.progress,
                    progressData.message
                );
            });

            eventSource.addEventListener('complete', (event) => {
                const progressData = JSON.parse(event.data);
                console.log('Complete:', progressData);

                // ปิด SSE connection
                eventSource.close();

                // อัปเดต timer ว่าเสร็จแล้ว
                if (progressData.status === 'completed') {
                    videoTimer.updateStep('เสร็จสมบูรณ์!', 100, 'สร้าง thumbnail สำเร็จ!');
                    videoTimer.finish(true);
                } else {
                    videoTimer.finish(false);
                }
            });

            eventSource.addEventListener('error', (event) => {
                console.error('SSE error:', event);
                eventSource.close();
            });
        }

        if (data.success) {
            // 🎉 เสร็จสมบูรณ์!

            // Show success
            showSuccess(successAlert, 'สร้าง Thumbnail สำเร็จ!');

            // Display result
            displayResult(
                data,
                'video-result-image',
                'video-result-filename',
                'video-result-layout',
                'video-result-chars',
                'video-result-time'
            );
            result.classList.add('show');

            // Show frames extracted
            if (data.metadata) {
                document.getElementById('video-result-frames').textContent =
                    data.metadata.num_characters || 'N/A';
            }
        } else {
            videoTimer.finish(false);
            showError(errorAlert, data.error || 'เกิดข้อผิดพลาด');
        }
    } catch (error) {
        console.error('Error:', error);
        videoTimer.finish(false);
        showError(errorAlert, 'ไม่สามารถเชื่อมต่อกับ API ได้: ' + error.message);

        // ปิด SSE connection ถ้ามี
        if (eventSource) {
            eventSource.close();
        }
    } finally {
        spinner.classList.remove('show');
        // ซ่อน progress bar เมื่อเสร็จ (หลัง 3 วินาที)
        setTimeout(() => {
            progressBar.classList.remove('show');
        }, 3000);
    }
});

// Generate from images
document.getElementById('generate-from-images-btn').addEventListener('click', async () => {
    const filesInput = document.getElementById('images-files');
    const title = document.getElementById('images-title').value;
    const subtitle = document.getElementById('images-subtitle').value;
    const numCharacters = document.getElementById('images-num-characters').value;

    const successAlert = document.getElementById('images-success-alert');
    const errorAlert = document.getElementById('images-error-alert');
    const spinner = document.getElementById('images-spinner');
    const result = document.getElementById('images-result');

    // Reset
    successAlert.classList.remove('show');
    errorAlert.classList.remove('show');
    result.classList.remove('show');

    // Validate
    if (!filesInput.files || filesInput.files.length === 0) {
        showError(errorAlert, 'กรุณาเลือกรูปภาพอย่างน้อย 1 รูป');
        return;
    }

    if (!title) {
        showError(errorAlert, 'กรุณากรอกชื่อ Thumbnail');
        return;
    }

    // Upload images first
    spinner.classList.add('show');

    try {
        // Upload images to workspace/raw
        const uploadFormData = new FormData();
        for (let file of filesInput.files) {
            uploadFormData.append('files', file);
        }

        // Note: Need to implement /upload-images endpoint
        // For now, generate directly

        const textStyle = document.getElementById('images-text-style').value;
        const generateData = {
            title: title,
            subtitle: subtitle,
            num_characters: parseInt(numCharacters),
            text_style: textStyle
        };

        const response = await fetch(`${API_BASE}/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(generateData)
        });

        const data = await response.json();

        if (data.success) {
            showSuccess(successAlert, 'สร้าง Thumbnail สำเร็จ!');
            displayResult(
                data,
                'images-result-image',
                'images-result-filename',
                'images-result-layout',
                null,
                null
            );
            result.classList.add('show');
        } else {
            showError(errorAlert, data.error || 'เกิดข้อผิดพลาด');
        }
    } catch (error) {
        console.error('Error:', error);
        showError(errorAlert, 'ไม่สามารถเชื่อมต่อกับ API ได้: ' + error.message);
    } finally {
        spinner.classList.remove('show');
    }
});

// Load gallery
document.getElementById('refresh-gallery-btn').addEventListener('click', loadGallery);

async function loadGallery() {
    const spinner = document.getElementById('gallery-spinner');
    const gallery = document.getElementById('thumbnail-gallery');

    spinner.classList.add('show');
    gallery.innerHTML = '';

    try {
        const response = await fetch(`${API_BASE}/thumbnails`);
        const thumbnails = await response.json();

        if (thumbnails.length === 0) {
            gallery.innerHTML = '<p style="text-align: center; color: #999;">ยังไม่มี Thumbnail</p>';
        } else {
            thumbnails.forEach(filename => {
                const item = createThumbnailItem(filename);
                gallery.appendChild(item);
            });
        }
    } catch (error) {
        console.error('Error loading gallery:', error);
        gallery.innerHTML = '<p style="text-align: center; color: #999;">ไม่สามารถโหลด Gallery ได้</p>';
    } finally {
        spinner.classList.remove('show');
    }
}

function createThumbnailItem(filename) {
    const div = document.createElement('div');
    div.className = 'thumbnail-item';

    div.innerHTML = `
        <img src="${API_BASE}/thumbnail/${filename}" alt="${filename}">
        <div class="thumbnail-item-info">
            <div class="thumbnail-item-title">${filename}</div>
            <div style="display: flex; gap: 10px; margin-top: 10px;">
                <a href="${API_BASE}/thumbnail/${filename}" download style="color: #667eea; text-decoration: none; flex: 1;">
                    📥 ดาวน์โหลด
                </a>
                <button class="delete-thumbnail-btn" data-filename="${filename}" style="background: #dc3545; color: white; border: none; padding: 5px 10px; border-radius: 5px; cursor: pointer; flex: 1;">
                    🗑️ ลบ
                </button>
            </div>
        </div>
    `;

    // Add delete handler
    const deleteBtn = div.querySelector('.delete-thumbnail-btn');
    deleteBtn.addEventListener('click', async (e) => {
        e.preventDefault();
        await deleteThumbnail(filename);
    });

    return div;
}

async function deleteThumbnail(filename) {
    if (!confirm(`คุณแน่ใจหรือไม่ว่าต้องการลบ "${filename}"?\n\nจะไม่สามารถกู้คืนได้!`)) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/thumbnail/${filename}`, {
            method: 'DELETE'
        });

        if (response.ok) {
            // Reload gallery
            loadGallery();

            // Show success message
            const message = document.createElement('div');
            message.className = 'alert alert-success show';
            message.textContent = `ลบ "${filename}" เรียบร้อยแล้ว`;
            message.style.position = 'fixed';
            message.style.top = '20px';
            message.style.right = '20px';
            message.style.zIndex = '9999';
            document.body.appendChild(message);

            setTimeout(() => {
                message.remove();
            }, 3000);
        } else {
            throw new Error('Failed to delete thumbnail');
        }
    } catch (error) {
        console.error('Error deleting thumbnail:', error);
        alert('ไม่สามารถลบ Thumbnail ได้: ' + error.message);
    }
}

function displayResult(data, imgId, filenameId, layoutId, charsId, timeId) {
    const img = document.getElementById(imgId);
    const filename = document.getElementById(filenameId);
    const layout = document.getElementById(layoutId);

    img.src = `${API_BASE}/thumbnail/${data.filename}`;
    filename.textContent = data.filename;

    if (data.metadata) {
        layout.textContent = data.metadata.layout_type || 'N/A';

        if (charsId) {
            const chars = document.getElementById(charsId);
            chars.textContent = data.metadata.num_characters || 'N/A';
        }

        // แสดง processing_time
        if (timeId && data.metadata.processing_time) {
            const timeElement = document.getElementById(timeId);
            timeElement.textContent = data.metadata.processing_time.formatted || 'N/A';
        }
    }
}

function showSuccess(element, message) {
    element.textContent = message;
    element.classList.add('show');
    setTimeout(() => element.classList.remove('show'), 5000);
}

function showError(element, message) {
    // แสดง error message บนหน้าเว็บ (ค้างไว้ 60 วินาที - อ่านได้สบาย)
    element.textContent = message;
    element.classList.add('show');
    setTimeout(() => element.classList.remove('show'), 60000); // เพิ่มเป็น 60 วินาที

    // แสดง error ใน console ด้วย (ไม่หาย)
    console.error('❌ ERROR:', message);
    console.error('Timestamp:', new Date().toLocaleString('th-TH'));
}

// Storage Management Functions
async function loadStorageInfo() {
    const spinner = document.getElementById('storage-spinner');
    spinner.classList.add('show');

    try {
        const response = await fetch(`${API_BASE}/storage-info`);
        const data = await response.json();

        // Update UI with storage info
        document.getElementById('total-size').textContent = data.total_size;
        document.getElementById('videos-size').textContent = data.folders.videos.size;
        document.getElementById('videos-count').textContent = data.folders.videos.file_count;
        document.getElementById('frames-size').textContent = data.folders.raw_frames.size;
        document.getElementById('frames-count').textContent = data.folders.raw_frames.file_count;
        document.getElementById('thumbnails-size').textContent = data.folders.thumbnails.size;
        document.getElementById('thumbnails-count').textContent = data.folders.thumbnails.file_count;
        document.getElementById('temp-size').textContent = data.folders.temp.size;
    } catch (error) {
        console.error('Error loading storage info:', error);
        showStorageError('ไม่สามารถโหลดข้อมูลพื้นที่ได้');
    } finally {
        spinner.classList.remove('show');
    }
}

// Clear videos
document.getElementById('clear-videos-btn').addEventListener('click', async () => {
    if (!confirm('คุณแน่ใจหรือไม่ว่าต้องการลบวิดีโอทั้งหมด?\n\nจะไม่สามารถกู้คืนได้!')) {
        return;
    }

    const spinner = document.getElementById('storage-spinner');
    spinner.classList.add('show');

    try {
        const response = await fetch(`${API_BASE}/clear-videos`, { method: 'POST' });
        const data = await response.json();

        showStorageSuccess(data.message);
        loadStorageInfo(); // Refresh storage info
    } catch (error) {
        console.error('Error:', error);
        showStorageError('ไม่สามารถลบวิดีโอได้: ' + error.message);
    } finally {
        spinner.classList.remove('show');
    }
});

// Clear workspace
document.getElementById('clear-workspace-btn').addEventListener('click', async () => {
    if (!confirm('คุณแน่ใจหรือไม่ว่าต้องการล้าง Workspace (Raw + Temp)?\n\nจะลบ frames และไฟล์ชั่วคราวทั้งหมด!')) {
        return;
    }

    const spinner = document.getElementById('storage-spinner');
    spinner.classList.add('show');

    try {
        const response = await fetch(`${API_BASE}/clear-workspace`, { method: 'POST' });
        const data = await response.json();

        showStorageSuccess(data.message);
        loadStorageInfo();
    } catch (error) {
        console.error('Error:', error);
        showStorageError('ไม่สามารถล้าง Workspace ได้: ' + error.message);
    } finally {
        spinner.classList.remove('show');
    }
});

// Clear all (videos + workspace)
document.getElementById('clear-all-btn').addEventListener('click', async () => {
    if (!confirm('⚠️ คำเตือน!\n\nคุณกำลังจะลบ:\n- วิดีโอทั้งหมด\n- Frames ทั้งหมด\n- ไฟล์ชั่วคราวทั้งหมด\n\nThumbnails จะไม่ถูกลบ\n\nคุณแน่ใจหรือไม่?')) {
        return;
    }

    const spinner = document.getElementById('storage-spinner');
    spinner.classList.add('show');

    try {
        // Clear videos first
        await fetch(`${API_BASE}/clear-videos`, { method: 'POST' });
        // Then clear workspace
        await fetch(`${API_BASE}/clear-workspace`, { method: 'POST' });

        showStorageSuccess('ลบไฟล์ทั้งหมดเรียบร้อยแล้ว (ยกเว้น Thumbnails)');
        loadStorageInfo();
    } catch (error) {
        console.error('Error:', error);
        showStorageError('ไม่สามารถลบไฟล์ทั้งหมดได้: ' + error.message);
    } finally {
        spinner.classList.remove('show');
    }
});

// Refresh storage info
document.getElementById('refresh-storage-btn').addEventListener('click', loadStorageInfo);

function showStorageSuccess(message) {
    const alert = document.getElementById('storage-success-alert');
    alert.textContent = message;
    alert.classList.add('show');
    setTimeout(() => alert.classList.remove('show'), 5000);
}

function showStorageError(message) {
    const alert = document.getElementById('storage-error-alert');
    alert.textContent = message;
    alert.classList.add('show');
    setTimeout(() => alert.classList.remove('show'), 8000);
}

// Batch Generation
document.getElementById('generate-batch-btn').addEventListener('click', async () => {
    const fileInput = document.getElementById('batch-file');
    const title = document.getElementById('batch-title').value;
    const subtitle = document.getElementById('batch-subtitle').value;
    const numCharacters = document.getElementById('batch-num-characters').value;
    const numVariants = document.getElementById('batch-num-variants').value;
    const textStyle = document.getElementById('batch-text-style').value;

    const successAlert = document.getElementById('batch-success-alert');
    const errorAlert = document.getElementById('batch-error-alert');
    const spinner = document.getElementById('batch-spinner');
    const result = document.getElementById('batch-result');
    const progress = document.getElementById('batch-progress');
    const progressFill = document.getElementById('batch-progress-fill');
    const thumbnailsGrid = document.getElementById('batch-thumbnails');

    // Reset
    successAlert.classList.remove('show');
    errorAlert.classList.remove('show');
    result.classList.remove('show');
    progress.classList.remove('show');
    thumbnailsGrid.innerHTML = '';

    // Validate
    if (!fileInput.files[0]) {
        showError(errorAlert, 'กรุณาเลือกไฟล์วิดีโอ');
        return;
    }

    if (!title) {
        showError(errorAlert, 'กรุณากรอกชื่อ Thumbnail');
        return;
    }

    // Prepare form data
    const formData = new FormData();
    formData.append('video', fileInput.files[0]);
    formData.append('title', title);
    if (subtitle && subtitle.trim() !== '') {
        formData.append('subtitle', subtitle);
    }
    formData.append('num_characters', numCharacters);
    formData.append('num_variants', numVariants);
    formData.append('text_style', textStyle);

    // Show progress
    spinner.classList.add('show');
    progress.classList.add('show');
    progressFill.style.width = '0%';
    progressFill.textContent = 'กำลังดึงเฟรม...';

    // 🚀 เริ่ม timer สำหรับ batch
    batchTimer.start();
    batchTimer.updateStep('อัปโหลดวิดีโอ', 10, 'กำลังอัปโหลดไฟล์วิดีโอไปยัง server...');

    try {
        const response = await fetch(`${API_BASE}/generate-batch-from-video`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            let errorMessage = 'เกิดข้อผิดพลาด';
            try {
                const errorData = await response.json();
                errorMessage = errorData.detail || errorData.error || errorMessage;
            } catch (e) {
                errorMessage = `HTTP ${response.status}: ${response.statusText}`;
            }
            showError(errorAlert, errorMessage);
            return;
        }

        const data = await response.json();

        progressFill.style.width = '100%';
        progressFill.textContent = '100%';

        if (data.success && data.thumbnails && data.thumbnails.length > 0) {
            // 🎉 เสร็จสมบูรณ์!
            batchTimer.updateStep('เสร็จสมบูรณ์!', 100, `สร้าง ${data.total_generated} thumbnails สำเร็จ!`);
            batchTimer.finish(true);

            // Show success
            showSuccess(successAlert, `✅ สร้างสำเร็จ ${data.total_generated}/${numVariants} thumbnails!`);

            // Display thumbnails in grid
            data.thumbnails.forEach((thumb, index) => {
                const item = document.createElement('div');
                item.className = 'thumbnail-item';
                item.innerHTML = `
                    <img src="${API_BASE}${thumb.url}" alt="${thumb.filename}">
                    <div class="thumbnail-item-info">
                        <div class="thumbnail-item-title">ตัวเลือก ${index + 1}</div>
                        <p style="font-size: 12px; color: #999; margin-top: 5px;">${thumb.filename}</p>
                        <a href="${API_BASE}${thumb.url}" download style="display: inline-block; margin-top: 10px; padding: 8px 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-decoration: none; border-radius: 5px; font-size: 14px;">
                            📥 ดาวน์โหลด
                        </a>
                    </div>
                `;
                thumbnailsGrid.appendChild(item);
            });

            result.classList.add('show');

            // Show errors if any failed
            if (data.failed > 0) {
                showError(errorAlert, `⚠️ เกิดข้อผิดพลาด ${data.failed} ตัว: ${data.errors.join(', ')}`);
            }
        } else {
            batchTimer.finish(false);
            showError(errorAlert, data.errors ? data.errors.join(', ') : 'ไม่สามารถสร้าง thumbnail ได้');
        }
    } catch (error) {
        console.error('Error:', error);
        batchTimer.finish(false);
        showError(errorAlert, 'ไม่สามารถเชื่อมต่อกับ API ได้: ' + error.message);
    } finally {
        spinner.classList.remove('show');
    }
});

// Analytics Functions
async function loadAnalytics() {
    const spinner = document.getElementById('analytics-spinner');
    spinner.classList.add('show');

    try {
        const response = await fetch(`${API_BASE}/analytics`);
        const data = await response.json();

        // Update stats
        document.getElementById('analytics-total').textContent = data.total_generated || 0;

        const successRate = data.total_generated > 0
            ? ((data.success_count / data.total_generated) * 100).toFixed(1)
            : 0;
        document.getElementById('analytics-success-rate').textContent = `${successRate}%`;

        document.getElementById('analytics-today').textContent = data.today_count || 0;
        document.getElementById('analytics-errors').textContent = data.error_count || 0;

        // Update last update time
        const now = new Date().toLocaleString('th-TH');
        document.getElementById('analytics-last-update').innerHTML = `<strong>Last updated:</strong> ${now}`;
    } catch (error) {
        console.error('Error loading analytics:', error);
        document.getElementById('analytics-total').textContent = 'Error';
        document.getElementById('analytics-success-rate').textContent = 'Error';
        document.getElementById('analytics-today').textContent = 'Error';
        document.getElementById('analytics-errors').textContent = 'Error';
    } finally {
        spinner.classList.remove('show');
    }
}

// Refresh analytics button
document.getElementById('refresh-analytics-btn').addEventListener('click', loadAnalytics);

// Load gallery on page load
window.addEventListener('load', () => {
    // Don't auto-load gallery, only load when tab is clicked
});
