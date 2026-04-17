// === PHASE 2: AUTO LABELING QUEUE ===

// Quản lý State (Trạng thái) của Phase 2
const BACKEND_URL = "http://127.0.0.1:8000";

let processedFolders = new Set(); // Dùng Set để lưu tên các thư mục đã xử lý (để lookup nhanh với độ phức tạp O(1))
let currentVideos = []; // Mảng chứa danh sách các object video được chọn [{name, sizeMB}]
let hasSelectedOutputFolder = false; // Cờ đánh dấu người dùng đã chọn Output Folder hay chưa
const DEFAULT_OUTPUT_PATH = "data/output_labels";
let selectedOutputFolder = DEFAULT_OUTPUT_PATH;

let pollingTimers = {};

let clientOutputDirHandle = null;

// 1. Hàm modal đổi Output Path
function showChangePathModal() {
    document.getElementById('new-path-input').value = selectedOutputPath;
    document.getElementById('path-error').classList.add('hidden');
    document.getElementById('change-path-modal').classList.remove('hidden');
    setTimeout(() => document.getElementById('new-path-input').focus(), 50);
}

function hideChangePathModal() {
    document.getElementById('change-path-modal').classList.add('hidden');
}

async function confirmChangePath() {
    const newPath = document.getElementById('new-path-input').value.trim();
    const errEl = document.getElementById('path-error');
    if (!newPath) {
        errEl.textContent = 'Vui lòng nhập đường dẫn.';
        errEl.classList.remove('hidden');
        return;
    }
    errEl.classList.add('hidden');
    const btn = document.getElementById('confirm-path-btn');
    btn.disabled = true;
    btn.textContent = 'Đang kiểm tra...';
    try {
        const res = await fetch('/api/phase2/list-output-folders', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ output_folder: newPath }),
        });
        if (!res.ok) {
            const data = await res.json();
            throw new Error(data.detail || 'Đường dẫn không hợp lệ trên server.');
        }
        const { processed_folders } = await res.json();
        processedFolders = new Set(processed_folders);
        selectedOutputPath = newPath;
        renderOutputPathUI(newPath);
        renderQueueUI();
        hideChangePathModal();
    } catch (err) {
        errEl.textContent = '⚠ ' + err.message;
        errEl.classList.remove('hidden');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Xác nhận';
    }
}

function renderOutputPathUI(path) {
    const parts = path.replace(/\\/g, '/').split('/');
    const short = parts.length > 2 ? '…/' + parts.slice(-2).join('/') : path;
    document.getElementById('output-folder-name').innerHTML =
        `<span class="font-mono text-emerald-700 font-medium" title="${path}">${short}</span>
<button onclick="showChangePathModal()" class="ml-1 text-xs text-emerald-600 hover:text-emerald-800 border border-emerald-200 hover:bg-emerald-100 px-1.5 py-0.5 rounded transition">Thay đổi</button>`;
}


// 1. Hàm xử lý khi chọn Output Folder (Sử dụng File System Access API hiện đại)
async function handleOutputFolderSelection() {
    try {
        // Mở hộp thoại hệ thống yêu cầu người dùng cấp quyền đọc thư mục
        // Ưu điểm: Chỉ lấy metadata (cấu trúc cây thư mục), KHÔNG load nội dung file vào RAM, chống treo trình duyệt.
        clientOutputDirHandle = await window.showDirectoryPicker({ mode: 'readwrite' });
        
        // Cập nhật tên thư mục lên UI và bật cờ
        document.getElementById('output-folder-name').innerHTML = `<span class="text-green-600 font-medium">✓ ${clientOutputDirHandle.name}</span>`;
        hasSelectedOutputFolder = true;

        // Xóa danh sách cũ nếu người dùng chọn lại folder khác
        processedFolders.clear();

        // Lặp qua các thành phần bên trong thư mục gốc vừa chọn
        for await (const entry of clientOutputDirHandle.values()) {
            // Nếu thành phần đó là một thư mục con (chứ không phải file), lưu tên nó vào Set
            if (entry.kind === 'directory') processedFolders.add(entry.name);
        }

        // Cập nhật lại giao diện bảng (Đối chiếu song song)
        renderQueueUI();

    } catch (error) {
        // Xử lý khi người dùng bấm Hủy (Cancel) hộp thoại hoặc trình duyệt không hỗ trợ API
        console.log("Permission error:", error);
        alert("Bạn cần cáp quyền truy cập thư mục để hệ thống có thể lưu kết quả về máy.")
    }
}

// 2. Hàm xử lý khi người dùng chọn nhiều file Video (.mp4)
async function handleVideoSelection(event) {
    const files = event.target.files;
    if (!files.length) return;

    const emptyRow = document.getElementById('empty-queue-row');
    if (emptyRow) emptyRow.remove();

    for (const file of files) {
        if (!file.name.endsWith('.mp4')) continue;

        if (currentVideos.some(v => v.name === file.name)) {
            continue;
        }

        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch(`${BACKEND_URL}/api/phase2/upload`, {
                method: 'POST',
                body: formData
            });

            if (!res.ok) throw new Error("Internet connection error when upload");
            const data = await res.json();

            currentVideos.push({
                name: file.name,
                sizeMB: data.size_mb ? `${data.size_mb.tofixed(2)} MB` : `${(file.size / (1024 * 1024)).toFixed(2)} MB`,
                tempPath: data.temp_path   // 🔥 QUAN TRỌNG
            });

            renderQueueUI();

        } catch (err) {
            console.error("Upload lỗi:", err);
        }
    }

    event.target.value = '';
}

// 3. Hàm cốt lõi: Đối chiếu chéo (Cross-check) và vẽ lại giao diện bảng Queue
function renderQueueUI() {
    const tbody = document.getElementById('queue-body');

    // Nếu có data thì xóa trắng bảng để vẽ lại từ đầu
    if (currentVideos.length > 0) tbody.innerHTML = '';

    // Duyệt qua mảng video đang có
    currentVideos.forEach((video, index) => {
        // Tạo tên folder cần tìm bằng cách xóa đuôi mở rộng (VD: "video1.mp4" -> "video1")
        const folderName = video.name.replace(/\.[^/.]+$/, "");

        // Kiểm tra xem folderName này có tồn tại trong Set Output Folder hay không
        const isProcessed = processedFolders.has(folderName);

        const tr = document.createElement('tr');

        if (isProcessed) {
            // KỊCH BẢN ĐÃ XỬ LÝ (DONE)
            tr.className = 'bg-slate-50 opacity-80'; // Làm mờ dòng
            tr.innerHTML = `
                <td class="p-3 text-center"><input type="checkbox" class="file-checkbox w-4 h-4 cursor-pointer" disabled></td>
                <td class="p-3 font-mono text-slate-500">${video.name}</td>
                <td class="p-3 text-slate-400">${video.sizeMB}</td>
                <td class="p-3 flex items-center justify-between gap-4">
                <span class="file-status inline-flex items-center gap-1 text-green-600 font-medium" title="Đã có thư mục: ${folderName}">✓ Done</span>
                <button onclick="deleteOutputFolder(this, '${folderName}')" class="text-xs text-red-500 hover:text-red-700 bg-red-50 hover:bg-red-100 px-2 py-1 rounded transition border border-red-200" title="Xóa folder ${folderName}">
                    🗑️ Delete Output
                </button>
                </td>
                `;
        } else {
            // KỊCH BẢN CHƯA XỬ LÝ (READY)
            tr.innerHTML = `
                <td class="p-3 text-center"><input type="checkbox" class="file-checkbox w-4 h-4 cursor-pointer"></td>
                <td class="p-3 font-mono font-medium">${video.name}</td>
                <td class="p-3 text-slate-500">${video.sizeMB}</td>
                <td class="p-3 file-status">
                    <span class="inline-flex items-center gap-1 text-slate-600 font-medium">⏳ Ready</span>
                </td>
            `;
        }
        tbody.appendChild(tr);
    });
}

// Hàm lưu kết quả sau khi xử lý xong video xuống Output Folder đã chọn trước đó
async function downloadAndSaveToUserDisk(videoNameStem, jsonResultFromBackend, imageBlobFromBackend) {
    if (!clientOutputDirHandle) {
        console.error("Error: Output folder have not been selected yet");
        return;
    }

    try {
        // 1. create subfolder with the same name as mp4 file in outputfolder
        const subDirHandle = await clientOutputDirHandle.getDirectoryHandle(videoNameStem, { create: true });
        
        let txtContent = "";

        // 1. video name
        txtContent +=  `${videoNameStem}.mp4\n`;

        let framesWithPeople = [];

        // 2. collect frames with detections
        if (jsonResultFromBackend.detections) {
            jsonResultFromBackend.detections.forEach(det => {
                if (det.bounding_boxes && det.bounding_boxes.length > 0) {
                    framesWithPeople.push(det);
                }
            });
        }

        // 3. number of frames
        txtContent += `${framesWithPeople.length}\n\n`;

        // 4. write each frame
        framesWithPeople.forEach(det => {

            const frameNum = det.frame || 0;
            const peopleCount = det.bounding_boxes.length;

            txtContent += `Frame ${frameNum} ${peopleCount}\n`;

            det.bounding_boxes.forEach(box => {
                
                const x1 = box.x1;
                const y1 = box.y1;
                const x2 = box.x2;
                const y2 = box.y2;

                txtContent += `${x1} ${y1} ${x2} ${y2}\n`;  
            });

            txtContent += `\n`;
        })

        // == save txt ==
        const txtFileHanlde = await subDirHandle.getFileHandle('detections.txt', {create:true});
        const writableTxt = await txtFileHanlde.createWritable();

        await writableTxt.write(txtContent);
        await writableTxt.close();

        // Handle Images
        let imageNames = [];
        if (jsonResultFromBackend.detections) {
            jsonResultFromBackend.detections.forEach(det => {
                if (det.bounding_boxes) {
                    det.bounding_boxes.forEach(box => {
                        if (box.image_file) imageNames.push(box.image_file.split('/').pop());
                    });
                }
            });
        }

        if (imageNames.length > 0) {
            const framesDirHandle = await subDirHandle.getDirectoryHandle('frames', { create: true });

            for (const imgName of imageNames) {
                try {
                    // call for Python API
                    let imgRes = await fetch(`${BACKEND_URL}/api/phase2/get/${videoNameStem}/frames/${imgName}`);

                    if (imgRes.ok) {
                        let blob = await imgRes.blob();

                        let imgHandle = await framesDirHandle.getFileHandle(imgName, { create: true });

                        let writableImg = await imgHandle.createWritable();
                        await writableImg.write(blob);
                        await writableImg.close();
                    }
                } catch (err) {
                    console.error(`Unstable connection when loading img ${imgName}: `, err);
                }
            }
        }

        // 4. notify
        console.log(`[Success] Save results of ${videoNameStem} successfully in your local device!`)

        try {
            await fetch('/api/phase2/delete-output',
                {
                    method: 'DELETE',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ folder_name: videoNameStem })
                });
            console.log("Cleared up server processed data.");
        } catch (e) {
            console.error("Error when delete data on server:", e);
        }

        processedFolders.add(videoNameStem);
        renderQueueUI();
    } catch (err) {
        console.error("Error when saving result files to user's local folder", err)
    }
}

// 4. CÁC HÀM TƯƠNG TÁC GIAO DIỆN (UI CONTROLS)

// Đồng bộ Checkbox: Check All / Uncheck All (Bỏ qua các ô đang bị disabled)
function toggleSelectAll(masterCheckbox) {
    const checkboxes = document.querySelectorAll('.file-checkbox:not(:disabled)');
    checkboxes.forEach(cb => cb.checked = masterCheckbox.checked);
}

// Validate dữ liệu trước khi hiện Modal Xác nhận chạy (Process)
function checkAndShowModal() {
    if (!clientOutputDirHandle) {
        alert('Vui lòng nhấn "Select Output Folder" và cấp quyền duyệt file trước khi chạy Process để hệ thống có thể lưu về máy của bạn!');
        return;
    }

    const checkboxes = document.querySelectorAll('.file-checkbox:checked:not(:disabled)');
    if (checkboxes.length === 0) {
        alert('Vui lòng chọn videos để xử lý!');
        return;
    }

    // Gắn số lượng file vào UI của Modal và hiển thị Modal
    document.getElementById('file-count').innerText = checkboxes.length;
    document.getElementById('confirm-modal').classList.remove('hidden');
}

// Ẩn Modal Process
function hideConfirmModal() { document.getElementById('confirm-modal').classList.add('hidden'); }

// Bắt đầu giả lập tiến trình xử lý
async function startProcessing() {
    hideConfirmModal();
    const checkboxes = document.querySelectorAll('.file-checkbox:checked:not(:disabled)');
    const selectedVideos = Array.from(checkboxes).map(cb => {
        const name = cb.closest('tr').querySelector('.font-mono').innerText;
        return currentVideos.find(v => v.name === name);
    }).filter(Boolean);
    const tempPaths = selectedVideos.map(v => v.tempPath).filter(Boolean);

    if (tempPaths.length === 0) {

        alert('Một số file chưa upload xong. Vui lòng chờ!');
        return;
    }
    checkboxes.forEach(cb => {
        const row = cb.closest('tr');
        row.classList.add('is-processing');
        row.querySelector('.file-status').innerHTML =
            '<span class="inline-flex items-center gap-1.5 text-blue-600 font-medium">' +
            '<div class="w-3 h-3 rounded-full border-2 border-blue-600 border-t-transparent animate-spin"></div>' +
            '⏳Waiting... <span class="job-progress text-xs text-blue-400">0%</span></span>';
        cb.checked = false;
        cb.disabled = true;
    });
    document.getElementById('select-all').checked = false;
    try {
        const res = await fetch('/api/phase2/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ videos: tempPaths }),
        });

        if (!res.ok) {
            const err = await res.json();
            let errorMsg = 'Process thất bại';

            if (typeof err.detail === 'string') {
                errorMsg = err.detail; // Lỗi từ HTTPException thường
            } else if (Array.isArray(err.detail)) {
                // Bóc tách mảng lỗi 422 của Pydantic FastAPI
                errorMsg = err.detail.map(e => `${e.loc.join('.')} - ${e.msg}`).join(', ');
            }

            throw new Error(errorMsg);
        }
        const { job_ids } = await res.json();
        // job_ids.forEach((jobId, i) => startPolling(jobId, selectedVideos[i]?.name));
        job_ids.forEach((jobId, index) => {
            const video = selectedVideos[index];
            if (video) startPolling(jobId, video.name);
        });
    } catch (err) {
        alert('Lỗi khi gọi API: ' + err.message);
    }
}

function startPolling(jobId, videoName) {
    if (pollingTimers[jobId]) clearInterval(pollingTimers[jobId]);

    pollingTimers[jobId] = setInterval(async () => {
        try {
            const res = await fetch(`/api/phase2/status/${jobId}`);
            const data = await res.json();

            // 🔥 tìm đúng row
            const rows = document.querySelectorAll('#queue-body tr');
            let targetRow = null;

            rows.forEach(row => {
                const name = row.querySelector('.font-mono')?.innerText;
                if (name === videoName) targetRow = row;
            });

            if (!targetRow) return;

            const statusCell = targetRow.querySelector('.file-status');

            if (data.status === 'processing') {
                statusCell.innerHTML = '<span class="inline-flex items-center gap-1.5 text-blue-600 font-medium">' +
                    '<div class="w-3 h-3 rounded-full border-2 border-blue-600 border-t-transparent animate-spin"></div>' +
                    ' Processing... <span class="job-progress text-xs text-blue-400">' + (data.progress || 0) + '%</span></span>';
            } else if (data.status === 'waiting') {
                statusCell.innerHTML = '<span class="inline-flex items-center gap-1 text-orange-500 font-medium">⏳ Waiting...</span>';
            }

            if (['done', 'error', 'stopped'].includes(data.status)) {
                clearInterval(pollingTimers[jobId]);
                delete pollingTimers[jobId];
                onJobFinished(videoName, data);
            }

        } catch (err) {
            console.error(err);
        }
    }, 2000);
}

async function onJobFinished(videoName, data) {
    const videoNameStem = videoName.replace(/\.[^/.]+$/, "");

    for (const row of document.querySelectorAll('#queue-body tr')) {
        const cell = row.querySelector('.font-mono');
        if (!cell || cell.innerText !== videoName) continue;

        row.classList.remove('is-processing');
        const statusCell = row.querySelector('.file-status');
        const cb = row.querySelector('.file-checkbox');

        if (data.status === 'done') {
            statusCell.innerHTML = `<span class="text-purple-600 font-medium">Downloading & Saving...</span>`;
            try {
                let resultRes = await fetch(`/api/phase2/result/${videoNameStem}`);
                if (resultRes.ok) {
                    let resultJson = await resultRes.json();
                    await downloadAndSaveToUserDisk(videoNameStem, resultJson, null);
                }
            } catch (err) {
                console.error('Error when download result back to user s device:', err);
            }

            // thong bao Done processing
            statusCell.innerHTML = `<span class="text-green-600 font-medium">✓ Done (${data.result?.total_detections ?? 0} detections)</span>`;
            processedFolders.add(videoName.replace(/\.[^/.]+$/, ''));
            row.classList.add('bg-slate-50', 'opacity-80');
        } else if (data.status === 'error') {
            statusCell.innerHTML = `<span class="text-red-500 font-medium">✗ Lỗi: ${data.error || 'Unknown'}</span>`;
            if (cb) cb.disabled = false;
        } else if (data.status === 'stopped') {
            statusCell.innerHTML = `<span class="text-yellow-600 font-medium">⏸ Đã dừng</span>`;
            if (cb) cb.disabled = false;
        }
        break;
    }
}

// Hiển thị Modal Cảnh báo khi bấm Stop All
// function promptStopProcessing() {
//     const processingRows = document.querySelectorAll('tr.is-processing');
//     if (processingRows.length === 0) {
//         alert('Không có tiến trình nào đang chạy!');
//         return;
//     }
//     document.getElementById('stop-count').innerText = processingRows.length;
//     document.getElementById('stop-confirm-modal').classList.remove('hidden');
// }

async function confirmStopProcessing() {
    hideStopModal();
    try {
        await fetch('/api/phase2/stop-all', { method: 'POST' });
    } catch (err) {
        console.error('Stop all lỗi:', err);
    }
    Object.values(pollingTimers).forEach(clearInterval);
    pollingTimers = {};
}

// Ẩn Modal Stop
function hideStopModal() { document.getElementById('stop-confirm-modal').classList.add('hidden'); }

// Xác nhận Dừng mọi tiến trình đang chạy
// function confirmStopProcessing() {
//     hideStopModal();
//     const processingRows = document.querySelectorAll('tr.is-processing');

//     processingRows.forEach(row => {
//         // Gỡ bỏ cờ đánh dấu 'is-processing'
//         row.classList.remove('is-processing');
//         const statusCell = row.querySelector('.file-status');
//         const cb = row.querySelector('.file-checkbox');

//         // Reset trạng thái hiển thị về lại Ready và mở khóa checkbox
//         statusCell.innerHTML = '<span class="inline-flex items-center gap-1 text-slate-600 font-medium">⏳ Ready</span>';
//         cb.disabled = false; 
//     });
// }

// Xóa thư mục kết quả (Mô phỏng UI)
// function deleteOutputFolder(btn, folderName) {
//     if(confirm(`Xác nhận xóa thư mục kết quả [ ${folderName} ] trên UI để chạy lại?`)) {
//         // Xóa tên thư mục khỏi Set bộ nhớ
//         processedFolders.delete(folderName);

//         // Render lại bảng, lúc này đối chiếu folderName sẽ ra False -> Chuyển dòng về Ready
//         renderQueueUI();

//         // NOTE: Thực tế hệ thống sẽ cần gọi fetch() xuống API Backend để xóa vật lý folder trên ổ cứng.
//     }
// }

async function deleteOutputFolder(btn, folderName) {
    if (!confirm(`Xác nhận xóa thư mục kết quả [ ${folderName} ] trên server?`)) return;
    try {
        const res = await fetch('/api/phase2/delete-output', {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ output_folder: selectedOutputPath, folder_name: folderName }),
        });
        if (!res.ok) {
            const err = await res.json();
            alert('Xóa thất bại: ' + (err.detail || 'Unknown error'));
            return;
        }
        processedFolders.delete(folderName);
        renderQueueUI();
    } catch (err) {
        alert('Lỗi kết nối: ' + err.message);
    }
}