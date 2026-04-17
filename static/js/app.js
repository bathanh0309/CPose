/* CPose Studio - Core Dashboard Logic */

const state = {
    running: false,
    mode: 'standby', // 'rtsp', 'multicam_folder'
    system: {
        resourcesLoaded: false,
        folderLoaded: false
    },
    cameras: {
        registrationCamOpen: false
    },
    preview: {
        rtspVisible: true,
        originalVisible: true,
        processedVisible: true
    },
    cams: [],
    clips: [],
    eventLogs: [], // Left summary logs
    metricLogs: [], // Middle technical logs
    maxLogs: 300,
    selectedClip: null,
    polling: {
        multicam: null,
        registration: null,
        original: null,
        processed: null
    }
};

const socket = io();

// ===================================================================
//                          CORE INITIALIZATION
// ===================================================================

document.addEventListener('DOMContentLoaded', () => {
    initSocket();
    bindEvents();
    refreshSystemState();
    pushEventLog("Hệ thống đã sẵn sàng", "SYS");
});

function initSocket() {
    socket.on('connect', () => {
        pushEventLog("Kết nối Socket.IO thành công", "WS");
        updateMainStatus("CONNECTED", "green text-emerald-500");
    });

    socket.on('disconnect', () => {
        pushEventLog("Mất kết nối Socket.IO - Hệ thống đang Offline", "WS");
        updateMainStatus("DISCONNECTED", "danger text-red-500");
    });

    socket.on('connect_error', (error) => {
        console.error("Socket Error:", error);
        updateMainStatus("OFFLINE", "danger");
    });

    socket.on('rec_status', (data) => {
        handleRecStatus(data);
    });

    socket.on('camera_status', (data) => {
        // Option to update cam list if needed
    });

    socket.on('clip_saved', (data) => {
        // data: { filename, cam_id, duration, path, ... }
        pushEventLog(`Clip mới: ${data.filename}`, data.cam_id);
        const newClip = {
            id: Date.now(),
            name: data.filename,
            cam: data.cam_id || "Unk",
            path: data.path,
            status: 'Done',
            progress: 100,
            timestamp: new Date().toLocaleTimeString()
        };
        state.clips.unshift(newClip);
        renderClipQueue();
    });

    socket.on('pose_progress', (data) => {
        // Technical real-time data
        pushMetricLog({
            time: nowTimeString(),
            cam: data.cam_id || "--",
            fps: data.fps || 0,
            frame: data.frame || 0,
            conf: data.conf ? (data.conf * 100).toFixed(0) + "%" : "--",
            adl: data.adl || "--",
            event: data.event || "Processing"
        });
    });

    socket.on('error', (data) => {
        showToast(data.message || "Lỗi hệ thống", "danger");
        pushEventLog(`Lỗi: ${data.message}`, "ERR");
    });
}

function bindEvents() {
    // Pipeline Controls (Refactored for Phase 1 & 2)
    document.getElementById('startRecBtn').addEventListener('click', startRecorder);
    document.getElementById('startAnalysisBtn').addEventListener('click', startAnalyzer);
    document.getElementById('stopAllBtn').addEventListener('click', stopAll);
    document.getElementById('refreshBtn').addEventListener('click', refreshSystemState);

    // File Inputs
    document.getElementById('resourcesInput').addEventListener('change', handleResourcesUpload);
    document.getElementById('multicamInput').addEventListener('change', handleMulticamFolderUpload);

    // Toggles
    document.getElementById('toggleRtspPreviewBtn').addEventListener('click', toggleMulticamPreview);
    document.getElementById('toggleOriginalPreviewBtn').addEventListener('click', toggleOriginalPreview);
    document.getElementById('toggleProcessedPreviewBtn').addEventListener('click', toggleProcessedPreview);
    document.getElementById('openWebcamBtn').addEventListener('click', toggleRegistrationWebcam);

    // Searches
    document.getElementById('leftEventSearch').addEventListener('input', (e) => renderLeftEventTable(e.target.value));
    document.getElementById('metricSearchInput').addEventListener('input', (e) => renderMetricTable(e.target.value));

    // Exports
    document.getElementById('leftExportCsvBtn').addEventListener('click', exportSummaryCsv);
    document.getElementById('metricExportCsvBtn').addEventListener('click', exportMetricCsv);

    // Clip Queue
    document.getElementById('batchProcessBtn').addEventListener('click', startAnalyzer);
}

// ===================================================================
//                          PIPELINE HANDLERS
// ===================================================================

async function startRecorder() {
    if (state.running) {
        showToast("Hệ thống đang bận", "warning");
        return;
    }
    
    if (!state.system.resourcesLoaded) {
        showToast("Vui lòng nạp resources.txt trước khi ghi hình", "warning");
        return;
    }

    setMainButtonsBusy(true);
    state.mode = 'rtsp';

    try {
        const payload = { cameras: state.cams.map(c => c.cam_id) };
        const res = await fetch('/api/recording/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        if (res.ok) {
            state.running = true;
            updateMainStatus("RECORDER ACTIVE", "blue");
            updateRecLamp("recording");
            pushEventLog("Bắt đầu Phase 1: RTSP Recorder", "SYS");
            startPreviewPolling();
        } else {
            throw new Error(data.error || "Lỗi khởi động Recorder");
        }
    } catch (err) {
        showToast(err.message, "danger");
        pushEventLog(`Lỗi Recorder: ${err.message}`, "ERR");
    } finally {
        setMainButtonsBusy(false);
    }
}

async function startAnalyzer() {
    if (state.running) {
        showToast("Hệ thống đang bận", "warning");
        return;
    }

    setMainButtonsBusy(true);
    state.mode = 'analyzer';

    try {
        const res = await fetch('/api/analysis/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ folder: "data/raw_videos" })
        });

        const data = await res.json();
        if (res.ok) {
            state.running = true;
            updateMainStatus("ANALYZER ACTIVE", "green");
            updateRecLamp("ready");
            pushEventLog("Bắt đầu Phase 2: Batch Analysis", "SYS");
            showToast(`Đang phân tích ${data.clips} clips...`, "success");
        } else {
            throw new Error(data.error || "Lỗi khởi động Analyzer");
        }
    } catch (err) {
        showToast(err.message, "danger");
        pushEventLog(`Lỗi Analyzer: ${err.message}`, "ERR");
    } finally {
        setMainButtonsBusy(false);
    }
}

async function stopAll() {
    if (!state.running) return;

    try {
        // Stop Recorder
        await fetch('/api/recording/stop', { method: 'POST' });
        // Stop Analyzer
        await fetch('/api/analysis/stop', { method: 'POST' });
        
        state.running = false;
        state.mode = 'standby';
        updateMainStatus("STOPPED", "danger");
        updateRecLamp("ready");
        pushEventLog("Đã dừng tất cả tiến trình", "SYS");
        stopPreviewPolling();
    } catch (err) {
        showToast("Lỗi khi dừng hệ thống", "danger");
    }
}

// ===================================================================
//                          FILE HANDLERS
// ===================================================================

async function handleResourcesUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    if (!file.name.endsWith('.txt')) {
        showToast("Chỉ nhận file .txt", "danger");
        resetFileInput('resourcesInput');
        return;
    }

    logSummary(`Đang nạp: ${file.name}`, "CFG");
    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/api/config/upload', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        
        if (res.ok) {
            state.cams = data.cameras || [];
            state.system.resourcesLoaded = true;
            state.system.folderLoaded = false;
            populateRtspSelect(state.cams);
            updateRecLamp("ready");
            showToast("Đã tải cấu hình RTSP", "success");
            pushEventLog(`Nạp thành công ${state.cams.length} Cams`, "CFG");
            updateMainStatus("RTSP READY");
        } else {
            throw new Error(data.error);
        }
    } catch (err) {
        showToast(err.message, "danger");
    } finally {
        resetFileInput('resourcesInput');
    }
}

function handleMulticamFolderUpload(e) {
    const files = Array.from(e.target.files);
    const videos = files.filter(f => f.name.match(/\.(mp4|avi|mov)$/i));

    if (videos.length === 0) {
        showToast("Thư mục không có video hợp lệ", "warning");
        return;
    }

    pushEventLog(`Nạp Folder Multicam: ${videos.length} clips`, "IO");
    
    // Cleanup old URLs
    cleanupClipObjectUrls(state.clips);

    // Multicam Sort: Time-first, Cam-second
    const sorted = videos.map(v => {
        const info = tryParseMulticamName(v.name);
        return { file: v, ...info };
    }).sort((a, b) => {
        if (a.time !== b.time) return a.time.localeCompare(b.time);
        return a.cam.localeCompare(b.cam);
    });

    state.clips = sorted.map(v => ({
        id: Math.random().toString(36).substr(2, 9),
        name: v.file.name,
        cam: v.cam,
        file: v.file,
        url: URL.createObjectURL(v.file),
        status: 'Ready',
        progress: 0,
        timestamp: nowTimeString()
    }));

    state.system.folderLoaded = true;
    state.system.resourcesLoaded = false;
    updateRecLamp("ready");
    renderClipQueue();
    updateMainStatus("FOLDER READY");
    showToast(`Đã nạp ${state.clips.length} clip multicam`, "success");
}

// ===================================================================
//                          UI TOGGLES & POLLING
// ===================================================================

function toggleMulticamPreview() {
    state.preview.rtspVisible = !state.preview.rtspVisible;
    const btn = document.getElementById('toggleRtspPreviewBtn');
    const icon = document.getElementById('rtspEyeIcon');
    const img = document.getElementById('slotSnap1');
    const empty = document.getElementById('slotEmpty1');

    if (state.preview.rtspVisible) {
        img.classList.remove('hidden');
        empty.classList.add('hidden');
        icon.className = 'fas fa-eye scale-90 text-blue-600';
    } else {
        img.classList.add('hidden');
        empty.classList.remove('hidden');
        icon.className = 'fas fa-eye-slash scale-90';
    }
}

function toggleOriginalPreview() {
    const img = document.getElementById('originalVideo');
    const empty = document.getElementById('originalVideoEmpty');
    const btn = document.getElementById('toggleOriginalPreviewBtn');
    const icon = btn.querySelector('i');

    if (state.preview.originalVisible) {
        state.preview.originalVisible = false;
        if (state.polling.original) clearInterval(state.polling.original);
        img.src = '';
        img.classList.add('hidden');
        empty.classList.remove('hidden');
        icon.classList.remove('fa-eye');
        icon.classList.add('fa-eye-slash');
        btn.classList.add('hidden-cam');
    } else {
        state.preview.originalVisible = true;
        empty.classList.add('hidden');
        img.classList.remove('hidden');
        icon.classList.remove('fa-eye-slash');
        icon.classList.add('fa-eye');
        btn.classList.remove('hidden-cam');
        startPreviewPolling();
    }
}

function toggleProcessedPreview() {
    const img = document.getElementById('processedVideo');
    const empty = document.getElementById('processedVideoEmpty');
    const btn = document.getElementById('toggleProcessedPreviewBtn');
    const icon = btn.querySelector('i');

    if (state.preview.processedVisible) {
        state.preview.processedVisible = false;
        if (state.polling.processed) clearInterval(state.polling.processed);
        img.src = '';
        img.classList.add('hidden');
        empty.classList.remove('hidden');
        icon.classList.remove('fa-eye');
        icon.classList.add('fa-eye-slash');
        btn.classList.add('hidden-cam');
    } else {
        state.preview.processedVisible = true;
        empty.classList.add('hidden');
        img.classList.remove('hidden');
        icon.classList.remove('fa-eye-slash');
        icon.classList.add('fa-eye');
        btn.classList.remove('hidden-cam');
        startPreviewPolling();
    }
}

async function toggleRegistrationWebcam() {
    const video = document.getElementById('liveWebcam');
    const empty = document.getElementById('webcamEmpty');
    const btn = document.getElementById('openWebcamBtn');

    if (state.cameras.registrationCamOpen) {
        // Close Webcam
        const stream = video.srcObject;
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
        video.srcObject = null;
        video.classList.add('hidden');
        empty.classList.remove('hidden');
        state.cameras.registrationCamOpen = false;
        btn.textContent = 'Open';
        btn.classList.remove('webcam-open', 'bg-red-500');
        btn.classList.add('bg-emerald-500');
        pushEventLog("Webcam closed", "REG");
    } else {
        // Open Webcam
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            video.classList.remove('hidden');
            empty.classList.add('hidden');
            state.cameras.registrationCamOpen = true;
            btn.textContent = 'Close';
            btn.classList.remove('bg-emerald-500');
            btn.classList.add('webcam-open');
            pushEventLog("Webcam opened", "REG");
        } catch (err) {
            showToast("Could not access webcam: " + err.message, "danger");
            pushEventLog("Webcam error: " + err.message, "REG");
        }
    }
}

// ===================================================================
//                          LOGGING & TABLES
// ===================================================================

function pushEventLog(event, camId = "--") {
    const tbody = document.getElementById('leftEventTableBody');
    if (!tbody) return;

    if (state.eventLogs.length === 0) tbody.innerHTML = ""; // Clear placeholder

    const time = new Date().toLocaleTimeString('vi-VN', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    
    const row = document.createElement('tr');
    row.innerHTML = `
        <td class="text-slate-400 font-mono text-[10px] p-2">${time}</td>
        <td class="font-medium text-slate-700 p-2">${escapeHtml(event)}</td>
        <td class="text-center p-2"><span class="bg-blue-50 text-blue-600 px-1.5 py-0.5 rounded text-[10px] font-bold">${camId}</span></td>
    `;
    
    tbody.prepend(row);
    state.eventLogs.unshift({ time, event, cam: camId });

    if (tbody.children.length > 50) tbody.removeChild(tbody.lastChild);
    if (state.eventLogs.length > state.maxLogs) state.eventLogs.pop();
}

function pushMetricLog(data) {
    const tbody = document.getElementById('metricLogsBody');
    if (!tbody) return;

    if (state.metricLogs.length === 0) tbody.innerHTML = ""; // Clear placeholder

    const row = document.createElement('tr');
    
    // Highlight ADL nếu có cảnh báo nghiêm trọng
    const isDangerous = data.adl && (data.adl.toLowerCase().includes('fall') || data.adl.toLowerCase().includes('lying'));
    const adlClass = isDangerous ? 'text-red-500 font-bold' : 'text-emerald-600 font-bold';

    row.innerHTML = `
        <td class="font-bold text-slate-700 p-2">CAM-${data.cam}</td>
        <td class="text-blue-600 font-black p-2">${data.fps}</td>
        <td class="text-slate-400 p-2">#${data.frame}</td>
        <td class="font-bold p-2">${data.conf}</td>
        <td class="${adlClass} uppercase p-2">${data.adl}</td>
        <td class="text-slate-500 italic truncate max-w-[200px] p-2">${escapeHtml(data.event || '--')}</td>
    `;

    tbody.prepend(row);
    state.metricLogs.unshift(data);

    if (tbody.children.length > 100) tbody.removeChild(tbody.lastChild);
    if (state.metricLogs.length > state.maxLogs) state.metricLogs.pop();
}

// ===================================================================
//                          HELPERS
// ===================================================================

function nowTimeString() {
    return new Date().toLocaleTimeString('vi-VN', { hour12: false });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function getConfClass(confStr) {
    const val = parseInt(confStr);
    if (val >= 80) return 'text-emerald-500';
    if (val >= 50) return 'text-yellow-500';
    return 'text-red-400';
}

function updateMainStatus(text, colorClass = "blue") {
    const el = document.getElementById('mStatus');
    if (el) {
        el.textContent = text;
        el.className = `pill ${colorClass}`;
    }
    
    const modeEl = document.getElementById('systemMode');
    if (modeEl) {
        modeEl.textContent = text.split(' ')[0];
        // Đồng bộ màu sắc cho mode summary
        if (text.includes("CONNECTED")) modeEl.className = "font-mono text-emerald-500 font-bold";
        if (text.includes("DISCONNECTED") || text.includes("OFFLINE")) modeEl.className = "font-mono text-red-500 font-bold";
    }
}

function updateRecLamp(stateName) {
    const lamp = document.getElementById('recIndicator');
    const text = document.getElementById('recIndicatorText');
    
    lamp.className = 'w-2 h-2 rounded-full';
    if (stateName === 'off') {
        lamp.classList.add('bg-red-400');
        text.textContent = 'OFF';
    } else if (stateName === 'ready') {
        lamp.classList.add('bg-emerald-400', 'active');
        text.textContent = 'READY';
    } else if (stateName === 'recording') {
        lamp.classList.add('bg-red-500', 'recording');
        text.textContent = 'REC';
    }
}

function showToast(msg, type = "info") {
    const container = document.getElementById('toasts');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = msg;
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

function resetFileInput(id) {
    document.getElementById(id).value = "";
}

function setMainButtonsBusy(flag) {
    document.getElementById('startRecBtn').disabled = flag;
    document.getElementById('startAnalysisBtn').disabled = flag;
    document.getElementById('stopAllBtn').disabled = !flag && !state.running;
}

function cleanupClipObjectUrls(clips) {
    clips.forEach(c => {
        if (c.url && c.url.startsWith('blob:')) URL.revokeObjectURL(c.url);
    });
}

function tryParseMulticamName(name) {
    // cam1_2026-01-29_16-26-25.mp4
    const match = name.match(/(cam\d+)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})/);
    if (match) return { cam: match[1], time: match[2] };
    return { cam: "Unk", time: name };
}

function populateRtspSelect(cams) {
    const sel = document.getElementById('slotSel1');
    sel.innerHTML = cams.map(c => `<option value="${c.cam_id}">${c.label || c.cam_id}</option>`).join('');
}

// ===================================================================
//                          CSV EXPORT
// ===================================================================

function exportSummaryCsv() {
    const headers = ["TIME", "EVENT", "CAM"];
    const rows = state.eventLogs.map(l => [l.time, l.event, l.cam]);
    downloadCsv("CPose_Summary_Logs.csv", headers, rows);
}

function exportMetricCsv() {
    const headers = ["TIME", "CAM", "FPS", "FRAME", "CONF", "ADL", "EVENT"];
    const rows = state.metricLogs.map(l => [l.time, l.cam, l.fps, l.frame, l.conf, l.adl, l.event]);
    downloadCsv("CPose_Technical_Logs.csv", headers, rows);
}

function downloadCsv(filename, headers, rows) {
    const escape = (v) => `"${String(v).replace(/"/g, '""')}"`;
    const content = [
        headers.map(escape).join(','),
        ...rows.map(r => r.map(escape).join(','))
    ].join('\n');
    
    const blob = new Blob(["\ufeff", content], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", filename);
    link.click();
    URL.revokeObjectURL(url);
}

// ===================================================================
//                          PREVIEW POLLING
// ===================================================================

function startPreviewPolling() {
    if (state.polling.original) return;
    
    state.polling.original = setInterval(() => {
        if (!state.preview.originalVisible) return;
        document.getElementById('originalVideo').src = `/api/pose/snapshot/original?t=${Date.now()}`;
    }, 200);

    state.polling.processed = setInterval(() => {
        if (!state.preview.processedVisible) return;
        document.getElementById('processedVideo').src = `/api/pose/snapshot/processed?t=${Date.now()}`;
    }, 200);
}

function stopPreviewPolling() {
    clearInterval(state.polling.original);
    clearInterval(state.polling.processed);
    state.polling.original = null;
    state.polling.processed = null;
    
    // Set to fixed dummy
    const dummy = "data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=";
    document.getElementById('originalVideo').src = dummy;
    document.getElementById('processedVideo').src = dummy;
}

function clearLogs() {
    state.eventLogs = [];
    state.metricLogs = [];
    document.getElementById('leftEventTableBody').innerHTML = '<tr><td colspan="3" class="text-center py-10 text-slate-300 italic">Chưa có sự kiện</td></tr>';
    document.getElementById('metricLogsBody').innerHTML = '<tr><td colspan="6" class="text-center py-10 text-slate-300 italic">Chờ dữ liệu...</td></tr>';
    showToast("Đã xóa trắng log", "info");
}

function renderClipQueue() {
    const container = document.getElementById('clipQueue');
    if (state.clips.length === 0) {
        container.innerHTML = '<div class="viewer-empty opacity-40 text-[10px] !h-32">Chưa có clip xử lý</div>';
        return;
    }

    container.innerHTML = state.clips.map(c => `
        <div class="bg-white p-3 rounded-2xl border border-pink-100 hover:border-pink-300 transition-all cursor-pointer group shadow-sm flex flex-col gap-2" onclick="playClip('${c.id}')">
            <div class="flex justify-between items-center">
                <span class="text-[10px] font-black text-pink-700 truncate max-w-[150px]">${c.name}</span>
                <span class="pill blue">${c.cam}</span>
            </div>
            <div class="flex justify-between items-center text-[9px] text-slate-400">
                <div class="flex items-center gap-1">
                    <i class="fas fa-clock opacity-50"></i> ${c.timestamp}
                </div>
                <div class="font-bold text-emerald-500">${c.status}</div>
            </div>
            <div class="h-1 bg-slate-100 rounded-full overflow-hidden">
                <div class="h-full bg-pink-500 transition-all duration-500" style="width: ${c.progress}%"></div>
            </div>
        </div>
    `).join('');
}

function playClip(id) {
    const clip = state.clips.find(c => c.id === id);
    if (!clip) return;
    
    state.selectedClip = clip;
    const player = document.getElementById('previewVideoLocal');
    const empty = document.getElementById('previewVideoEmpty');
    
    empty.classList.add('hidden');
    player.src = clip.url || `/api/video/${clip.path}`;
    player.play();
    
    document.getElementById('outputDirLabel').textContent = `PLAYING: ${clip.name}`;
    pushEventLog(`Mở xem clip: ${clip.name}`, clip.cam);
}

function refreshSystemState() {
    pushEventLog("Đang đồng bộ trạng thái hệ thống...", "SYS");
    // Fetch and sync UI
    fetch('/api/config/cameras')
        .then(res => res.json())
        .then(data => {
            if (data.cameras && data.cameras.length > 0) {
                state.cams = data.cameras;
                state.system.resourcesLoaded = true;
                populateRtspSelect(state.cams);
                updateRecLamp("ready");
            }
        });
}
