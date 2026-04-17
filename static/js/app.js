/* 
  CPose Dashboard - Integrated Management (Yellow/Pink Phase)
  Clean, Modular, High-Performance
*/

const $ = (id) => document.getElementById(id);
const setText = (id, val) => { const el = $(id); if (el) el.textContent = val; };

// Variables & State
let socket = null;
const state = {
    running: false,
    clips: [],
    logs: [],
    selectedClips: new Set(),
};

// --- Socket.IO ---
function initSocket() {
    if (!window.io) return;
    socket = io();
    window.appSocket = socket;

    socket.on("connect", () => console.info("Socket connected"));
    
    socket.on("rec_status", (msg) => {
        setText("recordingState", msg.is_recording ? "ON" : "OFF");
        if (msg.is_recording) {
            $("startBtn")?.classList.add("opacity-50", "pointer-events-none");
            $("stopBtn")?.classList.remove("opacity-50", "pointer-events-none");
        } else {
            $("startBtn")?.classList.remove("opacity-50", "pointer-events-none");
            $("stopBtn")?.classList.add("opacity-50", "pointer-events-none");
        }
    });

    socket.on("pose_progress", (msg) => {
        // High frequency UI updates
        const container = $("pipelineProgressContainer");
        if (container) container.classList.remove("hidden");
        setText("pipelineProgressText", `${msg.pct || 0}%`);
        const bar = $("pipelineProgressBar");
        if (bar) bar.style.width = `${msg.pct || 0}%`;
    });

    socket.on("analysis_progress", (msg) => {
        const clip = state.clips.find(c => c.name === msg.clip || c.name.startsWith(msg.clip));
        if (clip) {
            clip.status = "running";
            clip.progress = msg.pct;
            clip.currentFrame = msg.frame_id;
            clip.totalFrames = msg.total_frames;
            renderClipQueue();
        }
    });

    socket.on("analysis_complete", () => {
        state.clips.forEach(c => { if(c.status === "running") c.status = "done"; });
        renderClipQueue();
        toast("Xử lý phân tích hoàn tất", "success");
    });
}

// --- UI Utilities ---
function toast(msg, type = "info") {
    const t = document.createElement("div");
    t.className = `p-3 rounded-lg text-xs font-bold shadow-lg animate-bounce bg-white border-l-4 ${type === 'success' ? 'border-green-500' : type === 'error' ? 'border-red-500' : 'border-blue-500'}`;
    t.textContent = msg;
    $("toasts").appendChild(t);
    setTimeout(() => t.remove(), 3000);
}

function logEvent(event, cam = "--") {
    state.logs.unshift({ time: new Date().toLocaleTimeString('vi-VN'), event, cam });
    if (state.logs.length > 50) state.logs.pop();
    renderLogs();
}

function renderLogs() {
    const body = $("logTable");
    if (!body) return;
    const query = ($("searchInput")?.value || "").toLowerCase();
    const rows = state.logs.filter(l => l.event.toLowerCase().includes(query) || l.cam.toLowerCase().includes(query));

    body.innerHTML = rows.map(log => `
        <tr class="hover:bg-white/5 transition-colors border-b border-white/5">
            <td class="p-2 text-[10px] text-slate-500 font-mono">${log.time}</td>
            <td class="p-2 text-slate-200">
                <span class="px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400 text-[9px] font-bold mr-2">LOG</span>
                ${log.event}
            </td>
            <td class="p-2 text-blue-400 font-bold text-center">${log.cam}</td>
        </tr>
    `).join("");
}

function renderDynamicCameras() {
    const tbody = $("summaryTableBody");
    if (!tbody) return;
    tbody.innerHTML = [1,2,3,4].map(i => {
        const cam = `CAM0${i}`;
        return `
            <tr class="border-b border-black/5">
                <td class="py-2">
                    <div class="flex items-center gap-2">
                        <div class="w-2 h-2 rounded-full bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.5)]"></div>
                        <span class="font-bold">${cam}</span>
                    </div>
                </td>
                <td class="text-right py-2 text-slate-500 font-mono">ID 0</td>
            </tr>
        `;
    }).join("");
}

function renderClipQueue() {
    const q = $("clipQueue");
    if (!q) return;
    
    if (state.clips.length === 0) {
        q.innerHTML = '<div class="text-xs text-pink-400 text-center py-10 italic">Chưa có clip chờ xử lý</div>';
        return;
    }

    q.innerHTML = state.clips.map(c => `
        <div class="bg-white p-3 rounded-xl border border-pink-200 shadow-sm cursor-pointer hover:bg-pink-50 transition-colors clip-item" data-id="${c.id}">
            <div class="flex justify-between items-start mb-2">
                <div class="text-[11px] font-bold truncate pr-4 text-pink-900">${c.name}</div>
                <div class="text-[9px] px-1.5 py-0.5 rounded-full ${c.status === 'done' ? 'bg-green-100 text-green-700' : c.status === 'running' ? 'bg-blue-100 text-blue-700 animate-pulse' : 'bg-slate-100 text-slate-500'} font-bold uppercase">
                    ${c.status}
                </div>
            </div>
            <div class="flex items-center gap-3 text-[10px] text-pink-600/70">
                <span><i class="fas fa-video"></i> ${c.cam}</span>
                <span><i class="fas fa-clock"></i> ${c.progress || 0}%</span>
            </div>
            ${c.status === 'running' ? `
                <div class="h-1 w-full bg-blue-100 rounded-full mt-2 overflow-hidden">
                    <div class="h-full bg-blue-500" style="width: ${c.progress}%"></div>
                </div>
            ` : ''}
        </div>
    `).join("");

    // Event listener for clip preview
    q.querySelectorAll(".clip-item").forEach(item => {
        item.onclick = () => {
            const clip = state.clips.find(c => c.id === item.dataset.id);
            if (clip && $("previewVideoLocal")) {
                $("previewVideoLocal").src = clip.url;
                $("previewVideoLocal").play();
                $("previewVideoEmpty").style.display = 'none';
                logEvent(`Xem trước clip: ${clip.name}`, clip.cam);
            }
        };
    });
}

// --- Handlers ---
$("resourcesInput")?.addEventListener("change", (e) => {
    const files = Array.from(e.target.files);
    files.filter(f => f.name.endsWith('.mp4') || f.name.endsWith('.avi')).forEach(f => {
        const id = Math.random().toString(36).substr(2, 9);
        const camMatch = f.name.match(/cam\d+/i);
        state.clips.push({
            id,
            name: f.name,
            cam: camMatch ? camMatch[0].toUpperCase() : 'UNK',
            url: URL.createObjectURL(f),
            status: 'ready',
            progress: 0
        });
    });
    renderClipQueue();
    logEvent(`Đã nạp ${files.length} video tài nguyên`);
    toast(`Nạp thành công ${files.length} clip`, "success");
});

$("startBtn")?.addEventListener("click", () => {
    fetch("/api/pose/start", { method: "POST", body: JSON.stringify({ mode: "stream" }) });
    logEvent("Bắt đầu Pipeline chính");
});

$("stopBtn")?.addEventListener("click", () => {
    fetch("/api/pose/stop", { method: "POST" });
    logEvent("Dừng Pipeline");
});

$("batchProcessBtn")?.addEventListener("click", async () => {
    if (state.clips.length === 0) return toast("Không có clip nào để xử lý", "error");
    toast("Đang gửi yêu cầu xử lý hàng loạt...", "info");
    logEvent("Bắt đầu xử lý hàng loạt");
    // Mock call for now
    state.clips.forEach(c => { if(c.status === 'ready') c.status = 'queued'; });
    renderClipQueue();
});

$("openWebcamBtn")?.addEventListener("click", async () => {
    const video = $("liveWebcam");
    if (!video) return;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        video.play();
        $("webcamEmpty").classList.add("hidden");
        logEvent("Mở webcam đăng ký");
    } catch (e) {
        toast("Không thể truy cập Webcam", "error");
    }
});

// Init
window.addEventListener("DOMContentLoaded", () => {
    initSocket();
    logEvent("Hệ thống CPose Studio khởi động...");
    setText("totalCamsValue", "4");
});
