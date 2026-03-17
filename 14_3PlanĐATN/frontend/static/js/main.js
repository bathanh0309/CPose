// ─────────────────────────────────────────────────────────────────────────────
//  RTSP Monitor — main.js
//  Bugs fixed:
//    1. _streamRunning typo (_streamsRunning) → nút toggle không hoạt động
//    2. prefillFromConfig() phải chạy SAU buildSourceConfigs() — elements phải tồn tại trước
//    3. Bỏ startAll() / stopAll() thừa, chỉ dùng toggleStream()
// ─────────────────────────────────────────────────────────────────────────────



// ─── Collapsible ─────────────────────────────────────────────────────────────
function toggleBlock(id) {
    const body  = document.getElementById(id + '-body');
    const arrow = document.getElementById(id + '-arrow');
    const isOpen = !body.classList.contains('collapsed');
    body.classList.toggle('collapsed', isOpen);
    arrow.classList.toggle('open', !isOpen);
}

// ─── Camera Grid ─────────────────────────────────────────────────────────────
const CAM_LABELS = ["CAM-01", "CAM-02", "CAM-03", "CAM-04"];

function buildGrid() {
    const grid = document.getElementById("camGrid");
    grid.innerHTML = "";
    CAM_LABELS.forEach((lbl, i) => {
        grid.innerHTML += `
        <div class="cam-cell" id="cell${i}">
            <div class="cam-header">
                <span class="cam-label">${lbl}</span>
                <span class="cam-badge" id="badge${i}"></span>
                <span class="cam-person-badge" id="personBadge${i}"></span>
                <span class="cam-status st-offline" id="status${i}">○ OFFLINE</span>
            </div>
            <div class="cam-video-wrap">
                <img class="cam-video" id="video${i}" src="/video/${i}" alt="cam${i}">
            </div>
        </div>`;
    });
}

// ─── Source Config ────────────────────────────────────────────────────────────

function buildSourceConfigs() {
    const container = document.getElementById("sourceConfigs");
    container.innerHTML = "";
    for (let i = 0; i < 4; i++) {
        container.innerHTML += `
        <div class="source-row">
            <label>Camera ${i + 1}</label>
            <input class="url-input" id="urlInput${i}" type="text"
                    placeholder="rtsp://user:pass@192.168.1.${10 + i}:554/stream">
        </div>`;
    }
}

// ─── Prefill từ cameras.json (qua API) ───────────────────────────────────────
// BUG FIX 2: hàm này phải được gọi SAU buildSourceConfigs()
async function prefillFromConfig() {
    try {
        const config = await fetch("/api/camera_config").then(r => r.json());
        config.forEach(entry => {
            const i      = entry.id;
            const source = entry.source;
            if (source === "" || source === null || source === undefined) return;
            if (i < 0 || i >= 4) return;

            if(typeof source === "string" && source !==  ""){
                // URL
                document.getElementById(`url${i}`).checked = true;
                toggleSource(i);
                document.getElementById(`urlInput${i}`).value = source;
            }
        });
    } catch (e) {
        console.warn("prefillFromConfig failed:", e);
    }
}


// ─── Toggle Stream (Start / Stop) ────────────────────────────────────────────
// BUG FIX 1: dùng một tên biến nhất quán: _streamRunning
let _streamRunning = false;

async function toggleStream() {
    const btn = document.getElementById("mainToggleBtn");

    if (_streamRunning) {
        // ── STOP ──
        await fetch("/api/stop", { method: "POST" });
        for (let i = 0; i < 4; i++) {
            const img = document.getElementById(`video${i}`);
            img.src = "";
            // Delay nhỏ để browser ngắt connection MJPEG cũ trước khi reconnect
            setTimeout(() => { img.src = `/video/${i}?t=${Date.now()}`; }, 300);
        }
        _streamRunning = false;
        btn.className   = "btn btn-start";
        btn.textContent = "▶ START ALL";

    } else {
        // ── START: đọc giá trị từ input (có thể override so với cameras.json) ──
        const payload = [];
        for (let i = 0; i < 4; i++) {
            let source  = null;
            const v = document.getElementById(`urlInput${i}`).value.trim();
            source  = v || null;
            payload.push({ source });
        }

        await fetch("/api/start", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify(payload)
        });

        for (let i = 0; i < 4; i++) {
            document.getElementById(`video${i}`).src = `/video/${i}?t=${Date.now()}`;
        }

        // BUG FIX 1: set đúng biến _streamRunning (không phải _streamsRunning)
        _streamRunning  = true;
        btn.className   = "btn btn-stop";
        btn.textContent = "⏹ STOP ALL";
    }
}

// ─── Detection Toggle (per camera) ───────────────────────────────────────────
async function toggleDetection(cam_id, enabled) {
    await fetch("/api/detection_toggle", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ cam_id, enabled })
    });
}

// ─── Status Polling ───────────────────────────────────────────────────────────
const STATUS_MAP = {
    live:       { cls: "st-live",       txt: "● LIVE" },
    connecting: { cls: "st-connecting", txt: "◌ CONNECTING" },
    error:      { cls: "st-error",      txt: "✕ ERROR" },
    offline:    { cls: "st-offline",    txt: "○ OFFLINE" },
};

async function pollStatus() {
    try {
        const data  = await fetch("/api/status").then(r => r.json());
        const tbody = document.getElementById("infoBody");

        data.forEach((cam, i) => {
            const m = STATUS_MAP[cam.status] || STATUS_MAP.offline;

            // Header status text
            const st    = document.getElementById(`status${i}`);
            st.className   = "cam-status " + m.cls;
            st.textContent = m.txt;

            // Source badge (LOCAL / RTSP)
            const badge = document.getElementById(`badge${i}`);
            if (cam.info && cam.info.startsWith("LOCAL")) {
                badge.className = "cam-badge local"; badge.textContent = "LOCAL";
            } else if (cam.info && cam.info.startsWith("RTSP")) {
                badge.className = "cam-badge rtsp";  badge.textContent = "RTSP";
            } else {
                badge.className = "cam-badge";
            }

            // Person badge trên header camera
            const pb    = document.getElementById(`personBadge${i}`);
            const count = cam.person_count || 0;
            if (cam.status === "live" && cam.detection) {
                pb.className   = "cam-person-badge visible";
                pb.textContent = `👤 ${count}`;
            } else {
                pb.className = "cam-person-badge";
            }

            // Viền xanh khi detect được người
            const cell = document.getElementById(`cell${i}`);
            cell.classList.toggle("detecting", count > 0 && cam.status === "live");

            // Info table row
            const row = tbody.rows[i];
            row.cells[1].className   = m.cls;
            row.cells[1].textContent = cam.status.toUpperCase();

            const pc = row.cells[2];
            if (cam.status === "live" && cam.detection) {
                pc.className   = count > 0 ? "person-count-cell has-person" : "person-count-cell no-person";
                pc.textContent = String(count);
            } else {
                pc.className   = "person-count-cell no-person";
                pc.textContent = "—";
            }
        });
    } catch (e) { /* silent — server có thể đang khởi động */ }
}

// ─── Log Polling ──────────────────────────────────────────────────────────────
let _lastLogTime = "";

async function pollLogs() {
    try {
        const entries = await fetch("/api/logs?limit=30").then(r => r.json());
        if (!entries.length) return;

        const newestTime = entries[0]?.time || "";
        if (newestTime === _lastLogTime) return;   // không có gì mới
        _lastLogTime = newestTime;

        const list = document.getElementById("logList");
        document.getElementById("logCount").textContent = `${entries.length} event(s)`;

        list.innerHTML = entries.map(e => {
            const cropsHtml = (e.crops || []).map(f =>
                `<img class="log-crop-img"
                      src="/crops/${f}"
                      alt="crop"
                      onclick="openLightbox('/crops/${f}'); event.stopPropagation()"
                      onerror="this.style.display='none'">`
            ).join("");

            return `
            <div class="log-entry">
                <div class="log-entry-header">
                    <span class="log-cam">${e.cam_label}</span>
                    <span class="log-count">👤 ${e.count} person${e.count !== 1 ? "s" : ""}</span>
                    <span class="log-time">${e.time.replace("T", " ")}</span>
                </div>
                ${cropsHtml ? `<div class="log-crops">${cropsHtml}</div>` : ""}
            </div>`;
        }).join("");

    } catch (e) { /* silent */ }
}

async function clearLogs() {
    await fetch("/api/logs/clear", { method: "POST" });
    document.getElementById("logList").innerHTML = '<div class="log-empty">No detections yet.</div>';
    document.getElementById("logCount").textContent = "0 events";
    _lastLogTime = "";
}

// ─── Lightbox ─────────────────────────────────────────────────────────────────
function openLightbox(src) {
    document.getElementById("lightboxImg").src = src;
    document.getElementById("lightbox").classList.add("open");
}
function closeLightbox() {
    document.getElementById("lightbox").classList.remove("open");
}
document.addEventListener("keydown", e => { if (e.key === "Escape") closeLightbox(); });

// ─── Clock ────────────────────────────────────────────────────────────────────
function tickClock() {
    document.getElementById("clock").textContent =
        new Date().toISOString().replace("T", " ").slice(0, 19);
}

// ─── Init ─────────────────────────────────────────────────────────────────────
// BUG FIX 2: thứ tự đúng — buildGrid → buildSourceConfigs → prefillFromConfig
// Nếu đổi thứ tự, prefill sẽ tìm không thấy element và silently fail
buildGrid();
buildSourceConfigs();
prefillFromConfig();     // ← phải sau buildSourceConfigs
tickClock();
setInterval(tickClock,  1000);
setInterval(pollStatus, 1500);
setInterval(pollLogs,   3000);