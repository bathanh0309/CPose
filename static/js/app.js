/**
 * CPose — static/js/app.js
 * Dashboard logic: tab routing, API calls, Socket.IO real-time events.
 */

"use strict";

/* ═══════════════════════════════════════════════════════════════
   STATE
═══════════════════════════════════════════════════════════════ */
const state = {
  cameras: [],          // [{cam_id, url, width?, height?}]
  storageLimitGb: 10,
  recordingRunning: false,
  analysisRunning: false,
  clipCount: 0,
  detectCount: 0,
  selectedResolutions: {}, // {cam_id: {width, height}}
};

/* ═══════════════════════════════════════════════════════════════
   SOCKET.IO
═══════════════════════════════════════════════════════════════ */
const socket = io({ transports: ["websocket", "polling"] });

const dotSock  = document.getElementById("dot-sock");
const sockLbl  = document.getElementById("sock-lbl");

socket.on("connect", () => {
  dotSock.className = "dot dot-sm connected";
  sockLbl.textContent = "Đã kết nối";
  refreshStorageInfo();
});
socket.on("disconnect", () => {
  dotSock.className = "dot dot-sm disconnected";
  sockLbl.textContent = "Mất kết nối";
});

socket.on("camera_status", (d) => {
  updateCameraTile(d);
  addLog(`cam-${d.cam_id} ${d.status}${d.resolution ? " · " + d.resolution + " · " + d.fps + " fps" : ""}`,
    d.status === "error" ? "err" : "");
});

socket.on("detection_event", (d) => {
  state.detectCount++;
  document.getElementById("rec-detect-count").textContent = state.detectCount;
  addLog(`[cam-${d.cam_id}] ${d.count} người phát hiện`, "detect");
  pulseDotRec();
});

socket.on("clip_saved", (d) => {
  state.clipCount++;
  document.getElementById("rec-clip-count").textContent = state.clipCount;
  addLog(`[cam-${d.cam_id}] Clip lưu: ${d.filename}  ${d.size_mb} MB  ${d.duration_s}s`, "clip");
  refreshStorageInfo();
});

socket.on("analysis_progress", (d) => {
  showAnalysisProgress(d);
});

socket.on("analysis_complete", (d) => {
  finishAnalysis(d);
});

socket.on("storage_warning", (d) => {
  addLog(`⚠ Bộ nhớ ${d.used_gb}/${d.limit_gb} GB (${d.pct}%) — đang xóa clip cũ`, "warn");
  updateStorageBar(d.used_gb, d.limit_gb);
});

socket.on("error", (d) => {
  addLog(`Lỗi [${d.source}]: ${d.message}`, "err");
});

/* ═══════════════════════════════════════════════════════════════
   TAB NAVIGATION
═══════════════════════════════════════════════════════════════ */
const TAB_TITLES = {
  config:   "Cấu hình hệ thống",
  monitor:  "Live Monitor",
  analysis: "Phân tích — Phase 2",
  results:  "Kết quả",
};

document.querySelectorAll(".nav-item").forEach((el) => {
  el.addEventListener("click", (e) => {
    e.preventDefault();
    const tabId = el.dataset.tab;
    switchTab(tabId);
  });
});

function switchTab(tabId) {
  document.querySelectorAll(".nav-item").forEach((n) => n.classList.toggle("active", n.dataset.tab === tabId));
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${tabId}`));
  document.getElementById("page-title").textContent = TAB_TITLES[tabId] || tabId;

  if (tabId === "analysis") refreshVideoList();
  if (tabId === "results") refreshResults();
}

/* ═══════════════════════════════════════════════════════════════
   CONFIG TAB
═══════════════════════════════════════════════════════════════ */

// ── File Upload ──────────────────────────────────────────────
const dropZone  = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");

dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("over"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("over"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("over");
  const file = e.dataTransfer.files[0];
  if (file) uploadConfig(file);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) uploadConfig(fileInput.files[0]);
});

async function uploadConfig(file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch("/api/config/upload", { method: "POST", body: fd });
  const data = await res.json();
  const el = document.getElementById("config-result");
  el.classList.remove("hidden", "ok", "err");
  if (data.cameras) {
    el.classList.add("ok");
    el.textContent = `✓ Đã lưu — ${data.cameras.length} camera được tìm thấy`;
    state.cameras = data.cameras;
    renderCameraList(data.cameras);
  } else {
    el.classList.add("err");
    el.textContent = `✗ ${data.error || "Lỗi không xác định"}`;
  }
}

// ── Camera list ───────────────────────────────────────────────
async function fetchCameras() {
  const res  = await fetch("/api/config/cameras");
  const data = await res.json();
  if (data.cameras && data.cameras.length) {
    state.cameras = data.cameras;
    renderCameraList(data.cameras);
  }
}

function renderCameraList(cameras) {
  const el = document.getElementById("camera-list");
  if (!cameras.length) { el.innerHTML = '<p class="empty-hint">Chưa có camera</p>'; return; }
  el.innerHTML = cameras.map((c) => `
    <div class="camera-item" id="cam-item-${c.cam_id}">
      <span class="cam-id-tag">cam${c.cam_id}</span>
      <span class="cam-url" title="${c.url}">${c.url}</span>
      <span class="cam-res" id="res-tag-${c.cam_id}">—</span>
      <button class="btn btn-sm btn-ghost" onclick="probeCamera('${c.cam_id}','${c.url}')">Probe</button>
    </div>
  `).join("");
}

// ── Probe all ─────────────────────────────────────────────────
document.getElementById("btn-probe-all").addEventListener("click", async () => {
  for (const cam of state.cameras) {
    await probeCamera(cam.cam_id, cam.url, false);
  }
});

// ── Probe single ──────────────────────────────────────────────
async function probeCamera(camId, url, showModal = true) {
  if (showModal) {
    document.getElementById("probe-cam-id").textContent = `cam${camId}`;
    document.getElementById("probe-result").innerHTML = '<div class="loader"></div>';
    document.getElementById("modal-probe").classList.remove("hidden");
  }

  const res  = await fetch("/api/cameras/probe", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, cam_id: camId }),
  });
  const data = await res.json();

  // Update res tag in list
  const tag = document.getElementById(`res-tag-${camId}`);
  if (tag && !data.error) tag.textContent = `${data.width}×${data.height}`;

  if (!showModal) return;

  if (data.error) {
    document.getElementById("probe-result").innerHTML = `<p style="color:var(--danger)">${data.error}</p>`;
    return;
  }

  // Store default resolution (native)
  state.selectedResolutions[camId] = { width: data.width, height: data.height };

  document.getElementById("probe-result").innerHTML = `
    <dl class="probe-info">
      <dt>Camera</dt>     <dd>cam${data.cam_id}</dd>
      <dt>Độ phân giải</dt><dd>${data.width} × ${data.height}</dd>
      <dt>FPS</dt>        <dd>${data.fps} fps</dd>
      <dt>URL</dt>        <dd style="font-size:11px;font-family:var(--font-mono);word-break:break-all">${data.url}</dd>
    </dl>
    <p style="font-size:12px;color:var(--text-secondary);margin-bottom:6px">Chọn độ phân giải ghi:</p>
    <div class="res-options">
      ${data.resolutions.map((r) => `
        <button class="res-btn ${r.width === data.width ? "selected" : ""}"
          onclick="selectResolution('${camId}', ${r.width}, ${r.height}, this)">
          ${r.label}
        </button>
      `).join("")}
    </div>
  `;
}

function selectResolution(camId, width, height, btn) {
  state.selectedResolutions[camId] = { width, height };
  btn.closest(".res-options").querySelectorAll(".res-btn").forEach((b) => b.classList.remove("selected"));
  btn.classList.add("selected");
  const tag = document.getElementById(`res-tag-${camId}`);
  if (tag) tag.textContent = `${width}×${height}`;
}

document.getElementById("modal-close").addEventListener("click", () => {
  document.getElementById("modal-probe").classList.add("hidden");
});

// ── Storage limit ─────────────────────────────────────────────
document.getElementById("btn-set-limit").addEventListener("click", async () => {
  const gb = parseFloat(document.getElementById("storage-limit").value);
  state.storageLimitGb = gb;
  await fetch("/api/storage/limit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ limit_gb: gb }),
  });
  refreshStorageInfo();
});

async function refreshStorageInfo() {
  const res  = await fetch("/api/storage/info");
  const data = await res.json();
  const limit = state.storageLimitGb;
  document.getElementById("storage-text").textContent = `${data.used_gb} GB`;
  document.getElementById("storage-info-detail").textContent =
    `Đang dùng: ${data.used_gb} GB / ${limit} GB  (${data.file_count} clip)`;
  updateStorageBar(data.used_gb, limit);
}

function updateStorageBar(usedGb, limitGb) {
  const pct = Math.min(100, (usedGb / limitGb) * 100);
  const fill = document.getElementById("storage-fill");
  fill.style.width = `${pct}%`;
  fill.className = "storage-fill" + (pct > 90 ? " danger" : pct > 70 ? " warn" : "");
}

/* ═══════════════════════════════════════════════════════════════
   MONITOR TAB
═══════════════════════════════════════════════════════════════ */

// ── Start recording ───────────────────────────────────────────
document.getElementById("btn-start-rec").addEventListener("click", async () => {
  if (!state.cameras.length) {
    alert("Chưa có camera nào được cấu hình. Hãy tải lên resources.txt trước.");
    return;
  }

  const camerasPayload = state.cameras.map((c) => {
    const res = state.selectedResolutions[c.cam_id] || {};
    return { cam_id: c.cam_id, url: c.url, width: res.width, height: res.height };
  });

  const res = await fetch("/api/recording/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cameras: camerasPayload, storage_limit_gb: state.storageLimitGb }),
  });
  const data = await res.json();
  if (data.error) { addLog(`Lỗi: ${data.error}`, "err"); return; }

  state.recordingRunning = true;
  state.clipCount = 0; state.detectCount = 0;
  document.getElementById("rec-clip-count").textContent = "0";
  document.getElementById("rec-detect-count").textContent = "0";
  document.getElementById("btn-start-rec").disabled = true;
  document.getElementById("btn-stop-rec").disabled  = false;
  document.getElementById("dot-rec").className = "dot active";
  document.getElementById("lbl-rec").textContent = "Đang ghi…";

  renderCameraGrid(state.cameras);
  addLog(`Bắt đầu ghi ${state.cameras.length} camera`, "clip");
});

// ── Stop recording ────────────────────────────────────────────
document.getElementById("btn-stop-rec").addEventListener("click", async () => {
  await fetch("/api/recording/stop", { method: "POST" });
  state.recordingRunning = false;
  document.getElementById("btn-start-rec").disabled = false;
  document.getElementById("btn-stop-rec").disabled  = true;
  document.getElementById("dot-rec").className = "dot";
  document.getElementById("lbl-rec").textContent = "Đã dừng";
  addLog("Ghi dừng", "warn");
});

// ── Camera grid ───────────────────────────────────────────────
function renderCameraGrid(cameras) {
  const grid = document.getElementById("camera-grid");
  grid.innerHTML = cameras.map((c) => `
    <div class="cam-tile" id="tile-${c.cam_id}">
      <div class="cam-tile-header">
        <span><span class="cam-id-tag">cam${c.cam_id}</span></span>
        <span class="dot" id="tile-dot-${c.cam_id}"></span>
      </div>
      <div class="cam-tile-body">
        <div class="cam-tile-stat"><span>Trạng thái</span><span id="tile-st-${c.cam_id}">Đang kết nối…</span></div>
        <div class="cam-tile-stat"><span>Phân giải</span><span id="tile-res-${c.cam_id}">—</span></div>
        <div class="cam-tile-stat"><span>FPS</span><span id="tile-fps-${c.cam_id}">—</span></div>
      </div>
    </div>
  `).join("");
}

function updateCameraTile(d) {
  const st  = document.getElementById(`tile-st-${d.cam_id}`);
  const res = document.getElementById(`tile-res-${d.cam_id}`);
  const fps = document.getElementById(`tile-fps-${d.cam_id}`);
  const dot = document.getElementById(`tile-dot-${d.cam_id}`);
  if (!st) return;
  st.textContent  = d.status;
  res.textContent = d.resolution || "—";
  fps.textContent = d.fps ? `${d.fps} fps` : "—";
  dot.className   = "dot" + (d.status === "streaming" ? " active" : d.status === "error" ? " error" : "");
}

function pulseDotRec() {
  const dot = document.getElementById("dot-rec");
  dot.style.opacity = "0.3";
  setTimeout(() => { dot.style.opacity = "1"; }, 200);
}

// ── Log ───────────────────────────────────────────────────────
document.getElementById("btn-clear-log").addEventListener("click", () => {
  document.getElementById("event-log").innerHTML = '<p class="empty-hint">Chưa có sự kiện</p>';
});

function addLog(msg, cls = "") {
  const log = document.getElementById("event-log");
  const p = log.querySelector(".empty-hint");
  if (p) p.remove();

  const ts  = new Date().toLocaleTimeString("vi-VN");
  const div = document.createElement("div");
  div.className = `log-entry ${cls}`;
  div.innerHTML = `<span class="log-time">${ts}</span>${escHtml(msg)}`;
  log.prepend(div);

  // Keep max 200 entries
  const entries = log.querySelectorAll(".log-entry");
  if (entries.length > 200) entries[entries.length - 1].remove();
}

/* ═══════════════════════════════════════════════════════════════
   ANALYSIS TAB
═══════════════════════════════════════════════════════════════ */

document.getElementById("btn-start-analysis").addEventListener("click", async () => {
  const videoDir = document.getElementById("video-dir").value.trim() || "data/raw_videos";
  const res = await fetch("/api/analysis/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ video_dir: videoDir }),
  });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }

  state.analysisRunning = true;
  document.getElementById("btn-start-analysis").disabled = true;
  document.getElementById("btn-stop-analysis").disabled  = false;
  document.getElementById("dot-ana").className = "dot active";
  document.getElementById("lbl-ana").textContent = "Đang phân tích…";

  document.getElementById("analysis-idle").classList.add("hidden");
  document.getElementById("analysis-done").classList.add("hidden");
  document.getElementById("analysis-progress-section").classList.remove("hidden");

  showAnalysisProgress({ clip: "Khởi động…", clips_done: 0, clips_total: data.clips || 0, pct: 0 });
});

document.getElementById("btn-stop-analysis").addEventListener("click", async () => {
  await fetch("/api/analysis/stop", { method: "POST" });
  resetAnalysisUI();
});

function showAnalysisProgress(d) {
  const section = document.getElementById("analysis-progress-section");
  section.classList.remove("hidden");

  document.getElementById("ana-clip-name").textContent = d.clip || "—";
  document.getElementById("ana-pct").textContent = `${d.pct || 0}%`;
  document.getElementById("ana-bar").style.width = `${d.pct || 0}%`;
  if (d.clips_done !== undefined) document.getElementById("stat-clips").textContent = `${d.clips_done} / ${d.clips_total}`;
  if (d.frames_saved !== undefined) document.getElementById("stat-frames").textContent = d.frames_saved;
  if (d.labels_written !== undefined) document.getElementById("stat-labels").textContent = d.labels_written;
}

function finishAnalysis(d) {
  state.analysisRunning = false;
  document.getElementById("analysis-progress-section").classList.add("hidden");
  document.getElementById("analysis-done").classList.remove("hidden");
  document.getElementById("dot-ana").className = "dot";
  document.getElementById("lbl-ana").textContent = "Phân tích xong";
  resetAnalysisUI();
}

function resetAnalysisUI() {
  document.getElementById("btn-start-analysis").disabled = false;
  document.getElementById("btn-stop-analysis").disabled  = true;
}

// ── Video list ────────────────────────────────────────────────
document.getElementById("btn-refresh-videos").addEventListener("click", refreshVideoList);

async function refreshVideoList() {
  const res  = await fetch("/api/videos");
  const data = await res.json();
  const el   = document.getElementById("video-list-table");
  if (!data.videos || !data.videos.length) {
    el.innerHTML = '<p class="empty-hint">Chưa có clip nào được ghi</p>';
    return;
  }
  el.innerHTML = `
    <table>
      <thead><tr><th>Tên file</th><th>Dung lượng</th><th>Thời gian ghi</th><th></th></tr></thead>
      <tbody>
        ${data.videos.map((v) => `
          <tr>
            <td class="td-mono">${escHtml(v.filename)}</td>
            <td>${v.size_mb} MB</td>
            <td>${new Date(v.mtime * 1000).toLocaleString("vi-VN")}</td>
            <td><button class="btn btn-danger btn-del" onclick="deleteVideo('${escHtml(v.filename)}', this)">Xóa</button></td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

async function deleteVideo(filename, btn) {
  if (!confirm(`Xóa clip: ${filename}?`)) return;
  const res = await fetch(`/api/videos/${encodeURIComponent(filename)}`, { method: "DELETE" });
  if (res.ok) { btn.closest("tr").remove(); refreshStorageInfo(); }
}

/* ═══════════════════════════════════════════════════════════════
   RESULTS TAB
═══════════════════════════════════════════════════════════════ */
document.getElementById("btn-refresh-results").addEventListener("click", refreshResults);

async function refreshResults() {
  const res  = await fetch("/api/analysis/results");
  const data = await res.json();
  const grid = document.getElementById("results-grid");
  document.getElementById("result-count").textContent = `${data.results.length} clip kết quả`;

  if (!data.results.length) {
    grid.innerHTML = '<p class="empty-hint" style="margin:2rem">Chưa có kết quả — chạy Phase 2 để bắt đầu</p>';
    return;
  }
  grid.innerHTML = data.results.map((r) => `
    <div class="result-card">
      <div class="result-card-header"><h3>${escHtml(r.clip_stem)}</h3></div>
      <div class="result-card-body">
        <div class="result-stat"><span>Frames PNG</span><span>${r.frames}</span></div>
        <div class="result-stat"><span>Nhãn bounding box</span><span>${r.label_count}</span></div>
        <div class="result-stat"><span>File nhãn</span><span>${r.label_file || "—"}</span></div>
      </div>
    </div>
  `).join("");
}

/* ═══════════════════════════════════════════════════════════════
   UTILS
═══════════════════════════════════════════════════════════════ */
function escHtml(str) {
  return String(str).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

/* ═══════════════════════════════════════════════════════════════
   INIT
═══════════════════════════════════════════════════════════════ */
(async function init() {
  await fetchCameras();
  await refreshStorageInfo();
  await refreshVideoList();
})();
