"use strict";

const state = {
  cameras: [],
  storageLimitGb: 10,
  selectedResolutions: {},
  recordingRunning: false,
  analysisRunning: false,
  poseRunning: false,
  analysisClipsTotal: 0,
  poseClipsTotal: 0,
  clipCount: 0,
  detectCount: 0,
};

const TAB_TITLES = {
  config: "System Configuration",
  monitor: "Live Monitor",
  analysis: "Phase 2 Analysis",
  results: "Phase 2 Results",
  pose: "Phase 3 Pose and ADL",
};

const socket = io({ transports: ["websocket", "polling"] });

const dotSock = document.getElementById("dot-sock");
const sockLbl = document.getElementById("sock-lbl");

socket.on("connect", () => {
  dotSock.className = "dot dot-sm connected";
  sockLbl.textContent = "Connected";
  refreshStorageInfo();
});

socket.on("disconnect", () => {
  dotSock.className = "dot dot-sm disconnected";
  sockLbl.textContent = "Disconnected";
});

socket.on("camera_status", (payload) => {
  updateCameraTile(payload);
  const summary = payload.resolution ? `${payload.status} | ${payload.resolution}` : payload.status;
  addLog(`cam-${payload.cam_id}: ${summary}`, payload.status === "error" ? "err" : "");
});

socket.on("detection_event", (payload) => {
  const count = payload.person_count ?? payload.count ?? 0;
  state.detectCount += 1;
  document.getElementById("rec-detect-count").textContent = String(state.detectCount);
  addLog(`cam-${payload.cam_id}: detected ${count} person(s)`, "detect");
  pulseDotRec();
});

socket.on("clip_saved", (payload) => {
  state.clipCount += 1;
  document.getElementById("rec-clip-count").textContent = String(state.clipCount);
  addLog(`clip saved: ${payload.filename} (${payload.duration_s}s, ${payload.size_mb} MB)`, "clip");
  refreshStorageInfo();
  refreshVideoList();
});

socket.on("analysis_progress", showAnalysisProgress);
socket.on("analysis_complete", finishAnalysis);
socket.on("pose_progress", showPoseProgress);
socket.on("pose_complete", finishPose);

socket.on("storage_warning", (payload) => {
  addLog(
    `storage warning: ${payload.used_gb}/${payload.limit_gb} GB (${payload.pct}%)`,
    "warn"
  );
  updateStorageBar(payload.used_gb, payload.limit_gb);
});

socket.on("error", (payload) => {
  addLog(`error [${payload.source}]: ${payload.message}`, "err");
});

document.querySelectorAll(".nav-item").forEach((item) => {
  item.addEventListener("click", (event) => {
    event.preventDefault();
    switchTab(item.dataset.tab);
  });
});

function switchTab(tabId) {
  document.querySelectorAll(".nav-item").forEach((item) => {
    item.classList.toggle("active", item.dataset.tab === tabId);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `tab-${tabId}`);
  });
  document.getElementById("page-title").textContent = TAB_TITLES[tabId] || tabId;

  if (tabId === "analysis") {
    refreshVideoList();
  } else if (tabId === "results") {
    refreshResults();
  } else if (tabId === "pose") {
    refreshPoseResults();
  }
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  let data = {};
  try {
    data = await response.json();
  } catch (error) {
    data = {};
  }
  if (!response.ok) {
    throw new Error(data.error || data.message || `Request failed (${response.status})`);
  }
  return data;
}

const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("over"));
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("over");
  if (event.dataTransfer.files[0]) {
    uploadConfig(event.dataTransfer.files[0]);
  }
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) {
    uploadConfig(fileInput.files[0]);
  }
});

async function uploadConfig(file) {
  const formData = new FormData();
  formData.append("file", file);

  const status = document.getElementById("config-result");
  status.classList.remove("hidden", "ok", "err");

  try {
    const data = await api("/api/config/upload", { method: "POST", body: formData });
    state.cameras = data.cameras || [];
    status.classList.add("ok");
    status.textContent = `Saved resources.txt with ${state.cameras.length} camera(s).`;
    renderCameraList(state.cameras);
  } catch (error) {
    status.classList.add("err");
    status.textContent = error.message;
  }
}

async function fetchCameras() {
  try {
    const data = await api("/api/config/cameras");
    state.cameras = data.cameras || [];
    renderCameraList(state.cameras);
  } catch (error) {
    addLog(error.message, "err");
  }
}

function renderCameraList(cameras) {
  const list = document.getElementById("camera-list");
  if (!cameras.length) {
    list.innerHTML = '<p class="empty-hint">No cameras loaded yet</p>';
    return;
  }

  list.innerHTML = cameras.map((camera) => `
    <div class="camera-item" id="cam-item-${camera.cam_id}">
      <span class="cam-id-tag" title="ID: ${camera.cam_id}">${escHtml(camera.label || 'cam' + camera.cam_id)}</span>
      <span class="cam-url" title="${escHtml(camera.url)}">${escHtml(camera.url)}</span>
      <span class="cam-res" id="res-tag-${camera.cam_id}">-</span>
      <button class="btn btn-ghost btn-sm" onclick="probeCamera('${camera.cam_id}', '${camera.url.replace(/'/g, "\\'")}')">Probe</button>
    </div>
  `).join("");
}

document.getElementById("btn-probe-all").addEventListener("click", async () => {
  for (const camera of state.cameras) {
    await probeCamera(camera.cam_id, camera.url, false);
  }
});

async function probeCamera(camId, url, showModal = true) {
  if (showModal) {
    document.getElementById("probe-cam-id").textContent = `Camera cam${camId}`;
    document.getElementById("probe-result").innerHTML = '<div class="loader"></div>';
    document.getElementById("modal-probe").classList.remove("hidden");
  }

  try {
    const data = await api("/api/cameras/probe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cam_id: camId, url }),
    });

    state.selectedResolutions[camId] = { width: data.width, height: data.height };
    const tag = document.getElementById(`res-tag-${camId}`);
    if (tag) {
      tag.textContent = `${data.width}x${data.height}`;
    }

    if (!showModal) {
      return;
    }

    document.getElementById("probe-result").innerHTML = `
      <dl class="probe-info">
        <dt>Camera</dt><dd>cam${escHtml(data.cam_id)}</dd>
        <dt>Resolution</dt><dd>${data.width} x ${data.height}</dd>
        <dt>FPS</dt><dd>${data.fps}</dd>
      </dl>
      <div class="res-options">
        ${data.resolutions.map((option) => `
          <button
            class="res-btn ${option.width === data.width && option.height === data.height ? "selected" : ""}"
            onclick="selectResolution('${data.cam_id}', ${option.width}, ${option.height}, this)"
            type="button"
          >
            ${escHtml(option.label)}
          </button>
        `).join("")}
      </div>
    `;
  } catch (error) {
    if (showModal) {
      document.getElementById("probe-result").innerHTML = `<p class="empty-hint">${escHtml(error.message)}</p>`;
    }
  }
}

function selectResolution(camId, width, height, button) {
  state.selectedResolutions[camId] = { width, height };
  const tag = document.getElementById(`res-tag-${camId}`);
  if (tag) {
    tag.textContent = `${width}x${height}`;
  }
  button.closest(".res-options").querySelectorAll(".res-btn").forEach((item) => {
    item.classList.remove("selected");
  });
  button.classList.add("selected");
}

document.getElementById("modal-close").addEventListener("click", () => {
  document.getElementById("modal-probe").classList.add("hidden");
});

document.getElementById("btn-set-limit").addEventListener("click", async () => {
  const limitGb = parseFloat(document.getElementById("storage-limit").value);
  state.storageLimitGb = Number.isFinite(limitGb) ? limitGb : 10;
  try {
    await api("/api/storage/limit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ limit_gb: state.storageLimitGb }),
    });
    refreshStorageInfo();
  } catch (error) {
    addLog(error.message, "err");
  }
});

async function refreshStorageInfo() {
  try {
    const data = await api("/api/storage/info");
    document.getElementById("storage-text").textContent = `${data.used_gb} GB`;
    document.getElementById("storage-info-detail").textContent =
      `${data.used_gb} GB used out of ${state.storageLimitGb} GB (${data.file_count} clips)`;
    updateStorageBar(data.used_gb, state.storageLimitGb);
  } catch (error) {
    addLog(error.message, "err");
  }
}

function updateStorageBar(usedGb, limitGb) {
  const pct = limitGb > 0 ? Math.min(100, (usedGb / limitGb) * 100) : 0;
  const fill = document.getElementById("storage-fill");
  fill.style.width = `${pct}%`;
  fill.className = "storage-fill";
  if (pct > 90) {
    fill.classList.add("danger");
  } else if (pct > 70) {
    fill.classList.add("warn");
  }
}

document.getElementById("btn-start-rec").addEventListener("click", async () => {
  if (!state.cameras.length) {
    addLog("Upload resources.txt before starting Phase 1.", "warn");
    return;
  }

  const camerasPayload = state.cameras.map((camera) => {
    const resolution = state.selectedResolutions[camera.cam_id] || {};
    return {
      cam_id: camera.cam_id,
      url: camera.url,
      width: resolution.width,
      height: resolution.height,
    };
  });

  try {
    await api("/api/recording/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        cameras: camerasPayload,
        storage_limit_gb: state.storageLimitGb,
      }),
    });
    state.recordingRunning = true;
    state.clipCount = 0;
    state.detectCount = 0;
    document.getElementById("rec-clip-count").textContent = "0";
    document.getElementById("rec-detect-count").textContent = "0";
    document.getElementById("btn-start-rec").disabled = true;
    document.getElementById("btn-stop-rec").disabled = false;
    document.getElementById("dot-rec").className = "dot active";
    document.getElementById("lbl-rec").textContent = "Recording";
    renderCameraGrid(state.cameras);
    addLog(`Started recording on ${state.cameras.length} camera(s).`, "clip");
  } catch (error) {
    addLog(error.message, "err");
  }
});

document.getElementById("btn-stop-rec").addEventListener("click", async () => {
  try {
    await api("/api/recording/stop", { method: "POST" });
  } catch (error) {
    addLog(error.message, "err");
  } finally {
    state.recordingRunning = false;
    document.getElementById("btn-start-rec").disabled = false;
    document.getElementById("btn-stop-rec").disabled = true;
    document.getElementById("dot-rec").className = "dot";
    document.getElementById("lbl-rec").textContent = "Idle";
  }
});

function renderCameraGrid(cameras) {
  const grid = document.getElementById("camera-grid");
  if (!cameras.length) {
    grid.innerHTML = '<div class="camera-placeholder">No cameras configured.</div>';
    return;
  }

  grid.innerHTML = cameras.map((camera) => `
    <div class="cam-tile" id="tile-${camera.cam_id}">
      <div class="cam-tile-header">
        <span class="cam-id-tag">${escHtml(camera.label || 'cam' + camera.cam_id)}</span>
        <span class="dot" id="tile-dot-${camera.cam_id}"></span>
      </div>
      <div class="cam-tile-body">
        <div class="cam-tile-stat"><span>Status</span><span id="tile-st-${camera.cam_id}">Connecting</span></div>
        <div class="cam-tile-stat"><span>Resolution</span><span id="tile-res-${camera.cam_id}">-</span></div>
        <div class="cam-tile-stat"><span>FPS</span><span id="tile-fps-${camera.cam_id}">-</span></div>
      </div>
    </div>
  `).join("");
}

function updateCameraTile(payload) {
  const status = document.getElementById(`tile-st-${payload.cam_id}`);
  const resolution = document.getElementById(`tile-res-${payload.cam_id}`);
  const fps = document.getElementById(`tile-fps-${payload.cam_id}`);
  const dot = document.getElementById(`tile-dot-${payload.cam_id}`);
  if (!status) {
    return;
  }
  status.textContent = payload.status || "-";
  resolution.textContent = payload.resolution || "-";
  fps.textContent = payload.fps ? `${payload.fps} fps` : "-";
  dot.className = "dot";
  if (payload.status === "streaming") {
    dot.classList.add("active");
  } else if (payload.status === "error") {
    dot.classList.add("error");
  }
}

function pulseDotRec() {
  const dot = document.getElementById("dot-rec");
  dot.style.opacity = "0.35";
  window.setTimeout(() => {
    dot.style.opacity = "1";
  }, 180);
}

document.getElementById("btn-clear-log").addEventListener("click", () => {
  document.getElementById("event-log").innerHTML = '<p class="empty-hint">No events yet</p>';
});

function addLog(message, className = "") {
  const log = document.getElementById("event-log");
  const empty = log.querySelector(".empty-hint");
  if (empty) {
    empty.remove();
  }

  const entry = document.createElement("div");
  entry.className = `log-entry ${className}`.trim();
  const time = document.createElement("span");
  time.className = "log-time";
  time.textContent = new Date().toLocaleTimeString();
  entry.appendChild(time);
  entry.appendChild(document.createTextNode(message));
  log.prepend(entry);

  const entries = log.querySelectorAll(".log-entry");
  if (entries.length > 200) {
    entries[entries.length - 1].remove();
  }
}

document.getElementById("btn-start-analysis").addEventListener("click", async () => {
  const folder = document.getElementById("video-dir").value.trim() || "data/raw_videos";
  try {
    const data = await api("/api/analysis/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_dir: folder }),
    });
    state.analysisRunning = true;
    state.analysisClipsTotal = data.clips || 0;
    document.getElementById("btn-start-analysis").disabled = true;
    document.getElementById("btn-stop-analysis").disabled = false;
    document.getElementById("dot-ana").className = "dot active";
    document.getElementById("lbl-ana").textContent = "Running";
    document.getElementById("analysis-idle").classList.add("hidden");
    document.getElementById("analysis-done").classList.add("hidden");
    document.getElementById("analysis-progress-section").classList.remove("hidden");
    document.getElementById("stat-clips").textContent = `0 / ${state.analysisClipsTotal}`;
    document.getElementById("stat-frames").textContent = "0";
    document.getElementById("stat-labels").textContent = "0";
  } catch (error) {
    addLog(error.message, "err");
  }
});

document.getElementById("btn-stop-analysis").addEventListener("click", async () => {
  try {
    await api("/api/analysis/stop", { method: "POST" });
  } catch (error) {
    addLog(error.message, "err");
  } finally {
    resetAnalysisUi();
  }
});

function showAnalysisProgress(payload) {
  document.getElementById("analysis-progress-section").classList.remove("hidden");
  document.getElementById("ana-clip-name").textContent = payload.clip || "-";
  document.getElementById("ana-pct").textContent = `${payload.pct || 0}%`;
  document.getElementById("ana-bar").style.width = `${payload.pct || 0}%`;
  if (payload.frames_saved !== undefined) {
    document.getElementById("stat-frames").textContent = String(payload.frames_saved);
  }
  if (payload.labels_written !== undefined) {
    document.getElementById("stat-labels").textContent = String(payload.labels_written);
  }
}

function finishAnalysis(payload) {
  state.analysisRunning = false;
  document.getElementById("analysis-progress-section").classList.add("hidden");
  document.getElementById("analysis-done").classList.remove("hidden");
  document.getElementById("stat-clips").textContent = `${payload.clips_done || 0} / ${state.analysisClipsTotal}`;
  document.getElementById("stat-frames").textContent = String(payload.frames_saved || 0);
  document.getElementById("stat-labels").textContent = String(payload.labels_written || 0);
  resetAnalysisUi();
  refreshResults();
}

function resetAnalysisUi() {
  document.getElementById("btn-start-analysis").disabled = false;
  document.getElementById("btn-stop-analysis").disabled = true;
  document.getElementById("dot-ana").className = "dot";
  document.getElementById("lbl-ana").textContent = "Idle";
}

document.getElementById("btn-refresh-videos").addEventListener("click", refreshVideoList);

async function refreshVideoList() {
  const table = document.getElementById("video-list-table");
  try {
    const data = await api("/api/videos");
    if (!data.videos.length) {
      table.innerHTML = '<p class="empty-hint">No MP4 clips found.</p>';
      return;
    }
    table.innerHTML = `
      <table>
        <thead>
          <tr><th>Filename</th><th>Size</th><th>Updated</th><th></th></tr>
        </thead>
        <tbody>
          ${data.videos.map((video) => `
            <tr>
              <td class="td-mono">${escHtml(video.filename)}</td>
              <td>${video.size_mb} MB</td>
              <td>${new Date(video.mtime * 1000).toLocaleString()}</td>
              <td><button class="btn btn-danger btn-del" onclick="deleteVideo('${video.filename.replace(/'/g, "\\'")}')">Delete</button></td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
  } catch (error) {
    table.innerHTML = `<p class="empty-hint">${escHtml(error.message)}</p>`;
  }
}

async function deleteVideo(filename) {
  if (!window.confirm(`Delete clip ${filename}?`)) {
    return;
  }
  try {
    await api(`/api/videos/${encodeURIComponent(filename)}`, { method: "DELETE" });
    refreshVideoList();
    refreshStorageInfo();
  } catch (error) {
    addLog(error.message, "err");
  }
}

document.getElementById("btn-refresh-results").addEventListener("click", refreshResults);

async function refreshResults() {
  const grid = document.getElementById("results-grid");
  try {
    const data = await api("/api/analysis/results");
    document.getElementById("result-count").textContent = `${data.results.length} clip result(s)`;
    if (!data.results.length) {
      grid.innerHTML = '<p class="empty-hint">No Phase 2 outputs yet.</p>';
      return;
    }
    grid.innerHTML = data.results.map((result) => `
      <div class="result-card">
        <div class="result-card-header"><h3>${escHtml(result.clip_stem)}</h3></div>
        <div class="result-card-body">
          <div class="result-stat"><span>PNG frames</span><span>${result.frames}</span></div>
          <div class="result-stat"><span>Bounding boxes</span><span>${result.label_count}</span></div>
          <div class="result-stat"><span>Label file</span><span>${escHtml(result.label_file || "-")}</span></div>
          <div class="result-stat"><span>Preview frame</span><span>${escHtml(result.preview_frame || "-")}</span></div>
        </div>
      </div>
    `).join("");
  } catch (error) {
    grid.innerHTML = `<p class="empty-hint">${escHtml(error.message)}</p>`;
  }
}

document.getElementById("btn-start-pose").addEventListener("click", async () => {
  const folder = document.getElementById("pose-dir").value.trim() || "data/raw_videos";
  const saveOverlay = document.getElementById("pose-save-overlay").checked;
  try {
    const data = await api("/api/pose/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folder, save_overlay: saveOverlay }),
    });
    state.poseRunning = true;
    state.poseClipsTotal = data.total_clips || 0;
    document.getElementById("btn-start-pose").disabled = true;
    document.getElementById("btn-stop-pose").disabled = false;
    document.getElementById("dot-pose").className = "dot active";
    document.getElementById("lbl-pose").textContent = "Running";
    document.getElementById("pose-idle").classList.add("hidden");
    document.getElementById("pose-done").classList.add("hidden");
    document.getElementById("pose-progress-section").classList.remove("hidden");
    document.getElementById("pose-stat-clips").textContent = `0 / ${state.poseClipsTotal}`;
    document.getElementById("pose-stat-kps").textContent = "0";
    document.getElementById("pose-stat-adl").textContent = "0";
  } catch (error) {
    addLog(error.message, "err");
  }
});

document.getElementById("btn-stop-pose").addEventListener("click", async () => {
  try {
    await api("/api/pose/stop", { method: "POST" });
  } catch (error) {
    addLog(error.message, "err");
  } finally {
    resetPoseUi();
  }
});

function showPoseProgress(payload) {
  document.getElementById("pose-progress-section").classList.remove("hidden");
  document.getElementById("pose-clip-name").textContent = payload.clip || "-";
  document.getElementById("pose-pct").textContent = `${payload.pct || 0}%`;
  document.getElementById("pose-bar").style.width = `${payload.pct || 0}%`;
}

function finishPose(payload) {
  state.poseRunning = false;
  document.getElementById("pose-progress-section").classList.add("hidden");
  document.getElementById("pose-done").classList.remove("hidden");
  document.getElementById("pose-stat-clips").textContent = `${payload.clips_done || 0} / ${state.poseClipsTotal}`;
  document.getElementById("pose-stat-kps").textContent = String(payload.keypoints_written || 0);
  document.getElementById("pose-stat-adl").textContent = String(payload.adl_events || 0);
  resetPoseUi();
  refreshPoseResults();
}

function resetPoseUi() {
  document.getElementById("btn-start-pose").disabled = false;
  document.getElementById("btn-stop-pose").disabled = true;
  document.getElementById("dot-pose").className = "dot";
  document.getElementById("lbl-pose").textContent = "Idle";
}

document.getElementById("btn-refresh-pose").addEventListener("click", refreshPoseResults);

async function refreshPoseResults() {
  const grid = document.getElementById("pose-results-grid");
  try {
    const data = await api("/api/pose/results");
    if (!data.results.length) {
      grid.innerHTML = '<p class="empty-hint">No pose outputs yet.</p>';
      return;
    }
    grid.innerHTML = data.results.map((result) => `
      <div class="result-card">
        <div class="result-card-header"><h3>${escHtml(result.clip_stem)}</h3></div>
        <div class="result-card-body">
          <div class="result-stat"><span>Keypoint rows</span><span>${result.keypoints_count}</span></div>
          <div class="result-stat"><span>ADL events</span><span>${result.adl_events}</span></div>
          <div class="result-stat"><span>Overlay PNG</span><span>${result.overlays}</span></div>
          <div class="adl-summary">${renderAdlSummary(result.adl_summary)}</div>
        </div>
      </div>
    `).join("");
  } catch (error) {
    grid.innerHTML = `<p class="empty-hint">${escHtml(error.message)}</p>`;
  }
}

function renderAdlSummary(summary) {
  const entries = Object.entries(summary || {});
  if (!entries.length) {
    return '<span class="summary-chip">No ADL labels</span>';
  }
  return entries.map(([label, pct]) => {
    return `<span class="summary-chip">${escHtml(label)} ${pct}%</span>`;
  }).join("");
}

function escHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

window.probeCamera = probeCamera;
window.selectResolution = selectResolution;
window.deleteVideo = deleteVideo;

(async function init() {
  await fetchCameras();
  await refreshStorageInfo();
  await refreshVideoList();
  await refreshResults();
  await refreshPoseResults();
})();
