const FASTAPI_PORT = 8000;
const IS_LIVE_SERVER = location.port !== String(FASTAPI_PORT) && location.port !== "";
const API_BASE = IS_LIVE_SERVER ? `http://${location.hostname}:${FASTAPI_PORT}` : "";
const WS_BASE = IS_LIVE_SERVER ? `ws://${location.hostname}:${FASTAPI_PORT}` : `ws://${location.host}`;

const sockets = { 1: null, 2: null };
const sessionIds = { 1: null, 2: null };
const cameraSlots = { 1: null, 2: null };
const frameStats = { 1: null, 2: null };
let rtspCameras = [];

const activeModules = {
  1: new Set(),
  2: new Set()
};

async function handleRtspUpload(event, camId) {
  const file = event.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  addLog(camId, "Loading RTSP config file...", "system");
  try {
    const res = await fetch(`${API_BASE}/api/cameras/config`, {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Unable to load RTSP config");

    applyCameraList(data.cameras || []);
    selectDefaultCamera(camId);
    addLog(camId, "RTSP config loaded. Select a stream from the list, then press ON.", "system");
  } catch (err) {
    addLog(camId, `RTSP config load error: ${err.message || err}`, "error");
  } finally {
    event.target.value = "";
  }
}

async function loadConfiguredCameras() {
  try {
    const res = await fetch(`${API_BASE}/api/cameras`);
    const cameras = await res.json();
    applyCameraList(cameras || []);
  } catch (err) {
    addLog(1, "Unable to read camera list from server.", "error");
    addLog(2, "Unable to read camera list from server.", "error");
  }
}

function applyCameraList(cameras) {
  rtspCameras = cameras;

  [1, 2].forEach((camId) => {
    const selectEl = document.getElementById(`cam${camId}-display`);
    if (!selectEl) return;

    const previousValue = selectEl.value;
    selectEl.innerHTML = '<option value="">Select RTSP stream...</option>';

    cameras.forEach((camera) => {
      const option = document.createElement("option");
      option.value = camera.id;
      option.textContent = camera.display || camera.name || camera.id;
      option.title = camera.display || camera.name || camera.id;
      option.dataset.name = camera.name || "";
      option.dataset.urlMasked = camera.url_masked || "";
      selectEl.appendChild(option);
    });

    if (previousValue && cameras.some((camera) => camera.id === previousValue)) {
      selectEl.value = previousValue;
    } else if (!cameraSlots[camId] && cameras[camId - 1]) {
      selectEl.value = cameras[camId - 1].id;
    }

    selectRtspCamera(camId, { silent: true });
  });
}

function selectDefaultCamera(camId) {
  const selectEl = document.getElementById(`cam${camId}-display`);
  if (!selectEl || selectEl.value || !rtspCameras.length) return;

  selectEl.value = rtspCameras[Math.min(camId - 1, rtspCameras.length - 1)].id;
  selectRtspCamera(camId);
}

function selectRtspCamera(camId, options = {}) {
  const sourceEl = document.getElementById(`cam${camId}-source`);
  const selectEl = document.getElementById(`cam${camId}-display`);
  if (!sourceEl || !selectEl) return;

  const selectedId = selectEl.value;
  sourceEl.value = selectedId;
  cameraSlots[camId] = rtspCameras.find((camera) => camera.id === selectedId) || null;

  if (cameraSlots[camId] && !options.silent) {
    addLog(camId, `Selected stream: ${cameraSlots[camId].display || cameraSlots[camId].name}`, "system");
  }

  if (sockets[camId] && cameraSlots[camId]) {
    startStream(camId);
  }
}

async function handleCamUpload(event, camId) {
  const file = event.target.files[0];
  if (!file) return;

  if (sockets[camId]) {
    try {
      sockets[camId].send(JSON.stringify({ type: "stop" }));
    } catch (e) {
      // ignore close races
    }
    sockets[camId].close();
    sockets[camId] = null;
    addLog(camId, "Closed previous stream before loading uploaded video.", "system");
  }

  const formData = new FormData();
  formData.append("file", file);

  addLog(camId, `Uploading video: ${file.name}...`, "system");
  try {
    const res = await fetch(`${API_BASE}/api/upload`, {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Upload failed");

    const uploadedPath = data.source || data.file_path || data.filename || data.name;
    document.getElementById(`cam${camId}-source`).value = uploadedPath;
    setSelectToUploadedFile(camId, data.name || file.name, uploadedPath);
    cameraSlots[camId] = null;

    addLog(camId, "Upload completed. Select modules, then press ON.", "system");
  } catch (e) {
    addLog(camId, `Upload error: ${e.message || e}`, "error");
  } finally {
    event.target.value = "";
  }
}

function setSelectToUploadedFile(camId, fileName, sourcePath) {
  const selectEl = document.getElementById(`cam${camId}-display`);
  if (!selectEl) return;

  let option = selectEl.querySelector('option[data-upload="true"]');
  if (!option) {
    option = document.createElement("option");
    option.dataset.upload = "true";
    selectEl.appendChild(option);
  }
  option.value = sourcePath || "__uploaded_file__";
  option.textContent = fileName;
  option.title = sourcePath || fileName;
  option.dataset.source = sourcePath || "";
  selectEl.value = option.value;
}

function toggleModule(camId, mod) {
  const btn = document.getElementById(`btn-${mod}-${camId}`);
  if (!btn) return;

  if (activeModules[camId].has(mod)) {
    activeModules[camId].delete(mod);
    btn.classList.remove("active");
  } else {
    activeModules[camId].add(mod);
    btn.classList.add("active");
  }

  const mods = [...activeModules[camId]].join(",");
  addLog(
    camId,
    mods ? `Modules updated: ${mods}` : "No AI module selected, playing raw stream.",
    mods ? "system" : "warning"
  );

  if (sockets[camId] && sockets[camId].readyState === WebSocket.OPEN) {
    sockets[camId].send(JSON.stringify({
      type: "set_modules",
      modules: mods
    }));
  }
}

function startStream(camId) {
  const sourceEl = document.getElementById(`cam${camId}-source`);
  const videoSource = sourceEl.value.trim();

  if (!videoSource) {
    alert(`Please select an RTSP stream or upload a video for Camera ${camId}`);
    return;
  }

  if (sockets[camId]) {
    sockets[camId].close();
  }

  setStatus(camId, "Connecting...", "offline");
  document.getElementById(`placeholder-${camId}`).style.display = "none";
  const img = document.getElementById(`frame-${camId}`);
  img.style.display = "block";
  if (img._blobUrl) {
    URL.revokeObjectURL(img._blobUrl);
    img._blobUrl = null;
  }
  img.src = "";

  const mods = [...activeModules[camId]].join(",");
  if (!mods) {
    addLog(camId, "No AI module selected, playing raw stream.", "warning");
  }
  const wsUrl = `${WS_BASE}/ws/stream/${camId}?source=${encodeURIComponent(videoSource)}&modules=${mods}`;
  const displayName = getSelectedDisplayName(camId);
  addLog(camId, `Starting stream: ${displayName}`, "system");
  addLog(camId, `Analyzing modules: [${mods}]`, "system");

  const ws = new WebSocket(wsUrl);
  ws.binaryType = "arraybuffer";
  sockets[camId] = ws;
  initFrameStats(camId);

  ws.onopen = () => {
    setStatus(camId, "Live", "online");
  };

  ws.onmessage = (evt) => {
    if (evt.data instanceof ArrayBuffer) {
      renderBinaryFrame(camId, img, evt.data);
      return;
    }
    handleJsonMessage(camId, JSON.parse(evt.data));
  };

  ws.onerror = () => {
    addLog(camId, "WebSocket connection to AI server failed.", "error");
  };

  ws.onclose = () => {
    setStatus(camId, "Disconnected", "offline");
    img.style.display = "none";
    document.getElementById(`placeholder-${camId}`).style.display = "flex";
    toggleSaveButtons(camId, false);
    sessionIds[camId] = null;
    sockets[camId] = null;
    cleanupFrameStats(camId);
  };
}

function getSelectedDisplayName(camId) {
  const selectEl = document.getElementById(`cam${camId}-display`);
  return selectEl?.selectedOptions?.[0]?.textContent || `Camera ${camId}`;
}

function handleJsonMessage(camId, data) {
  if (data.type === "session") {
    sessionIds[camId] = data.session_id;
    toggleSaveButtons(camId, true);
  } else if (data.type === "metric") {
    const m = data.metrics || {};
    const modules = Array.isArray(m.modules) ? m.modules.join(",") : (m.module || "");
    if (Array.isArray(m.tracks) && m.tracks.length) {
      const trackText = m.tracks
        .map((track) => `t${track.track_id}=${Number(track.conf || 0).toFixed(2)}`)
        .join(", ");
      const mean = Number(m.mean_track_conf || 0).toFixed(2);
      addLog(camId, `TRACK_CONF: ${trackText} | mean=${mean}`, "metric");
    }
    addLog(
      camId,
      `METRIC ${modules}: fps=${m.fps ?? "-"} det=${m.detections ?? "-"} tracked=${m.tracked ?? "-"}${m.skip_ai ? " skip_ai=1" : ""}`,
      "metric"
    );
  } else if (data.type === "gallery") {
    addGalleryItem(camId, data);
  } else if (data.type === "ai") {
    addLog(camId, `AI: ${data.msg}`, "ai");
  } else if (data.type === "log" || data.type === "system") {
    addLog(camId, maskRtsp(data.msg), data.level || "system");
  }
}

function ensureVideoOverlays(camId) {
  const view = document.getElementById(`frame-${camId}`)?.closest(".cam-view");
  if (!view) return {};

  let fpsBadge = document.getElementById(`fps-badge-${camId}`);
  if (!fpsBadge) {
    fpsBadge = document.createElement("span");
    fpsBadge.id = `fps-badge-${camId}`;
    fpsBadge.className = "fps-badge fps-red";
    fpsBadge.textContent = "0 FPS";
    view.appendChild(fpsBadge);
  }

  let stale = document.getElementById(`stale-overlay-${camId}`);
  if (!stale) {
    stale = document.createElement("div");
    stale.id = `stale-overlay-${camId}`;
    stale.className = "stale-overlay";
    stale.textContent = "Weak signal";
    view.appendChild(stale);
  }

  return { fpsBadge, stale };
}

function initFrameStats(camId) {
  const overlays = ensureVideoOverlays(camId);
  frameStats[camId] = {
    count: 0,
    lastTick: performance.now(),
    lastFrame: performance.now(),
    fps: 0,
    staleTimer: window.setInterval(() => updateStaleOverlay(camId), 500),
    ...overlays
  };
  updateFpsBadge(camId, 0);
  setStaleVisible(camId, false);
  return frameStats[camId];
}

function cleanupFrameStats(camId) {
  const stats = frameStats[camId];
  if (!stats) return;
  if (stats.staleTimer) window.clearInterval(stats.staleTimer);
  const img = document.getElementById(`frame-${camId}`);
  if (img?._blobUrl) {
    URL.revokeObjectURL(img._blobUrl);
    img._blobUrl = null;
  }
  updateFpsBadge(camId, 0);
  setStaleVisible(camId, false);
  frameStats[camId] = null;
}

function renderBinaryFrame(camId, imgEl, arrayBuffer) {
  const prev = imgEl._blobUrl;
  imgEl._blobUrl = URL.createObjectURL(new Blob([arrayBuffer], { type: "image/jpeg" }));
  imgEl.src = imgEl._blobUrl;
  if (prev) requestAnimationFrame(() => URL.revokeObjectURL(prev));

  const now = performance.now();
  const stats = frameStats[camId] || initFrameStats(camId);
  frameStats[camId].count += 1;
  frameStats[camId].lastFrame = now;
  setStaleVisible(camId, false);

  if (now - frameStats[camId].lastTick >= 1000) {
    const seconds = (now - frameStats[camId].lastTick) / 1000;
    const fps = frameStats[camId].count / seconds;
    frameStats[camId].count = 0;
    frameStats[camId].lastTick = now;
    updateFpsBadge(camId, fps);
  }
}

function updateFpsBadge(camId, fps) {
  const badge = ensureVideoOverlays(camId).fpsBadge;
  if (!badge) return;
  badge.textContent = `${fps.toFixed(1)} FPS`;
  badge.className = "fps-badge " + (fps > 20 ? "fps-green" : fps >= 10 ? "fps-orange" : "fps-red");
}

function updateStaleOverlay(camId) {
  const stats = frameStats[camId];
  if (!stats) return;
  setStaleVisible(camId, performance.now() - stats.lastFrame > 2000);
}

function setStaleVisible(camId, visible) {
  const overlay = ensureVideoOverlays(camId).stale;
  if (!overlay) return;
  overlay.classList.toggle("visible", Boolean(visible));
}

function stopStream(camId) {
  if (sockets[camId]) {
    try {
      sockets[camId].send(JSON.stringify({ type: "stop" }));
    } catch (e) {
      // ignore close races
    }
    sockets[camId].close();
    addLog(camId, "Camera disconnected.", "system");
  }
}

function toggleSaveButtons(camId, enabled) {
  const btnVideo = document.getElementById(`save-btn-${camId}`);
  const btnExcel = document.getElementById(`save-excel-btn-${camId}`);
  if (btnVideo) btnVideo.disabled = !enabled;
  if (btnExcel) btnExcel.disabled = !enabled;
}

async function saveVideo(camId) {
  const sid = sessionIds[camId];
  if (!sid) return;

  addLog(camId, "Packaging and saving video...", "system");
  try {
    const res = await fetch(`${API_BASE}/api/save-video/${sid}`, { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || data.message || "Save Video failed");
    addLog(camId, `Video saved: ${data.path || data.saved || data.message || ""}`, "ai");
  } catch (err) {
    addLog(camId, `Save video error: ${err.message || err}`, "error");
  }
}

async function saveExcel(camId) {
  const sid = sessionIds[camId];
  if (!sid) return;

  addLog(camId, "Exporting data to Excel...", "system");
  try {
    const res = await fetch(`${API_BASE}/api/save-excel/${sid}`, { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || data.message || "Save Excel failed");
    addLog(camId, `Excel saved: ${data.file_path || data.message || ""}`, "ai");
  } catch (err) {
    addLog(camId, `Save Excel error: ${err.message || err}`, "error");
  }
}

function setStatus(camId, text, type) {
  const el = document.getElementById(`status-${camId}`);
  if (!el) return;
  el.textContent = text;
  el.className = `cam-status status-${type}`;
}

function addLog(camId, msg, level = "system") {
  const box = document.getElementById(`log-box-${camId}`);
  if (!box) return;

  const entry = document.createElement("div");
  const time = new Date().toLocaleTimeString("vi-VN");

  entry.className = `log-entry log-${level}`;
  entry.innerHTML = `<span class="log-time">${time}</span> ${escapeHtml(maskRtsp(String(msg || "")))}`;
  box.appendChild(entry);
  box.scrollTop = box.scrollHeight;

  while (box.children.length > 500) box.removeChild(box.firstChild);
}

function clearLog(camId) {
  const box = document.getElementById(`log-box-${camId}`);
  if (box) box.innerHTML = "";
}

function addGalleryItem(camId, data) {
  const box = document.getElementById(`gallery-box-${camId}`);
  if (!box || !data.crop_jpeg) return;

  const item = document.createElement("div");
  item.className = "gallery-item";

  const img = document.createElement("img");
  img.src = `data:image/jpeg;base64,${data.crop_jpeg}`;

  const meta = document.createElement("div");
  meta.className = "gallery-meta";

  const track = data.track_id ?? "-";
  const gid = data.global_id || "unknown";
  const conf = Number(data.conf || 0).toFixed(2);
  const adlLabel = data.adl_label || "";
  const ts = data.ts || new Date().toLocaleTimeString();

  meta.innerHTML = `
    <div>track=${escapeHtml(String(track))}</div>
    <div>gid=${escapeHtml(String(gid))}</div>
    <div>conf=${escapeHtml(conf)}</div>
    ${adlLabel ? `<div>ADL=${escapeHtml(String(adlLabel))}</div>` : ""}
    <div>${escapeHtml(ts)}</div>
  `;

  item.appendChild(img);
  item.appendChild(meta);
  box.prepend(item);

  while (box.children.length > 20) {
    box.removeChild(box.lastChild);
  }
}

function clearGallery(camId) {
  const box = document.getElementById(`gallery-box-${camId}`);
  if (box) box.innerHTML = "";
}

function maskRtsp(str) {
  return String(str || "").replace(/rtsp:\/\/[^\s"'<>]+/gi, (url) => {
    try {
      const parsed = new URL(url);
      const host = parsed.hostname.replace(/^(\d+\.\d+\.\d+)\.\d+$/, "$1.***");
      const port = parsed.port ? `:${parsed.port}` : "";
      const auth = parsed.username ? `${parsed.username}:***@` : "";
      return `${parsed.protocol}//${auth}${host}${port}${parsed.pathname}${parsed.search}${parsed.hash}`;
    } catch (e) {
      return url
        .replace(/:\/\/([^:@/\s]+):([^@/\s]+)@/, "://$1:***@")
        .replace(/(\d+\.\d+\.\d+)\.\d+/g, "$1.***");
    }
  });
}

function escapeHtml(str) {
  return str.replace(/[&<>"']/g, function(m) {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m];
  });
}

function bindUIEvents() {
  [1, 2].forEach((camId) => {
    const selectEl = document.getElementById(`cam${camId}-display`);
    if (selectEl) {
      selectEl.addEventListener("change", () => selectRtspCamera(camId));
    }

    const rtspInput = document.getElementById(`rtsp-upload-${camId}`);
    if (rtspInput) {
      rtspInput.addEventListener("change", (event) => handleRtspUpload(event, camId));
    }

    const fileInput = document.getElementById(`file-upload-${camId}`);
    if (fileInput) {
      fileInput.addEventListener("change", (event) => handleCamUpload(event, camId));
    }

    const onBtn = document.getElementById(`btn-on-${camId}`);
    if (onBtn) {
      onBtn.addEventListener("click", () => startStream(camId));
    }

    const offBtn = document.getElementById(`btn-off-${camId}`);
    if (offBtn) {
      offBtn.addEventListener("click", () => stopStream(camId));
    }

    const saveBtn = document.getElementById(`save-btn-${camId}`);
    if (saveBtn) {
      saveBtn.addEventListener("click", () => saveVideo(camId));
    }

    const saveExcelBtn = document.getElementById(`save-excel-btn-${camId}`);
    if (saveExcelBtn) {
      saveExcelBtn.addEventListener("click", () => saveExcel(camId));
    }

    const clearBtn = document.getElementById(`clear-log-${camId}`);
    if (clearBtn) {
      clearBtn.addEventListener("click", () => clearLog(camId));
    }

    const clearGalleryBtn = document.getElementById(`clear-gallery-${camId}`);
    if (clearGalleryBtn) {
      clearGalleryBtn.addEventListener("click", () => clearGallery(camId));
    }
  });

  document.querySelectorAll(".btn-module").forEach((btn) => {
    btn.addEventListener("click", () => {
      const camId = Number(btn.dataset.cam);
      const mod = btn.dataset.module;
      if (camId && mod) toggleModule(camId, mod);
    });
  });
}

Object.assign(window, {
  handleRtspUpload,
  handleCamUpload,
  selectRtspCamera,
  startStream,
  stopStream,
  toggleModule,
  saveVideo,
  saveExcel,
  clearLog,
  clearGallery,
  loadConfiguredCameras,
  bindUIEvents
});

window.addEventListener("DOMContentLoaded", () => {
  bindUIEvents();
  loadConfiguredCameras();
});

