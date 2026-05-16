const FASTAPI_PORT = 8000;
const IS_LIVE_SERVER = location.port !== String(FASTAPI_PORT) && location.port !== "";
const API_BASE = IS_LIVE_SERVER ? `http://${location.hostname}:${FASTAPI_PORT}` : "";
const WS_BASE = IS_LIVE_SERVER ? `ws://${location.hostname}:${FASTAPI_PORT}` : `ws://${location.host}`;

const sockets = { 1: null, 2: null };
const sessionIds = { 1: null, 2: null };
const cameraSlots = { 1: null, 2: null };
const frameStats = { 1: null, 2: null };
let rtspCameras = [];

// Module state per camera — persisted across uploads and restarts
const activeModules = {
  1: new Set(),
  2: new Set()
};

// Gallery dedup: key = "camId_trackId_bucketIdx"
const galleryKeys = { 1: new Set(), 2: new Set() };

// ── RTSP config upload ────────────────────────────────────────────────────────

async function handleRtspUpload(event, camId) {
  const file = event.target.files[0];
  if (!file) return;
  const formData = new FormData();
  formData.append("file", file);
  addLog(camId, "Loading RTSP config file...", "system");
  try {
    const res = await fetch(`${API_BASE}/api/cameras/config`, { method: "POST", body: formData });
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
    if (previousValue && cameras.some((c) => c.id === previousValue)) {
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
  cameraSlots[camId] = rtspCameras.find((c) => c.id === selectedId) || null;
  if (cameraSlots[camId] && !options.silent) {
    addLog(camId, `Selected stream: ${cameraSlots[camId].display || cameraSlots[camId].name}`, "system");
  }
  if (sockets[camId] && cameraSlots[camId]) startStream(camId);
}

// ── Video upload ──────────────────────────────────────────────────────────────

async function handleCamUpload(event, camId) {
  const file = event.target.files[0];
  if (!file) return;

  // Stop running stream – but do NOT clear activeModules
  if (sockets[camId]) {
    try { sockets[camId].send(JSON.stringify({ type: "stop" })); } catch (e) { /* ignore */ }
    sockets[camId].close();
    sockets[camId] = null;
    addLog(camId, "Previous stream stopped before loading uploaded video.", "system");
  }

  const formData = new FormData();
  formData.append("file", file);
  addLog(camId, `Uploading video: ${file.name}...`, "system");
  try {
    const res = await fetch(`${API_BASE}/api/upload`, { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Upload failed");

    const uploadedPath = data.source || data.file_path || data.filename || data.name;
    document.getElementById(`cam${camId}-source`).value = uploadedPath;
    setSelectToUploadedFile(camId, data.name || file.name, uploadedPath);
    cameraSlots[camId] = null;

    // Show currently selected modules so user knows they're preserved
    const mods = [...activeModules[camId]].join(", ");
    addLog(camId, `Upload completed: ${data.name || file.name}`, "system");
    addLog(
      camId,
      mods
        ? `Modules active: [${mods}]. Press ON to start.`
        : "No modules selected. Pick modules then press ON.",
      mods ? "system" : "warning"
    );
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

// ── Module toggle ─────────────────────────────────────────────────────────────

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

  // Log only when modules change – avoid false "No AI module" when user is
  // mid-selection. Only log the warning when the Set truly ends up empty.
  if (mods) {
    addLog(camId, `Modules selected: [${mods}]`, "system");
  } else {
    addLog(camId, "No AI module selected. Playing raw stream when ON is pressed.", "warning");
  }

  // If stream is already running, push the new module set to backend live
  if (sockets[camId] && sockets[camId].readyState === WebSocket.OPEN) {
    sockets[camId].send(JSON.stringify({ type: "set_modules", modules: mods }));
  }
}

// ── Stream start ──────────────────────────────────────────────────────────────

function startStream(camId) {
  const sourceEl = document.getElementById(`cam${camId}-source`);
  const videoSource = sourceEl ? sourceEl.value.trim() : "";

  if (!videoSource) {
    alert(`Please select an RTSP stream or upload a video for Camera ${camId}`);
    return;
  }

  if (sockets[camId]) sockets[camId].close();

  setStatus(camId, "Connecting...", "offline");
  const placeholder = document.getElementById(`placeholder-${camId}`);
  if (placeholder) placeholder.style.display = "none";
  const img = document.getElementById(`frame-${camId}`);
  if (img) {
    img.style.display = "block";
    if (img._blobUrl) { URL.revokeObjectURL(img._blobUrl); img._blobUrl = null; }
    img.src = "";
  }

  const mods = [...activeModules[camId]].join(",");

  // ── Build WebSocket URL with modules param ────────────────────────────
  const wsUrl = `${WS_BASE}/ws/stream/${camId}?source=${encodeURIComponent(videoSource)}&modules=${encodeURIComponent(mods)}`;

  const displayName = getSelectedDisplayName(camId);
  addLog(camId, `Starting stream: source=${displayName}, modules=${mods || "none"}`, "system");
  if (mods) {
    addLog(camId, `Modules: [${mods}]`, "system");
  } else {
    // Not an error — user may legitimately want raw stream
    addLog(camId, "No AI module selected, playing raw stream.", "warning");
  }

  const ws = new WebSocket(wsUrl);
  ws.binaryType = "arraybuffer";
  sockets[camId] = ws;
  initFrameStats(camId);

  ws.onopen = () => setStatus(camId, "Live", "online");

  ws.onmessage = (evt) => {
    if (evt.data instanceof ArrayBuffer) {
      renderBinaryFrame(camId, img, evt.data);
      return;
    }
    try { handleJsonMessage(camId, JSON.parse(evt.data)); } catch (e) { /* skip malformed */ }
  };

  ws.onerror = () => addLog(camId, "WebSocket connection to AI server failed.", "error");

  ws.onclose = () => {
    setStatus(camId, "Disconnected", "offline");
    if (img) { img.style.display = "none"; }
    if (placeholder) placeholder.style.display = "flex";
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

// ── JSON message handler ──────────────────────────────────────────────────────

function handleJsonMessage(camId, data) {
  switch (data.type) {
    case "session":
      sessionIds[camId] = data.session_id;
      toggleSaveButtons(camId, true);
      // Confirm active modules from backend echo
      if (Array.isArray(data.active_modules)) {
        syncModuleButtons(camId, data.active_modules);
        addLog(camId, `Session ready. Active modules: [${data.active_modules.join(",") || "none"}]`, "system");
      }
      break;

    case "metric": {
      const m = data.metrics || {};
      const modules = Array.isArray(m.modules) ? m.modules.join(",") : (m.module || "");
      addLog(
        camId,
        `METRIC ${modules}: fps=${m.fps ?? "-"} det=${m.detections ?? "-"} live=${m.live_tracked ?? m.tracked ?? "-"} display=${m.display_tracked ?? m.tracked ?? "-"} id_max=${m.id_max ?? "-"}${m.skip_ai ? " skip_ai=1" : ""}`,
        "metric"
      );
      break;
    }

    case "gallery":
      addGalleryItem(camId, data);
      break;

    case "log":
    case "system":
      addLog(camId, maskRtsp(data.msg || ""), data.level || "system");
      break;

    case "ai":
      addLog(camId, `AI: ${data.msg || ""}`, "ai");
      break;

    default:
      break;
  }
}

// ── Frame rendering ───────────────────────────────────────────────────────────

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
}

function cleanupFrameStats(camId) {
  const stats = frameStats[camId];
  if (!stats) return;
  if (stats.staleTimer) window.clearInterval(stats.staleTimer);
  const img = document.getElementById(`frame-${camId}`);
  if (img?._blobUrl) { URL.revokeObjectURL(img._blobUrl); img._blobUrl = null; }
  updateFpsBadge(camId, 0);
  setStaleVisible(camId, false);
  frameStats[camId] = null;
}

function renderBinaryFrame(camId, imgEl, arrayBuffer) {
  if (!imgEl) return;
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

function syncModuleButtons(camId, modules) {
  const next = new Set((modules || []).map((m) => String(m).toLowerCase()));
  activeModules[camId] = next;
  document.querySelectorAll(`.btn-module[data-cam="${camId}"]`).forEach((btn) => {
    btn.classList.toggle("active", next.has(btn.dataset.module));
  });
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

// ── Stream stop ───────────────────────────────────────────────────────────────

function stopStream(camId) {
  if (sockets[camId]) {
    try { sockets[camId].send(JSON.stringify({ type: "stop" })); } catch (e) { /* ignore */ }
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

// ── Save video ────────────────────────────────────────────────────────────────

async function saveVideo(camId) {
  const sid = sessionIds[camId];
  if (!sid) {
    addLog(camId, "No active session. Start a stream first.", "warning");
    return;
  }
  addLog(camId, "Saving video...", "system");
  try {
    const res = await fetch(`${API_BASE}/api/save-video/${sid}`, { method: "POST" });
    const data = await res.json();
    if (!res.ok) {
      // Friendly message when buffer is empty
      const msg = data.error || data.message || "Save failed";
      addLog(camId, `Save video: ${msg}`, "warning");
      return;
    }
    addLog(camId, `Video saved: ${data.path || data.saved || "OK"}`, "ai");
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

// ── UI helpers ────────────────────────────────────────────────────────────────

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
  const t = new Date().toLocaleTimeString("vi-VN");
  entry.className = `log-entry log-${level}`;
  entry.innerHTML = `<span class="log-time">${t}</span> ${escapeHtml(maskRtsp(String(msg || "")))}`;
  box.appendChild(entry);
  box.scrollTop = box.scrollHeight;
  while (box.children.length > 500) box.removeChild(box.firstChild);
}

function clearLog(camId) {
  const box = document.getElementById(`log-box-${camId}`);
  if (box) box.innerHTML = "";
}

// ── Gallery ───────────────────────────────────────────────────────────────────

function addGalleryItem(camId, data) {
  const box = document.getElementById(`gallery-box-${camId}`);
  if (!box || !data.crop_jpeg) return;

  const galleryIntervalFrames = 30;
  const bucketTs = Math.floor(Number(data.frame_idx || 0) / galleryIntervalFrames);
  const key = `${camId}_${data.track_id}_${bucketTs}`;
  if (galleryKeys[camId].has(key)) return;  // already shown in this bucket
  galleryKeys[camId].add(key);
  // Prevent unbounded Set growth
  if (galleryKeys[camId].size > 200) {
    const iter = galleryKeys[camId].values();
    galleryKeys[camId].delete(iter.next().value);
  }

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
  const reidScore = data.reid_score ? Number(data.reid_score).toFixed(2) : null;
  const ts = data.ts || new Date().toLocaleTimeString();

  meta.innerHTML = `
    <div>track=${escapeHtml(String(track))}</div>
    <div>gid=${escapeHtml(String(gid))}</div>
    ${reidScore ? `<div>reid=${escapeHtml(reidScore)}</div>` : ""}
    <div>conf=${escapeHtml(conf)}</div>
    ${adlLabel ? `<div>ADL=${escapeHtml(String(adlLabel))}</div>` : ""}
    <div>${escapeHtml(ts)}</div>
  `;

  item.appendChild(img);
  item.appendChild(meta);
  box.prepend(item);

  // Cap at max_gallery_items
  while (box.children.length > 20) box.removeChild(box.lastChild);
}

function clearGallery(camId) {
  const box = document.getElementById(`gallery-box-${camId}`);
  if (box) box.innerHTML = "";
  // Also reset dedup keys for this camera so new items appear immediately
  galleryKeys[camId] = new Set();
}

// ── Security helpers ──────────────────────────────────────────────────────────

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
  return String(str).replace(/[&<>"']/g, (m) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m])
  );
}

// ── Event binding ─────────────────────────────────────────────────────────────

function bindUIEvents() {
  [1, 2].forEach((camId) => {
    document.getElementById(`cam${camId}-display`)
      ?.addEventListener("change", () => selectRtspCamera(camId));

    document.getElementById(`rtsp-upload-${camId}`)
      ?.addEventListener("change", (e) => handleRtspUpload(e, camId));

    document.getElementById(`file-upload-${camId}`)
      ?.addEventListener("change", (e) => handleCamUpload(e, camId));

    document.getElementById(`btn-on-${camId}`)
      ?.addEventListener("click", () => startStream(camId));

    document.getElementById(`btn-off-${camId}`)
      ?.addEventListener("click", () => stopStream(camId));

    document.getElementById(`save-btn-${camId}`)
      ?.addEventListener("click", () => saveVideo(camId));

    document.getElementById(`save-excel-btn-${camId}`)
      ?.addEventListener("click", () => saveExcel(camId));

    // clear-log and clear-gallery are INDEPENDENT buttons
    document.getElementById(`clear-log-${camId}`)
      ?.addEventListener("click", () => clearLog(camId));

    document.getElementById(`clear-gallery-${camId}`)
      ?.addEventListener("click", () => clearGallery(camId));
  });

  document.querySelectorAll(".btn-module").forEach((btn) => {
    btn.addEventListener("click", () => {
      const camId = Number(btn.dataset.cam);
      const mod = btn.dataset.module;
      if (camId && mod) toggleModule(camId, mod);
    });
  });
}

// ── Exports ───────────────────────────────────────────────────────────────────

Object.assign(window, {
  handleRtspUpload, handleCamUpload, selectRtspCamera,
  startStream, stopStream, toggleModule,
  saveVideo, saveExcel, clearLog, clearGallery,
  loadConfiguredCameras, bindUIEvents
});

window.addEventListener("DOMContentLoaded", () => {
  bindUIEvents();
  loadConfiguredCameras();
});
