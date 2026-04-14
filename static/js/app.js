"use strict";
/* ════════════════════════════════════════════════════════════════════════
   CPose Dashboard — app.js
   Vanilla JS · Socket.IO · Camera snapshot polling · State machine aware
════════════════════════════════════════════════════════════════════════ */

// ── App state ──────────────────────────────────────────────────────────────
const S = {
  cameras:            [],    // parsed from resources.txt
  storageLimitGb:     10,
  selectedRes:        {},    // { cam_id: {width, height} }
  recRunning:         false,
  anaRunning:         false,
  poseRunning:        false,
  anaClipsTotal:      0,
  poseClipsTotal:     0,
  poseClipsDone:      0,    // Track clips processed for progress
  poseLampState:      {},
  poseClipQueue:      [],
  poseActiveCamera:   "",
  poseCurrentClip:    "",
  clipCount:          0,
  detectCount:        0,
  onlineCount:        0,
  recCount:           0,
  streamIntervals:    {},    // { cam_id: intervalId }
  activeTab:          "config",
  statusPollInterval: null,
  poseLampPollInterval: null,  // Real-time lamp state polling
  poseViewerInterval: null,    // Polling for video snapshots
};

const TAB_TITLES = {
  config:   "Setup — Configuration & Resources",
  pose:     "Sequential Multicam Demo — Pose & ADL",
};

// ── Socket.IO ──────────────────────────────────────────────────────────────
const POSE_CAMERA_ORDER = ["cam01", "cam02", "cam03", "cam04"];

const socket = io({ transports: ["websocket", "polling"] });
const $dotSock  = q("#dot-sock");
const $sockLbl  = q("#sock-label");

socket.on("connect", () => {
  cls($dotSock, "disconnected"); add($dotSock, "connected");
  $sockLbl.textContent = "Connected";
  if (S.statusPollInterval) { clearInterval(S.statusPollInterval); S.statusPollInterval = null; }
  fetchStorageInfo();
});
socket.on("disconnect", () => {
  cls($dotSock, "connected"); add($dotSock, "disconnected");
  $sockLbl.textContent = "Disconnected";
  if (!S.statusPollInterval) {
    S.statusPollInterval = setInterval(fetchCameraStatusFallback, 2000);
  }
});

socket.on("camera_status", (d) => {
  updateCamCard(d);
  updateOnlineCount();
  const msg = `${d.label || "cam"+d.cam_id}: ${d.status}${d.resolution ? " · "+d.resolution : ""}`;
  addLog(msg, d.cam_id, "");
});

socket.on("detection_event", (d) => {
  S.detectCount++;
  q("#kpi-detects").textContent = S.detectCount;
  addLog("person detected (conf " + (d.confidence_max || 0).toFixed(2) + ")", d.cam_id, "detect");
  flashDot(q("#dot-rec"));
});

socket.on("clip_saved", (d) => {
  S.clipCount++;
  q("#kpi-clips").textContent = S.clipCount;
  addLog(`clip saved: ${d.filename} · ${d.duration_s}s · ${d.size_mb} MB`, d.cam_id, "clip");
  fetchStorageInfo();
  fetchVideoList();
});

socket.on("rec_log", (d) => {
  addLog(d.message, d.cam_id, "info");
});

socket.on("analysis_progress",  showAnaProgress);
socket.on("analysis_complete",  doneAnalysis);
socket.on("pose_progress",      showPoseProgress);
socket.on("pose_complete",      donePose);
socket.on("pose_lamp_state",    syncPoseLampState);

socket.on("storage_warning", (d) => {
  addLog(`storage ${d.used_gb}/${d.limit_gb} GB (${d.pct}%)`, "system", "warn");
  applyStorageBar(d.used_gb, d.limit_gb);
});

socket.on("error", (d) => {
  addLog(`error [${d.source}]: ${d.message}`, "system", "err");
});

// ── Tab navigation ─────────────────────────────────────────────────────────
function initTabNavigation() {
  const items = document.querySelectorAll(".nav-item[data-tab]");
  console.log("🔍 initTabNavigation: Found", items.length, "nav-items");
  items.forEach(el => {
    console.log("  → Attaching listener to nav-item:", el.dataset.tab);
    el.addEventListener("click", ev => { 
      console.log("✅ Click triggered on nav-item:", el.dataset.tab);
      ev.preventDefault(); 
      switchTab(el.dataset.tab); 
    });
  });
  
  // Fallback: add event delegation on parent nav if direct listeners don't work
  const navParent = document.querySelector(".sidebar-nav");
  if (navParent) {
    navParent.addEventListener("click", ev => {
      const navItem = ev.target.closest(".nav-item[data-tab]");
      if (navItem) {
        console.log("⚡ Event delegation triggered on nav-item:", navItem.dataset.tab);
        ev.preventDefault();
        switchTab(navItem.dataset.tab);
      }
    });
  }
}

function switchTab(id) {
  S.activeTab = id;
  document.querySelectorAll(".nav-item").forEach(el => el.classList.toggle("active", el.dataset.tab === id));
  document.querySelectorAll(".tab-panel").forEach(el => el.classList.toggle("active", el.id === "tab-"+id));
  q("#page-title").textContent = TAB_TITLES[id] || id;
  if (id === "analysis") { fetchVideoList(); }
  if (id === "results")  { fetchResults(); }
  if (id === "pose") { 
    fetchPoseStatus(); 
    fetchPoseResults();
    // Auto-start Phase 3 if not already running
    if (!S.poseRunning) {
      setTimeout(() => autoStartPose(), 500);
    }
  }
  if (id === "monitor" && S.recRunning) { startAllStreams(); }
  if (id !== "monitor") { stopAllStreams(); }
}

// Auto-start Phase 3 with multicam folder
async function autoStartPose() {
  const folder = q("#pose-dir")?.value?.trim() || "data/multicam";
  const overlay = q("#pose-save-overlay")?.checked ?? true;
  
  try {
    const d = await api("/api/pose/start", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folder, save_overlay: overlay }),
    });
    
    // Immediately update UI with lamp state
    S.poseRunning = true;
    S.poseClipsTotal = d.total_clips || d.clips_total || 0;
    S.poseCurrentClip = "";
    
    q("#btn-start-pose").disabled = true;
    q("#btn-stop-pose").disabled = false;
    add(q("#dot-pose"), "online");
    q("#lbl-pose").textContent = "Running";
    q("#pose-top-status").textContent = "Running";
    add(q("#pose-top-status"), "badge-online");
    
    hide(q("#pose-idle-state"));
    hide(q("#pose-done-state"));
    
    q("#pose-stat-clips").textContent = "0/" + S.poseClipsTotal;
    q("#pose-stat-kps").textContent = "0";
    q("#pose-stat-adl").textContent = "0";
    q("#pose-stat-tracks").textContent = "0";
    q("#pose-pct-badge").textContent = "0%";
    
    q("#pv-fps").textContent = "0.0";
    q("#pv-frame").textContent = "0 / 0";
    
    // Sync lamp state immediately 
    syncPoseLampState(d);
    
    // Start real-time polling for lamp updates
    startPoseLampPolling();
    startPoseViewerPolling();
    
    addLog(`Auto-started Phase 3 demo from ${folder}`, "system", "info");
  } catch (e) {
    addLog(`Auto-start failed: ${e.message}`, "system", "err");
  }
}

// Ensure DOM is ready before setting up listeners
try {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initTabNavigation);
  } else {
    initTabNavigation();
  }
} catch (e) {
  console.error("Failed to init tab navigation:", e);
}

// ── Generic helpers ─────────────────────────────────────────────────────────
function q(sel)          { return document.querySelector(sel); }
function qAll(sel)       { return document.querySelectorAll(sel); }
function add(el, cls)    { el && el.classList.add(cls); }
function cls(el, ...cs)  { el && cs.forEach(c => el.classList.remove(c)); }
function show(el)        { el && el.classList.remove("hidden"); }
function hide(el)        { el && el.classList.add("hidden"); }
function esc(s)          { return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#39;"); }

async function api(url, opts = {}) {
  const r = await fetch(url, opts);
  let d = {};
  try { d = await r.json(); } catch {}
  if (!r.ok) throw new Error(d.error || d.message || `HTTP ${r.status}`);
  return d;
}

function defaultPoseLampState() {
  const state = {};
  POSE_CAMERA_ORDER.forEach(camId => { state[camId] = "IDLE"; });
  return state;
}

function syncPoseLampState(d = {}) {
  S.poseLampState = { ...defaultPoseLampState(), ...(d.lamp_state || {}) };
  S.poseClipQueue = Array.isArray(d.clip_queue) ? d.clip_queue : S.poseClipQueue;
  S.poseActiveCamera = d.active_camera || "";
  S.poseCurrentClip = d.current_clip || d.clip || S.poseCurrentClip || "";
  if (Number.isFinite(d.clips_total)) S.poseClipsTotal = d.clips_total;
  if (Number.isFinite(d.clips_done)) S.poseClipsDone = d.clips_done;
  renderPoseLamps();
  renderPoseQueue(Number.isFinite(d.clips_done) ? d.clips_done : null);
}

function renderPoseLamps() {
  POSE_CAMERA_ORDER.forEach(camId => {
    const lamp = q(`.lamp-card[data-cam="${camId}"]`);
    if (!lamp) return;
    const state = String(S.poseLampState[camId] || "IDLE").toUpperCase();
    cls(lamp, "lamp-idle", "lamp-active", "lamp-done", "lamp-alert");
    add(lamp, "lamp-" + state.toLowerCase());
    const label = lamp.querySelector(".lamp-state");
    if (label) label.textContent = state;
  });

  const badge = q("#pose-active-cam-badge");
  if (!badge) return;
  
  // Determine badge text based on overall lamp state
  const lampValues = Object.values(S.poseLampState);
  const allDone = POSE_CAMERA_ORDER.every(cam => S.poseLampState[cam] === "DONE");
  const anyAlert = lampValues.includes("ALERT");
  const anyActive = lampValues.includes("ACTIVE");
  
  if (allDone) {
    badge.textContent = "✓ Sequence Complete";
  } else if (anyAlert) {
    badge.textContent = "⚠ Alert";
  } else if (S.poseActiveCamera) {
    badge.textContent = `▶ Processing ${S.poseActiveCamera.toUpperCase()}`;
    q("#pose-top-cam").textContent = S.poseActiveCamera.toUpperCase();
  } else if (anyActive) {
    badge.textContent = "▶ Processing…";
  } else {
    badge.textContent = "Idle";
    q("#pose-top-cam").textContent = "—";
  }
}

function renderPoseQueue(clipsDone = null) {
  const queue = q("#pose-queue");
  const count = q("#pose-queue-count");
  if (!queue || !count) return;

  const items = Array.isArray(S.poseClipQueue) ? S.poseClipQueue : [];
  const doneCount = clipsDone == null ? -1 : clipsDone;
  const progressPct = S.poseClipsTotal > 0 ? Math.round((doneCount / S.poseClipsTotal) * 100) : 0;
  
  count.textContent = `${items.length} clip${items.length === 1 ? "" : "s"}`;
  if (!items.length) {
    queue.innerHTML = `<div class="empty" style="padding:16px"><div class="empty-sub">Start Phase 3 to build the multicam queue</div></div>`;
    return;
  }

  // Group by camera for better visual organization
  const byCam = {};
  items.forEach((item, index) => {
    const cam = item.cam_id || "unknown";
    if (!byCam[cam]) byCam[cam] = [];
    byCam[cam].push({ ...item, index });
  });
  
  let html = "";
  POSE_CAMERA_ORDER.forEach(cam => {
    if (!byCam[cam] || !byCam[cam].length) return;
    
    const camItems = byCam[cam];
    const lampState = S.poseLampState[cam] || "IDLE";
    const stateIcon = lampState === "DONE" ? "✓" : lampState === "ACTIVE" ? "▶" : lampState === "ALERT" ? "⚠" : "○";
    
    camItems.forEach((item, idx) => {
      const isActive = item.clip_name === S.poseCurrentClip || item.clip_stem === S.poseCurrentClip;
      const isDone = doneCount >= 0 && item.index < doneCount;
      const stateClass = isActive ? "is-active" : (isDone ? "is-done" : "");
      
      html += `
        <div class="pose-queue-item ${stateClass}">
          <span class="pose-queue-cam">${esc(cam)}</span>
          <span class="pose-queue-name">${esc(item.clip_name || item.clip_stem || "unknown")}</span>
          <span class="pose-queue-time">${esc(item.clip_time || "")}</span>
        </div>`;
    });
  });
  
  queue.innerHTML = html || `<div class="empty" style="padding:16px"><div class="empty-sub">No clips in queue</div></div>`;
}

// ── Config tab ─────────────────────────────────────────────────────────────
const $drop     = q("#drop-zone");
const $fileIn   = q("#file-input");
const $cfgMsg   = q("#config-msg");

$drop.addEventListener("click", () => $fileIn.click());
$drop.addEventListener("dragover", ev => { ev.preventDefault(); add($drop, "over"); });
$drop.addEventListener("dragleave", () => cls($drop, "over"));
$drop.addEventListener("drop", ev => {
  ev.preventDefault(); cls($drop, "over");
  if (ev.dataTransfer.files[0]) uploadConfig(ev.dataTransfer.files[0]);
});
$fileIn.addEventListener("change", () => $fileIn.files[0] && uploadConfig($fileIn.files[0]));

async function uploadConfig(file) {
  const fd = new FormData(); fd.append("file", file);
  cls($cfgMsg, "hidden", "ok", "err"); show($cfgMsg); $cfgMsg.textContent = "Uploading…";
  try {
    const d = await api("/api/config/upload", { method: "POST", body: fd });
    S.cameras = d.cameras || [];
    add($cfgMsg, "ok"); $cfgMsg.textContent = `✓ Loaded ${S.cameras.length} camera(s) from resources.txt`;
    renderConfigCamList(S.cameras);
  } catch (e) {
    add($cfgMsg, "err"); $cfgMsg.textContent = "✗ " + e.message;
  }
}

async function fetchCameras() {
  try {
    const d = await api("/api/config/cameras");
    S.cameras = d.cameras || [];
    renderConfigCamList(S.cameras);
  } catch {}
}

function renderConfigCamList(cams) {
  const el = q("#camera-list");
  q("#cam-count-badge").textContent = cams.length + " camera" + (cams.length !== 1 ? "s" : "");
  if (!cams.length) {
    el.innerHTML = `<div class="empty"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="36" height="36"><rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/></svg><div class="empty-title">No cameras loaded</div><div class="empty-sub">Upload resources.txt to get started</div></div>`;
    return;
  }
  el.innerHTML = cams.map(c => `
    <div class="cam-row" id="cfg-cam-${c.cam_id}">
      <span class="cam-id">${esc(c.cam_id)}</span>
      <span class="cam-label">${esc(c.label || "cam"+c.cam_id)}</span>
      <span class="cam-url" title="${esc(c.url)}">${maskRtsp(c.url)}</span>
      <span class="cam-res" id="res-tag-${c.cam_id}">—</span>
      <button class="btn btn-ghost btn-xs" onclick="probeCamera('${c.cam_id}','${c.url.replace(/'/g,"\\'")}')" type="button">Probe</button>
    </div>
  `).join("");
}

function maskRtsp(url) {
  return esc(url.replace(/(rtsp:\/\/)([^:]+):([^@]+)@/, "$1***:***@"));
}

q("#btn-probe-all").addEventListener("click", async () => {
  for (const c of S.cameras) await probeCamera(c.cam_id, c.url, false);
});

q("#btn-load-local-test").addEventListener("click", async () => {
  try {
    const d = await api("/api/config/load_local", { method: "POST" });
    S.cameras = d.cameras || [];
    renderConfigCams();
    msg(d.message || "Local videos loaded as cameras", true);
  } catch(e) {
    msg(e.message, false);
  }
});

async function probeCamera(camId, url, showModal = true) {
  if (showModal) {
    q("#probe-title").textContent = "Probing cam" + camId + "…";
    q("#probe-body").innerHTML = '<div class="spin"></div>';
    show(q("#modal-probe"));
  }
  try {
    const d = await api("/api/cameras/probe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cam_id: camId, url }),
    });
    S.selectedRes[camId] = { width: d.width, height: d.height };
    const tag = q(`#res-tag-${camId}`);
    if (tag) tag.textContent = `${d.width}×${d.height}`;
    if (!showModal) return;
    q("#probe-title").textContent = "Camera cam" + camId;
    q("#probe-body").innerHTML = `
      <dl class="probe-dl">
        <dt>ID</dt><dd>cam${esc(d.cam_id)}</dd>
        <dt>Resolution</dt><dd>${d.width} × ${d.height}</dd>
        <dt>FPS</dt><dd>${d.fps}</dd>
      </dl>
      <div class="form-label mb-4">Select recording resolution</div>
      <div class="res-list">
        ${(d.resolutions || []).map(r => `
          <button class="res-btn ${r.width===d.width&&r.height===d.height?"sel":""}"
            onclick="pickRes('${d.cam_id}',${r.width},${r.height},this)" type="button">
            ${esc(r.label)}
          </button>`).join("")}
      </div>`;
  } catch (e) {
    if (showModal) q("#probe-body").innerHTML = `<div class="empty"><div class="empty-title">Probe failed</div><div class="empty-sub">${esc(e.message)}</div><button class="btn btn-ghost mt-3" onclick="probeCamera('${camId}','${url}')">Retry</button></div>`;
  }
}

window.probeCamera = probeCamera;

window.pickRes = function(camId, w, h, btn) {
  S.selectedRes[camId] = { width: w, height: h };
  const tag = q(`#res-tag-${camId}`); if (tag) tag.textContent = `${w}×${h}`;
  btn.closest(".res-list").querySelectorAll(".res-btn").forEach(b => cls(b,"sel"));
  add(btn, "sel");
};

q("#modal-close").addEventListener("click", () => hide(q("#modal-probe")));

// ── Storage limit ──────────────────────────────────────────────────────────
q("#btn-set-limit").addEventListener("click", async () => {
  const v = parseFloat(q("#storage-limit").value);
  S.storageLimitGb = isFinite(v) ? v : 10;
  try {
    await api("/api/storage/limit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ limit_gb: S.storageLimitGb }),
    });
    fetchStorageInfo();
  } catch (e) { addLog(e.message, "system", "err"); }
});

async function fetchStorageInfo() {
  try {
    const d = await api("/api/storage/info");
    q("#storage-text").textContent = d.used_gb + " GB";
    q("#storage-detail").textContent = d.file_count + " clip" + (d.file_count !== 1 ? "s" : "");
    applyStorageBar(d.used_gb, S.storageLimitGb);
  } catch {}
}

function applyStorageBar(used, limit) {
  const pct = limit > 0 ? Math.min(100, (used/limit)*100) : 0;
  const fill = q("#storage-fill");
  fill.style.width = pct + "%";
  cls(fill, "hi", "crit");
  if (pct > 90) add(fill, "crit");
  else if (pct > 70) add(fill, "hi");
}

// ── Phase 1: Recording ─────────────────────────────────────────────────────
q("#btn-start-rec").addEventListener("click", async () => {
  if (!S.cameras.length) { addLog("Upload resources.txt first", "system", "warn"); return; }
  const camsPayload = S.cameras.map(c => ({
    cam_id: c.cam_id, url: c.url, label: c.label,
    ...(S.selectedRes[c.cam_id] || {}),
  }));
  try {
    await api("/api/recording/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cameras: camsPayload, storage_limit_gb: S.storageLimitGb }),
    });
    S.recRunning = true; S.clipCount = 0; S.detectCount = 0;
    q("#kpi-clips").textContent = "0";
    q("#kpi-detects").textContent = "0";
    q("#btn-start-rec").disabled = true;
    q("#btn-stop-rec").disabled = false;
    add(q("#dot-rec"), "online"); cls(q("#dot-rec"), "offline");
    q("#lbl-rec").textContent = "Running · person-triggered";
    q("#kpi-recording").textContent = "Active"; add(q("#kpi-recording"), "green");
    show(q("#nav-rec-badge"));
    renderCamGrid(S.cameras);
    addLog(`Recording started on ${S.cameras.length} camera(s)`, "system", "info");
    setGlobalStatus("recording");
  } catch (e) { addLog(e.message, "system", "err"); }
});

q("#btn-stop-rec").addEventListener("click", async () => {
  try { await api("/api/recording/stop", { method: "POST" }); } catch (e) { addLog(e.message, "system", "err"); }
  S.recRunning = false;
  q("#btn-start-rec").disabled = false;
  q("#btn-stop-rec").disabled = true;
  cls(q("#dot-rec"), "online", "recording"); add(q("#dot-rec"), "offline");
  q("#lbl-rec").textContent = "Idle";
  q("#kpi-recording").textContent = "Off"; cls(q("#kpi-recording"), "green", "red");
  hide(q("#nav-rec-badge"));
  stopAllStreams();
  setGlobalStatus("idle");
  addLog("Recording stopped", "system", "info");
});

function setGlobalStatus(s) {
  const el = q("#global-status");
  el.className = "badge";
  const map = { idle:"badge-idle", recording:"badge-recording", armed:"badge-armed" };
  add(el, map[s] || "badge-idle");
  el.textContent = { idle:"Idle", recording:"Recording", armed:"Armed" }[s] || s;
}

// ── Camera grid + live stream ─────────────────────────────────────────────
function renderCamGrid(cameras) {
  const grid = q("#camera-grid");
  if (!cameras.length) {
    grid.innerHTML = `<div class="cam-empty"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg><p class="empty-title">No cameras configured</p><p>Add cameras in the Configuration tab</p></div>`;
    return;
  }
  grid.innerHTML = cameras.map(c => camCardHTML(c)).join("");
  if (S.activeTab === "monitor" && S.recRunning) { startAllStreams(); }
}

function camCardHTML(c) {
  return `
  <div class="cam-card" id="cam-card-${c.cam_id}">
    <div class="cam-head">
      <span class="cam-name">${esc(c.label || "cam"+c.cam_id)}</span>
      <div class="cam-badges">
        <span class="badge badge-idle" id="cam-conn-${c.cam_id}">Connecting</span>
        <span class="badge badge-idle hidden" id="cam-person-${c.cam_id}">Person</span>
        <span class="badge badge-idle hidden" id="cam-rec-${c.cam_id}">Recording</span>
      </div>
      <div class="cam-actions">
        <button class="btn btn-icon" title="Refresh stream" onclick="refreshStream('${c.cam_id}')" type="button">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="13" height="13"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-4"/></svg>
        </button>
      </div>
    </div>
    <div class="cam-stream-wrap">
      <img class="cam-stream-img hidden" id="stream-img-${c.cam_id}" alt="Live stream cam${c.cam_id}">
      <div class="cam-stream-placeholder" id="stream-placeholder-${c.cam_id}">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg>
        <span>Connecting…</span>
      </div>
      <div class="cam-stream-overlay">
        <span class="badge badge-idle" id="cam-state-badge-${c.cam_id}">idle</span>
      </div>
    </div>
    <div class="cam-meta">
      <div class="cam-meta-item">
        <span class="cam-meta-label">Resolution</span>
        <span class="cam-meta-value" id="cam-res-${c.cam_id}">—</span>
      </div>
      <div class="cam-meta-item">
        <span class="cam-meta-label">FPS</span>
        <span class="cam-meta-value" id="cam-fps-${c.cam_id}">—</span>
      </div>
      <div class="cam-meta-item">
        <span class="cam-meta-label">Clip Duration</span>
        <span class="cam-meta-value" id="cam-dur-${c.cam_id}">—</span>
      </div>
    </div>
  </div>`;
}

window.refreshStream = function(camId) { stopStream(camId); startStream(camId); };

function startAllStreams() {
  if (!S.cameras.length) return;
  S.cameras.forEach(c => startStream(c.cam_id));
}

function startStream(camId) {
  stopStream(camId);
  let failCount = 0;
  const img = q(`#stream-img-${camId}`);
  const ph  = q(`#stream-placeholder-${camId}`);
  if (!img) return;

  function loadFrame() {
    const src = `/api/cameras/${camId}/snapshot?t=${Date.now()}`;
    const tmp = new Image();
    tmp.onload = () => {
      img.src = src;
      show(img); hide(ph);
      failCount = 0;
    };
    tmp.onerror = () => {
      failCount++;
      if (failCount >= 5) {
        hide(img); show(ph);
        if (ph) ph.querySelector("span").textContent = "No signal — retrying…";
      }
    };
    tmp.src = src;
  }

  loadFrame();
  S.streamIntervals[camId] = setInterval(loadFrame, 150); // ~6-7 fps preview
}

function stopStream(camId) {
  if (S.streamIntervals[camId]) {
    clearInterval(S.streamIntervals[camId]);
    delete S.streamIntervals[camId];
  }
}

function stopAllStreams() {
  Object.keys(S.streamIntervals).forEach(stopStream);
}

async function fetchCameraStatusFallback() {
  try {
    const d = await api("/api/cameras/status");
    (d.cameras || []).forEach(updateCamCard);
    updateOnlineCount();
  } catch {}
}

function updateCamCard(d) {
  const camId = d.cam_id;

  // Connection badge
  const connBadge = q(`#cam-conn-${camId}`);
  if (connBadge) {
    connBadge.className = "badge";
    const statusMap = {
      online:    ["badge-online",    "Online"],
      offline:   ["badge-offline",   "Offline"],
      recording: ["badge-online",    "Online"],
      error:     ["badge-offline",   "Error"],
      connecting:["badge-idle",      "Connecting"],
    };
    const [cls_, lbl] = statusMap[d.status] || ["badge-idle", d.status];
    add(connBadge, cls_);
    connBadge.textContent = lbl;
  }

  // Person badge
  const personBadge = q(`#cam-person-${camId}`);
  if (personBadge) {
    if (d.person_detected) {
      show(personBadge);
      personBadge.className = "badge badge-person";
      personBadge.textContent = "Person";
    } else {
      hide(personBadge);
    }
  }

  // Recording badge
  const recBadge = q(`#cam-rec-${camId}`);
  if (recBadge) {
    if (d.recording) {
      show(recBadge);
      recBadge.className = "badge badge-recording";
      recBadge.textContent = "● Rec";
    } else if (d.rec_state === "armed") {
      show(recBadge);
      recBadge.className = "badge badge-armed";
      recBadge.textContent = "Armed";
    } else {
      hide(recBadge);
    }
  }

  // State badge (overlay)
  const stateBadge = q(`#cam-state-badge-${camId}`);
  if (stateBadge) {
    const stateMap = {
      idle:      ["badge-idle",      "IDLE"],
      armed:     ["badge-armed",     "ARMED"],
      recording: ["badge-recording", "REC"],
      post_roll: ["badge-recording", "POST-ROLL"],
    };
    const [sc, sl] = stateMap[d.rec_state] || ["badge-idle", d.rec_state || "IDLE"];
    stateBadge.className = "badge " + sc;
    stateBadge.textContent = sl;
  }

  // Meta values
  const resEl = q(`#cam-res-${camId}`);
  const fpsEl = q(`#cam-fps-${camId}`);
  const durEl = q(`#cam-dur-${camId}`);
  if (resEl) resEl.textContent = d.resolution || "—";
  if (fpsEl) fpsEl.textContent = d.fps ? d.fps + " fps" : "—";
  if (durEl) durEl.textContent = d.recording && d.clip_duration > 0 ? d.clip_duration + "s" : "—";

  // Card highlight
  const card = q(`#cam-card-${camId}`);
  if (card) card.classList.toggle("is-recording", !!d.recording);
}

function updateOnlineCount() {
  // Count online from all tiles
  const online = document.querySelectorAll('.badge-online').length;
  q("#kpi-online").textContent = online;
  q("#kpi-online-sub").textContent = "of " + S.cameras.length + " configured";
}

function flashDot(el) {
  if (!el) return;
  el.style.opacity = ".3";
  setTimeout(() => el.style.opacity = "1", 200);
}

// ── Event log ──────────────────────────────────────────────────────────────
const btnClearLog = q("#btn-clear-log");
if (btnClearLog) {
  btnClearLog.addEventListener("click", () => {
    q("#log-wrap").innerHTML = `<div class="empty" style="padding:16px"><div class="empty-sub">Events will appear here during recording</div></div>`;
  });
}

function addLog(msg, camId, type = "") {
  const log = q("#log-wrap");
  if (!log) return;
  const empty = log.querySelector(".empty"); if (empty) empty.remove();
  const e = document.createElement("div");
  e.className = "log-entry " + type;
  const t = new Date().toLocaleTimeString("en-GB", { hour12: false });
  const camLabel = camId && camId !== "system" ? `<span class="log-cam">cam${camId}</span>` : "";
  e.innerHTML = `<span class="log-t">${t}</span>${camLabel}<span class="log-msg">${esc(msg)}</span>`;
  log.prepend(e);
  if (log.children.length > 250) log.removeChild(log.lastElementChild);
}

// ── Phase 2: Analysis ──────────────────────────────────────────────────────
q("#btn-start-analysis").addEventListener("click", async () => {
  const folder = q("#video-dir").value.trim() || "data/raw_videos";
  try {
    const d = await api("/api/analysis/start", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_dir: folder }),
    });
    S.anaRunning = true; S.anaClipsTotal = d.clips || 0;
    q("#btn-start-analysis").disabled = true;
    q("#btn-stop-analysis").disabled = false;
    add(q("#dot-ana"), "online"); q("#lbl-ana").textContent = "Running";
    show(q("#ana-running-state")); hide(q("#ana-idle-state")); hide(q("#ana-done-state"));
    q("#stat-clips").textContent = "0/" + S.anaClipsTotal;
    q("#stat-frames").textContent = "0";
    q("#stat-labels").textContent = "0";
  } catch (e) { addLog(e.message, "system", "err"); }
});

q("#btn-stop-analysis").addEventListener("click", async () => {
  try { await api("/api/analysis/stop", { method: "POST" }); } catch {}
  resetAnaUi();
});

function showAnaProgress(d) {
  show(q("#ana-running-state")); hide(q("#ana-idle-state"));
  q("#ana-clip-name").textContent = d.clip || "—";
  q("#ana-pct").textContent = (d.pct || 0) + "%";
  q("#ana-bar").style.width = (d.pct || 0) + "%";
  if (d.frames_saved !== undefined) q("#stat-frames").textContent = d.frames_saved;
  if (d.labels_written !== undefined) q("#stat-labels").textContent = d.labels_written;
}

function doneAnalysis(d) {
  S.anaRunning = false;
  hide(q("#ana-running-state")); show(q("#ana-done-state"));
  q("#stat-clips").textContent = `${d.clips_done||0}/${S.anaClipsTotal}`;
  q("#stat-frames").textContent = d.frames_saved || 0;
  q("#stat-labels").textContent = d.labels_written || 0;
  resetAnaUi();
  fetchResults();
}

function resetAnaUi() {
  q("#btn-start-analysis").disabled = false;
  q("#btn-stop-analysis").disabled = true;
  cls(q("#dot-ana"), "online"); q("#lbl-ana").textContent = "Idle";
}

q("#btn-refresh-videos").addEventListener("click", fetchVideoList);

async function fetchVideoList() {
  const area = q("#video-list-area");
  try {
    const d = await api("/api/videos");
    const videos = d.videos || [];
    q("#clip-count-badge").textContent = videos.length + " clips";
    if (!videos.length) {
      area.innerHTML = `<div class="empty"><div class="empty-sub">No MP4 clips found in the selected folder</div></div>`;
      return;
    }
    area.innerHTML = `
      <table>
        <thead><tr><th>Filename</th><th>Size</th><th>Modified</th><th></th></tr></thead>
        <tbody>${videos.map(v => `
          <tr>
            <td class="td-mono">${esc(v.filename)}</td>
            <td>${v.size_mb} MB</td>
            <td class="txt-muted">${new Date(v.mtime*1000).toLocaleString()}</td>
            <td class="td-right">
              <button class="btn btn-danger btn-xs" onclick="deleteVideo('${v.filename.replace(/'/g,"\\'")}')">Delete</button>
            </td>
          </tr>`).join("")}
        </tbody>
      </table>`;
  } catch (e) {
    area.innerHTML = errorState(e.message, "fetchVideoList()");
  }
}

window.deleteVideo = async function(fn) {
  if (!confirm("Delete " + fn + "?")) return;
  try {
    await api("/api/videos/" + encodeURIComponent(fn), { method: "DELETE" });
    fetchVideoList(); fetchStorageInfo();
  } catch (e) { addLog(e.message, "system", "err"); }
};

// ── Phase 2: Results ───────────────────────────────────────────────────────
q("#btn-refresh-results").addEventListener("click", fetchResults);

async function fetchResults() {
  const grid = q("#results-grid");
  try {
    const d = await api("/api/analysis/results");
    q("#result-count-label").textContent = (d.results||[]).length + " results";
    if (!d.results.length) {
      grid.innerHTML = `<div class="empty col-12"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="36" height="36"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg><div class="empty-title">No outputs yet</div><div class="empty-sub">Run Phase 2 Analysis to generate results</div></div>`;
      return;
    }
    grid.innerHTML = d.results.map(r => `
      <div class="result-card">
        <div class="result-card-head"><h3>${esc(r.clip_stem)}</h3></div>
        <div class="result-card-body">
          <div class="result-stat"><span>PNG frames</span><strong>${r.frames}</strong></div>
          <div class="result-stat"><span>Bounding boxes</span><strong>${r.label_count}</strong></div>
          <div class="result-stat"><span>Label file</span><strong>${esc(r.label_file||"—")}</strong></div>
        </div>
      </div>`).join("");
  } catch (e) {
    grid.innerHTML = errorState(e.message, "#btn-refresh-results");
  }
}

// ── Phase 3: Pose & ADL ────────────────────────────────────────────────────
q("#btn-start-pose").addEventListener("click", async () => {
  const folder = q("#pose-dir").value.trim() || "data/multicam";
  const overlay = q("#pose-save-overlay").checked;
  try {
    const d = await api("/api/pose/start", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folder, save_overlay: overlay }),
    });
    S.poseRunning = true; S.poseClipsTotal = d.total_clips || 0;
    S.poseCurrentClip = "";
    q("#btn-start-pose").disabled = true;
    q("#btn-stop-pose").disabled = false;
    add(q("#dot-pose"), "online"); q("#lbl-pose").textContent = "Running";
    q("#pose-top-status").textContent = "Running";
    q("#pose-top-status").className = "badge badge-online";
    q("#pose-top-clip").textContent = "—";
    hide(q("#pose-idle-state")); hide(q("#pose-done-state"));
    q("#pose-stat-clips").textContent = "0/" + S.poseClipsTotal;
    q("#pose-stat-kps").textContent = "0";
    q("#pose-stat-adl").textContent = "0";
    q("#pose-stat-tracks").textContent = "0";
    syncPoseLampState(d);
    startPoseLampPolling();
    startPoseViewerPolling();
  } catch (e) { addLog(e.message, "system", "err"); }
});

q("#btn-stop-pose").addEventListener("click", async () => {
  try { await api("/api/pose/stop", { method: "POST" }); } catch {}
  
  // Stop polling
  if (S.poseLampPollInterval) {
    clearInterval(S.poseLampPollInterval);
    S.poseLampPollInterval = null;
  }
  if (S.poseViewerInterval) {
    clearInterval(S.poseViewerInterval);
    S.poseViewerInterval = null;
  }
  
  q("#btn-stop-pose").disabled = true;
  q("#lbl-pose").textContent = "Stopping";
});

function showPoseProgress(d) {
  hide(q("#pose-idle-state"));
  q("#pose-top-clip").textContent = d.current_clip || d.clip || "—";
  q("#pose-pct-badge").textContent = (d.pct || d.progress_pct || 0) + "%";
  q("#pose-bar").style.width = (d.pct || d.progress_pct || 0) + "%";
  if (Number.isFinite(d.clips_total)) S.poseClipsTotal = d.clips_total;
  q("#pose-stat-clips").textContent = `${d.clips_done || 0}/${S.poseClipsTotal || 0}`;
  q("#pose-stat-kps").textContent = d.keypoints_written || 0;
  q("#pose-stat-adl").textContent = d.adl_events || 0;
  q("#pose-stat-tracks").textContent = d.active_tracks || 0;
  
  if (d.fps !== undefined) q("#pv-fps").textContent = d.fps;
  if (d.frame_id !== undefined && d.total_frames !== undefined) {
    q("#pv-frame").textContent = `${d.frame_id} / ${d.total_frames}`;
  }
  
  syncPoseLampState(d);
}

function donePose(d) {
  S.poseRunning = false;
  syncPoseLampState(d);
  
  // Stop polling when complete
  if (S.poseLampPollInterval) {
    clearInterval(S.poseLampPollInterval);
    S.poseLampPollInterval = null;
  }
  if (S.poseViewerInterval) {
    clearInterval(S.poseViewerInterval);
    S.poseViewerInterval = null;
  }
  
  show(q("#pose-done-state"));
  q("#pose-stat-clips").textContent = `${d.clips_done||0}/${S.poseClipsTotal}`;
  q("#pose-stat-kps").textContent = d.keypoints_written || 0;
  q("#pose-stat-adl").textContent = d.adl_events || 0;
  q("#pose-stat-tracks").textContent = "0";
  resetPoseUi("Complete");
  fetchPoseResults();
}

function startPoseLampPolling() {
  // Clear any existing polling
  if (S.poseLampPollInterval) {
    clearInterval(S.poseLampPollInterval);
  }
  
  // Poll every 500ms for real-time lamp updates
  S.poseLampPollInterval = setInterval(async () => {
    try {
      const d = await api("/api/pose/status");
      if (!d.running) {
        // Phase 3 completed
        clearInterval(S.poseLampPollInterval);
        S.poseLampPollInterval = null;
        donePose(d);
        return;
      }
      
      // Update progress and lamps
      showPoseProgress({
        current_clip: d.current_clip,
        clips_done: d.clips_done,
        clips_total: d.clips_total,
        progress_pct: d.progress_pct,
        pct: d.progress_pct,
        lamp_state: d.lamp_state,
        active_camera: d.active_camera,
        keypoints_written: d.keypoints_written,
        adl_events: d.adl_events,
      });
    } catch (e) {
      console.error("Polling error:", e);
    }
  }, 500);
}

function startPoseViewerPolling() {
  if (S.poseViewerInterval) clearInterval(S.poseViewerInterval);
  const imgOrig = q("#pv-orig");
  const imgProc = q("#pv-proc");
  const phOrig = q("#pv-orig-ph");
  const phProc = q("#pv-proc-ph");
  
  S.poseViewerInterval = setInterval(() => {
    if (!S.poseRunning) return;
    
    // Load original frame
    const srcO = `/api/pose/snapshot/original?t=${Date.now()}`;
    const tagO = new Image();
    tagO.onload = () => { imgOrig.src = srcO; imgOrig.style.opacity = 1; hide(phOrig); };
    tagO.onerror = () => { imgOrig.style.opacity = 0; show(phOrig); };
    tagO.src = srcO;

    // Load processed frame
    const srcP = `/api/pose/snapshot/processed?t=${Date.now()}`;
    const tagP = new Image();
    tagP.onload = () => { imgProc.src = srcP; imgProc.style.opacity = 1; hide(phProc); };
    tagP.onerror = () => { imgProc.style.opacity = 0; show(phProc); };
    tagP.src = srcP;
  }, 150); // ~6 fps roughly
}

function resetPoseUi(label = "Idle") {
  q("#btn-start-pose").disabled = false;
  q("#btn-stop-pose").disabled = true;
  cls(q("#dot-pose"), "online"); q("#lbl-pose").textContent = label;
  q("#pose-top-status").textContent = label;
  q("#pose-top-status").className = "badge badge-idle";
}

q("#btn-refresh-pose").addEventListener("click", async () => {
  await fetchPoseStatus();
  await fetchPoseResults();
});
q("#btn-refresh-pose-results").addEventListener("click", fetchPoseResults);

async function fetchPoseStatus() {
  try {
    const d = await api("/api/pose/status");
    S.poseRunning = !!d.running;
    S.poseClipsTotal = d.clips_total || 0;
    syncPoseLampState(d);
    q("#pose-stat-clips").textContent = `${d.clips_done || 0}/${S.poseClipsTotal || 0}`;
    q("#pose-stat-kps").textContent = d.keypoints_written || 0;
    q("#pose-stat-adl").textContent = d.adl_events || 0;
    q("#pose-top-clip").textContent = d.current_clip || "—";
    q("#pose-pct-badge").textContent = (d.progress_pct || 0) + "%";
    q("#pose-bar").style.width = (d.progress_pct || 0) + "%";

    if (d.running) {
      q("#btn-start-pose").disabled = true;
      q("#btn-stop-pose").disabled = false;
      add(q("#dot-pose"), "online");
      q("#lbl-pose").textContent = "Running";
      q("#pose-top-status").textContent = "Running";
      q("#pose-top-status").className = "badge badge-online";
      hide(q("#pose-idle-state")); hide(q("#pose-done-state"));
      if (!S.poseLampPollInterval) startPoseLampPolling();
      if (!S.poseViewerInterval) startPoseViewerPolling();
    } else if ((d.clips_done || 0) > 0) {
      resetPoseUi("Complete");
      hide(q("#pose-idle-state")); show(q("#pose-done-state"));
    } else {
      resetPoseUi("Idle");
      show(q("#pose-idle-state")); hide(q("#pose-done-state"));
    }
  } catch (e) {
    addLog(e.message, "system", "err");
  }
}

async function fetchPoseResults() {
  const grid = q("#pose-results-grid");
  try {
    const d = await api("/api/pose/results");
    if (!d.results.length) {
      grid.innerHTML = `<div class="empty"><div class="empty-sub">No pose outputs yet</div></div>`;
      return;
    }
    grid.innerHTML = d.results.map(r => `
      <div class="result-card">
        <div class="result-card-head flex-row" style="justify-content:space-between; padding-bottom:8px">
          <strong class="fs-12" style="word-break: break-all">${esc(r.clip_stem)}</strong>
          ${r.mp4_exists ? `<button class="btn btn-ghost btn-xs" onclick="showPreviewModal('${esc(r.clip_stem)}')">Preview <svg viewBox="0 0 24 24" width="12" height="12" stroke="currentColor" stroke-width="2" fill="none"><polygon points="5 3 19 12 5 21 5 3"/></svg></button>` : ''}
        </div>
        <div class="result-card-body">
          <div class="result-stat"><span>Keypoint rows</span><strong>${r.keypoints_count}</strong></div>
          <div class="result-stat"><span>ADL events</span><strong>${r.adl_events}</strong></div>
          <div class="result-stat"><span>Overlay frames</span><strong>${r.overlays}</strong></div>
          <div class="adl-chips mt-2">${renderAdlChips(r.adl_summary)}</div>
        </div>
      </div>`).join("");
  } catch (e) {
    grid.innerHTML = `<div class="empty"><div class="empty-sub">${esc(e.message)}</div></div>`;
  }
}

function showPreviewModal(clipStem) {
  const modal = q("#modal-preview");
  const video = q("#preview-video");
  // Assuming a static route serves this, but D:\Capstone_Project\data is exposed at /api/video/<subpath>? Let's use standard logic. Wait, do we have an endpoint for static video? 
  // Let me check app/api/routes.py to see how videos are served. 
  // Or I can point directly to `/static/...?` No, `data/` is not static by default in Flask. Let's see.
  video.src = `/api/video/output_process/${clipStem}/${clipStem}_processed.mp4`;
  q("#preview-title").textContent = clipStem;
  show(modal);
}

function closePreviewModal() {
  const modal = q("#modal-preview");
  const video = q("#preview-video");
  video.pause();
  video.src = "";
  hide(modal);
}

function renderAdlChips(summary) {
  const entries = Object.entries(summary || {});
  if (!entries.length) return '<span class="adl-chip">No ADL labels</span>';
  return entries.map(([k,v]) => `<span class="adl-chip">${esc(k)} ${v}%</span>`).join("");
}

// ── Error state helper ─────────────────────────────────────────────────────
function errorState(msg, retryFn) {
  return `<div class="empty">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="36" height="36"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
    <div class="empty-title">Failed to load</div>
    <div class="empty-sub">${esc(msg)}</div>
    ${retryFn ? `<button class="btn btn-ghost mt-3" onclick="${retryFn}">Retry</button>` : ""}
  </div>`;
}

// ── Init ───────────────────────────────────────────────────────────────────
(async function init() {
  S.poseLampState = defaultPoseLampState();
  renderPoseLamps();
  renderPoseQueue(0);
  await fetchCameras();
  await fetchStorageInfo();
  await fetchPoseStatus();
})();
