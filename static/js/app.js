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
  clipCount:          0,
  detectCount:        0,
  onlineCount:        0,
  recCount:           0,
  streamIntervals:    {},    // { cam_id: intervalId }
};

const TAB_TITLES = {
  config:   "System Configuration",
  monitor:  "Live Monitor — Phase 1",
  analysis: "Phase 2 Analysis",
  results:  "Phase 2 Results",
  pose:     "Phase 3 — Pose & ADL",
};

// ── Socket.IO ──────────────────────────────────────────────────────────────
const socket = io({ transports: ["websocket", "polling"] });
const $dotSock  = q("#dot-sock");
const $sockLbl  = q("#sock-label");

socket.on("connect", () => {
  cls($dotSock, "disconnected"); add($dotSock, "connected");
  $sockLbl.textContent = "Connected";
  fetchStorageInfo();
});
socket.on("disconnect", () => {
  cls($dotSock, "connected"); add($dotSock, "disconnected");
  $sockLbl.textContent = "Disconnected";
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

socket.on("storage_warning", (d) => {
  addLog(`storage ${d.used_gb}/${d.limit_gb} GB (${d.pct}%)`, "system", "warn");
  applyStorageBar(d.used_gb, d.limit_gb);
});

socket.on("error", (d) => {
  addLog(`error [${d.source}]: ${d.message}`, "system", "err");
});

// ── Tab navigation ─────────────────────────────────────────────────────────
document.querySelectorAll(".nav-item[data-tab]").forEach(el => {
  el.addEventListener("click", ev => { ev.preventDefault(); switchTab(el.dataset.tab); });
});

function switchTab(id) {
  document.querySelectorAll(".nav-item").forEach(el => el.classList.toggle("active", el.dataset.tab === id));
  document.querySelectorAll(".tab-panel").forEach(el => el.classList.toggle("active", el.id === "tab-"+id));
  q("#page-title").textContent = TAB_TITLES[id] || id;
  if (id === "analysis") { fetchVideoList(); }
  if (id === "results")  { fetchResults(); }
  if (id === "pose")     { fetchPoseResults(); }
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
  cameras.forEach(c => startStream(c.cam_id));
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
q("#btn-clear-log").addEventListener("click", () => {
  q("#event-log").innerHTML = `<div class="empty" style="padding:16px"><div class="empty-sub">Events will appear here during recording</div></div>`;
});

function addLog(msg, camId, type = "") {
  const log = q("#event-log");
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
  const folder = q("#pose-dir").value.trim() || "data/raw_videos";
  const overlay = q("#pose-save-overlay").checked;
  try {
    const d = await api("/api/pose/start", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folder, save_overlay: overlay }),
    });
    S.poseRunning = true; S.poseClipsTotal = d.total_clips || 0;
    q("#btn-start-pose").disabled = true;
    q("#btn-stop-pose").disabled = false;
    add(q("#dot-pose"), "online"); q("#lbl-pose").textContent = "Running";
    show(q("#pose-running-state")); hide(q("#pose-idle-state")); hide(q("#pose-done-state"));
    q("#pose-stat-clips").textContent = "0/" + S.poseClipsTotal;
    q("#pose-stat-kps").textContent = "0";
    q("#pose-stat-adl").textContent = "0";
  } catch (e) { addLog(e.message, "system", "err"); }
});

q("#btn-stop-pose").addEventListener("click", async () => {
  try { await api("/api/pose/stop", { method: "POST" }); } catch {}
  resetPoseUi();
});

function showPoseProgress(d) {
  show(q("#pose-running-state")); hide(q("#pose-idle-state"));
  q("#pose-clip-name").textContent = d.clip || "—";
  q("#pose-pct").textContent = (d.pct || 0) + "%";
  q("#pose-bar").style.width = (d.pct || 0) + "%";
}

function donePose(d) {
  S.poseRunning = false;
  hide(q("#pose-running-state")); show(q("#pose-done-state"));
  q("#pose-stat-clips").textContent = `${d.clips_done||0}/${S.poseClipsTotal}`;
  q("#pose-stat-kps").textContent = d.keypoints_written || 0;
  q("#pose-stat-adl").textContent = d.adl_events || 0;
  resetPoseUi();
  fetchPoseResults();
}

function resetPoseUi() {
  q("#btn-start-pose").disabled = false;
  q("#btn-stop-pose").disabled = true;
  cls(q("#dot-pose"), "online"); q("#lbl-pose").textContent = "Idle";
}

q("#btn-refresh-pose").addEventListener("click", fetchPoseResults);
q("#btn-refresh-pose-results").addEventListener("click", fetchPoseResults);

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
        <div class="result-card-head"><h3>${esc(r.clip_stem)}</h3></div>
        <div class="result-card-body">
          <div class="result-stat"><span>Keypoint rows</span><strong>${r.keypoints_count}</strong></div>
          <div class="result-stat"><span>ADL events</span><strong>${r.adl_events}</strong></div>
          <div class="result-stat"><span>Overlay frames</span><strong>${r.overlays}</strong></div>
          <div class="adl-chips">${renderAdlChips(r.adl_summary)}</div>
        </div>
      </div>`).join("");
  } catch (e) {
    grid.innerHTML = `<div class="empty"><div class="empty-sub">${esc(e.message)}</div></div>`;
  }
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
  await fetchCameras();
  await fetchStorageInfo();
})();
