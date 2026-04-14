"use strict";
/* ════════════════════════════════════════════════════════════════════════
   CPose Dashboard — app.js  (Workspace UI, 3-tab layout)
   Vanilla JS · Socket.IO · Sequential Multicam Demo
════════════════════════════════════════════════════════════════════════ */

// ── Helpers ────────────────────────────────────────────────────────────────
function q(sel)        { return document.querySelector(sel); }
function qAll(sel)     { return document.querySelectorAll(sel); }
function add(el, cls)  { el && el.classList.add(cls); }
function cls(el,...cs) { el && cs.forEach(c => el.classList.remove(c)); }
function show(el)      { el && el.classList.remove("hidden"); }
function hide(el)      { el && el.classList.add("hidden"); }
function esc(s)        { return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#39;"); }
/** Safe addEventListener — skips if element missing */
function on(id, event, fn) {
  const el = typeof id === "string" ? q(id) : id;
  if (el) el.addEventListener(event, fn);
}

async function api(url, opts = {}) {
  const r = await fetch(url, opts);
  let d = {};
  try { d = await r.json(); } catch {}
  if (!r.ok) throw new Error(d.error || d.message || `HTTP ${r.status}`);
  return d;
}

// ── App State ──────────────────────────────────────────────────────────────
const S = {
  cameras:           [],
  storageLimitGb:    10,
  recRunning:        false,
  anaRunning:        false,
  poseRunning:       false,
  poseClipsTotal:    0,
  poseClipsDone:     0,
  poseLampState:     {},
  poseClipQueue:     [],
  poseActiveCamera:  "",
  poseCurrentClip:   "",
  activeTab:         "workspace",
  poseLampPollInterval: null,
  poseViewerInterval:   null,
  statusPollInterval:   null,
  streamIntervals:      {},
};

const CAMERA_ORDER = ["cam01","cam02","cam03","cam04"];

const TAB_TITLES = {
  workspace: "Processing Workspace",
  results:   "Completed Results",
  settings:  "Settings & Data Prep",
};

// ── Socket.IO ──────────────────────────────────────────────────────────────
const socket = io({ transports: ["websocket","polling"] });

socket.on("connect", () => {
  const dot = q("#dot-sock"), lbl = q("#sock-label");
  if (dot) { cls(dot,"disconnected"); add(dot,"connected"); }
  if (lbl) lbl.textContent = "Connected";
  if (S.statusPollInterval) { clearInterval(S.statusPollInterval); S.statusPollInterval = null; }
  fetchStorageInfo();
});

socket.on("disconnect", () => {
  const dot = q("#dot-sock"), lbl = q("#sock-label");
  if (dot) { cls(dot,"connected"); add(dot,"disconnected"); }
  if (lbl) lbl.textContent = "Disconnected";
  if (!S.statusPollInterval) {
    S.statusPollInterval = setInterval(fetchStorageInfo, 5000);
  }
});

socket.on("pose_progress",    showPoseProgress);
socket.on("pose_complete",    donePose);
socket.on("pose_lamp_state",  syncPoseLampState);
socket.on("storage_warning",  (d) => {
  addLog(`storage ${d.used_gb}/${d.limit_gb} GB (${d.pct}%)`, "system", "warn");
  applyStorageBar(d.used_gb, d.limit_gb);
});
socket.on("error", (d) => {
  addLog(`error [${d.source}]: ${d.message}`, "system", "err");
});

// ── Tab Navigation ─────────────────────────────────────────────────────────
function switchTab(id) {
  S.activeTab = id;
  qAll(".nav-item").forEach(el => el.classList.toggle("active", el.dataset.tab === id));
  qAll(".tab-panel").forEach(el => el.classList.toggle("active", el.id === "tab-"+id));
  const pt = q("#page-title");
  if (pt) pt.textContent = TAB_TITLES[id] || id;

  if (id === "workspace")  { fetchPoseStatus(); }
  if (id === "results")    { fetchPoseResults(); }
  if (id === "settings")   { /* static — no fetch needed */ }
}

function initTabNav() {
  // Delegate on parent to avoid missing individual element issues
  const nav = q(".sidebar-nav");
  if (nav) {
    nav.addEventListener("click", ev => {
      const item = ev.target.closest(".nav-item[data-tab]");
      if (item) { ev.preventDefault(); switchTab(item.dataset.tab); }
    });
  }
}

// ── Storage ────────────────────────────────────────────────────────────────
async function fetchStorageInfo() {
  try {
    const d = await api("/api/storage/info");
    const st = q("#storage-text"); if (st) st.textContent = d.used_gb + " GB";
    const sd = q("#storage-detail"); if (sd) sd.textContent = d.file_count + " clip" + (d.file_count !== 1 ? "s" : "");
    applyStorageBar(d.used_gb, S.storageLimitGb);
  } catch {}
}

function applyStorageBar(used, limit) {
  const pct = limit > 0 ? Math.min(100, (used/limit)*100) : 0;
  const fill = q("#storage-fill");
  if (!fill) return;
  fill.style.width = pct + "%";
  cls(fill,"hi","crit");
  if (pct > 90) add(fill,"crit");
  else if (pct > 70) add(fill,"hi");
}

on("#btn-set-limit", "click", async () => {
  const v = parseFloat(q("#storage-limit")?.value);
  S.storageLimitGb = isFinite(v) ? v : 10;
  try {
    await api("/api/storage/limit", {
      method: "POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({limit_gb: S.storageLimitGb}),
    });
    fetchStorageInfo();
    addLog(`Storage limit set to ${S.storageLimitGb} GB`, "system", "info");
  } catch (e) { addLog(e.message, "system", "err"); }
});

// ── Settings: Camera config ────────────────────────────────────────────────
(function initSettings() {
  const $drop   = q("#drop-zone");
  const $fileIn = q("#file-input");
  const $cfgMsg = q("#config-msg");

  if ($drop && $fileIn) {
    $drop.addEventListener("click", () => $fileIn.click());
    $drop.addEventListener("dragover", ev => { ev.preventDefault(); add($drop,"over"); });
    $drop.addEventListener("dragleave", () => cls($drop,"over"));
    $drop.addEventListener("drop", ev => {
      ev.preventDefault(); cls($drop,"over");
      if (ev.dataTransfer.files[0]) uploadConfig(ev.dataTransfer.files[0]);
    });
    $fileIn.addEventListener("change", () => $fileIn.files[0] && uploadConfig($fileIn.files[0]));
  }

  async function uploadConfig(file) {
    const fd = new FormData(); fd.append("file", file);
    if ($cfgMsg) { cls($cfgMsg,"hidden","ok","err"); show($cfgMsg); $cfgMsg.textContent = "Uploading…"; }
    try {
      const d = await api("/api/config/upload", { method:"POST", body:fd });
      S.cameras = d.cameras || [];
      if ($cfgMsg) { add($cfgMsg,"ok"); $cfgMsg.textContent = `✓ Loaded ${S.cameras.length} camera(s)`; }
      addLog(`Loaded ${S.cameras.length} cameras from resources.txt`, "system", "info");
    } catch (e) {
      if ($cfgMsg) { add($cfgMsg,"err"); $cfgMsg.textContent = "✗ " + e.message; }
    }
  }
})();

on("#btn-load-local-test", "click", async () => {
  try {
    const d = await api("/api/config/load_local", { method:"POST" });
    S.cameras = d.cameras || [];
    addLog(d.message || `Loaded ${S.cameras.length} local videos`, "system", "info");
    const cfgMsg = q("#config-msg");
    if (cfgMsg) { cls(cfgMsg,"hidden","err"); add(cfgMsg,"ok"); cfgMsg.textContent = `✓ ${d.message}`; show(cfgMsg); }
  } catch(e) {
    addLog(e.message, "system", "err");
  }
});

// Phase 1 recording stubs (buttons exist in Settings)
on("#btn-start-rec", "click", async () => {
  if (!S.cameras.length) { addLog("Upload resources.txt first", "system", "warn"); return; }
  const camsPayload = S.cameras.map(c => ({cam_id:c.cam_id, url:c.url, label:c.label}));
  try {
    await api("/api/recording/start", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({cameras:camsPayload, storage_limit_gb:S.storageLimitGb}),
    });
    S.recRunning = true;
    const lbl = q("#lbl-rec"); if (lbl) lbl.textContent = "Running";
    addLog(`Recording started on ${S.cameras.length} cam(s)`, "system", "info");
  } catch (e) { addLog(e.message, "system", "err"); }
});

on("#btn-stop-rec", "click", async () => {
  try { await api("/api/recording/stop", {method:"POST"}); } catch {}
  S.recRunning = false;
  const lbl = q("#lbl-rec"); if (lbl) lbl.textContent = "Stopped";
  addLog("Recording stopped", "system", "info");
});

// Phase 2 analysis stubs
on("#btn-start-analysis", "click", async () => {
  try {
    const d = await api("/api/analysis/start", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({video_dir:"data/raw_videos"}),
    });
    S.anaRunning = true;
    const lbl = q("#lbl-ana"); if (lbl) lbl.textContent = "Running";
    addLog(`Analysis started: ${d.clips || 0} clips`, "system", "info");
  } catch (e) { addLog(e.message, "system", "err"); }
});

on("#btn-stop-analysis", "click", async () => {
  try { await api("/api/analysis/stop", {method:"POST"}); } catch {}
  S.anaRunning = false;
  const lbl = q("#lbl-ana"); if (lbl) lbl.textContent = "Stopped";
});

// ── Lamp state ─────────────────────────────────────────────────────────────
function defaultLampState() {
  const s = {}; CAMERA_ORDER.forEach(c => s[c] = "IDLE"); return s;
}

function syncPoseLampState(d = {}) {
  S.poseLampState   = { ...defaultLampState(), ...(d.lamp_state || {}) };
  S.poseClipQueue   = Array.isArray(d.clip_queue) ? d.clip_queue : S.poseClipQueue;
  S.poseActiveCamera = d.active_camera || "";
  S.poseCurrentClip  = d.current_clip || d.clip || S.poseCurrentClip || "";
  if (Number.isFinite(d.clips_total)) S.poseClipsTotal = d.clips_total;
  if (Number.isFinite(d.clips_done))  S.poseClipsDone  = d.clips_done;
  renderLamps();
  renderQueue();
}

function renderLamps() {
  CAMERA_ORDER.forEach(camId => {
    const lamp = q(`.lamp-card[data-cam="${camId}"]`);
    if (!lamp) return;
    const state = String(S.poseLampState[camId] || "IDLE").toUpperCase();
    cls(lamp,"lamp-idle","lamp-active","lamp-done","lamp-alert");
    add(lamp,"lamp-"+state.toLowerCase());
    const lbl = lamp.querySelector(".lamp-state");
    if (lbl) lbl.textContent = state;
  });

  // Update active cam badge in header
  const cam = q("#pose-top-cam");
  if (!cam) return;
  if (S.poseActiveCamera) {
    cam.textContent = S.poseActiveCamera.toUpperCase();
  } else {
    cam.textContent = "—";
  }
}

function renderQueue() {
  const queue = q("#pose-queue");
  const count = q("#pose-queue-count");
  if (!queue) return;

  const items = Array.isArray(S.poseClipQueue) ? S.poseClipQueue : [];
  if (count) count.textContent = `${items.length} clip${items.length === 1 ? "" : "s"}`;

  if (!items.length) {
    queue.innerHTML = `<div class="empty" style="padding:16px"><div class="empty-sub">Start Processing to build the queue</div></div>`;
    return;
  }

  let html = "";
  items.forEach((item, idx) => {
    const isActive = item.clip_name === S.poseCurrentClip || item.clip_stem === S.poseCurrentClip;
    const isDone   = idx < S.poseClipsDone;
    const cls2     = isActive ? "is-active" : isDone ? "is-done" : "";
    html += `<div class="pose-queue-item ${cls2}">
      <span class="pose-queue-cam">${esc(item.cam_id||"?")}</span>
      <span class="pose-queue-name">${esc(item.clip_name||item.clip_stem||"—")}</span>
      <span class="pose-queue-time">${esc(item.clip_time||"")}</span>
    </div>`;
  });
  queue.innerHTML = html;
}

// ── Workspace: Pose & ADL (Phase 3) ───────────────────────────────────────
on("#btn-start-pose", "click", async () => {
  const folder  = q("#pose-dir")?.value?.trim() || "data/multicam";
  const overlay = true; // always save overlay
  try {
    const d = await api("/api/pose/start", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({folder, save_overlay:overlay}),
    });
    S.poseRunning = true;
    S.poseClipsTotal = d.total_clips || 0;
    S.poseClipsDone  = 0;
    S.poseCurrentClip = "";

    const btn  = q("#btn-start-pose"); if (btn)  btn.disabled  = true;
    const stop = q("#btn-stop-pose");  if (stop) stop.disabled = false;
    add(q("#dot-pose"), "online");
    const lbl = q("#lbl-pose"); if (lbl) lbl.textContent = "Running";
    const topStatus = q("#pose-top-status");
    if (topStatus) { topStatus.textContent = "Running"; topStatus.className = "badge badge-online"; }
    const clips = q("#pose-stat-clips"); if (clips) clips.textContent = "0/" + S.poseClipsTotal;
    const kps   = q("#pose-stat-kps");   if (kps)   kps.textContent = "0";
    const adl   = q("#pose-stat-adl");   if (adl)   adl.textContent = "0";
    const trk   = q("#pose-stat-tracks");if (trk)   trk.textContent = "0";
    const pct   = q("#pose-pct-badge");  if (pct)   pct.textContent = "0%";
    const fps   = q("#pv-fps");         if (fps)   fps.textContent = "0.0";
    const frm   = q("#pv-frame");       if (frm)   frm.textContent = "0 / 0";
    const bar   = q("#pose-bar");       if (bar)   bar.style.width = "0%";

    syncPoseLampState(d);
    startLampPolling();
    startViewerPolling();
    addLog(`Started Phase 3 from ${folder} — ${d.total_clips} clips`, "system", "info");
  } catch (e) {
    addLog(`Start failed: ${e.message}`, "system", "err");
  }
});

on("#btn-stop-pose", "click", async () => {
  try { await api("/api/pose/stop", {method:"POST"}); } catch {}
  stopPolling();
  const stop = q("#btn-stop-pose"); if (stop) stop.disabled = true;
  const lbl = q("#lbl-pose"); if (lbl) lbl.textContent = "Stopping";
});

on("#btn-refresh-pose", "click", async () => {
  await fetchPoseStatus();
  await fetchPoseResults();
});

on("#btn-refresh-pose-results", "click", fetchPoseResults);

// ── Pose progress callbacks ────────────────────────────────────────────────
function showPoseProgress(d) {
  const pct = d.pct || d.progress_pct || 0;
  const bar = q("#pose-bar");       if (bar)  bar.style.width = pct + "%";
  const pb  = q("#pose-pct-badge"); if (pb)   pb.textContent  = pct + "%";
  const clp = q("#pose-stat-clips"); if (clp) clp.textContent = `${d.clips_done||0}/${S.poseClipsTotal||0}`;
  const kps = q("#pose-stat-kps");   if (kps) kps.textContent = d.keypoints_written || 0;
  const adl = q("#pose-stat-adl");   if (adl) adl.textContent = d.adl_events || 0;
  const trk = q("#pose-stat-tracks");if (trk) trk.textContent = d.active_tracks || 0;
  const clip = q("#pose-top-clip");  if (clip) clip.textContent = d.current_clip || d.clip || "—";
  const fps  = q("#pv-fps");        if (fps)  fps.textContent  = d.fps ?? "0.0";
  if (d.frame_id !== undefined && d.total_frames !== undefined) {
    const frm = q("#pv-frame"); if (frm) frm.textContent = `${d.frame_id} / ${d.total_frames}`;
  }
  syncPoseLampState(d);
}

function donePose(d) {
  S.poseRunning    = false;
  S.poseClipsDone  = d.clips_done || 0;
  syncPoseLampState(d);
  stopPolling();

  const btn  = q("#btn-start-pose"); if (btn)  btn.disabled  = false;
  const stop = q("#btn-stop-pose");  if (stop) stop.disabled = true;
  add(q("#dot-pose"), "online");
  const lbl = q("#lbl-pose"); if (lbl) lbl.textContent = "Complete";
  const topStatus = q("#pose-top-status");
  if (topStatus) { topStatus.textContent = "Complete"; topStatus.className = "badge badge-online"; }
  const clp = q("#pose-stat-clips"); if (clp) clp.textContent = `${d.clips_done||0}/${S.poseClipsTotal}`;
  const kps = q("#pose-stat-kps");   if (kps) kps.textContent = d.keypoints_written || 0;
  const adl = q("#pose-stat-adl");   if (adl) adl.textContent = d.adl_events || 0;
  const bar = q("#pose-bar");        if (bar) bar.style.width = "100%";
  const pb  = q("#pose-pct-badge"); if (pb)  pb.textContent  = "100%";
  addLog(`Phase 3 complete: ${d.clips_done} clips, ${d.keypoints_written} KP rows, elapsed ${d.elapsed_s}s`, "system", "info");
  fetchPoseResults();
}

// ── Lamp & Viewer Polling ─────────────────────────────────────────────────
function startLampPolling() {
  if (S.poseLampPollInterval) clearInterval(S.poseLampPollInterval);
  S.poseLampPollInterval = setInterval(async () => {
    try {
      const d = await api("/api/pose/status");
      if (!d.running) {
        clearInterval(S.poseLampPollInterval); S.poseLampPollInterval = null;
        if (S.poseRunning) donePose(d);
        return;
      }
      showPoseProgress({
        current_clip: d.current_clip,
        clips_done:   d.clips_done,
        clips_total:  d.clips_total,
        progress_pct: d.progress_pct,
        pct:          d.progress_pct,
        lamp_state:   d.lamp_state,
        active_camera:d.active_camera,
        keypoints_written: d.keypoints_written,
        adl_events:   d.adl_events,
      });
    } catch {}
  }, 1000);
}

function startViewerPolling() {
  if (S.poseViewerInterval) clearInterval(S.poseViewerInterval);
  const imgO = q("#pv-orig"), phO = q("#pv-orig-ph");
  const imgP = q("#pv-proc"), phP = q("#pv-proc-ph");

  function loadSnapshot(imgEl, phEl, url) {
    if (!imgEl) return;
    const tmp = new Image();
    tmp.onload = () => {
      imgEl.src = url;
      imgEl.style.opacity = 1;
      if (phEl) phEl.style.display = "none";
    };
    tmp.onerror = () => {
      imgEl.style.opacity = 0;
      if (phEl) phEl.style.display = "";
    };
    tmp.src = url;
  }

  S.poseViewerInterval = setInterval(() => {
    if (!S.poseRunning) return;
    const t = Date.now();
    loadSnapshot(imgO, phO, `/api/pose/snapshot/original?t=${t}`);
    loadSnapshot(imgP, phP, `/api/pose/snapshot/processed?t=${t}`);
  }, 200);
}

function stopPolling() {
  if (S.poseLampPollInterval) { clearInterval(S.poseLampPollInterval); S.poseLampPollInterval = null; }
  if (S.poseViewerInterval)   { clearInterval(S.poseViewerInterval);   S.poseViewerInterval   = null; }
}

// ── Pose status fetch ──────────────────────────────────────────────────────
async function fetchPoseStatus() {
  try {
    const d = await api("/api/pose/status");
    S.poseRunning    = !!d.running;
    S.poseClipsTotal  = d.clips_total || 0;
    S.poseClipsDone   = d.clips_done  || 0;
    syncPoseLampState(d);

    const clips = q("#pose-stat-clips"); if (clips) clips.textContent = `${d.clips_done||0}/${d.clips_total||0}`;
    const kps   = q("#pose-stat-kps");   if (kps)   kps.textContent   = d.keypoints_written || 0;
    const adl   = q("#pose-stat-adl");   if (adl)   adl.textContent   = d.adl_events || 0;
    const clip  = q("#pose-top-clip");   if (clip)  clip.textContent  = d.current_clip || "—";
    const pct   = q("#pose-pct-badge");  if (pct)   pct.textContent   = (d.progress_pct||0) + "%";
    const bar   = q("#pose-bar");        if (bar)   bar.style.width   = (d.progress_pct||0) + "%";

    const topStatus = q("#pose-top-status");
    if (d.running) {
      if (topStatus) { topStatus.textContent = "Running"; topStatus.className = "badge badge-online"; }
      const btn  = q("#btn-start-pose"); if (btn)  btn.disabled  = true;
      const stop = q("#btn-stop-pose");  if (stop) stop.disabled = false;
      add(q("#dot-pose"), "online");
      const lbl = q("#lbl-pose"); if (lbl) lbl.textContent = "Running";
      if (!S.poseLampPollInterval) startLampPolling();
      if (!S.poseViewerInterval)   startViewerPolling();
    } else if (d.clips_done > 0) {
      if (topStatus) { topStatus.textContent = "Complete"; topStatus.className = "badge badge-online"; }
      const lbl = q("#lbl-pose"); if (lbl) lbl.textContent = "Complete";
    } else {
      if (topStatus) { topStatus.textContent = "Idle"; topStatus.className = "badge badge-idle"; }
      const lbl = q("#lbl-pose"); if (lbl) lbl.textContent = "Idle";
    }
  } catch (e) {
    addLog("fetchPoseStatus: " + e.message, "system", "err");
  }
}

// ── Results tab ───────────────────────────────────────────────────────────
async function fetchPoseResults() {
  const grid = q("#pose-results-grid");
  if (!grid) return;
  try {
    const d = await api("/api/pose/results");
    if (!d.results || !d.results.length) {
      grid.innerHTML = `<div class="empty"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="36" height="36"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg><div class="empty-title">No outputs yet</div><div class="empty-sub">Run Phase 3 processing to generate results</div></div>`;
      return;
    }
    grid.innerHTML = d.results.map(r => `
      <div class="result-card">
        <div class="result-card-head flex-row" style="justify-content:space-between; padding-bottom:8px">
          <strong class="fs-12" style="word-break:break-all">${esc(r.clip_stem)}</strong>
          <div class="flex-row" style="gap:4px; flex-shrink:0">
            ${r.preview_exists && !r.mp4_exists ? `<button class="btn btn-primary btn-xs" onclick="savePoseResult('${esc(r.clip_stem)}')">💾 Save</button>` : ""}
            ${r.mp4_exists ? `<span class="badge badge-online" title="Permanently saved">Saved</span>` : ""}
            ${r.preview_exists || r.mp4_exists ? `<button class="btn btn-ghost btn-xs" onclick="openPreview('${esc(r.clip_stem)}',${!!r.mp4_exists})">▶ Preview</button>` : ""}
          </div>
        </div>
        <div class="result-card-body">
          <div class="result-stat"><span>Keypoint rows</span><strong>${r.keypoints_count}</strong></div>
          <div class="result-stat"><span>ADL events</span><strong>${r.adl_events}</strong></div>
          <div class="adl-chips mt-2">${renderAdlChips(r.adl_summary)}</div>
        </div>
      </div>`).join("");
  } catch (e) {
    grid.innerHTML = `<div class="empty"><div class="empty-sub">Error: ${esc(e.message)}</div><button class="btn btn-ghost mt-3" onclick="fetchPoseResults()">Retry</button></div>`;
  }
}

function renderAdlChips(summary) {
  const entries = Object.entries(summary || {});
  if (!entries.length) return `<span class="adl-chip">No ADL labels</span>`;
  return entries.map(([k,v]) => `<span class="adl-chip">${esc(k)} ${v}%</span>`).join("");
}

window.savePoseResult = async function(clipStem) {
  try {
    await api("/api/pose/save_result", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({clip_stem:clipStem}),
    });
    addLog(`Saved result: ${clipStem}`, "system", "info");
    fetchPoseResults();
  } catch(e) {
    addLog(`Save failed: ${e.message}`, "system", "err");
  }
};

window.openPreview = function(clipStem, isSaved) {
  const modal = q("#modal-preview");
  const video = q("#preview-video");
  const title = q("#preview-title");
  if (!modal || !video) return;
  video.src = isSaved
    ? `/api/video/output_process/${clipStem}/${clipStem}_processed.mp4`
    : `/api/video/output_pose/${clipStem}/${clipStem}_preview.mp4`;
  if (title) title.textContent = clipStem;
  show(modal);
};

window.closePreviewModal = function() {
  const modal = q("#modal-preview");
  const video = q("#preview-video");
  if (video) { video.pause(); video.src = ""; }
  hide(modal);
};

// ── Probe modal close ─────────────────────────────────────────────────────
on("#modal-close", "click", () => hide(q("#modal-probe")));

// ── Event log ────────────────────────────────────────────────────────────
function addLog(msg, camId, type = "") {
  const log = q("#log-wrap");
  if (!log) return;
  const empty = log.querySelector(".empty"); if (empty) empty.remove();
  const el = document.createElement("div");
  el.className = "log-entry " + type;
  const t = new Date().toLocaleTimeString("en-GB", {hour12:false});
  const camStr = camId && camId !== "system" ? `<span class="log-cam">${esc(camId)}</span>` : "";
  el.innerHTML = `<span class="log-t">${t}</span>${camStr}<span class="log-msg">${esc(msg)}</span>`;
  log.prepend(el);
  if (log.children.length > 200) log.removeChild(log.lastElementChild);
}

// ── Camera stream stubs (used internally if Phase 1 re-enabled) ───────────
function stopAllStreams() {
  Object.keys(S.streamIntervals).forEach(id => {
    clearInterval(S.streamIntervals[id]);
    delete S.streamIntervals[id];
  });
}

function updateOnlineCount() {}  // no-op, kpi elements removed

// ── Helper: esc is already defined, these stubs prevent older socket callbacks crashing ──
function setGlobalStatus(s) {
  const el = q("#global-status"); if (!el) return;
  el.className = "badge";
  const map = {idle:"badge-idle", recording:"badge-recording"};
  add(el, map[s] || "badge-idle");
  el.textContent = {idle:"Idle", recording:"Recording"}[s] || s;
}

// ── Init ──────────────────────────────────────────────────────────────────
(async function init() {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }

  async function boot() {
    initTabNav();
    S.poseLampState = defaultLampState();
    renderLamps();
    renderQueue();
    await fetchStorageInfo();
    await fetchPoseStatus();
    // Switch to workspace as default
    switchTab("workspace");
  }
})();
