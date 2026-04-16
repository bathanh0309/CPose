const $ = (id) => document.getElementById(id);

// Socket.IO client (will be initialized after socket script loads)
let socket = null;
let _pending_register_clip = null;

function initSocket() {
  if (!window.io) return;
  socket = io();

  socket.on("connect", () => {
    console.info("Socket connected to server");
  });

  socket.on("pose_progress", (msg) => {
    setText(refs.mFrame, `${msg.frame_id} / ${msg.total_frames}`);
    setText(refs.mFps, typeof msg.fps === "number" ? msg.fps.toFixed(1) : msg.fps);
    setText(refs.mAdl, msg.adl_events || 0);

    const progressContainer = document.getElementById("pipelineProgressContainer");
    const progressText = document.getElementById("pipelineProgressText");
    const progressBar = document.getElementById("pipelineProgressBar");

    if (progressContainer) progressContainer.classList.remove("hidden");
    if (progressText) progressText.textContent = `${msg.pct || 0}%`;
    if (progressBar) progressBar.style.width = `${msg.pct || 0}%`;
  });

  socket.on("pose_lamp_state", (msg) => {
    const lampsState = msg.lamp_state || {};
    uniqueCams.forEach(cam => {
      setLamp(cam, (lampsState[cam] || "idle").toLowerCase());
    });
  });

  socket.on("pose_complete", (msg) => {
     state.running = false;
     setWorkspaceState("READY", "ready");
     setText(refs.mStatus, "Ready");
     if (state.poller) {
         clearInterval(state.poller);
         state.poller = null;
     }

     const progressContainer = document.getElementById("pipelineProgressContainer");
     if (progressContainer) progressContainer.classList.add("hidden");

     toast("Backend xử lý hoàn tất!", "success");
  });

  socket.on("register_face_request", (msg) => {
    _pending_register_clip = msg.clip_stem || null;
    toast("Hệ thống nghẽn chờ đăng ký khuôn mặt!", "warning");
    const registerStatus = document.getElementById("registerStatus");
    if (registerStatus) registerStatus.classList.remove("hidden");
    if (refs.personName) refs.personName.focus();
  });
}

const refs = {
  startBtn: $("startBtn"),
  stopBtn: $("stopBtn"),
  refreshBtn: $("refreshBtn"),
  clearBtn: $("clearBtn"),
  exportBtn: $("exportBtn"),
  searchInput: $("searchInput"),
  multicamInput: $("multicamInput"),
  rawInput: $("rawInput"),
  openOriginal: $("openOriginal"),
  closeOriginal: $("closeOriginal"),
  openProcessed: $("openProcessed"),
  closeProcessed: $("closeProcessed"),
  toggleWebcam: null, // deleted from HTML
  originalShell: $("originalShell"),
  processedShell: $("processedShell"),
  webcamShell: $("webcamShell"),
  originalEmpty: $("originalEmpty"),
  processedEmpty: $("processedEmpty"),
  webcamEmpty: $("webcamEmpty"),
  originalVideo: $("originalVideo"),
  processedVideo: $("processedVideo"),
  liveWebcam: $("liveWebcam"),
  originalMeta: $("originalMeta"),
  processedMeta: $("processedMeta"),
  webcamMeta: $("webcamMeta"),
  workspaceState: $("workspaceState"),
  mStatus: $("mStatus"),
  mFps: $("mFps"),
  mFrame: $("mFrame"),
  mAdl: $("mAdl"),
  mOriginalInfo: $("mOriginalInfo"),
  mProcessedInfo: $("mProcessedInfo"),
  logTable: $("logTable"),
  toasts: $("toasts"),
  overlayCam: $("overlayCam"),
  overlayFps: $("overlayFps"),
  overlayFrame: $("overlayFrame"),
  overlayAdl: $("overlayAdl"),
  pendingList: $("pendingList"),
  clipQueue: $("clipQueue"),
  previewModal: $("previewModal"),
  previewVideo: $("previewVideo"),
  closePreviewBtn: $("closePreviewBtn"),
  modalOverlay: $("modalOverlay"),
  modalTitle: $("modalTitle"),
  openWebcamBtn: $("openWebcamBtn"),
  closeWebcamBtn: $("closeWebcamBtn"),
  personName: $("personName"),
  personAge: $("personAge"),
  personId: $("personId"),
  registerBtn: $("registerBtn"),
};

let lamps = {};
let uniqueCams = ["cam01", "cam02", "cam03", "cam04"];

const state = {
  clips: [],
  logs: [],
  pendingResults: {},
  activeClipId: null,
  running: false,
  stopRequested: false,
  originalVisible: false,
  processedVisible: false,
  webcamVisible: false,
  webcamReady: false,
  objectUrls: [],
  lampState: {},
};

const VIDEO_EXTENSIONS = [".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"];

function toast(message, kind = "info") {
  if (!refs.toasts) return;
  const node = document.createElement("div");
  node.className = `toast ${kind}`;
  node.textContent = message;
  refs.toasts.appendChild(node);
  setTimeout(() => node.remove(), 2600);
}

function safeText(value) {
  return String(value ?? "");
}

/** Null-safe textContent setter — no-op if element doesn't exist in DOM. */
function setText(el, val) {
  if (el) el.textContent = String(val ?? "");
}

function escapeHtml(value) {
  return safeText(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function timestamp() {
  const d = new Date();
  // Only show time (HH:MM:SS) per user request — date not required
  return [
    String(d.getHours()).padStart(2, "0"),
    String(d.getMinutes()).padStart(2, "0"),
    String(d.getSeconds()).padStart(2, "0"),
  ].join(":");
}

function parseCamera(name) {
  const match = safeText(name).match(/cam[_-]?(\d{1,2})/i);
  if (!match) return "--";
  return `cam${String(Number(match[1])).padStart(2, "0")}`;
}

function parseTimestampFromName(name) {
  const match = safeText(name).match(
    /(\d{4})[-_](\d{2})[-_](\d{2})[_-](\d{2})[-_](\d{2})[-_](\d{2})/,
  );
  if (!match) return null;
  const [, year, month, day, hour, minute, second] = match;
  const ts = new Date(`${year}-${month}-${day}T${hour}:${minute}:${second}`).getTime();
  return Number.isFinite(ts) ? ts : null;
}

function normalizeTime(value) {
  return Number.isFinite(value) ? value : Number.MAX_SAFE_INTEGER;
}

function compareByTimeAndName(a, b) {
  const t1 = normalizeTime(a.captureTime);
  const t2 = normalizeTime(b.captureTime);
  if (t1 !== t2) return t1 - t2;
  return safeText(a.sortName).localeCompare(safeText(b.sortName), "vi");
}

function cameraPriority(cam) {
  if (cam === "cam01") return 0;
  if (cam === "cam02") return 1;
  const value = Number((safeText(cam).match(/\d+/) || [])[0]);
  return Number.isFinite(value) ? 100 + value : 999;
}

function sortedQueue(clips) {
  const groups = new Map();
  clips.forEach((clip) => {
    const key = cameraPriority(clip.cam);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(clip);
  });

  const priorities = Array.from(groups.keys()).sort((a, b) => a - b);
  const result = [];

  for (const priority of priorities) {
    const group = groups.get(priority).slice().sort(compareByTimeAndName);
    if (!group.length) continue;

    let current = group.shift();
    result.push(current);

    while (group.length) {
      let bestIndex = 0;
      let bestDistance = Number.POSITIVE_INFINITY;

      for (let i = 0; i < group.length; i += 1) {
        const candidate = group[i];
        const distance = Math.abs(normalizeTime(candidate.captureTime) - normalizeTime(current.captureTime));

        if (distance < bestDistance) {
          bestDistance = distance;
          bestIndex = i;
          continue;
        }

        if (distance === bestDistance && compareByTimeAndName(candidate, group[bestIndex]) < 0) {
          bestIndex = i;
        }
      }

      current = group.splice(bestIndex, 1)[0];
      result.push(current);
    }
  }

  return result;
}

function isVideoFile(file) {
  if (!file) return false;
  if (file.type && file.type.startsWith("video/")) return true;
  const lower = safeText(file.name).toLowerCase();
  return VIDEO_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

function logEvent(id, event, conf = "--", cam = "--") {
  state.logs.unshift({
    time: timestamp(),
    id: safeText(id),
    event: safeText(event),
    conf: safeText(conf),
    cam: safeText(cam),
  });
  renderLogs();
}

function renderLogs() {
  if (!refs.logTable) return;
  const query = safeText(refs.searchInput?.value).trim().toLowerCase();
  const rows = state.logs.filter((row) => {
    if (!query) return true;
    return [row.time, row.event, row.cam].join(" ").toLowerCase().includes(query);
  });

  if (!rows.length) {
    refs.logTable.innerHTML = `
      <tr>
        <td colspan="3" style="text-align:center;padding:12px;color:#999">No logs yet</td>
      </tr>
    `;
    return;
  }

  refs.logTable.innerHTML = rows.map((row) => `
    <tr>
      <td title="${escapeHtml(row.time)}">${escapeHtml(row.time)}</td>
      <!-- display only first word for compact event column; full event text in title -->
      <td title="${escapeHtml(row.event)}">${escapeHtml((row.event || "").split(/\s+/)[0] || row.event)}</td>
      <td title="${escapeHtml(row.cam)}">${escapeHtml(row.cam)}</td>
    </tr>
  `).join("");
}

function renderClipQueue() {
  if (!refs.clipQueue) return;

  const registerStatus = document.getElementById("registerStatus");
  if (registerStatus) {
    const hasCam1 = state.clips.some(c => (/cam0?1/i.test(c.cam) || /cam0?1/i.test(c.name)));
    registerStatus.classList.toggle("hidden", !hasCam1);
  }

  if (!state.clips.length) {
    refs.clipQueue.innerHTML = `
      <div class="queue-empty">Chưa có video nào được tải</div>
    `;
    return;
  }

  refs.clipQueue.innerHTML = state.clips.map((clip) => {
    const fpsText = clip.fps ? clip.fps.toFixed(1) : "0.0";
    const frameText = `${clip.currentFrame} / ${clip.totalFrames}`;
    const progress = Math.max(0, Math.min(100, Math.round((clip.progress || 0) * 100)));
    const statusLabel = clip.status === "running" ? "Đang xử lý" : clip.status === "done" ? "Hoàn tất" : "Sẵn sàng";
    const sourceLabel = clip.source === "multicam" ? "Multicam" : "Raw";

    return `
      <div class="queue-card ${clip.status}">
        <details>
          <summary style="cursor: pointer; outline: none; margin-bottom: 4px;">
            <div class="queue-card-head" style="display: inline-flex; width: calc(100% - 20px); vertical-align: top;">
              <div class="queue-card-title" title="${escapeHtml(clip.name)}">${escapeHtml(clip.name)}</div>
              <span class="queue-status">${statusLabel}</span>
            </div>
          </summary>
          <div class="queue-card-sub">${escapeHtml(clip.cam)} • ${escapeHtml(sourceLabel)}</div>
          <div class="queue-grid" style="margin-top: 6px;">
            <div><strong>FPS</strong><span>${escapeHtml(fpsText)}</span></div>
            <div><strong>Frame</strong><span>${escapeHtml(frameText)}</span></div>
            <div><strong>ADL</strong><span>${escapeHtml(clip.adlEvents)}</span></div>
            <div><strong>KP</strong><span>${escapeHtml(clip.keypoints)}</span></div>
          </div>
          <div class="queue-actions" style="margin-top: 6px;">
            <button class="view-clip-btn" data-id="${clip.id}" style="width: 100%; padding: 4px; border-radius: 4px; background: #e2e8f0; border: none; font-size: 10px; cursor: pointer;">View Video</button>
          </div>
        </details>
        <div class="queue-progress"><span style="width:${progress}%"></span></div>
      </div>
    `;
  }).join("");

  refs.clipQueue.querySelectorAll(".view-clip-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const clipId = btn.dataset.id;
      const clip = state.clips.find(c => c.id === clipId);
      if (!clip) return;
      
      const res = Object.values(state.pendingResults).find(r => r.clip_stem === clip.sortName || r.clip_stem === clip.name);
      const url = clip.processedUrl || (res ? res.preview_video : clip.originalUrl);
      
      if (refs.previewVideoLocal) {
        refs.previewVideoLocal.src = url;
        refs.previewVideoLocal.play().catch(() => {});
        if (refs.previewVideoEmpty) refs.previewVideoEmpty.style.display = "none";
      }
    });
  });
}

function createClip(file, source) {
  const url = URL.createObjectURL(file);
  state.objectUrls.push(url);

  return {
    id: crypto.randomUUID(),
    file,
    source,
    name: file.name,
    sortName: safeText(file.name).toLowerCase(),
    cam: parseCamera(file.name),
    captureTime: parseTimestampFromName(file.name) ?? normalizeTime(file.lastModified),
    status: "ready",
    originalUrl: url,
    processedUrl: null,
    currentFrame: 0,
    totalFrames: 0,
    fps: 0,
    keypoints: 0,
    adlEvents: 0,
    latency: 0,
    tracks: 0,
    globalIds: 0,
    progress: 0,
  };
}

function addFiles(files, source) {
  const picked = Array.from(files || []);
  if (!picked.length) return;

  const videos = picked.filter(isVideoFile);
  const skipped = picked.length - videos.length;

  videos.forEach((file) => {
    const clip = createClip(file, source);
    state.clips.push(clip);
    logEvent("QUEUE", `Loaded ${source}: ${file.name}`, "0.99", clip.cam);
  });

  if (skipped > 0) {
    logEvent("QUEUE", `Skipped ${skipped} non-video files`, "0.80", "--");
    toast(`Bỏ qua ${skipped} tệp không phải video`, "warning");
  }

  if (videos.length) {
    toast(`Đã thêm ${videos.length} video`, "success");
    const loadedCams = new Set(state.clips.map(c => c.cam).filter(Boolean));
    const newCams = Array.from(loadedCams).sort();
    if (newCams.length > 0 && JSON.stringify(newCams) !== JSON.stringify(uniqueCams)) {
      uniqueCams = newCams;
      renderDynamicCameras();
    }
  }
  updateSummary();
}

function renderDynamicCameras() {
  const tbody = $("summaryTableBody");
  if (tbody) {
    tbody.innerHTML = uniqueCams.map(cam => `
      <tr>
        <td>
          <div style="display:flex; justify-content:center; align-items:center;">
             <div class="lamp ${state.lampState[cam] || 'idle'}" id="lamp_${cam}" style="width:24px; height:24px; border-width: 1px;"></div>
          </div>
        </td>
        <td>${cam}</td>
        <td id="${cam}_gid">--</td>
        <td id="${cam}_conf">--</td>
        <td id="${cam}_fps">--</td>
        <td id="${cam}_frame">--</td>
        <td id="${cam}_action">--</td>
      </tr>
    `).join("");
  }
  
  uniqueCams.forEach(cam => {
    lamps[cam] = $(`lamp_${cam}`);
    if (!state.lampState[cam]) state.lampState[cam] = "idle";
  });
}

function setLamp(cam, lampState) {
  if (!lamps[cam]) return;
  state.lampState[cam] = lampState;
  lamps[cam].className = `lamp ${lampState}`;
}

function setWorkspaceState(label, kind) {
  if (!refs.workspaceState) return;
  setText(refs.workspaceState, label);
  refs.workspaceState.className = `status-pill ${kind}`;
}

function showPreviewModal(videoUrl, title) {
  refs.previewVideo.src = videoUrl;
  refs.modalTitle.textContent = title || "Preview";
  refs.previewModal.classList.remove("hidden");
}

function hidePreviewModal() {
  refs.previewModal.classList.add("hidden");
  refs.previewVideo.pause();
  refs.previewVideo.src = "";
}

function renderPendingResults() {
  const items = Object.values(state.pendingResults);
  if (!items.length) {
    refs.pendingList.innerHTML = `<div style="font-size:11px;color:rgba(255,255,255,.8);text-align:center;padding:16px;">Không có kết quả chưa lưu</div>`;
    return;
  }

  refs.pendingList.innerHTML = items.map((result) => {
    const meta = result.metadata || {};
    const stemKey = encodeURIComponent(result.clip_stem);
    return `
      <div class="result-card">
        <details>
          <summary style="cursor: pointer; outline: none;">
            <div class="result-name" title="${escapeHtml(result.clip_stem)}" style="display:inline-block; vertical-align: top; width: calc(100% - 20px);">${escapeHtml(result.clip_stem)}</div>
          </summary>
          <div class="result-buttons">
            <button class="preview-btn" data-stem="${stemKey}">View</button>
            <button class="save-btn" data-stem="${stemKey}">Save</button>
          </div>
          <div class="result-meta">
            <div class="result-meta-row"><span class="result-meta-label">ID:</span><span class="result-meta-value">${escapeHtml(meta.id || "--")}</span></div>
            <div class="result-meta-row"><span class="result-meta-label">CAM:</span><span class="result-meta-value">${escapeHtml(meta.cam || "--")}</span></div>
            <div class="result-meta-row"><span class="result-meta-label">FPS:</span><span class="result-meta-value">${escapeHtml(meta.fps || "--")}</span></div>
            <div class="result-meta-row"><span class="result-meta-label">FRAME:</span><span class="result-meta-value">${escapeHtml(meta.frame || "--")}</span></div>
            <div class="result-meta-row"><span class="result-meta-label">KP:</span><span class="result-meta-value">${escapeHtml(meta.keypoints || "--")}</span></div>
            <div class="result-meta-row"><span class="result-meta-label">ADL:</span><span class="result-meta-value">${escapeHtml(meta.adl || "--")}</span></div>
            <div class="result-meta-row"><span class="result-meta-label">LATENCY:</span><span class="result-meta-value">${escapeHtml(meta.latency || "--")}</span></div>
            <div class="result-meta-row"><span class="result-meta-label">STATE:</span><span class="result-meta-value">${escapeHtml(meta.state || "--")}</span></div>
          </div>
        </details>
      </div>
    `;
  }).join("");

  refs.pendingList.querySelectorAll(".preview-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const stem = decodeURIComponent(btn.dataset.stem || "");
      const result = state.pendingResults[stem];
      
      if (result?.preview_video) {
        const url = result.preview_video;
        if (refs.processedVideo) {
          refs.processedVideo.src = url;
          refs.processedVideo.play().catch(() => {});
          openViewer("processed");
        }
        if (refs.previewVideoLocal) {
          refs.previewVideoLocal.src = url;
          refs.previewVideoLocal.play().catch(() => {});
          if (refs.previewVideoEmpty) refs.previewVideoEmpty.style.display = "none";
        }
      }
      
      const clip = state.clips.find(c => c.sortName === stem || c.name === stem);
      if (clip && refs.originalVideo) {
        refs.originalVideo.src = clip.originalUrl;
        refs.originalVideo.play().catch(() => {});
        openViewer("original");
      }
    });
  });

  refs.pendingList.querySelectorAll(".save-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const stem = decodeURIComponent(btn.dataset.stem || "");
      toast(`Lưu ${stem} thành công`, "success");
      delete state.pendingResults[stem];
      renderPendingResults();
    });
  });
}

function showViewer(type, visible) {
  if (type === "webcam") {
    const showVideo = visible && state.webcamReady;
    if (refs.webcamShell) refs.webcamShell.classList.toggle("hidden", !showVideo);
    if (refs.webcamEmpty) refs.webcamEmpty.classList.toggle("hidden", showVideo);
    return;
  }

  const isOriginal = type === "original";
  const shell = isOriginal ? refs.originalShell : refs.processedShell;
  const empty = isOriginal ? refs.originalEmpty : refs.processedEmpty;
  // Toggle shell and empty state visibility based on visible flag
  if (shell) shell.classList.toggle("hidden", !visible);
  if (empty) empty.classList.toggle("hidden", visible);
}

function openViewer(type) {
  if (type === "original") {
    if (!refs.originalVideo.getAttribute("src")) {
      if (state.clips.length > 0) {
        refs.originalVideo.src = state.clips[0].originalUrl;
        refs.originalVideo.play().catch(() => {});
      } else {
        toast("Vui lòng tải video trước", "info");
        return;
      }
    }
    state.originalVisible = true;
    showViewer("original", true);
    logEvent("VIEW", "Original viewer opened", "0.97", "--");
  } else if (type === "processed") {
    if (!refs.processedVideo.getAttribute("src")) {
      const firstProc = Object.values(state.pendingResults)[0];
      if (firstProc && firstProc.preview_video) {
        refs.processedVideo.src = firstProc.preview_video;
        refs.processedVideo.play().catch(() => {});
      } else {
        toast("Chưa có video xử lý để xem", "info");
        return;
      }
    }
    state.processedVisible = true;
    showViewer("processed", true);
    logEvent("VIEW", "Processed viewer opened", "0.97", "--");
  }
}

function closeViewer(type) {
  if (type === "original") {
    state.originalVisible = false;
    showViewer("original", false);
    logEvent("VIEW", "Original viewer closed", "0.97", "--");
  } else if (type === "processed") {
    state.processedVisible = false;
    showViewer("processed", false);
    logEvent("VIEW", "Processed viewer closed", "0.97", "--");
  }
}

function updateSummary() {
  const clip = state.clips.find((item) => item.id === state.activeClipId);

  if (!clip) {
    setText(refs.mStatus, state.running ? "Running" : "Ready");
    setText(refs.mFps, "0.0");
    setText(refs.mFrame, "0 / 0");
    setText(refs.mAdl, "0");
    // reset per-camera summary cells
    uniqueCams.forEach((c) => {
      const elGid = $(`${c}_gid`), elConf = $(`${c}_conf`), elFps = $(`${c}_fps`), elFrame = $(`${c}_frame`), elAction = $(`${c}_action`);
      if (elGid) elGid.textContent = "--";
      if (elConf) elConf.textContent = "--";
      if (elFps) elFps.textContent = "--";
      if (elFrame) elFrame.textContent = "--";
      if (elAction) elAction.textContent = "--";
    });
    setText(refs.originalMeta, "Waiting for start");
    setText(refs.processedMeta, "Waiting for start");
    setText(refs.overlayCam, "--");
    setText(refs.overlayFps, "FPS 0.0");
    setText(refs.overlayFrame, "0 / 0");
    setText(refs.overlayAdl, "ADL --");
    setText(refs.webcamMeta, state.webcamReady ? "Webcam đã mở" : "Chưa mở webcam");
    renderClipQueue();
    return;
  }

  const fpsText = clip.fps ? clip.fps.toFixed(1) : "0.0";
  const frameText = `${clip.currentFrame} / ${clip.totalFrames}`;
  const processedState = clip.status === "done" ? "Đã xử lý" : "Đang xử lý";
  const statusText = clip.status === "done" ? "Done" : clip.status === "running" ? "Running" : "Ready";

  setText(refs.mStatus, statusText);
  setText(refs.mFps, fpsText);
  setText(refs.mFrame, frameText);
  setText(refs.mAdl, `${clip.adlEvents}`);
  
  uniqueCams.forEach((c) => {
      const elGid = $(`${c}_gid`), elConf = $(`${c}_conf`), elFps = $(`${c}_fps`), elFrame = $(`${c}_frame`), elAction = $(`${c}_action`);
      if (elGid) elGid.textContent = "--";
      if (elConf) elConf.textContent = "--";
      if (elFps) elFps.textContent = "--";
      if (elFrame) elFrame.textContent = "--";
      if (elAction) elAction.textContent = "--";
  });

  const camKey = clip.cam || "cam01";
  const elGid = $(`${camKey}_gid`), elConf = $(`${camKey}_conf`), elFps = $(`${camKey}_fps`), elFrame = $(`${camKey}_frame`), elAction = $(`${camKey}_action`);
  if (elGid) elGid.textContent = (clip.status === "running" ? "unk" : "--");
  if (elConf) elConf.textContent = (clip.status === "running" ? "0.98" : "--");
  if (elFps) elFps.textContent = fpsText;
  if (elFrame) elFrame.textContent = frameText;
  if (elAction) elAction.textContent = (clip.status === "running" ? "Processing" : "--");

  setText(refs.originalMeta, `${clip.cam} • chưa xử lý`);
  setText(refs.processedMeta, `${clip.cam} • ${processedState.toLowerCase()}`);
  setText(refs.overlayCam, clip.cam);
  setText(refs.overlayFps, `FPS ${fpsText}`);
  setText(refs.overlayFrame, frameText);
  setText(refs.overlayAdl, `ADL ${clip.adlEvents}`);
  renderClipQueue();
}

async function startProcessing() {
  if (state.running) {
    toast("Pipeline đang chạy", "warning");
    return;
  }

  try {
    const payload = { folder: "data/multicam", save_overlay: true };
    const res = await fetch("/api/pose/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.status === "started") {
      state.running = true;
      setWorkspaceState("RUNNING", "running");
      setText(refs.mStatus, "Running");
      toast("Đã bắt đầu xử lý backend", "success");
      
      const registerStatus = document.getElementById("registerStatus");
      if (registerStatus) registerStatus.classList.remove("hidden");

      const progressContainer = document.getElementById("pipelineProgressContainer");
      if (progressContainer) progressContainer.classList.remove("hidden");

      if (!state.poller) {
        state.poller = setInterval(() => {
          if (state.running) {
            refs.originalVideo.src = `/api/pose/snapshot/original?t=${Date.now()}`;
            refs.processedVideo.src = `/api/pose/snapshot/processed?t=${Date.now()}`;
          }
        }, 120);
      }
    } else {
      toast(data.error || "Lỗi khởi động", "error");
    }
  } catch (error) {
    console.error("Lỗi:", error);
    toast("Lỗi kết nối backend", "error");
  }
}

async function stopProcessing() {
  if (!state.running) return;
  try {
    await fetch("/api/pose/stop", { method: "POST" });
    toast("Đã gửi yêu cầu dừng backend", "info");
  } catch (error) {
    console.error(error);
  }
  
  state.running = false;
  setWorkspaceState("READY", "ready");
  setText(refs.mStatus, "Ready");
  
  if (state.poller) {
    clearInterval(state.poller);
    state.poller = null;
  }

  const progressContainer = document.getElementById("pipelineProgressContainer");
  if (progressContainer) progressContainer.classList.add("hidden");
}

async function refreshProcessing() {
  stopProcessing();
  await new Promise(resolve => setTimeout(resolve, 300));
  
  if (state.clips.length === 0) {
    toast("Không có video để làm mới", "info");
    return;
  }
  
  state.clips.forEach(clip => {
    clip.status = "ready";
    clip.progress = 0;
    clip.currentFrame = 0;
    clip.keypoints = 0;
    clip.adlEvents = 0;
    clip.processedUrl = null;
    setLamp(clip.cam, "idle");
  });
  
  state.pendingResults = {};
  renderPendingResults();
  renderClipQueue();
  updateSummary();
  
  toast("Đang xử lý lại tất cả video...", "info");
  logEvent("SYS", "Restarted all processing", "1.00", "--");
  startProcessing();
}

function clearLogsOnly() {
  state.logs = [];
  renderLogs();
  toast("Đã xóa log", "success");
}

function exportCsv() {
  const rows = [["time", "event", "cam"]];
  state.logs.forEach((row) => rows.push([row.time, row.event, row.cam]));
  const csv = rows.map((row) => row.map((cell) => `"${safeText(cell).replace(/"/g, "\"\"")}"`).join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "cpose_logs.csv";
  link.click();
  URL.revokeObjectURL(url);
  logEvent("SYS", "Export CSV", "1.00", "--");
}

async function openWebcam() {
  toast("Đang mở webcam...", "info");
  if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
    toast("Trình duyệt không hỗ trợ webcam", "error");
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 360 } });
    const oldStream = refs.liveWebcam.srcObject;
    if (oldStream) oldStream.getTracks().forEach((track) => track.stop());

    refs.liveWebcam.srcObject = stream;
    state.webcamReady = true;
    state.webcamVisible = true;
    refs.liveWebcam.play().catch(e => console.error("Webcam play auto-rejected:", e));
    showViewer("webcam", true);
    setText(refs.webcamMeta, "Webcam đang phát");
    toast("Mở webcam thành công", "success");
  } catch (error) {
    console.error("Lỗi truy cập webcam:", error);
    setText(refs.webcamMeta, "Không thể mở webcam");
    toast("Không thể truy cập webcam", "error");
  }
}

function closeWebcam() {
  const stream = refs.liveWebcam && refs.liveWebcam.srcObject;
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    refs.liveWebcam.srcObject = null;
  }
  state.webcamReady = false;
  state.webcamVisible = false;
  showViewer("webcam", false);
  setText(refs.webcamMeta, "Webcam đã đóng");
  toast("Đã tắt webcam", "info");
}

function registerSubject() {
  const name = refs.personName.value.trim();
  const age = refs.personAge.value.trim();
  const id = refs.personId.value.trim();
  if (!name || !age || !id) {
    toast("Vui lòng nhập đầy đủ thông tin", "warning");
    return;
  }

  toast(`Đăng ký person: ${name} (${age} tuổi) - ID: ${id}`, "success");
  
  const payload = {
    clip_stem: _pending_register_clip || "",
    name: name,
    age: age,
    person_id: id,
  };
  
  if (socket && socket.connected) {
    socket.emit("register_face_done", payload);
  }

  logEvent(id, `Registered: ${name}`, "0.99", "--");
  
  const camKey = "cam01";
  const elGid = $(`${camKey}_gid`), elConf = $(`${camKey}_conf`), elAction = $(`${camKey}_action`);
  if (elGid) elGid.textContent = id;
  if (elConf) elConf.textContent = "0.99";
  if (elAction) elAction.textContent = "Registered";

  refs.personName.value = "";
  refs.personAge.value = "";
  refs.personId.value = "";
  
  const registerStatus = document.getElementById("registerStatus");
  if (registerStatus) registerStatus.classList.add("hidden");
  _pending_register_clip = null;
}

function boot() {
  // initialize Socket.IO after DOM ready
  initSocket();
  refs.startBtn.addEventListener("click", startProcessing);
  refs.stopBtn.addEventListener("click", stopProcessing);
  refs.refreshBtn.addEventListener("click", refreshProcessing);
  refs.clearBtn.addEventListener("click", clearLogsOnly);
  refs.exportBtn.addEventListener("click", exportCsv);

  refs.multicamInput.addEventListener("change", (event) => {
    addFiles(event.target.files, "multicam");
    event.target.value = "";
  });

  refs.rawInput.addEventListener("change", (event) => {
    addFiles(event.target.files, "raw");
    event.target.value = "";
  });

  refs.searchInput.addEventListener("input", renderLogs);
  if (refs.openOriginal) refs.openOriginal.addEventListener("click", () => openViewer("original"));
  if (refs.closeOriginal) refs.closeOriginal.addEventListener("click", () => closeViewer("original"));
  if (refs.openProcessed) refs.openProcessed.addEventListener("click", () => openViewer("processed"));
  if (refs.closeProcessed) refs.closeProcessed.addEventListener("click", () => closeViewer("processed"));

  refs.closePreviewBtn.addEventListener("click", hidePreviewModal);
  refs.modalOverlay.addEventListener("click", hidePreviewModal);

  refs.openWebcamBtn.addEventListener("click", openWebcam);
  if (refs.closeWebcamBtn) refs.closeWebcamBtn.addEventListener("click", closeWebcam);
  refs.registerBtn.addEventListener("click", registerSubject);

  // registration modal handlers
  const registerModal = document.getElementById("registerModal");
  const closeRegisterBtn = document.getElementById("closeRegisterBtn");
  const registerSubmitBtn = document.getElementById("registerSubmitBtn");

  function closeRegisterModal() {
    if (!registerModal) return;
    registerModal.classList.add("hidden");
  }

  function openRegisterModal(cam, clipStem) {
    if (!registerModal) return;
    document.getElementById("reg_name").value = "";
    document.getElementById("reg_age").value = "";
    document.getElementById("reg_id").value = "";
    registerModal.classList.remove("hidden");
  }

  window.openRegisterModal = openRegisterModal;

  if (closeRegisterBtn) closeRegisterBtn.addEventListener("click", closeRegisterModal);
  if (registerSubmitBtn) registerSubmitBtn.addEventListener("click", () => {
    const payload = {
      clip_stem: _pending_register_clip || "",
      name: document.getElementById("reg_name").value,
      age: document.getElementById("reg_age").value,
      person_id: document.getElementById("reg_id").value,
    };
    if (socket && socket.connected) {
      socket.emit("register_face_done", payload);
    }
    closeRegisterModal();
    _pending_register_clip = null;
  });

  renderDynamicCameras();
  showViewer("original", true);
  showViewer("processed", true);
  showViewer("webcam", false);
  renderLogs();
  renderClipQueue();
  renderPendingResults();
  logEvent("SYS", "Workspace ready", "--", "--");
  updateSummary();

  window.addEventListener("beforeunload", () => {
    state.objectUrls.forEach((url) => URL.revokeObjectURL(url));
    const stream = refs.liveWebcam.srcObject;
    if (stream) stream.getTracks().forEach((track) => track.stop());
  });
}

boot();
