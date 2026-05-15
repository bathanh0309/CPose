const FASTAPI_PORT = 8000;
const IS_LIVE_SERVER = location.port !== String(FASTAPI_PORT) && location.port !== "";
const API_BASE = IS_LIVE_SERVER ? `http://${location.hostname}:${FASTAPI_PORT}` : "";
const WS_BASE = IS_LIVE_SERVER ? `ws://${location.hostname}:${FASTAPI_PORT}` : `ws://${location.host}`;

const sockets = { 1: null, 2: null };
const sessionIds = { 1: null, 2: null };
const cameraSlots = { 1: null, 2: null };
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

  addLog(camId, "Dang nap file RTSP config...", "system");
  try {
    const res = await fetch(`${API_BASE}/api/cameras/config`, {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Khong the nap RTSP config");

    applyCameraList(data.cameras || []);
    selectDefaultCamera(camId);
    addLog(camId, "Da nap RTSP config. Chon luong trong danh sach roi nhan ON.", "system");
  } catch (err) {
    addLog(camId, `Loi nap RTSP config: ${err.message || err}`, "error");
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
    addLog(1, "Khong the doc danh sach camera tu server.", "error");
    addLog(2, "Khong the doc danh sach camera tu server.", "error");
  }
}

function applyCameraList(cameras) {
  rtspCameras = cameras;

  [1, 2].forEach((camId) => {
    const selectEl = document.getElementById(`cam${camId}-display`);
    if (!selectEl) return;

    const previousValue = selectEl.value;
    selectEl.innerHTML = '<option value="">Chon luong RTSP...</option>';

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
    addLog(camId, `Da chon luong: ${cameraSlots[camId].display || cameraSlots[camId].name}`, "system");
  }

  if (sockets[camId] && cameraSlots[camId]) {
    startStream(camId);
  }
}

async function handleCamUpload(event, camId) {
  const file = event.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  addLog(camId, `Dang tai len video: ${file.name}...`, "system");
  try {
    const res = await fetch(`${API_BASE}/api/upload`, {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Upload failed");

    const uploadedPath = data.source || data.file_path || data.filename || data.name;
    document.getElementById(`cam${camId}-source`).value = uploadedPath;
    setSelectToUploadedFile(camId, file.name);
    cameraSlots[camId] = null;

    addLog(camId, "Upload thanh cong. San sang xu ly tu file noi bo.", "system");
  } catch (e) {
    addLog(camId, `Loi upload: ${e.message || e}`, "error");
  } finally {
    event.target.value = "";
  }
}

function setSelectToUploadedFile(camId, fileName) {
  const selectEl = document.getElementById(`cam${camId}-display`);
  if (!selectEl) return;

  let option = selectEl.querySelector('option[data-upload="true"]');
  if (!option) {
    option = document.createElement("option");
    option.dataset.upload = "true";
    selectEl.appendChild(option);
  }
  option.value = "__uploaded_file__";
  option.textContent = fileName;
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
}

function startStream(camId) {
  const sourceEl = document.getElementById(`cam${camId}-source`);
  const videoSource = sourceEl.value.trim();

  if (!videoSource) {
    alert(`Vui long chon luong RTSP hoac upload video cho Camera ${camId}`);
    return;
  }

  if (sockets[camId]) {
    sockets[camId].close();
  }

  setStatus(camId, "Dang ket noi...", "offline");
  document.getElementById(`placeholder-${camId}`).style.display = "none";
  const img = document.getElementById(`frame-${camId}`);
  img.style.display = "block";
  img.src = "";

  const mods = [...activeModules[camId]].join(",");
  if (!mods) {
    addLog(camId, "Chua chon module AI nao, dang phat raw stream.", "warning");
  }
  const wsUrl = `${WS_BASE}/ws/stream/${camId}?source=${encodeURIComponent(videoSource)}&modules=${mods}`;
  const displayName = getSelectedDisplayName(camId);
  addLog(camId, `Bat dau tiep nhan luong: ${displayName}`, "system");
  addLog(camId, `Phan tich cac module: [${mods}]`, "system");

  const ws = new WebSocket(wsUrl);
  sockets[camId] = ws;

  ws.onopen = () => {
    setStatus(camId, "Dang Live", "online");
  };

  ws.onmessage = (evt) => {
    const data = JSON.parse(evt.data);

    if (data.type === "session") {
      sessionIds[camId] = data.session_id;
      toggleSaveButtons(camId, true);
    } else if (data.type === "image") {
      img.src = "data:image/jpeg;base64," + data.data;
    } else if (data.type === "metric") {
      const m = data.metrics || {};
      addLog(
        camId,
        `METRIC ${m.module || ""}: fps=${m.fps ?? "-"} det=${m.detections ?? "-"} tracked=${m.tracked ?? "-"}`,
        "metric"
      );
    } else if (data.type === "ai") {
      addLog(camId, `AI: ${data.msg}`, "ai");
    } else if (data.type === "log" || data.type === "system") {
      addLog(camId, maskRtsp(data.msg), data.level || "system");
    }
  };

  ws.onerror = () => {
    addLog(camId, "Loi ket noi WebSocket toi Server AI.", "error");
  };

  ws.onclose = () => {
    setStatus(camId, "Mat ket noi", "offline");
    img.style.display = "none";
    document.getElementById(`placeholder-${camId}`).style.display = "flex";
    toggleSaveButtons(camId, false);
    sessionIds[camId] = null;
    sockets[camId] = null;
  };
}

function getSelectedDisplayName(camId) {
  const selectEl = document.getElementById(`cam${camId}-display`);
  return selectEl?.selectedOptions?.[0]?.textContent || `Camera ${camId}`;
}

function stopStream(camId) {
  if (sockets[camId]) {
    try {
      sockets[camId].send(JSON.stringify({ type: "stop" }));
    } catch (e) {
      // ignore close races
    }
    sockets[camId].close();
    addLog(camId, "Da ngat ket noi Camera.", "system");
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

  addLog(camId, "Dang dong goi va luu video...", "system");
  try {
    const res = await fetch(`${API_BASE}/api/save-video/${sid}`, { method: "POST" });
    const data = await res.json();
    addLog(camId, `Da luu Video: ${data.path || data.saved || data.message || ""}`, "ai");
  } catch (err) {
    addLog(camId, `Loi luu video: ${err}`, "error");
  }
}

async function saveExcel(camId) {
  const sid = sessionIds[camId];
  if (!sid) return;

  addLog(camId, "Dang trich xuat du lieu ra file Excel...", "system");
  try {
    const res = await fetch(`${API_BASE}/api/save-excel/${sid}`, { method: "POST" });
    const data = await res.json();
    addLog(camId, `Da luu Excel: ${data.file_path || data.message || ""}`, "ai");
  } catch (err) {
    addLog(camId, `Loi luu Excel: ${err}`, "error");
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
  loadConfiguredCameras,
  bindUIEvents
});

window.addEventListener("DOMContentLoaded", () => {
  bindUIEvents();
  loadConfiguredCameras();
});
