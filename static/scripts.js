const FASTAPI_PORT = 8000;
const IS_LIVE_SERVER = location.port !== String(FASTAPI_PORT) && location.port !== "";
const API_BASE = IS_LIVE_SERVER ? `http://${location.hostname}:${FASTAPI_PORT}` : "";
const WS_BASE  = IS_LIVE_SERVER ? `ws://${location.hostname}:${FASTAPI_PORT}` : `ws://${location.host}`;

const sockets = { 1: null, 2: null };        
const sessionIds = { 1: null, 2: null };      

// Modules được bật riêng biệt cho mỗi Cam (Mặc định bật track, pose)
const activeModules = {
  1: new Set(["track", "pose"]),
  2: new Set(["track", "pose"])
};              

async function handleCamUpload(event, camId) {
  const file = event.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);
  
  addLog(camId, `Đang tải lên video: ${file.name}...`, "system");
  try {
    const res = await fetch(`${API_BASE}/api/upload`, {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    
    // Server trả về đường dẫn file sau khi upload (ví dụ: uploads/video.mp4)
    const uploadedPath = data.file_path || data.filename;

    // Tự động điền đường dẫn vừa upload vào ô Input Text
    const inputEl = document.getElementById(`cam${camId}-source`);
    inputEl.value = uploadedPath;
    
    addLog(camId, `Upload thành công. Sẵn sàng luồng xử lý từ file nội bộ.`, "system");
  } catch (e) {
    addLog(camId, `Lỗi upload: Không thể kết nối tới server xử lý file.`, "error");
  }
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
  // Lấy text người dùng nhập (link RTSP hoặc đường dẫn file đã upload)
  const inputEl = document.getElementById(`cam${camId}-source`);
  const url = inputEl.value.trim();
  
  if (!url) { alert(`Vui lòng nhập link RTSP hoặc Upload Video cho Camera ${camId}`); return; }

  if (sockets[camId]) {
    sockets[camId].close();
  }

  setStatus(camId, "Đang kết nối...", "offline");
  document.getElementById(`placeholder-${camId}`).style.display = "none";
  const img = document.getElementById(`frame-${camId}`);
  img.style.display = "block";
  img.src = "";

  const mods = [...activeModules[camId]].join(",");
  const wsUrl = `${WS_BASE}/ws/stream/${camId}?url=${encodeURIComponent(url)}&modules=${mods}`;
  addLog(camId, `Bắt đầu tiếp nhận luồng: ${url}`, "system");
  addLog(camId, `Phân tích các module: [${mods}]`, "system");

  const ws = new WebSocket(wsUrl);
  sockets[camId] = ws;

  ws.onopen = () => {
    setStatus(camId, "Đang Live", "online");
  };

  ws.onmessage = (evt) => {
    const data = JSON.parse(evt.data);

    if (data.type === "session") {
      sessionIds[camId] = data.session_id;
      // Bật nút lưu Excel/Video
      toggleSaveButtons(camId, true);
    } 
    else if (data.type === "image") {
      img.src = "data:image/jpeg;base64," + data.data;
    } 
    else if (data.type === "ai") {
      addLog(camId, `AI: ${data.msg}`, "ai");
    }
    else if (data.type === "log" || data.type === "system") {
      addLog(camId, data.msg, data.level || "system");
    }
  };

  ws.onerror = () => {
    addLog(camId, `Lỗi kết nối WebSocket tới Server AI.`, "error");
  };

  ws.onclose = () => {
    setStatus(camId, "Mất kết nối", "offline");
    img.style.display = "none";
    document.getElementById(`placeholder-${camId}`).style.display = "flex";
    toggleSaveButtons(camId, false);
    sessionIds[camId] = null;
    sockets[camId] = null;
  };
}

function stopStream(camId) {
  if (sockets[camId]) {
    sockets[camId].close();
    addLog(camId, `Đã ngắt kết nối Camera (Dừng xử lý buffer).`, "system");
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

  addLog(camId, `Đang xử lý đóng gói và lưu video...`, "system");
  try {
    const res = await fetch(`${API_BASE}/api/save-video/${sid}`, { method: "POST" });
    const data = await res.json();
    addLog(camId, `Đã lưu Video: ${data.file_path || data.message}`, "ai");
  } catch (err) {
    addLog(camId, `Lỗi lưu video: ${err}`, "error");
  }
}

async function saveExcel(camId) {
  const sid = sessionIds[camId];
  if (!sid) return;

  addLog(camId, `Đang trích xuất dữ liệu ra file Excel...`, "system");
  try {
    const res = await fetch(`${API_BASE}/api/save-excel/${sid}`, { method: "POST" });
    const data = await res.json();
    addLog(camId, `Đã lưu Excel: ${data.file_path || data.message}`, "ai");
  } catch (err) {
    addLog(camId, `Lỗi lưu Excel: ${err}`, "error");
  }
}

/* Utils: Thêm Log & Clear Log cho TỪNG Camera riêng biệt */
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
  const time  = new Date().toLocaleTimeString("vi-VN");

  entry.className = `log-entry log-${level}`;
  entry.innerHTML = `<span class="log-time">${time}</span> ${escapeHtml(msg)}`;
  box.appendChild(entry);
  box.scrollTop = box.scrollHeight;

  while (box.children.length > 500) box.removeChild(box.firstChild);
}

function clearLog(camId) {
  const box = document.getElementById(`log-box-${camId}`);
  if(box) box.innerHTML = "";
}

function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/[&<>"']/g, function(m) {
    return { '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' }[m];
  });
}