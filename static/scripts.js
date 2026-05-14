const API_BASE = "http://localhost:8000";
const WS_BASE = "ws://localhost:8000";
const wsConnections = { cam1: null, cam2: null };

document.addEventListener("DOMContentLoaded", () => {
    loadCameras();
    setupUploads();
});

async function loadCameras() {
    try {
        const response = await fetch(`${API_BASE}/api/cameras`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const cameras = await response.json();
        ["cam1", "cam2"].forEach(camId => {
            const selectEl = document.getElementById(`input-${camId}`);
            selectEl.innerHTML = '<option value="">-- Chọn Camera --</option>';

            cameras.forEach(cam => {
                const option = document.createElement("option");
                option.value = cam.id;
                option.textContent = cam.display || cam.name;
                option.dataset.label = cam.name;
                selectEl.appendChild(option);
            });
        });
    } catch (error) {
        console.error("Khong the load resources.txt. Hay chac chan backend dang chay.", error);
        ["cam1", "cam2"].forEach(camId => {
            writeLog(camId, "Không thể tải resources.txt. Hãy chạy backend FastAPI.", "error");
        });
    }
}

function getSelectedSourceLabel(camId) {
    const selectEl = document.getElementById(`input-${camId}`);
    const selectedOption = selectEl.options[selectEl.selectedIndex];
    return selectedOption?.dataset.label || selectedOption?.textContent || "Nguồn đã chọn";
}

function maskRtspUrl(value) {
    if (!value || !value.startsWith("rtsp://")) {
        return value;
    }

    return value.replace(/\/\/([^@/]+@)?([^:/]+)(:\d+)?/i, (_, auth = "", host = "", port = "") => {
        const maskedHost = host.replace(/(\d+)(?=\.\d+$)/, "xxx");
        return `//${auth ? "***:***@" : ""}${maskedHost}${port ? ":xxx" : ""}`;
    });
}

function writeLog(camId, message, type = "system") {
    const logOutput = document.getElementById(`log-${camId}`);
    if (!logOutput) return;

    const time = new Date().toLocaleTimeString();
    const p = document.createElement("p");
    p.className = `log-msg ${type}`;
    p.textContent = `[${time}] ${message}`;

    logOutput.appendChild(p);
    logOutput.scrollTop = logOutput.scrollHeight;
}

function getActiveModules(camId) {
    const modules = ["track", "pose", "reid", "adl"];
    const active = modules.filter(mod => document.getElementById(`${mod}-${camId}`).checked);
    return active.length > 0 ? active.map(mod => mod.toUpperCase()).join(", ") : "None";
}

function toggleCamera(camId, isOn) {
    const inputSource = document.getElementById(`input-${camId}`).value;
    const stream = document.getElementById(`stream-${camId}`);
    const placeholder = document.getElementById(`placeholder-${camId}`);
    const status = document.getElementById(`status-${camId}`);
    const btnOn = document.getElementById(`btn-on-${camId}`);
    const btnOff = document.getElementById(`btn-off-${camId}`);

    if (isOn) {
        if (!inputSource) {
            alert("Vui lòng chọn Camera từ danh sách hoặc tải file lên.");
            return;
        }

        if (wsConnections[camId]) {
            wsConnections[camId].close();
            wsConnections[camId] = null;
        }

        const sourceLabel = getSelectedSourceLabel(camId);

        btnOn.className = "btn btn-on-active";
        btnOff.className = "btn btn-default";
        placeholder.style.display = "none";
        stream.style.display = "block";
        status.textContent = "CONNECTING...";
        status.style.color = "var(--mod-yellow)";

        writeLog(camId, `Đang kết nối tới: ${sourceLabel}`, "system");
        writeLog(camId, `Các module AI kích hoạt: [${getActiveModules(camId)}]`, "system");

        const wsUrl = `${WS_BASE}/ws/stream/${camId}?source=${encodeURIComponent(inputSource)}`;
        const ws = new WebSocket(wsUrl);
        wsConnections[camId] = ws;

        ws.onopen = () => {
            status.textContent = "LIVE";
            status.style.color = "var(--mod-green)";
        };

        ws.onmessage = event => {
            const data = JSON.parse(event.data);
            if (data.type === "image") {
                stream.src = `data:image/jpeg;base64,${data.data}`;
            } else if (data.type === "log") {
                writeLog(camId, data.msg, data.level);
            }
        };

        ws.onerror = () => {
            writeLog(camId, "Lỗi kết nối WebSocket.", "error");
        };

        ws.onclose = () => {
            if (wsConnections[camId] === ws) {
                wsConnections[camId] = null;
            }
            if (status.textContent !== "OFFLINE") {
                writeLog(camId, "WebSocket đã ngắt kết nối.", "error");
                setCameraOffUi(camId);
            }
        };

        return;
    }

    if (wsConnections[camId]) {
        wsConnections[camId].close();
        wsConnections[camId] = null;
    }

    setCameraOffUi(camId);
    writeLog(camId, "Đã ngắt luồng video và giải phóng buffer.", "error");
}

function setCameraOffUi(camId) {
    const stream = document.getElementById(`stream-${camId}`);
    const placeholder = document.getElementById(`placeholder-${camId}`);
    const status = document.getElementById(`status-${camId}`);
    const btnOn = document.getElementById(`btn-on-${camId}`);
    const btnOff = document.getElementById(`btn-off-${camId}`);

    btnOff.className = "btn btn-off-active";
    btnOn.className = "btn btn-default";
    stream.src = "";
    stream.style.display = "none";
    placeholder.style.display = "block";
    status.textContent = "OFFLINE";
    status.style.color = "var(--text-dark)";
}

function setupUploads() {
    ["cam1", "cam2"].forEach(camId => {
        document.getElementById(`upload-${camId}`).addEventListener("change", async function() {
            if (!this.files[0]) return;

            const formData = new FormData();
            formData.append("file", this.files[0]);
            writeLog(camId, `Đang tải file lên backend: ${this.files[0].name}`, "system");

            try {
                const response = await fetch(`${API_BASE}/api/upload`, {
                    method: "POST",
                    body: formData,
                });
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }

                const uploaded = await response.json();
                const selectEl = document.getElementById(`input-${camId}`);
                const option = document.createElement("option");
                option.value = uploaded.source;
                option.textContent = `[Upload] ${uploaded.name}`;
                option.dataset.label = `[Upload] ${uploaded.name}`;
                selectEl.appendChild(option);
                selectEl.value = uploaded.source;

                writeLog(camId, `Đã chọn file: ${uploaded.name}`, "system");
            } catch (error) {
                console.error(error);
                writeLog(camId, "Không thể tải file lên backend.", "error");
            }
        });
    });
}
