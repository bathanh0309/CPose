/* CPose Studio - Face Registration Logic */

const regState = {
    source: 'local', // local | rtsp
    rtspUrl: '',
    name: '',
    age: '',
    personId: '',
    sessionId: null,
    polling: null,
    progress: 0
};

document.addEventListener('DOMContentLoaded', () => {
    bindRegistrationEvents();
});

function bindRegistrationEvents() {
    // Open Main Modal
    document.getElementById('registerBtn').addEventListener('click', openRegistrationFlow);

    // Source Modal
    document.getElementById('closeRegSourceBtn').addEventListener('click', () => toggleModal('regSourceModal', false));
    document.getElementById('toRegInfoBtn').addEventListener('click', moveToInfoStep);
    
    const sourceRadios = document.getElementsByName('reg_source');
    sourceRadios.forEach(r => r.addEventListener('change', (e) => {
        regState.source = e.target.value;
        document.getElementById('rtspSelectGroup').classList.toggle('hidden', regState.source !== 'rtsp');
        if (regState.source === 'rtsp') populateRtspRegList();
    }));

    // Info Modal
    document.getElementById('backToSourceBtn').addEventListener('click', () => {
        toggleModal('regInfoModal', false);
        toggleModal('regSourceModal', true);
    });
    document.getElementById('startCaptureBtn').addEventListener('click', startCaptureStep);

    // Capture Modal
    document.getElementById('stopRegBtn').addEventListener('click', stopRegistrationCapture);
}

// ===================================================================
//                          FLOW STEPS
// ===================================================================

async function openRegistrationFlow() {
    // Reset state
    regState.name = '';
    regState.age = '';
    regState.progress = 0;
    
    // Fetch Next ID
    try {
        const res = await fetch('/api/registration/next_id');
        const data = await res.json();
        regState.personId = data.next_id || "0001";
        document.getElementById('regUserId').value = regState.personId;
    } catch (err) {
        console.error("Failed to fetch next_id", err);
    }

    toggleModal('regSourceModal', true);
}

function moveToInfoStep() {
    if (regState.source === 'rtsp') {
        regState.rtspUrl = document.getElementById('regRtspSelect').value;
        if (!regState.rtspUrl) {
            showToast("Vui lòng chọn RTSP Camera", "warning");
            return;
        }
    }
    
    toggleModal('regSourceModal', false);
    toggleModal('regInfoModal', true);
}

async function startCaptureStep() {
    const name = document.getElementById('regUserName').value.trim();
    const age = document.getElementById('regUserAge').value.trim();

    if (!name) {
        showToast("Vui lòng nhập tên", "warning");
        return;
    }

    regState.name = name;
    regState.age = age || "??";

    // Prepare UI
    document.getElementById('regDisplayTitle').textContent = `ID: ${regState.personId} | Tên: ${regState.name} | Tuổi: ${regState.age}`;
    updateRegProgress(0);

    try {
        const res = await fetch('/api/registration/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                source: regState.source,
                rtsp_url: regState.rtspUrl,
                name: regState.name,
                age: regState.age,
                person_id: regState.personId
            })
        });

        const data = await res.json();
        if (res.ok) {
            regState.sessionId = data.session_id;
            toggleModal('regInfoModal', false);
            toggleModal('regCaptureModal', true);
            startCapturePolling();
            pushEventLog(`Bắt đầu đăng ký: ${regState.name}`, "REG");
        } else {
            throw new Error(data.error);
        }
    } catch (err) {
        showToast(err.message, "danger");
    }
}

async function stopRegistrationCapture() {
    if (!regState.sessionId) return;
    
    try {
        await fetch('/api/registration/stop', {
            method: 'POST',
            body: JSON.stringify({ session_id: regState.sessionId })
        });
        stopCapturePolling();
        toggleModal('regCaptureModal', false);
        pushEventLog(`Dừng đăng ký: ${regState.name}`, "REG");
    } catch (err) {
        console.error(err);
    }
}

// ===================================================================
//                          UI & POLLING
// ===================================================================

function startCapturePolling() {
    if (regState.polling) return;

    // Snapshot Polling
    regState.polling = setInterval(() => {
        const snap = document.getElementById('regSnapshot');
        snap.src = `/api/registration/snapshot/${regState.sessionId}?t=${Date.now()}`;
    }, 150);

    // Socket.io for progress
    socket.on('registration_progress', (data) => {
        if (data.session_id !== regState.sessionId) return;
        
        const pct = data.progress || 0;
        updateRegProgress(pct);
        
        const instr = document.getElementById('regInstructions');
        instr.textContent = data.message || "Vui lòng giữ nguyên vị trí";
    });

    socket.on('registration_done', (data) => {
        if (data.session_id !== regState.sessionId) return;
        
        stopCapturePolling();
        
        if (data.status === 'success') {
            updateRegProgress(100);
            showToast(`Đăng ký thành công: ${regState.name}`, "success");
            pushEventLog(`Hoàn tất đăng ký: ${regState.name}`, "REG");
            setTimeout(() => toggleModal('regCaptureModal', false), 1500);
        } else {
            showToast(`Lỗi đăng ký: ${data.message}`, "danger");
            toggleModal('regCaptureModal', false);
        }
    });
}

function stopCapturePolling() {
    clearInterval(regState.polling);
    regState.polling = null;
    socket.off('registration_progress');
    socket.off('registration_done');
    
    // Reset snapshot
    document.getElementById('regSnapshot').src = "data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=";
}

function updateRegProgress(pct) {
    const circle = document.getElementById('regProgressCircle');
    const label = document.getElementById('regTotalPct');
    
    const radius = circle.r.baseVal.value;
    const circumference = 2 * Math.PI * radius;
    
    const offset = circumference - (pct / 100) * circumference;
    circle.style.strokeDasharray = `${circumference} ${circumference}`;
    circle.style.strokeDashoffset = offset;
    
    label.textContent = `${Math.round(pct)}%`;
}

function toggleModal(id, show) {
    const el = document.getElementById(id);
    if (show) {
        el.classList.remove('hidden');
        el.classList.add('flex');
    } else {
        el.classList.add('hidden');
        el.classList.remove('flex');
    }
}

function populateRtspRegList() {
    const sel = document.getElementById('regRtspSelect');
    // Global state.cams from app.js
    if (window.state && window.state.cams) {
        sel.innerHTML = window.state.cams.map(c => `
            <option value="${c.cam_url || c.rtsp_url}">${c.label || c.cam_id}</option>
        `).join('');
    } else {
        sel.innerHTML = '<option value="">Chưa có camera RTSP</option>';
    }
}
