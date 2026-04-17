/**
 * Face Registration Module - Premium Thin Edition
 */

class FaceRegistration {
    constructor(socket) {
        this.socket = socket;
        this.sessionId = null;
        this.isCapturing = false;
        this.currentSnapshotInterval = null;

        // Elements
        this.sourceModal = document.getElementById('regSourceModal');
        this.infoModal = document.getElementById('regInfoModal');
        this.captureModal = document.getElementById('regCaptureModal');
        this.rtspGroup = document.getElementById('rtspSelectGroup');
        this.rtspSelect = document.getElementById('regRtspSelect');
        this.userNameInput = document.getElementById('regUserName');
        this.snapshotImg = document.getElementById('regSnapshot');
        this.totalFill = document.getElementById('regTotalFill');
        this.totalPct = document.getElementById('regTotalPct');
        this.instructions = document.getElementById('regInstructions');
        
        this.init();
    }

    init() {
        document.getElementById('registerBtn')?.addEventListener('click', () => this.openSourceModal());
        document.getElementById('closeRegSourceBtn').onclick = () => this.closeSourceModal();
        document.getElementById('toRegInfoBtn').onclick = () => this.openInfoModal();
        document.getElementById('backToSourceBtn').onclick = () => this.backToSource();
        document.getElementById('startCaptureBtn').onclick = () => this.startRegistration();
        document.getElementById('stopRegBtn').onclick = () => this.stopRegistration();
        
        document.getElementsByName('reg_source').forEach(r => {
            r.addEventListener('change', () => {
                this.rtspGroup.classList.toggle('hidden', r.value !== 'rtsp');
            });
        });

        this.socket.on('registration_progress', (data) => this.handleProgress(data));
        this.socket.on('registration_done', (data) => this.handleDone(data));
    }

    openSourceModal() {
        this.sourceModal.classList.remove('hidden');
        this.loadCameras();
    }

    closeSourceModal() {
        this.sourceModal.classList.add('hidden');
    }

    backToSource() {
        this.infoModal.classList.add('hidden');
        this.sourceModal.classList.remove('hidden');
    }

    openInfoModal() {
        this.sourceModal.classList.add('hidden');
        this.infoModal.classList.remove('hidden');
        this.userNameInput.focus();
    }

    async loadCameras() {
        try {
            const res = await fetch('/api/config/cameras');
            const data = await res.json();
            this.rtspSelect.innerHTML = data.cameras.map(c => `<option value="${c.source}">${c.id} - ${c.source}</option>`).join('');
        } catch (e) {}
    }

    async startRegistration() {
        const name = this.userNameInput.value.trim();
        if (!name) return alert("Vui lòng nhập tên");

        const source = document.querySelector('input[name="reg_source"]:checked').value;
        const rtsp_url = this.rtspSelect.value;

        try {
            const res = await fetch('/api/registration/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ source, rtsp_url, name })
            });
            const data = await res.json();
            if (data.session_id) {
                this.sessionId = data.session_id;
                this.isCapturing = true;
                this.infoModal.classList.add('hidden');
                this.captureModal.classList.remove('hidden');
                document.getElementById('regDisplayTitle').textContent = `ĐANG CHỤP: ${name}`;
                this.resetUI();
                this.startSnapshotLoop();
            }
        } catch (e) { alert("Lỗi kết nối server"); }
    }

    startSnapshotLoop() {
        this.currentSnapshotInterval = setInterval(() => {
            if (this.isCapturing && this.sessionId) {
                this.snapshotImg.src = `/api/registration/snapshot/${this.sessionId}?t=${Date.now()}`;
            }
        }, 100);
    }

    handleProgress(data) {
        if (data.session_id !== this.sessionId) return;
        this.instructions.textContent = data.instruction;
        const total = Object.values(data.counts || {}).reduce((a, b) => a + b, 0);
        const pct = Math.min(100, Math.round((total / 25) * 100));
        this.totalFill.style.width = `${pct}%`;
        this.totalPct.textContent = `${pct}%`;
    }

    handleDone(data) {
        if (data.session_id !== this.sessionId) return;
        this.isCapturing = false;
        clearInterval(this.currentSnapshotInterval);
        this.instructions.textContent = data.status === 'success' ? "HOÀN TẤT!" : "LỖI!";
        setTimeout(() => {
            alert(data.message);
            this.captureModal.classList.add('hidden');
            this.sessionId = null;
        }, 1500);
    }

    async stopRegistration() {
        this.isCapturing = false;
        clearInterval(this.currentSnapshotInterval);
        if (this.sessionId) {
            await fetch('/api/registration/stop', {
                method: 'POST',
                body: JSON.stringify({ session_id: this.sessionId })
            }).catch(()=>{});
        }
        this.captureModal.classList.add('hidden');
        this.sessionId = null;
    }

    resetUI() {
        this.totalFill.style.width = '0%';
        this.totalPct.textContent = '0%';
        this.instructions.textContent = "Chuẩn bị...";
    }
}

// Ensure init
const bootReg = setInterval(() => {
    if (window.appSocket) {
        window.faceRegistration = new FaceRegistration(window.appSocket);
        clearInterval(bootReg);
    }
}, 500);
