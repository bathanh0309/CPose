/**
 * Registration Module Logic
 */

class FaceRegistration {
    constructor(socket) {
        this.socket = socket;
        this.sessionId = null;
        this.isCapturing = false;
        this.currentSnapshotInterval = null;

        // Elements - Step 1
        this.sourceModal = document.getElementById('regSourceModal');
        this.sourceRadios = document.getElementsByName('reg_source');
        this.rtspGroup = document.getElementById('rtspSelectGroup');
        this.rtspSelect = document.getElementById('regRtspSelect');
        this.toInfoBtn = document.getElementById('toRegInfoBtn');
        this.closeSourceBtn = document.getElementById('closeRegSourceBtn');

        // Elements - Step 2
        this.infoModal = document.getElementById('regInfoModal');
        this.userNameInput = document.getElementById('regUserName');
        this.startCaptureBtn = document.getElementById('startCaptureBtn');
        this.backBtn = document.getElementById('backToSourceBtn');

        // Elements - Step 3 (Capture)
        this.captureModal = document.getElementById('regCaptureModal');
        this.snapshotImg = document.getElementById('regSnapshot');
        this.instructions = document.getElementById('regInstructions');
        this.displayTitle = document.getElementById('regDisplayTitle');
        this.stopRegBtn = document.getElementById('stopRegBtn');
        this.totalBar = document.getElementById('regTotalBar');
        this.totalPct = document.getElementById('regTotalPct');

        this.init();
    }

    init() {
        // Main button outside (assuming it exists in app.js or index.html)
        const mainRegBtn = document.getElementById('registerBtn');
        if (mainRegBtn) {
            mainRegBtn.addEventListener('click', () => this.openSourceModal());
        }

        // Source Modal Events
        this.sourceRadios.forEach(r => r.addEventListener('change', () => this.toggleRtspSelect()));
        this.closeSourceBtn.onclick = () => this.closeSourceModal();
        this.toInfoBtn.onclick = () => this.openInfoModal();

        // Info Modal Events
        this.backBtn.onclick = () => {
            this.infoModal.classList.add('hidden');
            this.sourceModal.classList.remove('hidden');
        };
        this.startCaptureBtn.onclick = () => this.startRegistration();

        // Capture Modal Events
        this.stopRegBtn.onclick = () => this.stopRegistration();

        // Init dots
        this.initDots();

        // Socket listeners
        this.socket.on('registration_progress', (data) => this.handleProgress(data));
        this.socket.on('registration_done', (data) => this.handleDone(data));
    }

    initDots() {
        const angles = ['center', 'left', 'right', 'up', 'down'];
        angles.forEach(angle => {
            const container = document.getElementById(`dots-${angle}`);
            if (container) {
                container.innerHTML = '';
                for (let i = 0; i < 5; i++) {
                    const dot = document.createElement('div');
                    dot.className = 'dot';
                    container.appendChild(dot);
                }
            }
        });
    }

    openSourceModal() {
        this.sourceModal.classList.remove('hidden');
        this.loadCameras();
    }

    closeSourceModal() {
        this.sourceModal.classList.add('hidden');
    }

    toggleRtspSelect() {
        const selected = document.querySelector('input[name="reg_source"]:checked').value;
        if (selected === 'rtsp') {
            this.rtspGroup.classList.remove('hidden');
        } else {
            this.rtspGroup.classList.add('hidden');
        }
    }

    async loadCameras() {
        try {
            const res = await fetch('/api/config/cameras');
            const data = await res.json();
            this.rtspSelect.innerHTML = '';
            data.cameras.forEach(cam => {
                const opt = document.createElement('option');
                opt.value = cam.source;
                opt.textContent = `${cam.id} - ${cam.source}`;
                this.rtspSelect.appendChild(opt);
            });
        } catch (e) {
            console.error("Failed to load cameras", e);
        }
    }

    openInfoModal() {
        this.sourceModal.classList.add('hidden');
        this.infoModal.classList.remove('hidden');
        this.userNameInput.focus();
    }

    async startRegistration() {
        const name = this.userNameInput.value.trim();
        if (!name) {
            alert("Vui lòng nhập tên");
            return;
        }

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
                this.displayTitle.textContent = `Đăng ký: ${name}`;
                this.resetUI();
                this.startSnapshotLoop();
            } else {
                alert("Lỗi khi khởi tạo session: " + (data.error || "Unknown"));
            }
        } catch (e) {
            alert("Lỗi kết nối server");
        }
    }

    async stopRegistration() {
        if (!this.sessionId) return;
        
        this.isCapturing = false;
        clearInterval(this.currentSnapshotInterval);
        
        try {
            await fetch('/api/registration/stop', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: this.sessionId })
            });
        } catch (e) {}

        this.captureModal.classList.add('hidden');
        this.sessionId = null;
    }

    startSnapshotLoop() {
        if (this.currentSnapshotInterval) clearInterval(this.currentSnapshotInterval);
        this.currentSnapshotInterval = setInterval(() => {
            if (!this.isCapturing || !this.sessionId) return;
            this.snapshotImg.src = `/api/registration/snapshot/${this.sessionId}?t=${Date.now()}`;
        }, 100); // 10fps preview
    }

    handleProgress(data) {
        if (data.session_id !== this.sessionId) return;

        this.instructions.textContent = data.instruction;
        
        if (data.counts) {
            let totalCaptured = 0;
            Object.entries(data.counts).forEach(([angle, count]) => {
                totalCaptured += count;
                const container = document.getElementById(`dots-${angle}`);
                if (container) {
                    const dots = container.querySelectorAll('.dot');
                    dots.forEach((dot, idx) => {
                        if (idx < count) dot.classList.add('filled');
                        else dot.classList.remove('filled');
                    });
                }
            });

            const pct = Math.round((totalCaptured / 25) * 100);
            this.totalBar.style.width = `${pct}%`;
            this.totalPct.textContent = `${pct}%`;
        }
    }

    handleDone(data) {
        if (data.session_id !== this.sessionId) return;
        
        this.isCapturing = false;
        clearInterval(this.currentSnapshotInterval);
        
        if (data.status === 'success') {
            this.instructions.textContent = "Hoàn tất!";
            this.instructions.style.borderColor = "#22c55e";
            setTimeout(() => {
                alert(data.message);
                this.captureModal.classList.add('hidden');
                location.reload(); // Refresh to update face database in UI if needed
            }, 1000);
        } else {
            alert("Lỗi: " + data.message);
            this.captureModal.classList.add('hidden');
        }
    }

    resetUI() {
        this.initDots();
        this.totalBar.style.width = '0%';
        this.totalPct.textContent = '0%';
        this.instructions.textContent = "Vui lòng điều chỉnh mặt vào khung";
        this.instructions.style.borderColor = "rgba(34, 197, 94, 0.5)";
    }
}

// Export or initialize if socket available
if (window.appSocket) {
    window.faceRegistration = new FaceRegistration(window.appSocket);
} else {
    // Wait for socket to be initialized in app.js
    const checkSocket = setInterval(() => {
        if (window.appSocket) {
            window.faceRegistration = new FaceRegistration(window.appSocket);
            clearInterval(checkSocket);
        }
    }, 500);
}
