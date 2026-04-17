// === PHASE 1: GLOBAL SERVICE & UPLOAD RTSP 

// Biến cờ kiểm tra xem cấu hình camera đã được nạp thành công chưa
let isConfigLoaded = false;

// Hàm thay đổi trạng thái UI của tiến trình thu thập dữ liệu chạy ngầm
function toggleService(state) {
    // Nếu người dùng muốn Start nhưng chưa nạp cấu hình -> Chặn lại
    if (state === 'start' && !isConfigLoaded) {
        alert('Vui lòng nạp cấu hình nguồn (resources.txt) hợp lệ trước khi Start Service!');
        return;
    }

    const statusText = document.getElementById('service-status');
    if (state === 'start') {
        statusText.innerHTML = '<span class="text-green-600">Active (Recording)</span>';
    } else {
        statusText.innerHTML = '<span class="text-red-600">Stopped</span>';
    }
}

// Hàm xử lý khi người dùng upload file cấu hình nguồn camera (resources.txt)
function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return; // Thoát nếu không có file

    // Cập nhật thông báo đã nạp file thành công
    document.getElementById('upload-status').innerHTML = `<span class="text-green-600 font-medium">✓ Đã nạp: ${file.name}</span>`;

    // Sử dụng FileReader API để đọc nội dung file text
    const reader = new FileReader();
    reader.onload = function (e) {
        const text = e.target.result;
        const lines = text.split(/\r?\n/); // Tách file thành từng dòng

        // Lấy reference đến bảng Camera Status và dropdown Live Monitor
        const tbody = document.getElementById('camera-status-body');
        const select = document.getElementById('live-cam-select');

        // Xóa trắng dữ liệu cũ trước khi nạp mới
        tbody.innerHTML = '';
        select.innerHTML = '';

        let camCount = 0;

        // Duyệt qua từng dòng trong file txt
        lines.forEach(line => {
            if (line.trim() === '') return; // Bỏ qua dòng trống

            // Phân tách dữ liệu theo chuỗi "__" (Tên_Cam__Link_RTSP)
            const parts = line.split('__');
            if (parts.length >= 2) {
                const camName = parts[0].trim();
                camCount++;

                // Tạo và chèn dòng mới (tr) vào bảng Camera Status (chỉ show Tên và Trạng thái, ẩn RTSP link để bảo mật)
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td class="p-3 font-medium">${camName}</td>
                    <td class="p-3"><span class="inline-flex items-center gap-1.5 py-1 px-2 rounded-md text-xs font-medium bg-yellow-50 text-yellow-700">Connecting...</span></td>
                    <td class="p-3 font-mono">--</td>
                `;
                tbody.appendChild(tr);

                // Tạo và chèn option mới vào dropdown Live Monitor
                const option = document.createElement('option');
                option.text = camName;
                option.value = camName;
                select.appendChild(option);
            }
        });

        // Cảnh báo nếu file sai format hoặc không có dữ liệu
        if (camCount === 0) {
            isConfigLoaded = false; // Đánh dấu cấu hình chưa hợp lệ
            alert("Không tìm thấy camera nào hợp lệ. Vui lòng kiểm tra lại cấu trúc file!");
            document.getElementById('upload-status').innerHTML = '<span class="text-red-500">Lỗi format file</span>';

            // Nếu service đang chạy mà up file sai lên thì tự động dừng service cho an toàn
            toggleService('stop');
        } else {
            isConfigLoaded = true; // Đánh dấu cấu hình đã nạp thành công
        }
    };
    // Yêu cầu đọc file dưới dạng Text
    reader.readAsText(file);

    // Reset giá trị input để có thể chọn lại chính file đó nếu cần
    event.target.value = '';
}