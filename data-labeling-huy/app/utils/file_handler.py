# Hàm đọc file resources.txt, tạo/xóa thư mục...
import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

BASE_DATA_DIR = Path("data")
TEMP_DIR = BASE_DATA_DIR / "temp"
OUTPUT_DIR = BASE_DATA_DIR / "output_labels"

TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

# upload / temp
def save_upload_temp(file_obj, original_filename:str) ->str:
    safe_name = _sanitize_filename(original_filename)
    unique_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
    dest_path = TEMP_DIR / unique_name
    
    with open(dest_path, "wb") as out_f:
        shutil.copyfileobj(file_obj, out_f)
    
    return str(dest_path.resolve())


def delete_temp_file(file_path: str, retries: int = 3, delay: float = 1.0) -> bool:
    """Xóa file tạm, có cơ chế thử lại nếu file đang bị Windows khóa."""
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        return False
        
    for attempt in range(retries):
        try:
            p.unlink()
            logger.info(f"Đã xóa file tạm thành công: {file_path}")
            return True
        except PermissionError as e:
            logger.warning(f"File {file_path} đang bận (lần thử {attempt + 1}/{retries}). Thử lại sau {delay}s...")
            time.sleep(delay)
        except Exception as e:
            logger.error(f"Lỗi không xác định khi xóa file {file_path}: {e}")
            break
            
    logger.error(f"Bó tay! Không thể xóa file tạm {file_path} sau {retries} lần thử.")
    return False


def cleanup_temp_dir() -> int:
    # free up space in tempt folder regulary
    # return the number of mp4 files had cleaned up
    
    count = 0
    for item in TEMP_DIR.iterdir():
        if item.is_file():
            item.unlink()
            count += 1
    return count


# OUTPUT FOLDER
def ensure_output_subdir(base_output_folder: str, video_name: str) -> str:
    # create sub folder for each mp4 files to store its data
    folder_name = Path(video_name).stem
    out_dir = Path(base_output_folder) / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir.resolve())


def delete_output_folder(target_path: str) -> bool:
    # delete 1 output folder when user click on delete button
    p = Path(target_path)
    if p.exists() and p.is_dir():
        shutil.rmtree(p)
        return True
    return False


def list_output_subfolders(base_output_folder: str) -> list[str]:
    # return a list of all the subfolder in output folder
    p = Path(base_output_folder)
    if not p.exists():
        return []
    return [entry.name for entry in p.iterdir() if entry.is_dir()]


def get_output_json_path(base_output_folder: str, video_name: str) -> Optional[str]:
    # return a dir of a json file if it exist
    folder_name = Path(video_name).stem
    json_path = Path(base_output_folder) / folder_name / "detections.json"
    return str(json_path.resolve()) if json_path.exists() else None


# Validation and helpers
def is_valid_mp4(file_path: str) -> bool:
    # check if the mp4 file is valid or not
    p = Path(file_path)
    if not p.exists() or p.stat().st_size == 0:
        return False
    
    with open(p, "rb") as f:
        f.seek(4)
        magic = f.read(4)
    return magic in (b"ftyp", b"moov", b"mdat", b"free")


def get_file_size_mb(file_path: str) -> float:
    return round(Path(file_path).stat().st_size / (1024 * 1024), 2)


def _sanitize_filename(filename: str) -> str:
    # clean up unwanted character in filename
    safe = "".join(c for c in Path(filename).name if c.isalnum() or c in ("_","-","."))
    return safe if safe else "unnamed_file"