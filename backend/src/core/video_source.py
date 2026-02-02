"""
Video Source Abstraction
Supports: Single file, Directory watching, RTSP streams
"""

import cv2
import time
import os
import glob
from pathlib import Path
from typing import Optional, Tuple, List
from abc import ABC, abstractmethod
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class VideoSource(ABC):
    """Abstract base class for video sources"""
    
    @abstractmethod
    def read(self) -> Tuple[bool, Optional[cv2.Mat], float]:
        """
        Read next frame
        Returns: (success, frame, timestamp_ms)
        """
        pass
    
    @abstractmethod
    def release(self):
        """Release video source"""
        pass
    
    @abstractmethod
    def is_opened(self) -> bool:
        """Check if source is opened"""
        pass
    
    @abstractmethod
    def get_fps(self) -> float:
        """Get FPS of video source"""
        pass
    
    @abstractmethod
    def get_info(self) -> dict:
        """Get source information"""
        pass


class FileVideoSource(VideoSource):
    """Single video file source"""
    
    def __init__(self, video_path: str, start_time: Optional[datetime] = None):
        self.video_path = video_path
        self.start_time = start_time or datetime.now()
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Opened video: {video_path} (FPS: {self.fps}, Frames: {self.total_frames})")
    
    def read(self) -> Tuple[bool, Optional[cv2.Mat], float]:
        ret, frame = self.cap.read()
        if not ret:
            return False, None, 0.0
        
        # Get timestamp from video position
        pos_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        
        # Calculate absolute timestamp
        timestamp_ms = self.start_time.timestamp() * 1000 + pos_ms
        
        return True, frame, timestamp_ms
    
    def release(self):
        if self.cap:
            self.cap.release()
    
    def is_opened(self) -> bool:
        return self.cap.isOpened()
    
    def get_fps(self) -> float:
        return self.fps
    
    def get_info(self) -> dict:
        return {
            'type': 'file',
            'path': self.video_path,
            'fps': self.fps,
            'total_frames': self.total_frames,
            'start_time': self.start_time.isoformat()
        }


class DirectoryVideoSource(VideoSource):
    """
    Directory-based video source with automatic file discovery and switching
    Supports watch mode for continuous monitoring
    """
    
    def __init__(
        self,
        directory: str,
        extensions: List[str] = ['.mp4', '.avi', '.mkv', '.mov'],
        sort_method: str = 'name',  # 'name' or 'created_time'
        watch_mode: bool = False,
        watch_interval: float = 5.0
    ):
        self.directory = Path(directory)
        self.extensions = extensions
        self.sort_method = sort_method
        self.watch_mode = watch_mode
        self.watch_interval = watch_interval
        
        if not self.directory.exists():
            raise ValueError(f"Directory does not exist: {directory}")
        
        # State
        self.current_source: Optional[FileVideoSource] = None
        self.video_files: List[Path] = []
        self.current_index = -1
        self.last_scan_time = 0.0
        self.base_timestamp = datetime.now().timestamp() * 1000
        
        # Initial scan
        self._scan_videos()
        self._load_next_video()
    
    def _scan_videos(self):
        """Scan directory for video files"""
        all_files = []
        for ext in self.extensions:
            all_files.extend(self.directory.glob(f"*{ext}"))
        
        # Sort files
        if self.sort_method == 'created_time':
            all_files.sort(key=lambda x: x.stat().st_ctime)
        else:  # 'name'
            all_files.sort()
        
        # Check for new files
        new_files = [f for f in all_files if f not in self.video_files]
        if new_files:
            logger.info(f"Found {len(new_files)} new video file(s) in {self.directory}")
            for f in new_files:
                logger.debug(f"  - {f.name}")
        
        self.video_files = all_files
        self.last_scan_time = time.time()
    
    def _load_next_video(self) -> bool:
        """Load next video file"""
        # Release current source
        if self.current_source:
            self.current_source.release()
            self.current_source = None
        
        # Check if we need to rescan
        if self.watch_mode and (time.time() - self.last_scan_time) > self.watch_interval:
            old_count = len(self.video_files)
            self._scan_videos()
            if len(self.video_files) > old_count:
                logger.info(f"New videos detected: {len(self.video_files) - old_count}")
        
        # Move to next file
        self.current_index += 1
        
        if self.current_index >= len(self.video_files):
            if self.watch_mode:
                # Wait for new files
                logger.info(f"No more videos. Waiting for new files in {self.directory}...")
                return False
            else:
                logger.info("All videos processed.")
                return False
        
        # Load next video
        video_path = self.video_files[self.current_index]
        
        # Try to extract timestamp from filename (e.g., cam2_20250131_143000.mp4)
        start_time = self._extract_timestamp_from_filename(video_path)
        
        try:
            self.current_source = FileVideoSource(str(video_path), start_time)
            logger.info(f"Loaded video [{self.current_index + 1}/{len(self.video_files)}]: {video_path.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load video {video_path}: {e}")
            return self._load_next_video()  # Try next file
    
    def _extract_timestamp_from_filename(self, path: Path) -> datetime:
        """
        Extract timestamp from filename
        Format: cam2_20250131_143000.mp4 -> 2025-01-31 14:30:00
        """
        try:
            name = path.stem
            parts = name.split('_')
            
            # Look for date (YYYYMMDD) and time (HHMMSS) patterns
            for i, part in enumerate(parts):
                if len(part) == 8 and part.isdigit():  # Date
                    date_str = part
                    if i + 1 < len(parts) and len(parts[i + 1]) == 6 and parts[i + 1].isdigit():
                        time_str = parts[i + 1]
                        dt_str = f"{date_str}_{time_str}"
                        return datetime.strptime(dt_str, "%Y%m%d_%H%M%S")
        except Exception as e:
            logger.debug(f"Cannot extract timestamp from {path.name}: {e}")
        
        # Fallback: use file creation time
        return datetime.fromtimestamp(path.stat().st_ctime)
    
    def read(self) -> Tuple[bool, Optional[cv2.Mat], float]:
        """Read next frame, automatically switch to next video if needed"""
        while True:
            # No current source
            if not self.current_source:
                if not self._load_next_video():
                    if self.watch_mode:
                        time.sleep(1.0)  # Wait before rescanning
                        self._scan_videos()
                        continue
                    else:
                        return False, None, 0.0
            
            # Try to read frame
            ret, frame, timestamp_ms = self.current_source.read()
            
            if ret:
                return True, frame, timestamp_ms
            else:
                # End of current video, try next
                logger.info(f"Finished video: {self.video_files[self.current_index].name}")
                if not self._load_next_video():
                    if self.watch_mode:
                        time.sleep(1.0)
                        self._scan_videos()
                        continue
                    else:
                        return False, None, 0.0
    
    def release(self):
        if self.current_source:
            self.current_source.release()
    
    def is_opened(self) -> bool:
        return self.current_source is not None and self.current_source.is_opened()
    
    def get_fps(self) -> float:
        if self.current_source:
            return self.current_source.get_fps()
        return 30.0  # Default
    
    def get_info(self) -> dict:
        return {
            'type': 'directory',
            'directory': str(self.directory),
            'total_files': len(self.video_files),
            'current_index': self.current_index,
            'watch_mode': self.watch_mode,
            'current_file': self.video_files[self.current_index].name if self.current_index >= 0 else None
        }


class RTSPVideoSource(VideoSource):
    """RTSP stream source (future implementation)"""
    
    def __init__(self, rtsp_url: str, reconnect_interval: float = 5.0):
        self.rtsp_url = rtsp_url
        self.reconnect_interval = reconnect_interval
        self.cap = cv2.VideoCapture(rtsp_url)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open RTSP stream: {rtsp_url}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.last_reconnect = 0.0
        
        logger.info(f"Connected to RTSP: {rtsp_url}")
    
    def read(self) -> Tuple[bool, Optional[cv2.Mat], float]:
        ret, frame = self.cap.read()
        
        if not ret:
            # Try to reconnect
            if time.time() - self.last_reconnect > self.reconnect_interval:
                logger.warning(f"RTSP connection lost. Reconnecting to {self.rtsp_url}...")
                self.cap.release()
                self.cap = cv2.VideoCapture(self.rtsp_url)
                self.last_reconnect = time.time()
            return False, None, 0.0
        
        # Use system time for live streams
        timestamp_ms = time.time() * 1000
        
        return True, frame, timestamp_ms
    
    def release(self):
        if self.cap:
            self.cap.release()
    
    def is_opened(self) -> bool:
        return self.cap.isOpened()
    
    def get_fps(self) -> float:
        return self.fps
    
    def get_info(self) -> dict:
        return {
            'type': 'rtsp',
            'url': self.rtsp_url,
            'fps': self.fps
        }


def create_video_source(config: dict, camera_id: str) -> VideoSource:
    """
    Factory function to create appropriate video source
    
    Args:
        config: Configuration dict from YAML
        camera_id: Camera ID (e.g., 'cam2')
    
    Returns:
        VideoSource instance
    """
    data_source = config.get('data_source', {})
    mode = data_source.get('mode', 'folder')
    
    if mode == 'folder':
        # Folder-based
        data_root = data_source.get('data_root')
        if not data_root:
            raise ValueError("data_root is required for folder mode")
        
        # Find camera folder name from config
        camera_folder = None
        for cam in config.get('cameras', []):
            if cam['id'] == camera_id:
                camera_folder = cam.get('folder_name', camera_id)
                break
        
        if not camera_folder:
            raise ValueError(f"Camera {camera_id} not found in config")
        
        camera_dir = os.path.join(data_root, camera_folder)
        
        return DirectoryVideoSource(
            directory=camera_dir,
            extensions=data_source.get('video_extensions', ['.mp4', '.avi', '.mkv']),
            sort_method=data_source.get('sort_method', 'name'),
            watch_mode=data_source.get('watch_mode', False)
        )
    
    elif mode == 'files':
        # File-based (backward compatible)
        video_files = data_source.get('video_files', {})
        video_path = video_files.get(camera_id)
        
        if not video_path:
            raise ValueError(f"Video path for {camera_id} not found in config")
        
        return FileVideoSource(video_path)
    
    else:
        raise ValueError(f"Unknown data source mode: {mode}")
