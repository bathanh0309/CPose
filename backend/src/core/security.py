"""
Security Manager - Dangerous Zones & Objects Detection

Features:
1. Forbidden/Danger Zone detection using point-in-polygon
2. Dangerous object detection (knife, gun, etc.)
3. Anti-false-positive: N consecutive frames before alert
4. Event logging (JSON Lines format)
"""

import cv2
import numpy as np
import json
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class DangerZone:
    """Represents a dangerous/forbidden zone"""
    name: str
    polygon: np.ndarray  # Nx2 array of points
    color: Tuple[int, int, int]
    alert_consecutive_frames: int
    camera_id: str


@dataclass
class DangerousObject:
    """Detected dangerous object"""
    class_name: str
    bbox: List[float]
    confidence: float
    camera_id: str
    timestamp: float


@dataclass
class Alert:
    """Security alert"""
    alert_type: str  # 'zone_intrusion' or 'dangerous_object'
    camera_id: str
    timestamp: float
    description: str
    global_id: Optional[int] = None
    track_id: Optional[int] = None
    location: Optional[List[float]] = None  # bbox or point
    metadata: Dict = field(default_factory=dict)


class SecurityManager:
    """
    Security Manager for multi-camera surveillance
    
    Handles:
    - Danger zone monitoring
    - Dangerous object detection
    - Alert generation with anti-false-positive
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.security_config = config.get('security', {})
        self.enabled = self.security_config.get('enabled', True)
        
        if not self.enabled:
            logger.info("Security module disabled")
            return
        
        # Danger zones
        self.danger_zones: Dict[str, List[DangerZone]] = {}
        self._load_danger_zones()
        
        # Dangerous objects
        self.dangerous_objects_config = self.security_config.get('dangerous_objects', {})
        self.dangerous_enabled = self.dangerous_objects_config.get('enabled', True)
        self.dangerous_classes = set(self.dangerous_objects_config.get('objects', []))
        self.dangerous_conf_threshold = self.dangerous_objects_config.get('confidence_threshold', 0.5)
        self.dangerous_alert_frames = self.dangerous_objects_config.get('alert_consecutive_frames', 5)
        
        # Class name mapping (COCO classes)
        self.class_names = {
            43: 'knife',
            76: 'scissors',  # Temporary for gun
            # Add more as needed
        }
        
        # Anti-false-positive tracking
        # For zones: track (camera_id, track_id) -> consecutive frames inside zone
        self.zone_violations: Dict[Tuple[str, int], int] = defaultdict(int)
        
        # For objects: track (camera_id, class_name, approximate_location) -> consecutive frames
        self.object_detections: Dict[Tuple[str, str, Tuple[int, int]], int] = defaultdict(int)
        
        # Recent alerts (to avoid spam)
        self.recent_alerts: deque = deque(maxlen=100)
        self.alert_cooldown = 5.0  # Seconds between same type of alert
        
        # Alert logging
        self.alert_log_path = self.security_config.get('alert_log_path', 'backend/data/security_alerts.jsonl')
        Path(self.alert_log_path).parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"SecurityManager initialized. Zones: {sum(len(v) for v in self.danger_zones.values())}, "
                   f"Dangerous objects: {self.dangerous_classes}")
    
    def _load_danger_zones(self):
        """Load danger zones from config"""
        zones_config = self.security_config.get('danger_zones', {})
        
        for camera_id, zones in zones_config.items():
            self.danger_zones[camera_id] = []
            
            for zone_data in zones:
                polygon = np.array(zone_data['polygon'], dtype=np.int32)
                
                zone = DangerZone(
                    name=zone_data['name'],
                    polygon=polygon,
                    color=tuple(zone_data.get('color', [0, 0, 255])),
                    alert_consecutive_frames=zone_data.get('alert_consecutive_frames', 10),
                    camera_id=camera_id
                )
                
                self.danger_zones[camera_id].append(zone)
                logger.info(f"Loaded danger zone: {camera_id}/{zone.name}")
    
    def check_zones(
        self,
        camera_id: str,
        detections: List[Dict],
        timestamp: float
    ) -> List[Alert]:
        """
        Check if any person enters danger zones
        
        Args:
            camera_id: Camera ID
            detections: List of detections with 'bbox', 'track_id', 'global_id'
            timestamp: Current timestamp
        
        Returns:
            List of alerts
        """
        if not self.enabled or camera_id not in self.danger_zones:
            return []
        
        alerts = []
        zones = self.danger_zones[camera_id]
        
        # Track which track_ids are currently in zones
        current_in_zone = set()
        
        for detection in detections:
            bbox = detection['bbox']
            track_id = detection.get('track_id')
            global_id = detection.get('global_id')
            
            if track_id is None:
                continue
            
            # Get bottom-center point of bbox (foot position)
            x1, y1, x2, y2 = bbox
            point = (int((x1 + x2) / 2), int(y2))
            
            # Check each zone
            for zone in zones:
                if self._point_in_polygon(point, zone.polygon):
                    current_in_zone.add((camera_id, track_id))
                    
                    # Increment counter
                    key = (camera_id, track_id)
                    self.zone_violations[key] += 1
                    
                    # Check if threshold reached
                    if self.zone_violations[key] >= zone.alert_consecutive_frames:
                        # Generate alert (but check cooldown)
                        if self._should_alert('zone_intrusion', camera_id, track_id, timestamp):
                            alert = Alert(
                                alert_type='zone_intrusion',
                                camera_id=camera_id,
                                timestamp=timestamp,
                                description=f"Person entered {zone.name}",
                                global_id=global_id,
                                track_id=track_id,
                                location=bbox,
                                metadata={
                                    'zone_name': zone.name,
                                    'consecutive_frames': self.zone_violations[key]
                                }
                            )
                            alerts.append(alert)
                            self._log_alert(alert)
        
        # Reset counters for tracks not in zone anymore
        keys_to_remove = []
        for key in self.zone_violations.keys():
            if key not in current_in_zone:
                self.zone_violations[key] = 0
                if self.zone_violations[key] == 0:
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.zone_violations[key]
        
        return alerts
    
    def check_dangerous_objects(
        self,
        camera_id: str,
        frame: np.ndarray,
        yolo_results,
        timestamp: float
    ) -> List[Alert]:
        """
        Check for dangerous objects in YOLO detections
        
        Args:
            camera_id: Camera ID
            frame: Video frame (for visualization)
            yolo_results: YOLO detection results
            timestamp: Current timestamp
        
        Returns:
            List of alerts
        """
        if not self.enabled or not self.dangerous_enabled:
            return []
        
        alerts = []
        current_objects = set()
        
        # Parse YOLO results
        if yolo_results is None or len(yolo_results) == 0:
            return []
        
        boxes = yolo_results[0].boxes
        
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Check if this is a dangerous class
            if cls not in self.class_names:
                continue
            
            class_name = self.class_names[cls]
            
            if class_name not in self.dangerous_classes:
                continue
            
            if conf < self.dangerous_conf_threshold:
                continue
            
            # Get bbox
            xyxy = box.xyxy[0].cpu().numpy()
            bbox = [float(x) for x in xyxy]
            
            # Get approximate location (for tracking)
            center_x = int((bbox[0] + bbox[2]) / 2 / 100) * 100  # Quantize to 100px grid
            center_y = int((bbox[1] + bbox[3]) / 2 / 100) * 100
            location_key = (center_x, center_y)
            
            key = (camera_id, class_name, location_key)
            current_objects.add(key)
            
            # Increment counter
            self.object_detections[key] += 1
            
            # Check if threshold reached
            if self.object_detections[key] >= self.dangerous_alert_frames:
                if self._should_alert('dangerous_object', camera_id, class_name, timestamp):
                    alert = Alert(
                        alert_type='dangerous_object',
                        camera_id=camera_id,
                        timestamp=timestamp,
                        description=f"Dangerous object detected: {class_name.upper()}",
                        location=bbox,
                        metadata={
                            'class': class_name,
                            'confidence': conf,
                            'consecutive_frames': self.object_detections[key]
                        }
                    )
                    alerts.append(alert)
                    self._log_alert(alert)
        
        # Reset counters for objects not detected anymore
        keys_to_remove = []
        for key in self.object_detections.keys():
            if key not in current_objects:
                self.object_detections[key] = max(0, self.object_detections[key] - 1)
                if self.object_detections[key] == 0:
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.object_detections[key]
        
        return alerts
    
    def _point_in_polygon(self, point: Tuple[int, int], polygon: np.ndarray) -> bool:
        """
        Check if point is inside polygon using ray casting algorithm
        
        Args:
            point: (x, y) coordinates
            polygon: Nx2 numpy array of polygon vertices
        
        Returns:
            True if point is inside polygon
        """
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _should_alert(self, alert_type: str, camera_id: str, identifier, timestamp: float) -> bool:
        """Check if we should generate alert (cooldown check)"""
        # Create alert signature
        signature = f"{alert_type}_{camera_id}_{identifier}"
        
        # Check recent alerts
        for alert_data in self.recent_alerts:
            if alert_data['signature'] == signature:
                if timestamp - alert_data['timestamp'] < self.alert_cooldown:
                    return False
        
        # Add to recent alerts
        self.recent_alerts.append({
            'signature': signature,
            'timestamp': timestamp
        })
        
        return True
    
    def _log_alert(self, alert: Alert):
        """Log alert to file"""
        try:
            alert_dict = {
                'alert_type': alert.alert_type,
                'camera_id': alert.camera_id,
                'timestamp': alert.timestamp,
                'datetime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp)),
                'description': alert.description,
                'global_id': alert.global_id,
                'track_id': alert.track_id,
                'location': alert.location,
                'metadata': alert.metadata
            }
            
            with open(self.alert_log_path, 'a') as f:
                f.write(json.dumps(alert_dict) + '\n')
            
            logger.warning(f"ALERT [{alert.camera_id}]: {alert.description}")
        
        except Exception as e:
            logger.error(f"Failed to log alert: {e}")
    
    def draw_zones(self, frame: np.ndarray, camera_id: str) -> np.ndarray:
        """Draw danger zones on frame"""
        if not self.enabled or camera_id not in self.danger_zones:
            return frame
        
        overlay = frame.copy()
        
        for zone in self.danger_zones[camera_id]:
            # Draw filled polygon with transparency
            cv2.fillPoly(overlay, [zone.polygon], zone.color)
            
            # Draw border
            cv2.polylines(frame, [zone.polygon], isClosed=True, color=zone.color, thickness=2)
            
            # Draw label
            label_pos = tuple(zone.polygon[0])
            cv2.putText(
                frame,
                zone.name,
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                zone.color,
                2
            )
        
        # Blend overlay with frame (30% transparency)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        return frame
    
    def draw_alerts(
        self,
        frame: np.ndarray,
        alerts: List[Alert],
        detections: List[Dict]
    ) -> np.ndarray:
        """Draw alert overlays on frame"""
        if not alerts:
            return frame
        
        for alert in alerts:
            if alert.alert_type == 'zone_intrusion':
                # Find corresponding detection
                for det in detections:
                    if det.get('track_id') == alert.track_id:
                        bbox = det['bbox']
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        
                        # Draw red box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        
                        # Draw label
                        label = "DANGER ZONE"
                        cv2.putText(
                            frame,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2
                        )
                        break
            
            elif alert.alert_type == 'dangerous_object':
                bbox = alert.location
                x1, y1, x2, y2 = [int(v) for v in bbox]
                
                # Draw red box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
                # Draw label
                label = alert.metadata['class'].upper()
                conf = alert.metadata['confidence']
                text = f"{label} {conf:.2f}"
                
                cv2.putText(
                    frame,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )
        
        return frame
    
    def get_statistics(self) -> dict:
        """Get statistics for logging"""
        return {
            'enabled': self.enabled,
            'active_zone_violations': len([k for k, v in self.zone_violations.items() if v > 0]),
            'active_object_detections': len([k for k, v in self.object_detections.items() if v > 0]),
            'total_alerts': len(self.recent_alerts)
        }
