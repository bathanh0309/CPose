from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Optional

class ProjectConfig(BaseModel):
    name: str = "CPose"
    dashboard_url: str = "http://localhost:5000"

class Phase1Config(BaseModel):
    model: str
    conf_threshold: float
    pre_buffer_sec: int
    post_buffer_sec: int
    inference_every: int
    min_clip_duration: float
    storage_limit_gb: float
    person_conf_threshold: float
    trigger_min_consecutive: int
    pre_roll_seconds: int
    post_roll_seconds: int
    rearm_cooldown_seconds: int
    min_clip_seconds: float
    max_clip_seconds: float
    min_box_area_ratio: float
    reconnect_delay_seconds: int
    jpeg_quality: int
    person_class_id: int
    snapshot_fps: float
    snapshot_active_ttl_s: float
    rtsp_transport: str
    ffmpeg_capture_options: str = ""

class Phase2Config(BaseModel):
    model: str
    person_class_id: int
    conf_threshold: float
    progress_every: int

class ADLThresholds(BaseModel):
    knee_bend_angle: float
    hip_angle_lying: float
    shoulder_raise: float
    velocity_walk: float

class Phase3Config(BaseModel):
    model: str
    person_class_id: int
    conf_threshold: float
    keypoint_conf_min: float
    window_size: int
    progress_every: int
    save_overlay: bool
    adl_classes: List[str]
    thresholds: ADLThresholds

class PoseUtilsConfig(BaseModel):
    knee_bend_angle: float
    shoulder_raise: float
    velocity_walk: float
    min_visible_keypoints: int
    falling_torso_angle: float
    falling_velocity_multiplier: float
    lying_aspect_ratio: float
    bending_velocity_multiplier: float
    confidence_unknown: float
    confidence_falling: float
    confidence_lying_down: float
    confidence_sitting: float
    confidence_bending: float
    confidence_reaching: float
    confidence_walking: float
    confidence_standing: float

class ADLConfig(BaseModel):
    torso_angle_laying: float
    aspect_ratio_laying: float
    knee_angle_sitting: float
    movement_threshold_ratio: float
    movement_walking_multiplier: float
    keypoint_conf: float
    hand_raise_frames: int
    posture_voting_frames: int
    position_history_maxlen: int
    posture_history_maxlen: int
    event_history_maxlen: int
    default_knee_angle: float
    assumed_fps: int

class GlobalIDConfig(BaseModel):
    strong_threshold: float
    weak_threshold: float
    confirm_frames: int
    top_k_candidates: int
    use_hungarian: bool
    max_unk_per_video: int
    iou_resurrection_threshold: float
    quality_update_threshold: float
    transition_windows: Dict[str, List[int]]

class ReidConfig(BaseModel):
    threshold: float
    max_features: int
    confirm_frames: int
    min_crop_height: int
    min_crop_width: int
    top_k_similarity: int
    pending_track_ttl_seconds: float
    confirmed_track_ttl_seconds: float

class TrackerConfig(BaseModel):
    max_age: int
    n_init: int
    max_iou_distance: float
    max_cosine_distance: float
    half: bool

class DetectorsConfig(BaseModel):
    yolo_path: str
    face_data_path: str
    face_min_size: int
    face_similarity_threshold: float
    face_det_width: int
    face_det_height: int
    body_conf_threshold: float
    person_class_id: int

class PersistenceConfig(BaseModel):
    embedding_dim: int
    initial_memmap_size: int
    expand_step: int
    ema_alpha: float

class VectorDBConfig(BaseModel):
    search_top_k: int
    medium_dataset_threshold: int
    large_dataset_threshold: int
    hnsw_m: int
    hnsw_ef_construction: int
    hnsw_ef_search: int
    ivf_nlist: int
    ivf_nprobe: int

class StorageConfig(BaseModel):
    prune_target_ratio: float

class StreamProbeConfig(BaseModel):
    timeout_ms: int
    common_resolutions: List[Tuple[int, int, str]]

class AppConfig(BaseModel):
    project: ProjectConfig
    phase1: Phase1Config
    phase2: Phase2Config
    phase3: Phase3Config
    pose_utils: PoseUtilsConfig
    adl: ADLConfig
    global_id: GlobalIDConfig
    reid: ReidConfig
    tracker: TrackerConfig
    detectors: DetectorsConfig
    persistence: PersistenceConfig
    vector_db: VectorDBConfig
    storage: StorageConfig
    stream_probe: StreamProbeConfig
