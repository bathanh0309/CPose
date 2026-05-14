# CPose

Real-time computer vision pipeline:

`YOLO11-Pose -> ByteTrack -> FastReID -> Pose Buffer -> PoseC3D/ADL -> visualized video`


| Module | File |
| --- | --- |
| YOLO11-Pose | `models/yolo11n-pose.pt` |
| FastReID | `models/fastreid_market_R50.pth` |
| PoseC3D | `models/posec3d_r50_ntu60.pth` |
