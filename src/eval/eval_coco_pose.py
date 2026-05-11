from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLO pose on COCO val2017 keypoints")
    parser.add_argument("--model", default="models/pose_estimation/yolov8n-pose.pt")
    parser.add_argument("--data", default="configs/eval/coco_pose.yaml")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help='GPU index, "cpu", or Ultralytics device string')
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    metrics = model.val(
        data=args.data,
        split="val",
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
    )

    print(f"Pose AP50:    {metrics.pose.ap50:.3f}")
    print(f"Pose AP50-95: {metrics.pose.ap:.3f}")
    print(f"Pose mAP:     {metrics.pose.map:.3f}")


if __name__ == "__main__":
    main()
