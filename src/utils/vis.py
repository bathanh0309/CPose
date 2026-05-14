import cv2


COCO_EDGES = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]


def draw_detection(frame, det, label=None, color=(0, 220, 0)):
    x1, y1, x2, y2 = map(int, det["bbox"])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if label:
        cv2.putText(
            frame, label, (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
        )

    kpts = det.get("keypoints")
    kpt_scores = det.get("keypoint_scores")

    if kpts is not None:
        for i, kp in enumerate(kpts):
            x, y = int(kp[0]), int(kp[1])
            conf = 1.0 if kpt_scores is None else float(kpt_scores[i])
            if conf > 0.2:
                cv2.circle(frame, (x, y), 3, (0, 140, 255), -1)

        for a, b in COCO_EDGES:
            if a < len(kpts) and b < len(kpts):
                xa, ya = int(kpts[a][0]), int(kpts[a][1])
                xb, yb = int(kpts[b][0]), int(kpts[b][1])
                ca = 1.0 if kpt_scores is None else float(kpt_scores[a])
                cb = 1.0 if kpt_scores is None else float(kpt_scores[b])
                if ca > 0.2 and cb > 0.2:
                    cv2.line(frame, (xa, ya), (xb, yb), (255, 180, 0), 2)

    return frame