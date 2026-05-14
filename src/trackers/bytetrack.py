class ByteTrackWrapper:
    """Thin wrapper for Ultralytics ByteTrack via YoloPoseTracker.

    The actual tracker state lives inside YOLO.track(..., persist=True). This
    class exists to keep the module boundary explicit without introducing an
    external ByteTrack checkpoint dependency.
    """

    def __init__(self, detector):
        if not hasattr(detector, "infer"):
            raise TypeError("ByteTrackWrapper expects a detector with infer(frame, persist=True)")
        self.detector = detector

    def update(self, frame):
        """Return (detections, raw_result)."""
        return self.detector.infer(frame, persist=True)
