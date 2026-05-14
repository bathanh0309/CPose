class ByteTrackWrapper:
    def __init__(self, detector):
        self.detector = detector

    def update(self, frame):
        """Return (detections, raw_result)."""
        return self.detector.infer(frame, persist=True)
