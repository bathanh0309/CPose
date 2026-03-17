from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    def __init__(self, embedder_name = 'mobilenet'):
        self.model = DeepSort(
            max_age=30,
            n_init=3,
            max_iou_distance=0.7,
            max_cosine_distance=0.4,
            embedder=embedder_name,
            half=True
        )
        
    def update_tracks(self, detections, frame):
        return self.model.update_tracks(detections, frame=frame)