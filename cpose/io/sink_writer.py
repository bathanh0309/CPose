import os
import json
import numpy as np

class SinkWriter:
    """
    Writer for research-related outputs (skeletons, research logs, datasets).
    """
    def __init__(self, output_root='adl-pose/outputs'):
        self.output_root = output_root
        os.makedirs(output_root, exist_ok=True)

    def save_skeleton_npy(self, skeletons, filename):
        """Save raw skeleton data in NPY format for research/training."""
        path = os.path.join(self.output_root, 'skeletons')
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, filename), skeletons)

    def save_benchmark_result(self, metrics, name):
        """Save evaluation metrics in JSON format."""
        path = os.path.join(self.output_root, 'benchmarks')
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f"{name}.json"), 'w') as f:
            json.dump(metrics, f, indent=4)

    def log_experiment(self, config, result):
        """Log research experiment details."""
        pass
