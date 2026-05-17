import cv2
import numpy as np
import onnxruntime as ort
import time

class RTMPoseONNX:
    def __init__(self, model_path: str):
        self.input_size = (192, 256) # W, H
        providers = ['CPUExecutionProvider']
        so = ort.SessionOptions()
        so.inter_op_num_threads = 2
        so.intra_op_num_threads = 4
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(model_path, so, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        
        print("[POSE] RTMPose ONNX loaded")
        print(f"[POSE] provider={providers[0]}")
        print(f"[POSE] input={self.input_size[1]}x{self.input_size[0]}")
        print("[POSE] ready")

    def infer(self, person_crop: np.ndarray):
        h, w = person_crop.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((17, 2), dtype=np.float32), np.zeros(17, dtype=np.float32)
            
        target_w, target_h = self.input_size
        
        scale_x = target_w / w
        scale_y = target_h / h
        scale = min(scale_x, scale_y)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(person_crop, (new_w, new_h))
        
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        
        padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(128,128,128))
        
        img = padded[:, :, ::-1].astype(np.float32)
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        img = (img - mean) / std
        
        tensor = img.transpose(2, 0, 1)[None]
        
        outputs = self.session.run(None, {self.input_name: tensor})
        out_kpts = outputs[0][0]
        
        if len(outputs) > 1:
            out_scores = outputs[1][0]
        else:
            if out_kpts.shape[-1] == 3:
                out_scores = out_kpts[:, 2]
                out_kpts = out_kpts[:, :2]
            else:
                out_scores = np.ones(17, dtype=np.float32)
                
        kpts_x = (out_kpts[:, 0] - pad_left) / scale
        kpts_y = (out_kpts[:, 1] - pad_top) / scale
        
        kpts = np.stack([kpts_x, kpts_y], axis=1)
        return kpts, out_scores
