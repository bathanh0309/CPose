import numpy as np
import onnxruntime as ort
import sys
from typing import List, Dict
from .preprocess import preprocess_batch

def process_with_logits(raw_logits: np.ndarray, threshold: float) -> Dict:
    """Convert raw logits to real/spoof classification"""
    real_logit = float(raw_logits[0])
    spoof_logit = float(raw_logits[1])
    
    # === Soft max ===
    logits = np.array([real_logit, spoof_logit])
    probs = np.exp(logits) / np.sum(np.exp(logits))
    
    prob_real = float(probs[0])
    prob_spoof = float(probs[1])
    
    confidence = abs(prob_real - prob_spoof)    
    is_real = confidence >= threshold
    
    return {
        "is_real": bool(is_real),
        "status": "real" if is_real else "spoof",
        
        # probability used for voting
        "prob_real": float(prob_real),
        "prob_spoof": float(prob_spoof),
        
        # log used for debug
        "real_logit": float(real_logit),
        "spof_logit": float(spoof_logit),
        
        "confidence": float(confidence),
    }
    

def infer(
            face_crops: List[np.ndarray],
            ort_session: ort.InferenceSession,
            input_name: str,
            model_img_size: int,
        ) -> List[np.ndarray]:
    
    """Run batch inference on cropped face images. Return list of logits per face."""
    if not face_crops or ort_session is None:
        raise ValueError("Invalid input to infer")

    batch_input = preprocess_batch(face_crops, model_img_size)
    
    logits = ort_session.run([], {input_name: batch_input})
    
    if logits is None or len(logits) == 0:
        raise RuntimeError("Model return empty output")
    
    logits = logits[0]
    
    # === fix shape ===
    logits = np.squeeze(logits)
    
    if logits.ndim == 1:
        logits = np.expand_dims(logits, axis = 0)
        
    if logits.shape[0] != len(face_crops):
        print("⚠️ Shape mismatch → fixing")
        logits = np.tile(logits[0], (len(face_crops), 1))
        
    return [logits[i] for i in range(len(face_crops))]