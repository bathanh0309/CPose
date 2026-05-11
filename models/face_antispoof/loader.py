"""ONNX model loader with provider auto-detection."""

import onnxruntime as ort
from typing import Tuple, Optional
from pathlib import Path
import os

def load_model(model_path: str) -> Tuple[Optional[ort.InferenceSession], Optional[str]]:
    """Load ONNX model. Return (session, input_name) or (None, None) on failure."""
    print("🔍 Loading model from:", model_path)
    
    model_path = Path(model_path)
    print(os.path.getsize(str(model_path)))
    
    if not Path(model_path).exists():
        print("❌ Model file NOT FOUND")
        print("Resolved path:", model_path.resolve())
        return None, None
    
    print("✅ Model file exists")
    
    try:
        
        # ===== Debug providers =====
        available_providers = ort.get_available_providers()
        print("Available providers:", available_providers)

        preferred_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in preferred_providers if p in available_providers]

        if not providers:
            print("⚠️ No preferred providers → using all available")
            providers = available_providers

        print("Using providers:", providers)
        
        # ==== Session options ====
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # ==== Load model ====
        available_providers = ort.get_available_providers()
        preferred_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in preferred_providers if p in available_providers]

        if not providers:
            providers = available_providers

        ort_session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options, 
            providers=providers
        )

        print("✅ Model loaded successfully")

        # ==== Debug input name ====
        inputs = ort_session.get_inputs()
        if not inputs:
            print("❌ Model has NO inputs")
            return None, None
        
        input_name = inputs[0].name
        
        return ort_session, input_name

    except Exception as e:
        print("🔥 LOAD MODEL ERROR:", e)
        return None, None