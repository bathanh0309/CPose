import os
import sys
import time
import argparse
import logging
import json
from pathlib import Path
import numpy as np
import cv2
from typing import Dict, Any, List, Optional
from collections import Counter

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics (YOLO) not installed.")
    sys.exit(1)

# try to import sklearn for metrics, fallback if not available
try:
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("AdvancedBenchmark")

class ADLEvaluator:
    def __init__(self):
        self.y_true = []
        self.y_pred = []
        self.confidences = []
        self.frame_results = []
        self.label_timeline = []

    def add_entry(self, pred_label: str, true_label: Optional[str], confidence: float):
        self.y_pred.append(pred_label)
        self.confidences.append(confidence)
        self.label_timeline.append(pred_label)
        if true_label:
            self.y_true.append(true_label)

    def calculate_metrics(self) -> Dict[str, Any]:
        results = {}
        
        # Performance/System Metrics
        if self.label_timeline:
            switches = sum(1 for i in range(1, len(self.label_timeline)) if self.label_timeline[i] != self.label_timeline[i-1])
            results["Label Stability (Switches)"] = switches
            results["Stability Index"] = f"{1.0 - (switches / len(self.label_timeline)):.2%}"

        # ML Metrics (if Ground Truth is available)
        if self.y_true and len(self.y_true) == len(self.y_pred):
            if HAS_SKLEARN:
                results["Accuracy"] = f"{accuracy_score(self.y_true, self.y_pred):.4f}"
                results["F1-Score (Macro)"] = f"{f1_score(self.y_true, self.y_pred, average='macro', zero_division=0):.4f}"
                results["Precision (Macro)"] = f"{precision_score(self.y_true, self.y_pred, average='macro', zero_division=0):.4f}"
                results["Recall (Macro)"] = f"{recall_score(self.y_true, self.y_pred, average='macro', zero_division=0):.4f}"
            else:
                correct = sum(1 for gt, pd in zip(self.y_true, self.y_pred) if gt == pd)
                results["Accuracy (Manual)"] = f"{correct / len(self.y_true):.4f}"
        
        results["Avg Confidence"] = f"{np.mean(self.confidences):.2%}"
        return results

    def print_confusion_matrix(self):
        if not self.y_true or not HAS_SKLEARN:
            return
        
        labels = sorted(list(set(self.y_true + self.y_pred)))
        cm = confusion_matrix(self.y_true, self.y_pred, labels=labels)
        
        print("\n" + "-"*60)
        print("CONFUSION MATRIX")
        print("-"*60)
        header = " " * 12 + " ".join([f"{l[:8]:>8}" for l in labels])
        print(header)
        for i, row in enumerate(cm):
            row_str = f"{labels[i][:10]:<10} | " + " ".join([f"{count:>8}" for count in row])
            print(row_str)
        print("-"*60)

    def print_classification_report(self):
        if not self.y_true or not HAS_SKLEARN:
            return
        print("\nDETAILED CLASSIFICATION REPORT")
        print(classification_report(self.y_true, self.y_pred, zero_division=0))

def run_research_benchmark(model_path: str, source: str, gt_path: Optional[str] = None, device: str = "cpu"):
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return

    # Load Ground Truth if available
    gt_data = {}
    if gt_path and os.path.exists(gt_path):
        try:
            with open(gt_path, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)
            logger.info(f"Loaded {len(gt_data)} ground truth labels from {gt_path}")
        except Exception as e:
            logger.error(f"Error loading Ground Truth: {e}")

    logger.info(f"Research Setup: Model={Path(model_path).name}, Source={Path(source).name}, Device={device}")
    
    # Load Model
    start_load = time.time()
    model = YOLO(model_path).to(device)
    load_time = time.time() - start_load
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open: {source}")
        return

    evaluator = ADLEvaluator()
    inf_times = []
    frame_count = 0
    
    start_bench = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        t1 = time.time()
        # In research mode, we often process pose to get ADL
        results = model.predict(frame, verbose=False)
        t2 = time.time()
        
        inf_times.append((t2 - t1) * 1000)
        
        # Extract pseudo-ADL (for benchmark demo)
        # In a real pipeline, we'd call the CTR-GCN engine here.
        # Here we simulate/extract dominant class if it's a pose model
        top_label = "unknown"
        confidence = 0.0
        
        if results and len(results[0].boxes) > 0:
            # Placeholder: getting class name from YOLO results
            # If it's a pose model, it detects the 'person' class (id 0)
            res = results[0]
            if hasattr(res, 'keypoints') and res.keypoints is not None:
                # Simulated ADL logic based on pose height (very simple baseline)
                # In production this uses CTR-GCN
                kpts = res.keypoints.data[0].cpu().numpy() # [17, 3]
                head_y = kpts[0, 1]
                ankle_y = np.mean(kpts[15:17, 1])
                height = ankle_y - head_y
                
                if height < 100: top_label = "falling"
                elif height < 200: top_label = "sitting"
                else: top_label = "standing"
                confidence = float(res.boxes.conf[0])
            else:
                top_label = model.names[int(res.boxes.cls[0])]
                confidence = float(res.boxes.conf[0])

        # Match with Ground Truth
        # Format can be "frame_0", "frame_1" or just index
        gt_label = gt_data.get(f"frame_{frame_count}") or gt_data.get(str(frame_count)) or gt_data.get("all")
        
        evaluator.add_entry(top_label, gt_label, confidence)
        frame_count += 1
        
        if frame_count % 50 == 0:
            logger.info(f"Progress: {frame_count} frames | Avg Latency: {np.mean(inf_times):.2f}ms")

    total_time = time.time() - start_bench
    cap.release()

    # Final Stats
    metrics = evaluator.calculate_metrics()
    
    print("\n" + "="*50)
    print("      SCIENTIFIC RESEARCH BENCHMARK REPORT")
    print("="*50)
    print(f"Model             : {Path(model_path).name}")
    print(f"Total Frames      : {frame_count}")
    print(f"Inference Time Avg: {np.mean(inf_times):.2f} ms")
    print(f"Total Throughput  : {frame_count / total_time:.2f} FPS")
    print("-" * 50)
    for k, v in metrics.items():
        print(f"{k:<20}: {v:>25}")
    print("="*50)

    if gt_data:
        evaluator.print_classification_report()
        evaluator.print_confusion_matrix()
    else:
        print("\n[NOTE] Ground Truth file not provided. Detailed Accuracy/F1 metrics unavailable.")
        print("To enable full metrics, prepare a JSON file with frame labels and use --gt <path>.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced ADL Research Benchmark CLI")
    parser.add_argument("--model", type=str, default="models/product/yolov8n-pose.pt", help="Model path (.pt) - defaults to yolov8n-pose.pt")
    parser.add_argument("--source", type=str, help="Video path (optional, will auto-select if omitted)")
    parser.add_argument("--gt", type=str, help="Ground Truth JSON file")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    
    args = parser.parse_args()

    # Priority-based auto-selection
    if not args.source or not os.path.exists(args.source):
        raw_dir = Path("data/raw_videos")
        multi_dir = Path("data/multicam")
        
        raw_clips = sorted(list(raw_dir.glob("*.mp4"))) if raw_dir.exists() else []
        multi_clips = sorted(list(multi_dir.glob("*.mp4"))) if multi_dir.exists() else []
        
        if raw_clips:
            args.source = str(raw_clips[0])
            logger.info(f"[MODE: RAW] Auto-selected: {args.source}")
        elif multi_clips:
            args.source = str(multi_clips[0])
            logger.info(f"[MODE: MULTI] Auto-selected: {args.source}")
        else:
            logger.error("No source video found in raw_videos or multicam.")
            sys.exit(1)
    else:
        # User specified a source, detect type
        path_str = str(Path(args.source).resolve())
        if "raw_videos" in path_str:
            logger.info(f"[MODE: RAW] Using user-specified raw video: {args.source}")
        elif "multicam" in path_str:
            logger.info(f"[MODE: MULTI] Using user-specified multi video: {args.source}")
        else:
            logger.info(f"[MODE: EXTERNAL] Using video: {args.source}")

    run_research_benchmark(args.model, args.source, args.gt, args.device)
