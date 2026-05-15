from pathlib import Path
import argparse
import shutil
import traceback

from ultralytics import YOLO


MODELS_DIR = Path(__file__).resolve().parent

# Only Ultralytics YOLO .pt models should be exported by this script.
YOLO_MODELS = [
    "yolo11n.pt",
    "yolo11n-pose.pt",
    "tracking.pt",
]

# These are not Ultralytics YOLO checkpoints.
SKIP_MODELS = [
    "posec3d_r50_ntu60.pth",
    "fastreid_market_R50.pth",
]


def export_one_model(
    model_path: Path,
    imgsz: int = 640,
    half: bool = False,
    dynamic: bool = False,
    force: bool = False,
) -> Path | None:
    """
    Export one Ultralytics YOLO .pt model to OpenVINO format.

    Output folder:
        models/<model_stem>_openvino_model/

    Example:
        yolo11n.pt
        -> yolo11n_openvino_model/
    """
    if not model_path.exists():
        print(f"[SKIP] Not found: {model_path}")
        return None

    if model_path.suffix.lower() != ".pt":
        print(f"[SKIP] Not a YOLO .pt file: {model_path.name}")
        return None

    out_dir = model_path.parent / f"{model_path.stem}_openvino_model"

    if out_dir.exists():
        if force:
            print(f"[INFO] Removing existing folder: {out_dir}")
            shutil.rmtree(out_dir)
        else:
            print(f"[SKIP] Already exists: {out_dir}")
            return out_dir

    print()
    print("=" * 80)
    print(f"[EXPORT] {model_path.name}")
    print(f"[INPUT ] {model_path}")
    print(f"[OUTPUT] {out_dir}")
    print("=" * 80)

    try:
        model = YOLO(str(model_path))

        exported_path = model.export(
            format="openvino",
            imgsz=imgsz,
            half=half,
            dynamic=dynamic,
        )

        exported_path = Path(exported_path)

        # Ultralytics usually exports to the same parent folder.
        # This block guarantees the final folder name is exactly:
        # <stem>_openvino_model
        if exported_path.resolve() != out_dir.resolve():
            if out_dir.exists():
                shutil.rmtree(out_dir)
            shutil.move(str(exported_path), str(out_dir))

        print(f"[DONE] OpenVINO model saved to: {out_dir}")

        xml_files = list(out_dir.glob("*.xml"))
        bin_files = list(out_dir.glob("*.bin"))

        if xml_files:
            print(f"[OK] XML: {xml_files[0].name}")
        else:
            print("[WARN] No .xml file found in output folder.")

        if bin_files:
            print(f"[OK] BIN: {bin_files[0].name}")
        else:
            print("[WARN] No .bin file found in output folder.")

        return out_dir

    except Exception as exc:
        print(f"[ERROR] Failed to export {model_path.name}: {exc}")
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Export Ultralytics YOLO .pt models to OpenVINO IR."
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", help="Use FP16 if supported. Default: FP32.")
    parser.add_argument("--dynamic", action="store_true", help="Export dynamic input shape.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing OpenVINO folders.")
    args = parser.parse_args()

    print(f"[MODELS_DIR] {MODELS_DIR}")

    exported = []

    for name in YOLO_MODELS:
        result = export_one_model(
            model_path=MODELS_DIR / name,
            imgsz=args.imgsz,
            half=args.half,
            dynamic=args.dynamic,
            force=args.force,
        )
        if result is not None:
            exported.append(result)

    print()
    print("=" * 80)
    print("[SKIPPED NON-ULTRALYTICS CHECKPOINTS]")
    print("=" * 80)

    for name in SKIP_MODELS:
        path = MODELS_DIR / name
        if path.exists():
            print(f"[SKIP] {name} cannot be exported by Ultralytics YOLO.export().")
        else:
            print(f"[SKIP] Not found: {name}")

    print()
    print("=" * 80)
    print("[SUMMARY]")
    print("=" * 80)

    if exported:
        for path in exported:
            print(f"[OK] {path}")
    else:
        print("[WARN] No model was exported.")

    print()
    print("Expected output folders:")
    print(f"  {MODELS_DIR / 'yolo11n_openvino_model'}")
    print(f"  {MODELS_DIR / 'yolo11n-pose_openvino_model'}")
    print(f"  {MODELS_DIR / 'tracking_openvino_model'}")


if __name__ == "__main__":
    main()