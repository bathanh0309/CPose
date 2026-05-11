from __future__ import annotations

import argparse
from pathlib import Path


def export_osnet(model_name: str, output: Path) -> Path:
    import torch
    import torchreid

    output.parent.mkdir(parents=True, exist_ok=True)
    model = torchreid.models.build_model(model_name, num_classes=0, pretrained=True)
    model.eval()
    dummy = torch.randn(1, 3, 256, 128)
    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=18,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Export torchreid OSNet to ONNX")
    parser.add_argument("--model", default="osnet_x0_25", choices=["osnet_x0_25", "osnet_x1_0"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    output = Path(args.output or f"models/global_reid/{args.model}_market.onnx")
    print(f"Exported: {export_osnet(args.model, output)}")


if __name__ == "__main__":
    main()
