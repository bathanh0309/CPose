# Global ReID Models

Place OSNet ONNX files here:

- `osnet_x0_25_market.onnx`
- `osnet_x1_0_market.onnx`

`src.core.body_embedder.BodyEmbedder` uses these ONNX files when present. If they are missing, it attempts to load a pretrained OSNet model through `torchreid`.

Current exported files were generated from `torchreid` ImageNet-pretrained OSNet weights. Replace them with Market-1501 fine-tuned ONNX files when you need true Market-1501 Rank-1/mAP evaluation.
