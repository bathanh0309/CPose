from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OSNet on Market-1501 with torchreid")
    parser.add_argument("--root", default="dataset")
    parser.add_argument("--model", default="osnet_x0_25")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--batch-size-test", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    import torch
    import torchreid

    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())

    datamanager = torchreid.data.ImageDataManager(
        root=args.root,
        sources="market1501",
        targets="market1501",
        height=args.height,
        width=args.width,
        batch_size_test=args.batch_size_test,
        workers=args.workers,
    )

    model = torchreid.models.build_model(
        name=args.model,
        num_classes=datamanager.num_train_pids,
        pretrained=True,
    )
    model = model.cuda() if use_cuda else model.cpu()

    engine = torchreid.engine.ImageSoftmaxEngine(datamanager, model, optimizer=None)
    engine.run(test_only=True)


if __name__ == "__main__":
    main()
