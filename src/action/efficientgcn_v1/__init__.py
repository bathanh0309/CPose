"""Local EfficientGCNv1 model definition for NTU120 checkpoints."""

import math

from torch import nn

from .activations import HardSwish, Swish
from .nets import EfficientGCN


_ACTIVATIONS = {
    "relu": nn.ReLU(inplace=True),
    "relu6": nn.ReLU6(inplace=True),
    "hswish": HardSwish(inplace=True),
    "swish": Swish(inplace=True),
}


def _rescale_block(block_args, scale_args, scale_factor):
    channel_scaler = math.pow(scale_args[0], scale_factor)
    depth_scaler = math.pow(scale_args[1], scale_factor)
    new_block_args = []
    for channel, stride, depth in block_args:
        channel = max(int(round(channel * channel_scaler / 16)) * 16, 16)
        depth = int(round(depth * depth_scaler))
        new_block_args.append([channel, stride, depth])
    return new_block_args


def create_efficientgcn_b0(data_shape, num_class, A, parts):
    model_args = {
        "stem_channel": 64,
        "block_args": [[48, 1, 0.5], [24, 1, 0.5], [64, 2, 1], [128, 2, 1]],
        "fusion_stage": 2,
        "act_type": "swish",
        "att_type": "stja",
        "layer_type": "Sep",
        "drop_prob": 0.25,
        "kernel_size": [5, 2],
        "scale_args": [1.2, 1.35],
        "expand_ratio": 2,
        "reduct_ratio": 4,
        "bias": True,
        "edge": True,
    }
    model_args["act"] = _ACTIVATIONS[model_args.pop("act_type")]
    model_args["block_args"] = _rescale_block(
        model_args["block_args"],
        model_args.pop("scale_args"),
        0,
    )
    return EfficientGCN(
        data_shape=data_shape,
        num_class=num_class,
        A=A,
        parts=parts,
        **model_args,
    )
