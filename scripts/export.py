#!/usr/bin/env python
from pathlib import Path

import hydra
import onnx
import tensorrt as trt
import torch
from omegaconf import DictConfig

from brain_tumor_segmentation.models.segresnet import get_model


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model
    model = get_model(cfg).to(device)
    checkpoint_path = Path(cfg.checkpoint_dir) / "model.pth"
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # Create dummy input for export
    dummy_input = torch.randn(1, cfg.in_channels, *cfg.roi_size).to(device)

    # Export to ONNX
    onnx_path = Path(cfg.checkpoint_dir) / "model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"Model exported to ONNX at {onnx_path}")

    # Verify ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Export to TensorRT
    trt_path = Path(cfg.checkpoint_dir) / "model.trt"
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    ) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_batch_size = 1
        builder.max_workspace_size = 1 << 30  # 1GB
        with open(onnx_path, "rb") as model:
            parser.parse(model.read())
        engine = builder.build_cuda_engine(network)
        with open(trt_path, "wb") as f:
            f.write(engine.serialize())
    print(f"Model exported to TensorRT at {trt_path}")


if __name__ == "__main__":
    main()
