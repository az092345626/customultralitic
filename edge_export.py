#!/usr/bin/env python3
"""
Edge Deployment Export Script for YOLO-Edge Models

This script provides optimized export for various edge deployment targets:
- ONNX (with/without opset optimization)
- TensorRT (FP16/INT8)
- OpenVINO (FP16/INT8)
- CoreML
- TFLite (FP16/INT8)
- ONNX Runtime (with optimizations)

Example usage:
    # Export to ONNX
    python edge_export.py --weights yolo-edge-n.pt --format onnx --imgsz 640
    
    # Export to TensorRT INT8
    python edge_export.py --weights yolo-edge-n.pt --format engine --imgsz 640 --int8
    
    # Export to TFLite with INT8 quantization
    python edge_export.py --weights yolo-edge-n.pt --format tflite --imgsz 640 --int8 --calib-images 100
"""

import argparse
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn

# Add ultralytics to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.nn.modules import Detect_Edge, Detect_Small, Detect_Edge_End2End, RepGhostBlock, RepConv_Edge
from ultralytics.utils.torch_utils import fuse_conv_and_bn


def fuse_repgghost_blocks(model: nn.Module) -> nn.Module:
    """Fuse RepGhostBlock and RepConv_Edge for deployment.
    
    This converts multi-branch training structure to single-branch inference.
    """
    for m in model.model.modules():
        if isinstance(m, RepGhostBlock):
            m.fuse()
        elif isinstance(m, RepConv_Edge):
            m.switch_to_deploy()
    return model


def export_onnx(model, imgsz, half=False, simplify=True, opset=12):
    """Export to ONNX format with optimizations.
    
    Args:
        model: YOLO model
        imgsz: Input image size
        half: Use FP16
        simplify: Use ONNX simplifier
        opset: ONNX opset version
    """
    print(f"\n📦 Exporting to ONNX (opset={opset})...")
    
    f = model.export(
        format="onnx",
        imgsz=imgsz,
        half=half,
        simplify=simplify,
        opset=opset,
        dynamic=False,  # Static shapes for edge
    )
    print(f"✅ ONNX export complete: {f}")
    return f


def export_tensorrt(model, imgsz, half=False, int8=False, workspace=4, calib_images=None):
    """Export to TensorRT engine.
    
    Args:
        model: YOLO model
        imgsz: Input image size
        half: Use FP16
        int8: Use INT8 quantization
        workspace: Max workspace size in GB
        calib_images: Number of calibration images for INT8
    """
    print(f"\n🚀 Exporting to TensorRT (FP16={half}, INT8={int8})...")
    
    args = {
        "format": "engine",
        "imgsz": imgsz,
        "half": half,
        "int8": int8,
        "workspace": workspace,
        "dynamic": False,
    }
    
    if int8 and calib_images:
        args["data"] = "coco128.yaml"  # Default calibration dataset
        
    f = model.export(**args)
    print(f"✅ TensorRT export complete: {f}")
    return f


def export_openvino(model, imgsz, half=False, int8=False):
    """Export to OpenVINO IR format.
    
    Args:
        model: YOLO model
        imgsz: Input image size
        half: Use FP16
        int8: Use INT8 quantization
    """
    print(f"\n🔷 Exporting to OpenVINO (FP16={half}, INT8={int8})...")
    
    f = model.export(
        format="openvino",
        imgsz=imgsz,
        half=half,
        int8=int8,
        dynamic=False,
    )
    print(f"✅ OpenVINO export complete: {f}")
    return f


def export_tflite(model, imgsz, half=False, int8=False, calib_images=100):
    """Export to TensorFlow Lite.
    
    Args:
        model: YOLO model
        imgsz: Input image size
        half: Use FP16
        int8: Use INT8 quantization
        calib_images: Number of calibration images
    """
    print(f"\n📱 Exporting to TensorFlow Lite (FP16={half}, INT8={int8})...")
    
    args = {
        "format": "tflite",
        "imgsz": imgsz,
        "half": half,
        "int8": int8,
    }
    
    if int8:
        args["data"] = "coco128.yaml"
        
    f = model.export(**args)
    print(f"✅ TFLite export complete: {f}")
    return f


def export_coreml(model, imgsz, half=False, nms=True):
    """Export to CoreML format (for Apple devices).
    
    Args:
        model: YOLO model
        imgsz: Input image size
        half: Use FP16
        nms: Include NMS in model
    """
    print(f"\n🍎 Exporting to CoreML (FP16={half})...")
    
    f = model.export(
        format="coreml",
        imgsz=imgsz,
        half=half,
        nms=nms,
    )
    print(f"✅ CoreML export complete: {f}")
    return f


def optimize_for_edge(model, fuse_bn=True, reparameterize=True):
    """Apply edge optimization techniques.
    
    Args:
        model: YOLO model
        fuse_bn: Fuse Conv and BatchNorm layers
        reparameterize: Fuse RepGhost blocks
        
    Returns:
        Optimized model
    """
    print("\n⚡ Optimizing model for edge deployment...")
    
    if fuse_bn:
        print("  → Fusing Conv + BatchNorm layers...")
        model.fuse()
        
    if reparameterize:
        print("  → Reparameterizing RepGhost blocks...")
        model = fuse_repgghost_blocks(model)
        
    print("✅ Optimization complete")
    return model


def benchmark_model(model_path, imgsz, warmup=10, iterations=100):
    """Benchmark model inference speed.
    
    Args:
        model_path: Path to exported model
        imgsz: Input size
        warmup: Number of warmup iterations
        iterations: Number of benchmark iterations
    """
    import time
    
    print(f"\n⏱️  Benchmarking {model_path}...")
    
    # Load model based on extension
    suffix = Path(model_path).suffix.lower()
    
    if suffix == '.onnx':
        import onnxruntime as ort
        session = ort.InferenceSession(str(model_path), 
                                       providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        dummy_input = torch.randn(1, 3, imgsz, imgsz).numpy()
        
        # Warmup
        for _ in range(warmup):
            session.run(None, {input_name: dummy_input})
            
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            session.run(None, {input_name: dummy_input})
            times.append((time.perf_counter() - start) * 1000)
            
    elif suffix == '.engine':
        print("  TensorRT benchmark requires pycuda, skipping...")
        return
    else:
        # PyTorch model
        model = YOLO(model_path)
        dummy_input = torch.randn(1, 3, imgsz, imgsz)
        
        # Warmup
        for _ in range(warmup):
            model.predict(dummy_input, verbose=False)
            
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            model.predict(dummy_input, verbose=False)
            times.append((time.perf_counter() - start) * 1000)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"  Average: {avg_time:.2f}ms ({1000/avg_time:.1f} FPS)")
    print(f"  Min: {min_time:.2f}ms | Max: {max_time:.2f}ms")


def main():
    parser = argparse.ArgumentParser(description='Export YOLO-Edge models for edge deployment')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights (.pt)')
    parser.add_argument('--format', type=str, default='onnx', 
                        choices=['onnx', 'engine', 'openvino', 'tflite', 'coreml', 'all'],
                        help='Export format')
    parser.add_argument('--imgsz', type=int, nargs='+', default=[640], help='Image size')
    parser.add_argument('--half', action='store_true', help='FP16 quantization')
    parser.add_argument('--int8', action='store_true', help='INT8 quantization')
    parser.add_argument('--no-fuse', action='store_true', help='Skip Conv+BN fusion')
    parser.add_argument('--no-reparam', action='store_true', help='Skip reparameterization')
    parser.add_argument('--workspace', type=int, default=4, help='TensorRT workspace (GB)')
    parser.add_argument('--calib-images', type=int, default=100, help='Calibration images for INT8')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark after export')
    parser.add_argument('--simplify', action='store_true', help='Simplify ONNX model')
    parser.add_argument('--opset', type=int, default=12, help='ONNX opset version')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔧 YOLO-Edge Export Tool")
    print("=" * 60)
    print(f"Weights: {args.weights}")
    print(f"Format: {args.format}")
    print(f"Image size: {args.imgsz}")
    print(f"FP16: {args.half} | INT8: {args.int8}")
    print("=" * 60)
    
    # Load model
    print(f"\n📥 Loading model from {args.weights}...")
    model = YOLO(args.weights)
    
    # Apply edge optimizations
    if not args.no_fuse or not args.no_reparam:
        model = optimize_for_edge(
            model, 
            fuse_bn=not args.no_fuse,
            reparameterize=not args.no_reparam
        )
    
    # Export based on format
    exported_files = []
    
    if args.format == 'onnx' or args.format == 'all':
        f = export_onnx(model, args.imgsz, args.half, args.simplify, args.opset)
        exported_files.append(f)
        
    if args.format == 'engine' or args.format == 'all':
        f = export_tensorrt(model, args.imgsz, args.half, args.int8, 
                           args.workspace, args.calib_images)
        exported_files.append(f)
        
    if args.format == 'openvino' or args.format == 'all':
        f = export_openvino(model, args.imgsz, args.half, args.int8)
        exported_files.append(f)
        
    if args.format == 'tflite' or args.format == 'all':
        f = export_tflite(model, args.imgsz, args.half, args.int8, args.calib_images)
        exported_files.append(f)
        
    if args.format == 'coreml' or args.format == 'all':
        f = export_coreml(model, args.imgsz, args.half)
        exported_files.append(f)
    
    # Benchmark if requested
    if args.benchmark:
        for f in exported_files:
            if f and Path(f).exists():
                benchmark_model(f, args.imgsz[0] if isinstance(args.imgsz, list) else args.imgsz)
    
    print("\n" + "=" * 60)
    print("✅ Export complete!")
    print("=" * 60)
    
    return exported_files


if __name__ == '__main__':
    main()
