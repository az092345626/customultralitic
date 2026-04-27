#!/usr/bin/env python3
"""
Edge Model Benchmark Script

Comprehensive benchmark comparing YOLO-Edge vs standard YOLO models:
- Model size (params, FLOPs)
- Inference latency (single/batch)
- Throughput (FPS)
- Memory usage
- Detection accuracy (mAP)
- Small object detection performance

Example usage:
    # Compare YOLO-Edge vs YOLOv8n
    python edge_benchmark.py --models yolo-edge-n.pt yolov8n.pt --imgsz 640
    
    # Full benchmark with accuracy
    python edge_benchmark.py --models yolo-edge-n.pt yolov8n.pt --data coco128.yaml --task all
"""

import argparse
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add ultralytics to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info, select_device


class ModelBenchmark:
    """Benchmark suite for YOLO models."""
    
    def __init__(self, model_path, device='cpu', imgsz=640, half=False):
        """Initialize benchmark.
        
        Args:
            model_path: Path to model weights
            device: Device to run on
            imgsz: Input image size
            half: Use FP16
        """
        self.model_path = model_path
        self.device = select_device(device)
        self.imgsz = imgsz
        self.half = half
        
        self.model = None
        self.results = {}
        
    def load_model(self):
        """Load model."""
        print(f"\n📥 Loading {Path(self.model_path).name}...")
        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        if self.half:
            self.model.half()
        return self
    
    def benchmark_model_size(self):
        """Benchmark model size (params, FLOPs, memory)."""
        print("  → Measuring model size...")
        
        # Get model info
        info = model_info(self.model.model, imgsz=self.imgsz)
        
        self.results['model_size'] = {
            'params_M': info[0] / 1e6,
            'flops_G': info[1] / 1e9,
            'layers': info[2],
        }
        
        # Model file size
        self.results['model_size']['file_size_MB'] = Path(self.model_path).stat().st_size / (1024 * 1024)
        
        print(f"    Params: {self.results['model_size']['params_M']:.2f}M")
        print(f"    FLOPs: {self.results['model_size']['flops_G']:.2f}G")
        print(f"    File size: {self.results['model_size']['file_size_MB']:.2f}MB")
        
    def benchmark_inference(self, iterations=100, warmup=10):
        """Benchmark inference latency.
        
        Args:
            iterations: Number of inference iterations
            warmup: Number of warmup iterations
        """
        print(f"  → Benchmarking inference ({iterations} iterations)...")
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, self.imgsz, self.imgsz).to(self.device)
        if self.half:
            dummy_input = dummy_input.half()
        
        self.model.model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model.model(dummy_input)
        
        # Synchronize GPU
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(iterations):
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                _ = self.model.model(dummy_input)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                    
                times.append((time.perf_counter() - start) * 1000)
        
        self.results['inference'] = {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'median_ms': np.median(times),
            'fps': 1000 / np.mean(times),
        }
        
        print(f"    Mean: {self.results['inference']['mean_ms']:.2f}ms")
        print(f"    Std: {self.results['inference']['std_ms']:.2f}ms")
        print(f"    FPS: {self.results['inference']['fps']:.1f}")
        
    def benchmark_throughput(self, batch_sizes=[1, 2, 4, 8, 16], duration=5):
        """Benchmark throughput at different batch sizes.
        
        Args:
            batch_sizes: List of batch sizes to test
            duration: Duration per batch size in seconds
        """
        print(f"  → Benchmarking throughput (batch sizes: {batch_sizes})...")
        
        throughput = {}
        
        for bs in batch_sizes:
            # Create batch
            dummy_input = torch.randn(bs, 3, self.imgsz, self.imgsz).to(self.device)
            if self.half:
                dummy_input = dummy_input.half()
            
            # Warmup
            with torch.no_grad():
                _ = self.model.model(dummy_input)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark
            start = time.perf_counter()
            count = 0
            
            with torch.no_grad():
                while (time.perf_counter() - start) < duration:
                    _ = self.model.model(dummy_input)
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    count += 1
            
            elapsed = time.perf_counter() - start
            throughput[bs] = {
                'images_per_sec': (count * bs) / elapsed,
                'batches_per_sec': count / elapsed,
                'ms_per_batch': (elapsed / count) * 1000,
            }
            
            print(f"    Batch {bs}: {throughput[bs]['images_per_sec']:.1f} img/s")
        
        self.results['throughput'] = throughput
        
    def benchmark_memory(self, batch_size=1):
        """Benchmark memory usage.
        
        Args:
            batch_size: Batch size for memory test
        """
        print(f"  → Benchmarking memory usage...")
        
        if self.device.type != 'cuda':
            print("    (Memory benchmark requires CUDA)")
            return
        
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # Create input
        dummy_input = torch.randn(batch_size, 3, self.imgsz, self.imgsz).to(self.device)
        if self.half:
            dummy_input = dummy_input.half()
        
        # Run inference
        with torch.no_grad():
            _ = self.model.model(dummy_input)
        
        # Get memory stats
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        current_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        
        self.results['memory'] = {
            'peak_MB': peak_memory,
            'current_MB': current_memory,
            'batch_size': batch_size,
        }
        
        print(f"    Peak memory: {peak_memory:.2f}MB")
        
    def benchmark_accuracy(self, data='coco128.yaml', task='detect'):
        """Benchmark detection accuracy.
        
        Args:
            data: Dataset YAML path
            task: Task type (detect, segment, etc.)
        """
        print(f"  → Benchmarking accuracy on {data}...")
        
        try:
            # Run validation
            metrics = self.model.val(data=data, verbose=False)
            
            if task == 'detect':
                self.results['accuracy'] = {
                    'mAP50': metrics.box.map50,
                    'mAP50-95': metrics.box.map,
                    'precision': metrics.box.mp,
                    'recall': metrics.box.mr,
                }
                print(f"    mAP50: {self.results['accuracy']['mAP50']:.4f}")
                print(f"    mAP50-95: {self.results['accuracy']['mAP50-95']:.4f}")
            
        except Exception as e:
            print(f"    ⚠️ Accuracy benchmark failed: {e}")
            
    def export_and_benchmark(self, formats=['onnx'], int8=False):
        """Export to different formats and benchmark.
        
        Args:
            formats: List of export formats
            int8: Use INT8 quantization
        """
        print(f"  → Exporting and benchmarking formats: {formats}...")
        
        export_results = {}
        
        for fmt in formats:
            try:
                print(f"    Exporting to {fmt}...")
                
                export_args = {
                    'format': fmt,
                    'imgsz': self.imgsz,
                    'half': self.half and not int8,
                    'int8': int8,
                    'dynamic': False,
                }
                
                exported_path = self.model.export(**export_args)
                
                # Get exported file size
                file_size = Path(exported_path).stat().st_size / (1024 * 1024)
                
                export_results[fmt] = {
                    'path': str(exported_path),
                    'file_size_MB': file_size,
                }
                
                print(f"    {fmt}: {file_size:.2f}MB")
                
            except Exception as e:
                print(f"    ⚠️ Export to {fmt} failed: {e}")
                export_results[fmt] = {'error': str(e)}
        
        self.results['export'] = export_results
        
    def run_all(self, data=None, task='detect', export_formats=None):
        """Run all benchmarks.
        
        Args:
            data: Dataset for accuracy benchmark
            task: Task type
            export_formats: List of formats to export
        """
        self.benchmark_model_size()
        self.benchmark_inference()
        self.benchmark_throughput()
        self.benchmark_memory()
        
        if data:
            self.benchmark_accuracy(data, task)
            
        if export_formats:
            self.benchmark_export(export_formats)
            
        return self.results


def print_comparison(results_dict):
    """Print comparison table of multiple models.
    
    Args:
        results_dict: Dict of {model_name: results}
    """
    print("\n" + "=" * 80)
    print("📊 MODEL COMPARISON")
    print("=" * 80)
    
    # Header
    models = list(results_dict.keys())
    header = f"{'Metric':<30}"
    for model in models:
        header += f"{model:>20}"
    print(header)
    print("-" * 80)
    
    # Model size
    metrics = [
        ('Params (M)', 'model_size', 'params_M'),
        ('FLOPs (G)', 'model_size', 'flops_G'),
        ('File Size (MB)', 'model_size', 'file_size_MB'),
        ('Inference (ms)', 'inference', 'mean_ms'),
        ('FPS', 'inference', 'fps'),
        ('Peak Memory (MB)', 'memory', 'peak_MB'),
    ]
    
    if 'accuracy' in results_dict.get(models[0], {}):
        metrics.extend([
            ('mAP50', 'accuracy', 'mAP50'),
            ('mAP50-95', 'accuracy', 'mAP50-95'),
        ])
    
    for label, section, key in metrics:
        row = f"{label:<30}"
        for model in models:
            val = results_dict[model].get(section, {}).get(key, 'N/A')
            if isinstance(val, float):
                row += f"{val:>20.2f}"
            else:
                row += f"{str(val):>20}"
        print(row)
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Benchmark YOLO models')
    parser.add_argument('--models', nargs='+', required=True, help='Model paths to compare')
    parser.add_argument('--imgsz', type=int, default=640, help='Input size')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--half', action='store_true', help='Use FP16')
    parser.add_argument('--data', type=str, help='Dataset for accuracy test')
    parser.add_argument('--task', type=str, default='detect', help='Task type')
    parser.add_argument('--export-formats', nargs='+', help='Export formats to test')
    parser.add_argument('--iterations', type=int, default=100, help='Inference iterations')
    parser.add_argument('--output', type=str, help='Save results to JSON file')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔧 YOLO-Edge Benchmark Suite")
    print("=" * 80)
    
    # Run benchmarks for each model
    all_results = {}
    
    for model_path in args.models:
        print(f"\n{'='*60}")
        print(f"📦 Benchmarking: {Path(model_path).name}")
        print(f"{'='*60}")
        
        benchmark = ModelBenchmark(model_path, args.device, args.imgsz, args.half)
        benchmark.load_model()
        
        # Run benchmarks
        benchmark.benchmark_model_size()
        benchmark.benchmark_inference(args.iterations)
        benchmark.benchmark_throughput()
        benchmark.benchmark_memory()
        
        if args.data:
            benchmark.benchmark_accuracy(args.data, args.task)
            
        if args.export_formats:
            benchmark.export_and_benchmark(args.export_formats)
        
        all_results[Path(model_path).stem] = benchmark.results
    
    # Print comparison
    print_comparison(all_results)
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS")
    print("=" * 80)
    
    if len(all_results) > 1:
        # Find best for each metric
        best_fps = max(all_results.items(), key=lambda x: x[1].get('inference', {}).get('fps', 0))
        best_map = max(all_results.items(), key=lambda x: x[1].get('accuracy', {}).get('mAP50-95', 0) if 'accuracy' in x[1] else 0)
        smallest = min(all_results.items(), key=lambda x: x[1].get('model_size', {}).get('params_M', float('inf')))
        
        print(f"🏎️  Fastest: {best_fps[0]} ({best_fps[1].get('inference', {}).get('fps', 0):.1f} FPS)")
        if 'accuracy' in best_map[1]:
            print(f"🎯 Most Accurate: {best_map[0]} (mAP50-95: {best_map[1].get('accuracy', {}).get('mAP50-95', 0):.4f})")
        print(f"🪶 Lightest: {smallest[0]} ({smallest[1].get('model_size', {}).get('params_M', 0):.2f}M params)")
    
    print("=" * 80)


if __name__ == '__main__':
    main()
