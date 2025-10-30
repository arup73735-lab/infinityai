"""
Convert trained model to ONNX format for optimized inference.

ONNX provides:
- Cross-platform compatibility
- Hardware-specific optimizations
- Reduced inference latency
- Smaller model size with quantization
"""

import os
import logging
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
from optimum.onnxruntime.configuration import OptimizationConfig, AutoQuantizationConfig
import click

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--model-path', required=True, help='Path to trained model')
@click.option('--output-path', required=True, help='Output path for ONNX model')
@click.option('--quantize', is_flag=True, help='Apply dynamic quantization')
@click.option('--optimize', is_flag=True, help='Apply graph optimizations')
def convert_to_onnx(
    model_path: str,
    output_path: str,
    quantize: bool,
    optimize: bool
):
    """
    Convert PyTorch model to ONNX format.
    
    Args:
        model_path: Path to trained PyTorch model
        output_path: Output directory for ONNX model
        quantize: Whether to apply quantization
        optimize: Whether to apply graph optimizations
    """
    logger.info(f"Loading model from {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # ONNX export requires float32
    )
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Export to ONNX
    logger.info("Exporting to ONNX format")
    
    ort_model = ORTModelForCausalLM.from_pretrained(
        model_path,
        export=True,
    )
    
    # Apply optimizations
    if optimize:
        logger.info("Applying graph optimizations")
        optimization_config = OptimizationConfig(
            optimization_level=2,  # 0=disable, 1=basic, 2=extended, 99=all
        )
        ort_model = ort_model.optimize(optimization_config)
    
    # Apply quantization
    if quantize:
        logger.info("Applying dynamic quantization")
        quantization_config = AutoQuantizationConfig.avx512_vnni(
            is_static=False,
            per_channel=True,
        )
        ort_model = ort_model.quantize(quantization_config)
    
    # Save ONNX model
    logger.info(f"Saving ONNX model to {output_path}")
    ort_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Test inference
    logger.info("Testing ONNX model inference")
    test_input = "Hello, this is a test"
    inputs = tokenizer(test_input, return_tensors="pt")
    
    with torch.no_grad():
        outputs = ort_model.generate(**inputs, max_new_tokens=20)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Test output: {generated_text}")
    
    # Get model size
    onnx_file = Path(output_path) / "model.onnx"
    if onnx_file.exists():
        size_mb = onnx_file.stat().st_size / (1024 * 1024)
        logger.info(f"ONNX model size: {size_mb:.2f} MB")
    
    logger.info("Conversion complete!")


@click.command()
@click.option('--model-path', required=True, help='Path to ONNX model')
@click.option('--prompt', default="Hello, how are you?", help='Test prompt')
def test_onnx_model(model_path: str, prompt: str):
    """Test ONNX model inference."""
    logger.info(f"Loading ONNX model from {model_path}")
    
    # Load model and tokenizer
    model = ORTModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Run inference
    logger.info(f"Running inference with prompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt")
    
    import time
    start = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    
    latency = time.time() - start
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    logger.info(f"Generated text: {generated_text}")
    logger.info(f"Latency: {latency:.3f}s")
    logger.info(f"Tokens: {len(outputs[0])}")
    logger.info(f"Tokens/sec: {len(outputs[0])/latency:.1f}")


@click.group()
def cli():
    """ONNX conversion and testing CLI."""
    pass


cli.add_command(convert_to_onnx, name='convert')
cli.add_command(test_onnx_model, name='test')


if __name__ == '__main__':
    cli()
