import torch
import onnxruntime

def export_to_onnx(model, input_shape, output_path):
    """
    Export PyTorch model to ONNX format
    
    Args:
        model: PyTorch model
        input_shape: Tuple of input dimensions (batch_size, channels, height, width)
        output_path: Path to save the ONNX model
    
    Returns:
        Path to the exported ONNX model
    """
    model.eval()
    
    # Create dummy input tensor
    dummy_input = torch.randn(input_shape)
    
    # Export the model to ONNX
    torch.onnx.export(
        model,                     # Model being run
        dummy_input,               # Model input
        output_path,               # Output file
        export_params=True,        # Store the trained parameter weights
        opset_version=12,          # ONNX version
        do_constant_folding=True,  # Optimize for inference
        input_names=['input'],     # Model's input names
        output_names=['output'],   # Model's output names
        dynamic_axes={             # Variable length axes
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {output_path}")
    return output_path

def verify_onnx_model(onnx_path, model, sample_input):
    """
    Verify that the ONNX model produces the same output as the PyTorch model
    
    Args:
        onnx_path: Path to the ONNX model
        model: PyTorch model
        sample_input: Sample input tensor
    
    Returns:
        bool: True if verification passes
    """

    
    # Get PyTorch output
    model.eval()
    with torch.no_grad():
        pytorch_output = model(sample_input)
    
    # Get ONNX output
    ort_session = onnxruntime.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: sample_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    onnx_output = torch.from_numpy(ort_outputs[0])
    
    # Compare outputs
    max_diff = torch.max(torch.abs(pytorch_output - onnx_output)).item()
    print(f"Maximum difference between PyTorch and ONNX outputs: {max_diff}")
    
    if max_diff < 1e-5:
        print("✅ ONNX model matches PyTorch model")
        return True
    else:
        print(f"❌ ONNX model output differs. Max difference: {max_diff}")
        return False