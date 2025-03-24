#!/bin/bash

# Directory containing .onnx files
ONNX_DIR="./promising_models_3/downscale_4"  # Change this to your folder path

# Directory for output .c files
OUTPUT_DIR="./converted_models_downscale4"   # Change this to your desired output path

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through all .onnx files in the input directory
for onnx_file in "$ONNX_DIR"/*.onnx; do
    # Extract filename without path and extension
    filename=$(basename "$onnx_file" .onnx)
    
    echo "Converting $filename.onnx to C..."
    
    # Run the docker command for each file
    sudo docker run --rm -v $(pwd):/data onnx2c-fixed "/data/$ONNX_DIR/$filename.onnx" > "$OUTPUT_DIR/$filename.c"
    
    echo "Conversion complete: $filename.c"
done

echo "All models converted successfully!"