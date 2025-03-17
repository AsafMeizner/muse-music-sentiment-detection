#!/usr/bin/env python
import os
from pathlib import Path
import tensorflow as tf
import tf2onnx

def convert_model(keras_model_path: Path, onnx_model_path: Path, custom_objects=None):
    """
    Load a Keras model from keras_model_path and convert it to ONNX format,
    saving the result at onnx_model_path.
    """
    if not keras_model_path.exists():
        print(f"ERROR: Keras model file not found: {keras_model_path}")
        return

    print(f"Loading model from {keras_model_path} ...")
    model = tf.keras.models.load_model(str(keras_model_path), custom_objects=custom_objects)
    
    # Create an input signature from the model's first input shape.
    spec = (tf.TensorSpec(model.inputs[0].shape, tf.float32, name="input"),)
    
    print("Converting model to ONNX format ...")
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=str(onnx_model_path))
    print(f"ONNX model saved to {onnx_model_path}\n")

if __name__ == "__main__":
    # Get the directory containing this script.
    base_dir = Path(__file__).parent.absolute()
    
    # Build absolute paths using pathlib.
    sentiment_keras_path = base_dir / "setiment-results" / "best_multiseg_vgg.h5"
    genre_keras_path = base_dir / "genre_results" / "best_model.keras"
    
    sentiment_onnx_path = base_dir / "setiment-results" / "best_multiseg_vgg.onnx"
    genre_onnx_path = base_dir / "genre_results" / "best_model.onnx"
    
    print("Sentiment model absolute path:", sentiment_keras_path)
    print("Genre model absolute path:", genre_keras_path)
    
    # If your model uses any custom layers (e.g. ScalingLayer), import and add them here.
    custom_objects = {}
    try:
        # Update the module name as necessary.
        from your_custom_module import ScalingLayer  # type: ignore
        custom_objects["ScalingLayer"] = ScalingLayer
        print("Custom object 'ScalingLayer' loaded.")
    except ImportError:
        print("No custom objects to load or 'ScalingLayer' not found. Continuing without custom objects.")

    # Convert the sentiment model
    convert_model(sentiment_keras_path, sentiment_onnx_path, custom_objects=custom_objects)
    
    # Convert the genre model
    convert_model(genre_keras_path, genre_onnx_path, custom_objects=custom_objects)
