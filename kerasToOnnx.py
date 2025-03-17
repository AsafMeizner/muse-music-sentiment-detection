#!/usr/bin/env python
import os
from pathlib import Path
import tensorflow as tf
import tf2onnx

# ----------------------------
# Custom ScalingLayer Definition
# ----------------------------
class ScalingLayer(tf.keras.layers.Layer):
    def __init__(self, initial_value=1.0, **kwargs):
        super(ScalingLayer, self).__init__(**kwargs)
        self.initial_value = initial_value

    def build(self, input_shape):
        self.alpha = self.add_weight(
            name='alpha',
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.initial_value),
            trainable=True
        )
        super(ScalingLayer, self).build(input_shape)

    def call(self, inputs):
        return inputs * self.alpha

    def get_config(self):
        config = super(ScalingLayer, self).get_config()
        config.update({"initial_value": self.initial_value})
        return config

# ----------------------------
# Conversion Function
# ----------------------------
def convert_model(keras_model_path: Path, onnx_model_path: Path, custom_objects=None):
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

# ----------------------------
# Main Script
# ----------------------------
if __name__ == "__main__":
    # Get the directory containing this script.
    base_dir = Path(__file__).parent.absolute()
    
    # Build absolute paths relative to the working directory.
    sentiment_keras_path = base_dir / "setiment-results" / "best_multiseg_vgg.h5"
    genre_keras_path = base_dir / "genre_results" / "best_model.keras"
    
    sentiment_onnx_path = base_dir / "setiment-results" / "best_multiseg_vgg.onnx"
    genre_onnx_path = base_dir / "genre_results" / "best_model.onnx"
    
    print("Sentiment model absolute path:", sentiment_keras_path)
    print("Genre model absolute path:", genre_keras_path)
    
    # Provide custom_objects with the ScalingLayer so the sentiment model loads correctly.
    custom_objects = {"ScalingLayer": ScalingLayer}
    
    # Convert the sentiment model.
    convert_model(sentiment_keras_path, sentiment_onnx_path, custom_objects=custom_objects)
    
    # Convert the genre model.
    convert_model(genre_keras_path, genre_onnx_path, custom_objects=custom_objects)
