import os
import tensorflow as tf
import numpy as np

class ModelLoader:
    def __init__(self):
        # Define base paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.genre_model_path = os.path.join(self.base_dir, "genre_results", "best_model.keras")
        self.sentiment_model_path = os.path.join(self.base_dir, "setiment-results", "best_multiseg_vgg.h5")
        
        # Initialize models as None
        self.genre_model = None
        self.sentiment_model = None
        
        # Class/genre labels
        self.genre_classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 
                             'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    def load_genre_model(self):
        """Load the genre classification model"""
        if self.genre_model is None:
            try:
                self.genre_model = tf.keras.models.load_model(self.genre_model_path)
                print(f"Genre model loaded from {self.genre_model_path}")
            except Exception as e:
                print(f"Error loading genre model: {e}")
                raise e
        return self.genre_model
    
    def load_sentiment_model(self):
        """Load the sentiment prediction model"""
        if self.sentiment_model is None:
            try:
                # Define custom objects including metrics and layers
                custom_objects = {
                    'ScalingLayer': self._get_scaling_layer(),
                    'mse': tf.keras.losses.MeanSquaredError(),
                    'mae': tf.keras.metrics.MeanAbsoluteError()
                }
                
                # Load model with custom objects
                self.sentiment_model = tf.keras.models.load_model(
                    self.sentiment_model_path,
                    custom_objects=custom_objects,
                    compile=False
                )
                
                # Recompile the model with the same optimizer and loss
                self.sentiment_model.compile(
                    optimizer=tf.keras.optimizers.Adam(1e-4),
                    loss=tf.keras.losses.MeanSquaredError(),
                    metrics=[tf.keras.metrics.MeanAbsoluteError()]
                )
                
                print(f"Sentiment model loaded from {self.sentiment_model_path}")
            except Exception as e:
                print(f"Error loading sentiment model: {e}")
                raise e
        return self.sentiment_model
    
    def _get_scaling_layer(self):
        """Recreate the ScalingLayer class from sentiment_train.py"""
        class ScalingLayer(tf.keras.layers.Layer):
            def __init__(self, initial_value=1.0, **kwargs):
                super(ScalingLayer, self).__init__(**kwargs)
                self.initial_value = initial_value

            def build(self, input_shape):
                self.alpha = self.add_weight(name='alpha', shape=(1,), 
                                            initializer=tf.keras.initializers.Constant(self.initial_value),
                                            trainable=True)
                super(ScalingLayer, self).build(input_shape)

            def call(self, inputs):
                return inputs * self.alpha

            def get_config(self):
                config = super(ScalingLayer, self).get_config()
                config.update({"initial_value": self.initial_value})
                return config
        
        return ScalingLayer

# Create singleton instance
model_loader = ModelLoader() 