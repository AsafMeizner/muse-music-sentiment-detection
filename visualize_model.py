import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import ImageFont, Image
from collections import defaultdict

# Create directory for visualizations
os.makedirs('visualizations', exist_ok=True)

# First, install visualkeras if not already installed
import visualkeras

def load_keras_model(model_path):
    try:
        if model_path.endswith('.h5'):
            model = tf.keras.models.load_model(model_path, compile=False)
        else:
            model = tf.keras.models.load_model(model_path)
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

def get_layer_text(layer_index, layer):
    """
    Create descriptive text for each layer showing its type, shape and name
    """
    # Get layer output shape
    try:
        output_shape = [x for x in list(layer.output_shape) if x is not None]
        
        # Handle nested shapes
        if isinstance(output_shape[0], tuple):
            output_shape = list(output_shape[0])
            output_shape = [x for x in output_shape if x is not None]
            
        # Create shape text
        shape_text = ""
        for i in range(len(output_shape)):
            shape_text += str(output_shape[i])
            if i < len(output_shape) - 2:
                shape_text += "x"
            if i == len(output_shape) - 2:
                shape_text += "\n"
                
        # Add the layer name and type
        layer_type = layer.__class__.__name__
        return f"{shape_text}\n{layer_type}\n{layer.name}", bool(layer_index % 2)
    except:
        return f"{layer.__class__.__name__}\n{layer.name}", bool(layer_index % 2)

def create_enhanced_visualkeras_diagram(model, model_name):
    """Create enhanced visualizations of a model using visualkeras with more details"""
    if model is None:
        print(f"Cannot visualize {model_name} - model not available")
        return

    # Get model summary as string for reference
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    model_summary = "\n".join(stringlist)
    
    # Save model summary
    with open(f"visualizations/{model_name}_summary.txt", 'w', encoding='utf-8') as f:
        f.write(model_summary)
    print(f"✓ Model summary saved to visualizations/{model_name}_summary.txt")

    # Define custom color map for different layer types (with more distinctive colors)
    color_map = defaultdict(dict)
    color_map[tf.keras.layers.Conv2D]['fill'] = '#00a8e8'
    color_map[tf.keras.layers.Conv1D]['fill'] = '#007EA7' 
    color_map[tf.keras.layers.Dense]['fill'] = '#ff5a5f'
    color_map[tf.keras.layers.MaxPooling2D]['fill'] = '#a8df65'
    color_map[tf.keras.layers.MaxPooling1D]['fill'] = '#7FB948'
    color_map[tf.keras.layers.AveragePooling2D]['fill'] = '#73d2de'
    color_map[tf.keras.layers.AveragePooling1D]['fill'] = '#59A5AF'
    color_map[tf.keras.layers.Flatten]['fill'] = '#fcbf49'
    color_map[tf.keras.layers.Dropout]['fill'] = '#d4a373'
    color_map[tf.keras.layers.BatchNormalization]['fill'] = '#f4acb7'
    color_map[tf.keras.layers.Activation]['fill'] = '#9d4edd'
    color_map[tf.keras.layers.Add]['fill'] = '#06d6a0'
    color_map[tf.keras.layers.LSTM]['fill'] = '#ff9e00'
    color_map[tf.keras.layers.GRU]['fill'] = '#E08914'
    color_map[tf.keras.layers.InputLayer]['fill'] = '#b5e48c'
    color_map[tf.keras.layers.Reshape]['fill'] = '#8338ec'
    color_map[tf.keras.layers.Concatenate]['fill'] = '#3A86FF'
    color_map[tf.keras.layers.ZeroPadding2D]['fill'] = '#E0E0E0'

    # Try to find a font that works on Windows
    try:
        try:
            # Try Arial first as it's common on Windows
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            # Fall back to a system font
            font = ImageFont.truetype("C:\\Windows\\Fonts\\segoeui.ttf", 14)
    except:
        # If no font found, use default
        font = None
        print("Warning: Custom font not found, using default font.")

    # Visualization 1: 3D layered view with legend and dimensions
    try:
        viz_path = f"visualizations/{model_name}_3d_layers_with_dimensions.png"
        visualkeras.layered_view(model, to_file=viz_path, legend=True, 
                                font=font, color_map=color_map, 
                                spacing=50, scale_xy=1.5, show_dimensions=True,
                                padding=50)
        print(f"✓ 3D layered visualization with dimensions saved to {viz_path}")
    except Exception as e:
        print(f"Error creating 3D visualization with dimensions: {e}")

    # Visualization 2: 2D flat view with dimensions
    try:
        viz_path = f"visualizations/{model_name}_2d_flat_with_dimensions.png"
        visualkeras.layered_view(model, to_file=viz_path, legend=True,
                                font=font, color_map=color_map,
                                draw_volume=False, spacing=30, show_dimensions=True,
                                padding=50)
        print(f"✓ 2D flat visualization with dimensions saved to {viz_path}")
    except Exception as e:
        print(f"Error creating 2D flat visualization with dimensions: {e}")

    # Visualization 3: Graph view with detailed information
    try:
        viz_path = f"visualizations/{model_name}_graph.png"
        visualkeras.graph_view(model, to_file=viz_path, font=font)
        print(f"✓ Graph visualization saved to {viz_path}")
    except Exception as e:
        print(f"Error creating graph visualization: {e}")

    # Visualization 4: Layered view with layer info text
    try:
        viz_path = f"visualizations/{model_name}_with_layer_info.png"
        visualkeras.layered_view(model, to_file=viz_path,
                                font=font, color_map=color_map,
                                text_callable=get_layer_text, padding=100,
                                spacing=70, scale_xy=1.5)
        print(f"✓ Layered visualization with layer info saved to {viz_path}")
    except Exception as e:
        print(f"Error creating layered visualization with layer info: {e}")

    # Visualization 5: True scale visualization
    try:
        viz_path = f"visualizations/{model_name}_true_scale.png"
        visualkeras.layered_view(model, to_file=viz_path, legend=True,
                                font=font, color_map=color_map,
                                scale_xy=1, scale_z=1, max_z=1000)
        print(f"✓ True scale visualization saved to {viz_path}")
    except Exception as e:
        print(f"Error creating true scale visualization: {e}")

    # Visualization 6: Simplified view (hiding some layers)
    try:
        # Define layers to hide - can be customized based on the specific model
        hide_layers = [tf.keras.layers.Dropout, tf.keras.layers.BatchNormalization]
        viz_path = f"visualizations/{model_name}_simplified.png"
        visualkeras.layered_view(model, to_file=viz_path, legend=True,
                                font=font, color_map=color_map,
                                type_ignore=hide_layers)
        print(f"✓ Simplified visualization saved to {viz_path}")
    except Exception as e:
        print(f"Error creating simplified visualization: {e}")

    # Create a figure showing parameters by layer type
    try:
        plt.figure(figsize=(12, 8))
        layer_types = {}
        
        for layer in model.layers:
            layer_type = layer.__class__.__name__
            if layer_type not in layer_types:
                layer_types[layer_type] = {"params": 0, "count": 0}
                
            trainable_params = np.sum([tf.keras.backend.count_params(w) for w in layer.trainable_weights])
            non_trainable_params = np.sum([tf.keras.backend.count_params(w) for w in layer.non_trainable_weights])
            
            layer_types[layer_type]["params"] += trainable_params + non_trainable_params
            layer_types[layer_type]["count"] += 1
            
        # Create bar chart for parameters
        plt.subplot(2, 1, 1)
        types = list(layer_types.keys())
        params = [layer_types[t]["params"]/1000 for t in types]  # Convert to thousands
        
        colors = [color_map[getattr(tf.keras.layers, t)].get('fill', '#CCCCCC') 
                 if hasattr(tf.keras.layers, t) else '#CCCCCC' for t in types]
        
        plt.bar(types, params, color=colors)
        plt.title(f'{model_name}: Parameters by Layer Type')
        plt.ylabel('Parameters (thousands)')
        plt.xticks(rotation=45, ha='right')
        
        # Create bar chart for layer counts
        plt.subplot(2, 1, 2)
        counts = [layer_types[t]["count"] for t in types]
        
        plt.bar(types, counts, color=colors)
        plt.title(f'{model_name}: Layer Count by Type')
        plt.ylabel('Number of Layers')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f'visualizations/{model_name}_layer_stats.png')
        plt.close()
        print(f"✓ Layer statistics visualization saved to visualizations/{model_name}_layer_stats.png")
    except Exception as e:
        print(f"Error creating layer statistics visualization: {e}")

# Model paths
model1_path = os.path.join('server', 'models', 'best_model.keras')  # genre model
model2_path = os.path.join('server', 'models', 'best_multiseg_vgg.h5')  # sentiment model

# Load and visualize models
print("Loading models...")
model1 = load_keras_model(model1_path)
model2 = load_keras_model(model2_path)

print("\nGenerating enhanced visualizations with visualkeras...")
print("\n--- GENRE MODEL VISUALIZATIONS ---")
create_enhanced_visualkeras_diagram(model1, "genre_model")

print("\n--- SENTIMENT MODEL VISUALIZATIONS ---")
create_enhanced_visualkeras_diagram(model2, "sentiment_model")

print("\nVisualization complete! Check the 'visualizations' directory for all diagrams.")

# Create a combined model comparison image (if both models loaded successfully)
if model1 and model2:
    try:
        # Create side-by-side visualization of both models
        model1_img = visualkeras.layered_view(model1, legend=True, font=font, 
                                             color_map=color_map, show_dimensions=True)
        model2_img = visualkeras.layered_view(model2, legend=True, font=font, 
                                             color_map=color_map, show_dimensions=True)
        
        # Determine the size of the combined image
        width = model1_img.width + model2_img.width
        height = max(model1_img.height, model2_img.height)
        
        # Create a new image with white background
        comparison = Image.new('RGBA', (width, height), (255, 255, 255, 255))
        
        # Paste both model visualizations side by side
        comparison.paste(model1_img, (0, 0))
        comparison.paste(model2_img, (model1_img.width, 0))
        
        # Save the comparison
        comparison.save('visualizations/model_comparison.png')
        print("✓ Model comparison visualization saved to visualizations/model_comparison.png")
    except Exception as e:
        print(f"Error creating model comparison: {e}")