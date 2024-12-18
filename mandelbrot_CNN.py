### MANDELBROT ###
### Kristijonas and Julio pair project Artificial Intelligence Course. 

## CNN

import numpy as np #type:ignore
import matplotlib.pyplot as plt #type:ignore
from PIL import Image as img #type:ignore
from PIL import Image #type:ignore
import os

import tensorflow #type:ignore
from tensorflow.keras import layers, models #type:ignore
import glob

# To generate a GIF as an output.
import imageio #type:ignore

import argparse # For the image input

image_size = 480
number_samples = 1000

############--------------############
#            Mandelbrot              #
############--------------############
def mandelbrot(h, w, max_iter, x_min, x_max, y_min, y_max):

    y, x = np.ogrid[y_min:y_max:h*1j, x_min:x_max:w*1j]
    c = x + y*1j
    z = c
    divtime = max_iter + np.zeros(z.shape, dtype=int)
    
    for i in range(max_iter):
        z = z**2 + c
        diverge = z * np.conj(z) > 2**2
        div_now = diverge & (divtime == max_iter)
        divtime[div_now] = i
        z[diverge] = 2
        
    return divtime

############--------------############
#            Neural Network          #
############--------------############

### 2. Train the NN. 
### Using tensorflow. 

# The description of the model in in the jupyter notebook. 
# Here it is only loaded. 
from tensorflow.keras.models import load_model # type: ignore
model = load_model("mandelbrot_CNN_model.keras")

##IMPORTANT for different models, modelled with different image sizes.

## mandelbrot_CNN_model.keras, size = 64
## mandelbrot_CNN_K.keras, size = 64
## mandelbrot_CNN_model_updated.keras, size = 480
size = 64


############--------------############
#            ZOOM FUNCTION           #
############--------------############

#### Zoom function
number_steps = 30


def zoom_in(snippet, model, full_width=800, full_height=600, zoom_steps=number_steps, gif_filename="zoom_animation_model.gif"):
    
    # Predict bounding box
    snippet_resized = np.expand_dims(np.array(snippet.resize((size, size))) / 255.0, axis=(0, -1))
    x_min, x_max, y_min, y_max = model.predict(snippet_resized)[0]
    
    frames = []

    for step in range(zoom_steps):
        interp_x_min = x_min + (-2 - x_min) * (1 - step / zoom_steps)
        interp_x_max = x_max + (0.8 - x_max) * (1 - step / zoom_steps)
        interp_y_min = y_min + (-1.4 - y_min) * (1 - step / zoom_steps)
        interp_y_max = y_max + (1.4 - y_max) * (1 - step / zoom_steps)

        mandel_zoom = mandelbrot(full_height, full_width, 100, 
                                 interp_x_min, interp_x_max, interp_y_min, interp_y_max)
        
        # Plot the Mandelbrot set for the current step
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(mandel_zoom, cmap='magma', extent=[interp_x_min, interp_x_max, interp_y_min, interp_y_max])
        ax.set_title(f'Zoom Step {step + 1}/{zoom_steps}')
        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
        plt.colorbar(ax.imshow(mandel_zoom, cmap='magma'), label='Iteration count')

        # Draw the plot to the canvas and capture the image
        fig.canvas.draw()  # Draw the plot
        image_buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_buf = image_buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))  # Convert to RGB shape
        
        frames.append(image_buf)  # Append the frame

        plt.close(fig)  # Close the figure to prevent display in the notebook
    
    # Save all frames as a GIF
    imageio.mimsave(gif_filename, frames, duration=0.1)  # Adjust duration for speed of the GIF

    print(f"GIF saved as {gif_filename}")

## Execute the process. 
# Take input image and gives the output zoom as GIF. 
def take_input(image_path):
    snippet_img = Image.open(image_path).convert('L')
    zoom_in(snippet_img, model)

if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Generate a zoom-in animation of the Mandelbrot set")
    parser.add_argument('image', type=str, help="Path to the input image (PNG)")
    args = parser.parse_args()

    # Call the main function with the provided image file
    take_input(args.image)


#snippet_img = Image.open("search_mandelbrot.png").convert('L')
#zoom_in(snippet_img, model)