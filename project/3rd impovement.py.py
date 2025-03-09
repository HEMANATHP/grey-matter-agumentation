import numpy as np
import cv2
import os
import time
import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD

# Constants for model paths
PROTOTXT_PATH = './model/colorization_deploy_v2.prototxt'
MODEL_PATH = './model/colorization_release_v2.caffemodel'
PTS_PATH = './model/pts_in_hull.npy'
IMAGE_SIZE = (224, 224)  # Default size for resizing

def load_model(prototxt_path, model_path, pts_path):
    """Load the colorization model and points."""
    print("Loading models...")
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    pts = np.load(pts_path).transpose().reshape(2, 313, 1, 1)
    
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype='float32')]
    
    return net

def preprocess_image(image_path, target_size=IMAGE_SIZE):
    """Read and preprocess the image for colorization."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    # Resize and normalize L channel in one step
    resized = cv2.resize(lab, target_size)
    L_channel = cv2.split(resized)[0] - 50  # Normalize L channel
    
    return image, L_channel

def colorize_image(net, L_channel, original_image):
    """Run the model to colorize the L channel."""
    net.setInput(cv2.dnn.blobFromImage(L_channel))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # Resize the A and B channels to match the original image dimensions
    original_shape = original_image.shape[:2]  # (height, width)
    ab = cv2.resize(ab, (original_shape[1], original_shape[0]))  # (width, height)

    L_channel = cv2.split(original_image)[0]
    colorized = np.concatenate((L_channel[:, :, np.newaxis], ab), axis=2)

    return cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)

def display_images(original_image, colorized_image):
    """Display the original and colorized images."""
    cv2.imshow("Original", original_image)
    cv2.imshow("Colorized", colorized_image)

def save_image(colorized_image, default_path='./colorized/cat_colorized.jpg'):
    """Save the colorized image if the user wants to."""
    save_option = input("Do you want to save the colorized image? (yes/no): ").strip().lower()
    if save_option == 'yes':
        save_path = input(f"Enter the path to save the colorized image (default: {default_path}): ").strip() or default_path
        
        # Check if the path has a valid extension
        if not save_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            print("Invalid file extension. Please use .png, .jpg, .jpeg, or .bmp.")
            return
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        cv2.imwrite(save_path, colorized_image)
        print(f"Colorized image saved to: {save_path}")

def process_image(image_path):
    """Process the image file."""
    # Load the model
    net = load_model(PROTOTXT_PATH, MODEL_PATH, PTS_PATH)

    # Start timing the process
    start_time = time.time()

    # Preprocess the image
    original_image, L_channel = preprocess_image(image_path)

    # Colorize the image
    colorized_image = colorize_image(net, L_channel, original_image)

    # Clip and convert to uint8
    colorized_image = np.clip(colorized_image, 0, 1)
    colorized_image = (255 * colorized_image).astype("uint8")

    # Display the images
    display_images(original_image, colorized_image)

    # Briefly wait before prompting for save option
    cv2.waitKey(1000)  # Display images for 1 second

    # Ask the user to save the colorized image
    save_image(colorized_image)

    # Clean up windows
    cv2.destroyAllWindows()

    # End timing the process
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")

def on_drop(event):
    """Handle file drop event."""
    file_path = event.data.strip()
    if os.path.isfile(file_path):
        process_image(file_path)

def create_gui():
    """Create a simple GUI for drag and drop."""
    root = TkinterDnD.Tk()
    root.title("Image Colorizer")
    root.geometry("400x200")

    label = tk.Label(root, text="Drag and drop an image file here", padx=10, pady=10)
    label.pack(expand=True)

    # Register drop target
    root.drop_target_register(DND_FILES)
    root.dnd_bind('<<Drop>>', on_drop)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
