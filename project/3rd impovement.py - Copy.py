import numpy as np
import cv2
import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD

# Constants for model paths
PROTOTXT_PATH = './model/colorization_deploy_v2.prototxt'
MODEL_PATH = './model/colorization_release_v2.caffemodel'
PTS_PATH = './model/pts_in_hull.npy'
IMAGE_SIZE = (224, 224)  # Default size for resizing

def load_model(prototxt_path, model_path, pts_path):
    print("Loading models...")
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    pts = np.load(pts_path).transpose().reshape(2, 313, 1, 1)
    
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype='float32')]
    
    return net

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, IMAGE_SIZE)
    L_channel = cv2.split(resized)[0] - 50  # Normalize L channel
    
    return image, L_channel

def colorize_image(net, L_channel, original_image):
    net.setInput(cv2.dnn.blobFromImage(L_channel))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    original_shape = original_image.shape[:2]
    ab = cv2.resize(ab, (original_shape[1], original_shape[0]))

    L_channel = cv2.split(original_image)[0]
    colorized = np.concatenate((L_channel[:, :, np.newaxis], ab), axis=2)

    return cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)

def display_image(window_name, image):
    scale = 1.0
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 600, 600)

    while True:
        resized_image = cv2.resize(image, None, fx=scale, fy=scale)
        cv2.imshow(window_name, resized_image)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('+'):
            scale *= 1.1  # Zoom in
        elif key == ord('-'):
            scale /= 1.1  # Zoom out

    cv2.destroyAllWindows()

def save_image(colorized_image):
    save_path = filedialog.asksaveasfilename(
        defaultextension=".jpg",
        filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("BMP files", "*.bmp")],
        title="Save Colorized Image"
    )
    if save_path:
        cv2.imwrite(save_path, colorized_image)
        messagebox.showinfo("Success", f"Colorized image saved to: {save_path}")

def process_image(image_path):
    try:
        net = load_model(PROTOTXT_PATH, MODEL_PATH, PTS_PATH)

        start_time = time.time()

        original_image, L_channel = preprocess_image(image_path)
        colorized_image = colorize_image(net, L_channel, original_image)

        colorized_image = np.clip(colorized_image, 0, 255).astype("uint8")

        # Display images in separate windows
        display_image("Original Image", original_image)
        display_image("Colorized Image", colorized_image)

        if messagebox.askyesno("Save Image", "Do you want to save the colorized image?"):
            save_image(colorized_image)

        end_time = time.time()
        print(f"Total processing time: {end_time - start_time:.2f} seconds")

    except Exception as e:
        messagebox.showerror("Error", str(e))

def on_drop(event):
    file_path = event.data.strip()
    if os.path.isfile(file_path):
        process_image(file_path)
    else:
        messagebox.showerror("Invalid File", "Please drop a valid image file.")

def create_gui():
    root = TkinterDnD.Tk()
    root.title("Image Colorizer")
    root.geometry("600x400")

    label = tk.Label(root, text="Drag and drop an image file here", padx=10, pady=10)
    label.pack(expand=True, fill=tk.BOTH)

    root.drop_target_register(DND_FILES)
    root.dnd_bind('<<Drop>>', on_drop)

    exit_button = tk.Button(root, text="Exit", command=root.quit)
    exit_button.pack(pady=10)

    instructions = tk.Label(root, text="Drop an image file to colorize it.", padx=10, pady=10)
    instructions.pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
