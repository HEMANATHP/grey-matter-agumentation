import numpy as np
import cv2
import os

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

def preprocess_image(image_path):
    """Read and preprocess the image for colorization."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L_channel = cv2.split(resized)[0]
    L_channel -= 50  # Normalize L channel
    
    return image, L_channel, lab

def colorize_image(net, L_channel, original_image):
    """Run the model to colorize the L channel."""
    net.setInput(cv2.dnn.blobFromImage(L_channel))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    
    # Get the original image shape
    original_shape = original_image.shape[:2]  # (height, width)

    # Resize the A and B channels to match the original image dimensions
    ab = cv2.resize(ab, (original_shape[1], original_shape[0]))  # (width, height)
    
    L_channel = cv2.split(original_image)[0]
    colorized = np.concatenate((L_channel[:, :, np.newaxis], ab), axis=2)

    return cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)

def main(image_path, save_path=None):
    """Main function to perform image colorization."""
    # Load the model
    net = load_model('./model/colorization_deploy_v2.prototxt', 
                     './model/colorization_release_v2.caffemodel', 
                     './model/pts_in_hull.npy')

    # Preprocess the image
    original_image, L_channel, lab_image = preprocess_image(image_path)

    # Colorize the image
    colorized_image = colorize_image(net, L_channel, original_image)

    # Clip and convert to uint8
    colorized_image = np.clip(colorized_image, 0, 1)
    colorized_image = (255 * colorized_image).astype("uint8")

    # Display the images
    cv2.imshow("Original", original_image)
    cv2.imshow("Colorized", colorized_image)

    if save_path:
        cv2.imwrite(save_path, colorized_image)
        print(f"Colorized image saved to: {save_path}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    main('./images/cat.jfif', save_path='./images/cat_colorized.jpg')
