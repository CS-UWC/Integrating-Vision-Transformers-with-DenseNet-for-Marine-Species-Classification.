import os
from PIL import Image

def remove_truncated_images(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith(('png', 'jpg', 'jpeg')):
            filepath = os.path.join(directory, filename)
            try:
                with Image.open(filepath) as img:
                    img.convert("RGB")
            except (OSError, IOError) as e:
                print(f"Error with image {filename}: {e}")
                os.remove(filepath)
                print(f"Removed truncated image: {filename}")

if __name__ == "__main__":
    # Replace 'your_image_directory' with the path to your image directory
    image_directory = 'fish______3/test'
    remove_truncated_images(image_directory)
