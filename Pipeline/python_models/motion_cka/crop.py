import os
from PIL import Image

# Amount of pixels to remove from the right
crop_right = 300  # change this to your desired amount

# Supported image file extensions
image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

# Process all images in current folder
for filename in os.listdir("."):
    if filename.lower().endswith(image_extensions):
        try:
            img = Image.open(filename)
            width, height = img.size

            # Ensure crop_right is valid
            new_width = max(1, width - crop_right)

            # Crop: (left, top, right, bottom)
            cropped = img.crop((0, 0, new_width, height))

            # Save with a suffix or overwrite
            output_name = f"{filename}"  # or just filename to overwrite
            cropped.save(output_name)
            print(f"✅ Cropped and saved: {output_name}")
        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")