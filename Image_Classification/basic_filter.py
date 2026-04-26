from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
import os

def apply_blur_filter(image_path, output_path="blurred_image.png"):
    try:
        img = Image.open(image_path)
        img_resized = img.resize((128, 128))
        img_blurred = img_resized.filter(ImageFilter.GaussianBlur(radius=2))

        plt.imshow(img_blurred)
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Processed image saved as '{output_path}'.")

    except Exception as e:
        print(f"Error processing image: {e}")


def apply_vignette_noise_filter(image_path, output_path="vignette_noise_image.png"):
    try:
        img = Image.open(image_path).convert("RGB")
        img_resized = img.resize((256, 256))

        img_array = np.array(img_resized).astype(np.float32)

        height, width = img_array.shape[:2]

        # Create vignette mask
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        xv, yv = np.meshgrid(x, y)

        distance = np.sqrt(xv ** 2 + yv ** 2)

        vignette_strength = .8
        vignette = 1 - vignette_strength * (distance ** 2)
        vignette = np.clip(vignette, 0.15, 1)

        img_array = img_array * vignette[:, :, np.newaxis]

        # Add medium noise
        noise_strength = 10
        noise = np.random.normal(0, noise_strength, img_array.shape)
        img_array = img_array + noise

        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        final_img = Image.fromarray(img_array)

        plt.imshow(final_img)
        plt.axis("off")
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close()

        print(f"Processed image saved as '{output_path}'.")

    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    print("Image Blur Processor (type 'exit' to quit)\n")
    while True:
        image_path = input("Enter image filename (or 'exit' to quit): ").strip()
        if image_path.lower() == 'exit':
            print("Goodbye!")
            break
        if not os.path.isfile(image_path):
            print(f"File not found: {image_path}")
            continue
        # derive output filename
        base, ext = os.path.splitext(image_path)
        output_file = f"{base}_blurred{ext}"
        apply_vignette_noise_filter(image_path, output_file)