import numpy as np
import matplotlib.pyplot as plt

try:
    import cv2
except ImportError:
    print("OpenCV import failed. Falling back to Pillow.")
    from PIL import Image, ImageEnhance

def load_image(file_path):
    try:
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except:
        image = np.array(Image.open(file_path).convert('RGB'))
    return image / 255.0

def adjust_gamma(image, gamma=1.0):
    return np.power(image, 1/gamma)

def remove_color_cast(image):
    avg_color = np.mean(image, axis=(0, 1))
    max_channel = np.max(avg_color)
    scaling = max_channel / avg_color
    return np.clip(image * scaling, 0, 1)

def enhance_contrast(image):
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val)

def color_balance(image):
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
    r_avg, g_avg, b_avg = np.mean(r), np.mean(g), np.mean(b)
    avg = (r_avg + g_avg + b_avg) / 3
    r = np.clip(r * (avg / r_avg), 0, 1)
    g = np.clip(g * (avg / g_avg), 0, 1)
    b = np.clip(b * (avg / b_avg), 0, 1)
    return np.dstack((r, g, b))

def process_image(image):
    # Remove color cast
    image_corrected = remove_color_cast(image)
    
    # Enhance contrast
    image_corrected = enhance_contrast(image_corrected)
    
    # Color balance
    image_corrected = color_balance(image_corrected)
    
    # Adjust gamma
    image_corrected = adjust_gamma(image_corrected, 1.2)
    
    # Final adjustments
    image_corrected = np.clip(image_corrected * 1.2, 0, 1)  # Brighten
    
    return image_corrected

def main():
    image = load_image('image/Screenshot 2024-09-23 215336.png')
    processed_image = process_image(image)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(image)
    ax1.set_title('Original Underwater Image')
    ax1.axis('off')
    ax2.imshow(processed_image)
    ax2.set_title('Processed Image')
    ax2.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()