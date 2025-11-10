"""
Script to generate synthetic demo logo images and test samples.
Creates reference logos and test images (real + fake variants) for the demo.
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def create_simple_logo(name, color, shape='circle', size=(200, 200)):
    """Create a simple synthetic logo with text and shape."""
    img = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(img)
    
    if shape == 'circle':
        draw.ellipse([40, 40, 160, 160], fill=color, outline='black', width=3)
    elif shape == 'square':
        draw.rectangle([40, 40, 160, 160], fill=color, outline='black', width=3)
    elif shape == 'triangle':
        draw.polygon([(100, 40), (40, 160), (160, 160)], fill=color, outline='black', width=3)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), name, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = (size[0] - text_width) // 2
    text_y = size[1] - 40
    
    draw.text((text_x, text_y), name, fill='black', font=font)
    
    return np.array(img)

def add_noise_and_compression(img, quality=30):
    """Add compression artifacts to simulate fake logo."""
    pil_img = Image.fromarray(img)
    pil_img.save('/tmp/temp_compressed.jpg', 'JPEG', quality=quality)
    compressed = Image.open('/tmp/temp_compressed.jpg')
    return np.array(compressed)

def modify_colors(img, hue_shift=20):
    """Modify logo colors to create fake variant."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

def warp_logo(img, strength=0.1):
    """Apply slight geometric distortion to logo."""
    h, w = img.shape[:2]
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    offset = int(w * strength)
    pts2 = np.float32([[offset, 0], [w-offset, offset], [offset, h-offset], [w, h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (w, h), borderValue=(255, 255, 255))

# Create output directories
os.makedirs('data/logos_db', exist_ok=True)
os.makedirs('data/samples', exist_ok=True)

# Generate 6 reference brand logos
logos = [
    ("TechCo", (0, 120, 255), 'circle'),
    ("ShopMart", (255, 100, 0), 'square'),
    ("FastFood", (255, 200, 0), 'circle'),
    ("AutoDrive", (200, 50, 50), 'triangle'),
    ("SoftNet", (100, 200, 100), 'square'),
    ("MediaPlay", (150, 50, 200), 'circle'),
]

print("Generating reference logos...")
for i, (name, color, shape) in enumerate(logos):
    logo = create_simple_logo(name, color, shape)
    cv2.imwrite(f'data/logos_db/logo_{i+1}_{name.lower()}.png', cv2.cvtColor(logo, cv2.COLOR_RGB2BGR))
    print(f"  Created: logo_{i+1}_{name.lower()}.png")

# Generate test samples (5 real + 5 fake)
print("\nGenerating test samples...")

# Real logos (pristine copies)
for i in range(5):
    name, color, shape = logos[i]
    logo = create_simple_logo(name, color, shape)
    cv2.imwrite(f'data/samples/real_logo{i+1}.jpg', cv2.cvtColor(logo, cv2.COLOR_RGB2BGR))
    print(f"  Created: real_logo{i+1}.jpg")

# Fake logos (modified versions)
fake_modifications = [
    (0, 'compressed', lambda img: add_noise_and_compression(img, quality=25)),
    (1, 'color_shifted', lambda img: modify_colors(img, hue_shift=30)),
    (2, 'warped', lambda img: warp_logo(img, strength=0.12)),
    (3, 'compressed_color', lambda img: modify_colors(add_noise_and_compression(img, quality=20), hue_shift=25)),
    (4, 'all_modified', lambda img: warp_logo(modify_colors(add_noise_and_compression(img, quality=15), hue_shift=35), strength=0.15)),
]

for i, (logo_idx, mod_name, mod_func) in enumerate(fake_modifications):
    name, color, shape = logos[logo_idx]
    logo = create_simple_logo(name, color, shape)
    fake_logo = mod_func(logo)
    cv2.imwrite(f'data/samples/fake_logo{i+1}_{mod_name}.jpg', cv2.cvtColor(fake_logo, cv2.COLOR_RGB2BGR))
    print(f"  Created: fake_logo{i+1}_{mod_name}.jpg")

print("\nDemo assets generated successfully!")
print(f"  - {len(logos)} reference logos in data/logos_db/")
print(f"  - 10 test samples in data/samples/ (5 real + 5 fake)")
