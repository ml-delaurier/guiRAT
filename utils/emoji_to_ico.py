"""
Utility script to convert emoji to .ico file for the guiRAT application.
This script uses the Pillow library to create an icon file from emoji text.
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_emoji_icon(emoji="🐀", size=256, output_path="assets/icon.ico", padding_percent=15):
    """
    Convert emoji to .ico file
    
    Args:
        emoji (str): The emoji character to convert
        size (int): Size of the icon in pixels
        output_path (str): Path where to save the .ico file
        padding_percent (int): Percentage of the image size to use as padding
    """
    # Create a new image with transparency
    image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    
    # Calculate the effective size with padding
    padding = int(size * (padding_percent / 100))
    effective_size = size - (2 * padding)
    
    # Try to find a suitable font that supports emojis
    try:
        # Windows emoji font
        font = ImageFont.truetype("seguiemj.ttf", effective_size-20)
    except OSError:
        try:
            # Alternative font paths
            font = ImageFont.truetype("Arial Unicode.ttf", effective_size-20)
        except OSError:
            print("Error: Could not find a suitable emoji font")
            return False
    
    # Calculate text position to center the emoji
    bbox = draw.textbbox((0, 0), emoji, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = padding + ((effective_size - w) / 2)
    y = padding + ((effective_size - h) / 2)
    
    # Draw the emoji
    draw.text((x, y), emoji, font=font, embedded_color=True)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as ICO file
    image.save(output_path, format='ICO', sizes=[(size, size)])
    return True

if __name__ == "__main__":
    # Create the icon
    if create_emoji_icon():
        print("Icon created successfully!")
    else:
        print("Failed to create icon")
