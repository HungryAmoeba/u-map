from PIL import Image, ImageDraw, ImageFont
import numpy as np

def return_numpy_array_for_letter(letter, size = 8):
    # Define the font and size
    font_path = 'arial.ttf'
    font_size = size
    font = ImageFont.truetype(font_path, font_size)

    # Create a blank image
    image_width = size
    image_height = size
    # mode L is for 8-bit pizels, grayscale
    image = Image.new('L', (image_width, image_height), color='white')
    draw = ImageDraw.Draw(image)

    # Define the starting position for drawing
    x = 2
    y = -2

    # Draw the letter
    draw.text((x,y),letter, font=font, fill='black')


    # Resize the image
    image = image.resize((size, size))

    # Convert the image to a numpy array
    # ensure that the background is given a value of 0
    image_array = 255 - np.array(image)

    # Convert the image to grayscale
    #image_array = np.mean(image_array, axis=2)

    return image_array
