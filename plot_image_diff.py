# Python example program for image subtraction

from PIL import Image, ImageChops
import numpy as np
import matplotlib.pyplot as plt
 
image_code = "4_0"

# Paths of two image frames
image1Path = f"./dfal-imgs/nx_{image_code}.png"
image2Path = f"./dfal-imgs/nx_new_{image_code}.png"

 
# Open the images

image1     = Image.open(image1Path)
image2     = Image.open(image2Path)
image2     = image2.convert("RGBA")
image1     = image1.convert("RGBA")

image3 = ImageChops.subtract(image1, image2, scale=0.1, offset=255)
image3.save(f'./dfal-imgs/nx_{image_code}_diff.png')
