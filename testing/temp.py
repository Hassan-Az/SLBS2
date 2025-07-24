import cv2
import numpy as np

img_path = "shore_jpg.jpg"# input("image name: ")
img = cv2.imread(img_path)  # Reading an image
flattenimg = img.flatten()

print(flattenimg.dtype==np.uint8)