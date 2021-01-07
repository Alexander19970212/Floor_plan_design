
from PIL import Image


names = [f"Scrins/band{band}.jpg" for band in range(0, 2500, 10)]
images = [Image.open(f) for f in names]
images = [image.convert("P", palette=Image.ADAPTIVE) for image in images]
fp_out = "image.gif"

img = images[0]
img.save(fp=fp_out, format="GIF", append_images=images[1:], save_all=True, duration=0.1, loop=0)