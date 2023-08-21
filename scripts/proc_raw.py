from PIL import Image
import numpy as np
raw = np.fromfile('./frame.raw', np.uint8)

# Save raw view
raw = raw.reshape(480,1504)
img = Image.fromarray(raw)
img.save("test.png")

# Save left/right view
raw = raw.reshape(480,752,2)
left = Image.fromarray(raw[:,:,0])
right = Image.fromarray(raw[:,:,1])
left.save("left.png")
right.save("right.png")


