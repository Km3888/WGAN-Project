import numpy as np
import matplotlib.pyplot as plt

img=np.loadtxt('GAN_images_i600_time1094.301936864853v1')
plt.imshow(img, cmap='gray')

plt.show()