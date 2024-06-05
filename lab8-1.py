import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'd:/VScode/lab7/variant-5.jpg'
image = cv2.imread(image_path)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # изначально картинка вроде как в BRG, поэтому переводим в RGB

noize_1 = 0
noize_2 = 100
gaussian = np.random.normal(noize_1, noize_2, image_rgb.shape)
noisy_image = image_rgb + gaussian

noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8) # диапазон значений пикселей

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image_rgb)
plt.axis('off') # избавляемся от самой полезной штуки в мире

plt.subplot(1, 2, 2)
plt.title('Noisy Image')
plt.imshow(noisy_image)
plt.axis('off') 

plt.show()