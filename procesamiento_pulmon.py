# Autor: MQA

import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Cargar imagen
# -----------------------------
image_path = "RxDZ.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# -----------------------------
# 2. Mejora de imagen (CLAHE)
# -----------------------------
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_enhanced = clahe.apply(img)

# -----------------------------
# 3. Realce de bordes (Canny)
# -----------------------------
edges = cv2.Canny(img_enhanced, 50, 150)

# -----------------------------
# 4. Segmentación simple (máscara)
# -----------------------------
# Threshold automático
_, mask = cv2.threshold(img_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Invertir si es necesario (pulmones suelen ser más oscuros)
mask = cv2.bitwise_not(mask)

# Operaciones morfológicas para limpiar
kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# -----------------------------
# 5. Aplicar máscara (ROI)
# -----------------------------
roi = cv2.bitwise_and(img_enhanced, img_enhanced, mask=mask)

# -----------------------------
# 6. Visualización
# -----------------------------
titles = ['Original', 'Mejorada (CLAHE)', 'Bordes', 'Máscara', 'ROI']
images = [img, img_enhanced, edges, mask, roi]

plt.figure(figsize=(12,6))
for i in range(5):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()