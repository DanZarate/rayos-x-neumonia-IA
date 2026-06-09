import os
import cv2
import numpy as np
from tqdm import tqdm

def preprocess_image(img):
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
        
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_enhanced = clahe.apply(img_gray)
    
    # Segmentación para separar fondo (negro) del cuerpo
    _, binary = cv2.threshold(img_enhanced, 15, 255, cv2.THRESH_BINARY)
    
    # Encontrar contornos para quedarse con el cuerpo (el componente más grande)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img_enhanced)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    else:
        mask = binary
    
    # Operaciones morfológicas para limpiar los bordes de la máscara
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Aplicar máscara (ROI)
    roi = cv2.bitwise_and(img_enhanced, img_enhanced, mask=mask)
    
    return roi

def main():
    source_dir = "Chest-X-Rays-4"
    target_dir = "Chest-X-Rays-4-Preprocessed"
    
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} not found.")
        return
        
    for split in ["train", "test", "valid"]:
        split_dir = os.path.join(source_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        for class_name in ["NORMAL", "PNEUMONIA"]:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            out_class_dir = os.path.join(target_dir, split, class_name)
            os.makedirs(out_class_dir, exist_ok=True)
            
            files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"Processing {split}/{class_name} ({len(files)} files)...")
            
            for file in tqdm(files):
                img_path = os.path.join(class_dir, file)
                out_path = os.path.join(out_class_dir, file)
                
                # Check if it already exists to resume interrupted runs
                if os.path.exists(out_path):
                    continue
                    
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                roi = preprocess_image(img)
                cv2.imwrite(out_path, roi)

if __name__ == "__main__":
    main()
