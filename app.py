import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import uvicorn
from ultralytics import YOLO
import base64

app = FastAPI()

os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

ORIGINAL_MODEL_PATH = 'runs/classify/runs_classification/original_model/weights/best.pt'
PREPROCESSED_MODEL_PATH = 'runs/classify/runs_classification/preprocessed_model/weights/best.pt'

model_orig = YOLO(ORIGINAL_MODEL_PATH) if os.path.exists(ORIGINAL_MODEL_PATH) else YOLO('yolov8n-cls.pt')
model_prep = YOLO(PREPROCESSED_MODEL_PATH) if os.path.exists(PREPROCESSED_MODEL_PATH) else YOLO('yolov8n-cls.pt')

def preprocess_contorno(img_gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_enhanced = clahe.apply(img_gray)
    
    _, binary = cv2.threshold(img_enhanced, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img_enhanced)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    else:
        mask = binary
        
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask, img_enhanced

def preprocess_pulmon(img_gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_enhanced = clahe.apply(img_gray)
    
    _, mask = cv2.threshold(img_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_not(mask)
    
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask, img_enhanced

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    preprocess: str = Form("no"),
    model_choice: str = Form("original")
):
    try:
        contents = await file.read()
        if not contents:
            return templates.TemplateResponse(request=request, name="index.html", context={"error": "Por favor, carga una imagen válida."})
            
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return templates.TemplateResponse(request=request, name="index.html", context={"error": "Imagen no válida o formato no soportado."})

        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img

        # Apply preprocessing if selected
        if preprocess == "contorno":
            mask, enhanced = preprocess_contorno(img_gray)
            roi = cv2.bitwise_and(enhanced, enhanced, mask=mask)
            processed_img = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        elif preprocess == "pulmon":
            mask, enhanced = preprocess_pulmon(img_gray)
            roi = cv2.bitwise_and(enhanced, enhanced, mask=mask)
            processed_img = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        elif preprocess == "ambos":
            mask_c, enhanced = preprocess_contorno(img_gray)
            mask_p, _ = preprocess_pulmon(img_gray)
            mask_ambos = cv2.bitwise_and(mask_c, mask_p)
            roi = cv2.bitwise_and(enhanced, enhanced, mask=mask_ambos)
            processed_img = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        else:
            processed_img = img

        # Select model
        model = model_prep if model_choice == "preprocessed" else model_orig

        # Inference
        results = model.predict(processed_img, verbose=False)
        
        # Extract prediction
        probs = results[0].probs
        top1_idx = probs.top1
        top1_conf = probs.top1conf.item()
        class_name = results[0].names[top1_idx]

        # Encode image to show in HTML
        _, buffer = cv2.imencode('.png', processed_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        img_data_url = f"data:image/png;base64,{img_base64}"

        return templates.TemplateResponse(request=request, name="index.html", context={
            "prediction": class_name,
            "confidence": f"{top1_conf*100:.2f}%",
            "image_url": img_data_url,
            "preprocess_used": preprocess,
            "model_used": model_choice
        })
    except Exception as e:
        return templates.TemplateResponse(request=request, name="index.html", context={"error": f"Error procesando la imagen: {str(e)}"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
