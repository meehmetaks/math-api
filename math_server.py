import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LABELS = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9',
          10:'+', 11:'-', 12:'x', 13:'/'}

model = None

def find_and_load_model():
    print("--- MODEL YÜKLENİYOR ---")
    cwd = os.getcwd()
    for root, dirs, files in os.walk(cwd):
        for f in files:
            if f.endswith(".keras") or f.endswith(".h5"):
                path = os.path.join(root, f)
                try:
                    # Keras 3 uyumluluğu için safe_mode=False eklendi
                    loaded = tf.keras.models.load_model(path, safe_mode=False)
                    print(f"✅ Model başarıyla yüklendi: {path}")
                    return loaded
                except Exception as e:
                    print(f"❌ Yükleme hatası: {e}")
    return None

model = find_and_load_model()

def preprocess_for_api(image_np):
    img = cv2.resize(image_np, (28, 28))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

def merge_and_group(boxes, dist=10):
    if not boxes: return []
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = []
    if len(boxes) > 0:
        curr = list(boxes[0])
        for i in range(1, len(boxes)):
            nxt = boxes[i]
            if nxt[0] < curr[0] + curr[2] + dist:
                new_x = min(curr[0], nxt[0])
                new_y = min(curr[1], nxt[1])
                new_w = max(curr[0] + curr[2], nxt[0] + nxt[2]) - new_x
                new_h = max(curr[1] + curr[3], nxt[1] + nxt[3]) - new_y
                curr = [new_x, new_y, new_w, new_h]
            else:
                merged.append(curr)
                curr = list(nxt)
        merged.append(curr)
    return merged

@app.post("/solve")
async def solve(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model sunucuda yüklü değil."}
    
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('L')
        image_np = np.array(image)
        
        # Görüntü işleme ve segmentasyon
        _, thresh = cv2.threshold(image_np, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2] > 5]
        boxes = merge_and_group(boxes)
        boxes = sorted(boxes, key=lambda b: b[0])

        expression = ""
        for x, y, w, h in boxes:
            roi = thresh[y:y+h, x:x+w]
            processed = preprocess_for_api(roi)
            pred = model.predict(processed, verbose=0)
            label = LABELS.get(np.argmax(pred), '?')
            expression += label

        # Hesaplama
        calc_expr = expression.replace('x', '*')
        result = "Hata"
        try:
            if set(calc_expr).issubset(set("0123456789+-*/. ")):
                result = eval(calc_expr)
        except:
            result = "Hesaplanamadı"

        return {
            "expression": expression,
            "result": str(result)
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}
