import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import os

# --- FastAPI setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Yükleme ---
MODEL_PATH = "best_math_cnn_model.keras"
model = None

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model yüklendi: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Model yüklenemedi: {e}")
    model = None

# --- Etiketler ---
LABELS = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9',
          10:'+', 11:'-', 12:'x', 13:'/'}

# --- Görüntü İşleme Fonksiyonları ---
def preprocess_for_api(roi):
    h, w = roi.shape
    diff = abs(h - w)
    pad_1, pad_2 = diff // 2, diff - (diff // 2)
    if h > w:
        padded = cv2.copyMakeBorder(roi, 0, 0, pad_1, pad_2, cv2.BORDER_CONSTANT, value=0)
    else:
        padded = cv2.copyMakeBorder(roi, pad_1, pad_2, 0, 0, cv2.BORDER_CONSTANT, value=0)
    resized = cv2.resize(padded, (20, 20), interpolation=cv2.INTER_AREA)
    final_img = np.zeros((28, 28), dtype=np.uint8)
    final_img[4:24, 4:24] = resized
    final_img = final_img.astype('float32') / 255.0
    final_img = np.expand_dims(final_img, axis=-1)
    final_img = np.expand_dims(final_img, axis=0)
    return final_img

def merge_and_group(boxes, y_tol=30):
    if not boxes: return []
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = []
    while len(boxes) > 0:
        curr = list(boxes.pop(0))
        changed = True
        while changed:
            changed = False
            for i in range(len(boxes)):
                x2, y2, w2, h2 = boxes[i]
                overlap = min(curr[0] + curr[2], x2 + w2) - max(curr[0], x2)
                y_dist = abs(curr[1] - y2)
                if overlap > min(curr[2], w2) * 0.5 and y_dist < y_tol:
                    nx, ny = min(curr[0], x2), min(curr[1], y2)
                    nw, nh = max(curr[0] + curr[2], x2 + w2) - nx, max(curr[1] + curr[3], y2 + h2) - ny
                    curr = [nx, ny, nw, nh]
                    boxes.pop(i)
                    changed = True
                    break
        merged.append(tuple(curr))
    return merged

# --- API ---
@app.get("/")
def read_root():
    return {"message": "Math API Calisiyor"}

@app.post("/solve")
async def solve(file: UploadFile = File(...)):
    global model
    if model is None:
        return {"result": "Err", "expression": "MODEL_NOT_LOADED", "debug_boxes": []}

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('L')
        img_np = np.array(image)

        # Arka plan beyazsa tersle
        if np.mean(img_np) > 127:
            img_np = cv2.bitwise_not(img_np)

        _, thresh = cv2.threshold(img_np, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        initial_boxes = [cv2.boundingRect(cnt) for cnt in contours if cv2.boundingRect(cnt)[2] > 5]
        final_boxes = merge_and_group(initial_boxes)
        final_boxes = sorted(final_boxes, key=lambda b: b[0])

        expression = ""
        debug_boxes = []
        min_x, max_x = 10000, 0

        for box in final_boxes:
            x, y, w, h = box
            min_x = min(min_x, x)
            max_x = max(max_x, x + w)

            roi = thresh[y:y+h, x:x+w]
            processed = preprocess_for_api(roi)
            pred = model.predict(processed, verbose=0)
            label = LABELS.get(np.argmax(pred), '?')

            expression += label
            debug_boxes.append({"rect": [int(x), int(y), int(w), int(h)], "label": label})

        # Hesaplama
        try:
            calc_expr = expression.replace('x', '*')
            if set(calc_expr).issubset(set("0123456789+-*/. ")):
                py_result = eval(calc_expr)
                if isinstance(py_result, float) and py_result.is_integer():
                    result = int(py_result)
                else:
                    result = py_result
            else:
                result = "?"
        except:
            result = "?"

        anchor_x_val = (min_x + max_x) / 2 if max_x > 0 else 50.0

        return {
            "expression": expression,
            "expr": expression,
            "result": str(result),
            "anchorX": float(anchor_x_val),
            "debug_boxes": debug_boxes
        }

    except Exception as e:
        print(f"Hata: {e}")
        return {"result": "Err", "expr": "Hata", "anchorX": 0.0, "debug_boxes": []}
