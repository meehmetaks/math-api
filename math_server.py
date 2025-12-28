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
MODEL_PATH = "best_math_cnn_model.keras"

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"✅ Model yüklendi: {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Model yüklenemedi: {e}")
            model = None
    else:
        print("❌ Model dosyası bulunamadı!")

load_model()

def preprocess_for_api(roi):
    h, w = roi.shape
    diff = abs(h - w)
    pad1, pad2 = diff // 2, diff - diff // 2
    if h > w:
        padded = cv2.copyMakeBorder(roi, 0, 0, pad1, pad2, cv2.BORDER_CONSTANT, value=0)
    else:
        padded = cv2.copyMakeBorder(roi, pad1, pad2, 0, 0, cv2.BORDER_CONSTANT, value=0)
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
    while boxes:
        curr = list(boxes.pop(0))
        changed = True
        while changed:
            changed = False
            for i in range(len(boxes)):
                x2, y2, w2, h2 = boxes[i]
                overlap = min(curr[0]+curr[2], x2+w2) - max(curr[0], x2)
                y_dist = abs(curr[1] - y2)
                if overlap > min(curr[2], w2)*0.5 and y_dist < y_tol:
                    nx, ny = min(curr[0], x2), min(curr[1], y2)
                    nw, nh = max(curr[0]+curr[2], x2+w2)-nx, max(curr[1]+curr[3], y2+h2)-ny
                    curr = [nx, ny, nw, nh]
                    boxes.pop(i)
                    changed = True
                    break
        merged.append(tuple(curr))
    return merged

@app.get("/")
def read_root():
    return {"message": "Math API Çalışıyor"}

@app.post("/solve")
async def solve(file: UploadFile = File(...)):
    global model
    if model is None:
        load_model()
        if model is None:
            return {"result":"Err","expr":"MODEL_NOT_LOADED","anchorX":0.0,"debug_boxes":[]}

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('L')
        img_np = np.array(image)

        if np.mean(img_np) > 127:
            img_np = cv2.bitwise_not(img_np)

        _, thresh = cv2.threshold(img_np,127,255,cv2.THRESH_BINARY)
        contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2]>5]
        boxes = merge_and_group(boxes)
        boxes = sorted(boxes,key=lambda b:b[0])

        expression = ""
        debug_boxes = []
        min_x, max_x = 10000, 0

        for x,y,w,h in boxes:
            min_x = min(min_x,x)
            max_x = max(max_x,x+w)
            roi = thresh[y:y+h,x:x+w]
            processed = preprocess_for_api(roi)
            pred = model.predict(processed,verbose=0)
            label = LABELS.get(np.argmax(pred),'?')
            expression += label
            debug_boxes.append({"rect":[int(x),int(y),int(w),int(h)],"label":label})

        try:
            calc_expr = expression.replace('x','*')
            if set(calc_expr).issubset(set("0123456789+-*/. ")):
                py_result = eval(calc_expr)
                if isinstance(py_result,float) and py_result.is_integer():
                    result=int(py_result)
                else:
                    result=py_result
            else:
                result="?"
        except:
            result="?"

        anchor_x = (min_x+max_x)/2 if max_x>0 else 50.0
        return {"expression":expression,"expr":expression,"result":str(result),"anchorX":anchor_x,"debug_boxes":debug_boxes}

    except Exception as e:
        print(f"Hata: {e}")
        return {"result":"Err","expr":"Hata","anchorX":0.0,"debug_boxes":[]}
