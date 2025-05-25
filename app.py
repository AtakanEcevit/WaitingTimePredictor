###################################
# app.py — tahmin API'si
###################################
import os
import pandas as pd
import numpy as np
import joblib

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI()

# Model yükle
MODEL_PATH = "rf_pipe.joblib"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Model feature_in_:", model.feature_names_in_)
else:
    model = None
    print("Uyarı: Model dosyası yok! Lütfen train_model.py çalıştır.")

# Kullanıcıdan alınacak giriş yapısı
class PatientInput(BaseModel):
    Doktor_ID: str
    Randevuya_Gelis_Sure: float
    Saatlik_Doluluk: int
    Gun: str
    Saat: int

@app.get("/", response_class=HTMLResponse)
def serve_index():
    # index.html 'templates/' altinda
    template_path = os.path.join("templates", "index.html")
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h3>index.html bulunamadı!</h3>", status_code=404)

@app.post("/tahmin")
def predict_wait_time(inp: PatientInput):
    if model is None:
        return {"detail": "Model yüklenemedi. Lütfen train_model.py çalıştırın."}

    # 1. Data
    df = pd.DataFrame([inp.model_dump()])

    # 2. Feature engineering
    df["Randevuya_Gelis_Mutlak"] = df["Randevuya_Gelis_Sure"].abs()
    df["Randevuya_Gelis_ErkenMi"] = (df["Randevuya_Gelis_Sure"] < 0).astype(int)
    df["Doluluk_GelisMutlak"] = df["Saatlik_Doluluk"] * df["Randevuya_Gelis_Mutlak"]
    df["Saat_GelisErken"] = df["Saat"] * df["Randevuya_Gelis_ErkenMi"]

    # 3. Kolon isimlerinin eşlenmesi
    rename_map = {
        "Doktor_ID": "Doktor ID",
        "Gun": "Gün"
    }
    df.rename(columns=rename_map, inplace=True)

    # 4. One-hot encode — eksik olan feature’ları sıfırla
    cat_cols = ["Gün", "Doktor ID"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0
    df = df[model.feature_names_in_]

    # 5. Tahmin
    y_pred_log = model.predict(df)[0]
    bekleme_tahmin = float(np.expm1(y_pred_log))  # log → orijinal
    
    
    # TODO, Eldeki veri ve gelecek veri ile detaylı analizi yaplacak
    gun_etkisi = {
    "Monday": 1.0,
    "Tuesday": 0.8,
    "Wednesday": 0.5,
    "Thursday": 0.7,
    "Friday": 1.2,
    "Saturday": -0.8,
    }
    gun = inp.Gun
    bekleme_tahmin += gun_etkisi.get(gun, 0)

    return {"Tahmini_Bekleme_Suresi (dk)": round(bekleme_tahmin, 1)}


 # Host
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
