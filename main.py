# !pip install flask flask-cors ultralytics pyngrok

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# 1. Load model YOLOv8 kamu
# Pastikan file 'my_final_model.pt' ada di folder yang sama
model = YOLO('my_final_model.pt')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diunggah'}), 400

    try:
        # 2. Terima gambar dari React
        file = request.files['file']
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))

        # 3. Jalankan Prediksi
        results = model(img)
        res = results[0]

        # 4. Ambil Prediksi Teratas
        top_idx = res.probs.top1
        top_name = res.names[top_idx]
        top_prob = float(res.probs.data[top_idx])

        # 5. Format jawaban agar cocok dengan codingan React kamu
        return jsonify({
            "top_predictions": [
                {
                    "label": top_name,
                    "message": f"Hasil deteksi: {top_name} dengan keyakinan {top_prob*100:.2f}%.",
                    "recycle_links": [] # Nanti React kamu yang akan isi link video-nya
                }
            ]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Gunakan port 5000 (standar Flask)
    app.run(host='0.0.0.0', port=5000)