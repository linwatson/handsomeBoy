from flask import Flask, request, jsonify, Response, render_template, send_file, send_from_directory
import cv2
import numpy as np
from PIL import Image
import io
import time
import threading
import os

app = Flask(__name__, static_folder="static", template_folder="templates")

UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

progress_status = {}  # 存儲進度


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': '沒有上傳文件'})

    file = request.files['file']
    task_id = str(int(time.time()))  # 以時間戳為 ID

    original_filename = f"{task_id}.png"
    processed_filename = f"{task_id}_processed.png"

    original_path = os.path.join(UPLOAD_FOLDER, original_filename)
    processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)

    file.save(original_path)  # 存原始圖片

    image = Image.open(original_path).convert('RGB')
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    progress_status[task_id] = 0  # 初始化進度

    def process_image():
        processed_image = convert_to_contours(image_cv, task_id)
        cv2.imwrite(processed_path, processed_image)  # 存轉換後圖片
        progress_status[task_id] = 100  # 轉換完成

    threading.Thread(target=process_image).start()

    return jsonify({
        'task_id': task_id,
        'original': original_filename,
        'processed': processed_filename
    })


@app.route('/progress/<task_id>')
def progress(task_id):
    def generate():
        while progress_status.get(task_id, 0) < 100:
            yield f"data: {progress_status.get(task_id, 0)}\n\n"
            time.sleep(0.2)
        yield "data: 100\n\n"
    return Response(generate(), mimetype="text/event-stream")


@app.route('/download/<task_id>')
def download(task_id):
    processed_path = os.path.join(PROCESSED_FOLDER, f"{task_id}_processed.png")
    if os.path.exists(processed_path):
        return send_file(processed_path, mimetype='image/png')
    return jsonify({'error': '圖片未處理完成'})


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)


def convert_to_contours(image, task_id, binarize_channel='L'):
    """
    只保留圖片的輪廓，將圖片轉換為只有邊緣的圖像。

    參數:
        image (numpy.ndarray): 輸入的 BGR 圖片。
        task_id (str): 任務 ID (用於進度監控)。
        binarize_channel (str): 選擇 "H" (色調), "S" (飽和度), "L" (亮度) 進行二值化。

    回傳:
        numpy.ndarray: 只有邊緣的圖像。
    """
    # 轉換為 HSL 色彩空間
    hsl = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    # 選擇使用哪個通道來二值化
    if binarize_channel == 'H':
        gray = hsl[:, :, 0]  # 使用色相
    elif binarize_channel == 'S':
        gray = hsl[:, :, 2]  # 使用飽和度
    else:
        gray = hsl[:, :, 1]  # 使用亮度 (預設)

    # 轉換為 8-bit 灰階
    gray = gray.astype(np.uint8)

    # 使用 Canny 邊緣檢測
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)

    # 查找輪廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 創建一個全黑的圖像來繪製輪廓
    contour_image = np.zeros_like(image)

    # 在圖像上繪製輪廓，顏色為白色
    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)

    return contour_image


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
