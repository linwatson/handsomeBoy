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
        processed_image = convert_to_dithered(image_cv, task_id)
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


def convert_to_dithered(image, task_id,):
    tile_size = 1

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    total_pixels = height * width
    processed_pixels = 0

    # 執行二值化
    for y in range(height - 1):
        for x in range(1, width - 1):
            old_pixel = gray[y, x]
            new_pixel = 255 if old_pixel > 128 else 0  # 將灰階轉換為 0 或 255
            gray[y, x] = new_pixel
            error = old_pixel - new_pixel

            # 擴散誤差
            gray[y, x + 1] = np.clip(gray[y, x + 1] + error * 7 / 16, 0, 255)
            gray[y + 1, x -
                 1] = np.clip(gray[y + 1, x - 1] + error * 3 / 16, 0, 255)
            gray[y + 1, x] = np.clip(gray[y + 1, x] + error * 5 / 16, 0, 255)
            gray[y + 1, x +
                 1] = np.clip(gray[y + 1, x + 1] + error * 1 / 16, 0, 255)

            processed_pixels += 1
            if processed_pixels % (total_pixels // 100) == 0:
                progress_status[task_id] = int(
                    (processed_pixels / total_pixels) * 100)

    # 強制二值化
    binary_image = np.where(gray > 128, 255, 0).astype(np.uint8)

    # 開始將二值化影像切割成 "方點"
    tiled_image = np.copy(binary_image)
    height, width = binary_image.shape

    # 根據tile_size，將圖像劃分成小區塊
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            # 計算每個方塊的範圍
            tile = binary_image[y:y+tile_size, x:x+tile_size]

            # 計算該區域的平均顏色，若區域大部分為白色，則將區域設為白色，反之為黑色
            avg_color = np.mean(tile)

            # 如果平均顏色大於 128，則該區塊設為白色
            if avg_color > 128:
                tiled_image[y:y+tile_size, x:x+tile_size] = 255
            else:
                tiled_image[y:y+tile_size, x:x+tile_size] = 0

    return tiled_image


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
