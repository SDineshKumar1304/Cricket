from flask import Flask, request, jsonify, send_from_directory
import os
import subprocess
import cv2
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt,strip_optimizer,xyxy2xywh
from utils.plots import output_to_keypoint, plot_skeleton_kpts,colors,plot_one_box_kpt

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed_videos'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    print("upload")
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    processed_video_path = os.path.join(PROCESSED_FOLDER, 'processed_' + file.filename)
    import logging
    logging.basicConfig(
    level=logging.DEBUG,  # Show all logs from DEBUG level and above
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        # Run the YOLOv7 video processing with subprocess
        result = subprocess.run(['python', 'yolov7_pose.py', '--source', file_path, '--poseweights', 'yolov7-w6-pose.pt', '--device', request.form.get('device', 'cpu')], check=True, capture_output=True, text=True)
        logging.info(f"Processing video: {file.filename} with device: {request.form.get('device', 'cpu')}")

        print(result.stdout)
        print(result.stderr)
        os.rename(file_path + '_keypoint.mp4', processed_video_path)
    except subprocess.CalledProcessError as e:
        print(e.stdout)
        print(e.stderr)
        logging.error(f"Error processing video: {str(e)}")
        return jsonify({'error': 'Video processing failed. Please check the logs for more details.'}), 500


    return jsonify({
        'status': 'Processing complete',
        'processed_video': processed_video_path.replace("\\", "/")
    })

@app.route('/processed_videos/<filename>')
def download_video(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
