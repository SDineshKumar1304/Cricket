from flask import Flask, request, jsonify, send_from_directory,render_template
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
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import random
import logging
logging.basicConfig(level=logging.DEBUG)

os.environ["OPENCV_FFMPEG_DLL_PATH"] = "./openh264-1.8.0-win64.dll"

app = Flask(__name__)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 9929,
    "response_mime_type": "application/json",
}

# Configure API Key
key = 'AIzaSyCCh8odfwIp1ok4IdgKrJczN0YXd3J95cw'
genai.configure(api_key=key)
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
        # result = subprocess.run(['python', 'yolov7_pose.py', '--source', file_path, '--poseweights', 'yolov7-w6-pose.pt', '--device', request.form.get('device', 'cpu')], check=True, capture_output=True, text=True)
        # logging.info(f"Processing video: {file.filename} with device: {request.form.get('device', 'cpu')}")

        # print(result.stdout)
        # print(result.stderr)
        # os.rename(file_path + '_keypoint.mp4', processed_video_path)
        opt = parse_opt()
        opt.source = file_path
        opt.device = request.form.get('device', 'cpu')
        logging.info(f"Processing video: {file.filename} with device: {opt.device}")
        run(**vars(opt))
        os.rename(file_path + '_keypoint.mp4', processed_video_path)

    except subprocess.CalledProcessError as e:
        # print(e.stdout)
        # print(e.stderr)
        logging.error(f"Error processing video: {str(e)}")
        return jsonify({'error': 'Video processing failed. Please check the logs for more details.'}), 500


    return jsonify({
        'status': 'Processing complete',
        'processed_video': processed_video_path.replace("\\", "/")
    })

@app.route('/processed_videos/<filename>')
def download_video(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

@torch.no_grad()
def run(poseweights="yolov7-w6-pose.pt", source="test9.mp4", device='cpu', view_img=False,
        save_conf=False, line_thickness=3, hide_labels=False, hide_conf=True):

    frame_count = 0  
    total_fps = 0  
    time_list = []  
    fps_list = []    

    device = select_device(device) 
    half = device.type != 'cpu'

   # model = attempt_load(poseweights, map_location=device)  
    ckpt = torch.load(poseweights, map_location=device, weights_only=False)  # load
    model = ckpt['model']
    _ = model.eval()
    names = model.module.names if hasattr(model, 'module') else model.names  

    if source.isnumeric():    
        cap = cv2.VideoCapture(int(source))    
    else:
        cap = cv2.VideoCapture(source)   

    if not cap.isOpened():  
        print('Error while trying to read video. Please check path again')
        raise SystemExit()

    frame_width = int(cap.get(3))  
    frame_height = int(cap.get(4))
        
    vid_write_image = letterbox(cap.read()[1], (frame_width), stride=64, auto=True)[0] 
    resize_height, resize_width = vid_write_image.shape[:2]
    out_video_name = f"{source.split('/')[-1].split('.')[0]}"
    out = cv2.VideoWriter(f"{source}_keypoint.mp4",
                            cv2.VideoWriter_fourcc(*'avc1'), 30,
                            (resize_width, resize_height))

    while cap.isOpened(): 
        print(f"Frame {frame_count+1} Processing")
        ret, frame = cap.read()  
        if ret:
            orig_image = frame
            image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            image = letterbox(image, (frame_width), stride=64, auto=True)[0]
            image_ = image.copy()
            image = transforms.ToTensor()(image)
            image = torch.tensor(np.array([image.numpy()]))
            
            image = image.to(device)
            image = image.float()
            start_time = time.time()
            
            with torch.no_grad():
                output_data, _ = model(image)

            output_data = non_max_suppression_kpt(output_data, 0.25, 0.65,
                                                  nc=model.yaml['nc'],
                                                  nkpt=model.yaml['nkpt'],
                                                  kpt_label=True)

            output = output_to_keypoint(output_data)

            im0 = image[0].permute(1, 2, 0) * 255
            im0 = im0.cpu().numpy().astype(np.uint8)
            im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
            
            for i, pose in enumerate(output_data):
                if len(output_data):
                    for c in pose[:, 5].unique():
                        n = (pose[:, 5] == c).sum()
                        print(f"No of Objects in Current Frame: {n}")
                        
                    for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:, :6])):
                        c = int(cls)
                        kpts = pose[det_index, 6:]
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        plot_one_box_kpt(xyxy, im0, label=label, color=colors(c, True),
                                          line_thickness=line_thickness, kpt_label=True, kpts=kpts, steps=3,
                                          orig_shape=im0.shape[:2])

            end_time = time.time()
            fps = 1 / (end_time - start_time)
            total_fps += fps
            frame_count += 1

            fps_list.append(fps)
            time_list.append(end_time - start_time)
            
            if view_img:
                cv2.imshow("YOLOv7 Pose Estimation Demo", im0)
                cv2.waitKey(1)
            out.write(im0)
        else:
            break

    cap.release()
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")

    plot_fps_time_comparision(time_list=time_list, fps_list=fps_list)


nmodel = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
    system_instruction="""
        Imagine you are a cricket coach. Your task is to give cricket advice to students and learners to improve their performance.
        Provide actionable advice based on the given shot.
    """
)

import json

@app.route('/generateadvice', methods=['GET'])
def generateadvice():
    shots = ['cut', 'pull', 'cover', 'flick', 'sweep']
    selected_shot = random.choice(shots)

    try:
        response = nmodel.generate_content(f"Cricket {selected_shot} drive")
        raw_text = response.candidates[0].content.parts[0].text
        
        # Parse the JSON-like content
        advice_data = json.loads(raw_text)
        advice = advice_data.get('advice', [])
        
    except Exception as e:
        advice = []
        print(f"Error generating advice: {e}")

    return render_template('cricket.html', advice=advice, shot=selected_shot)



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='test9.mp4', help='video/0 for webcam') #video source
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')   #device arugments
    parser.add_argument('--view-img', action='store_true', help='display results')  #display results
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels') #save confidence in txt writing
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)') #box linethickness
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels') #box hidelabel
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences') #boxhideconf
    opt = parser.parse_args()
    return opt

#function for plot fps and time comparision graph
def plot_fps_time_comparision(time_list,fps_list):
    plt.figure()
    plt.xlabel('Time (s)')
    plt.ylabel('FPS')
    plt.title('FPS and Time Comparision Graph')
    plt.plot(time_list, fps_list,'b',label="FPS & Time")
    plt.savefig("FPS_and_Time_Comparision_pose_estimate.png")
    

#main function
def main(opt):
    run(**vars(opt))

if __name__ == '__main__':
    # opt = parse_opt()
    # strip_optimizer(opt.device,opt.poseweights)
    # main(opt)
    app.run(debug=True)