# Authors: Vincenzo, Nikita

# python combined.py --yolo_model yolov5/weights/crowdhuman_yolov5m.pt --classes 0 --imgs 100

# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')
import cv2
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, 
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from yolov5.utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
import socket
import io
import numpy as np
from PIL import Image
import threading
import ujson
from math import floor
import firebase_admin as fa
from firebase_admin import credentials 
from firebase_admin import db
import multiprocessing as mp
import struct
import subprocess
import datetime
from pathlib import Path


HOST = socket.gethostbyname(socket.gethostname())
PORT = 50001
ADDR = (HOST, PORT)
DISCONNECT = str('BYEBYE').encode('utf-8')
BATTERY = str('BATTERY!').encode('utf-8')
SKIP = 1


now = datetime.datetime.now()

print("Current date and time : ")
print(now.strftime("%Y-%m-%d %H:%M:%S"))

dayOfWeek = now.strftime("%A")
print("Day of week:", dayOfWeek)

# current_date = "2023-04-01"
current_date = now.strftime('%Y-%m-%d')

cred = credentials.Certificate('firebase_sdk.json')
fa.initialize_app(cred, {
    'databaseURL': 'https://group-11-fall2022-spring2023-default-rtdb.firebaseio.com/'
})

cameras_ref = db.reference('root/cameras')
date_ref = db.reference('root/cameras/cam0/date')
count_ref = db.reference('root/cameras/cam0/count')
db_date = date_ref.get()


# if the database date is different than the current date, the count will be reset to 0
# and the database date will be updated to reflect the current date
if db_date != current_date:
    print(f"Last date of use: {db_date} \n Current date: {current_date}")
    cameras_ref.update({
        "cam0/count": 0,
        "cam0/date" : current_date
    })
    print("Resetting Count to 0")


# TRACK BARS - Vincenzo

def nothing(*_,**__):
    pass


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

in_count = 0
out_count = 0
net_count = 0

about_to_enter = []
about_to_exit = []

in_data = []
out_data = []


def detect(opt):

    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, project, name, exist_ok= \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')
    
    # Manually resets the DB count to 0 if reset_count argument is active
    if opt.reset_count == True:
        cameras_ref.update({
        "cam0/count": 0
        })
        print("Reset Count flag active, Resetting Count to 0")

    init_count = count_ref.get()
    last_net_count = init_count

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(deep_sort_model,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names


    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0

    # create a window named "Tracker" to hold trackbars to adjust the position of the thresholding lines
    cv2.namedWindow("Tracker")
    cv2.namedWindow("Tracker", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Tracker", 960, 540)

    cv2.namedWindow("Count")
    count_img = np.zeros((400, 400, 3), np.uint8)
    
    # create four trackbars to adjust the y min and max values for the thresholding lines
    cv2.createTrackbar("in_ymax", "Tracker", 453, 1080, nothing)
    cv2.createTrackbar("in_ymin", "Tracker", 264, 1080, nothing)

    cv2.createTrackbar("out_ymax", "Tracker", 292, 1080, nothing)
    cv2.createTrackbar("out_ymin", "Tracker", 70, 1080, nothing)

    cv2.createTrackbar("in_x", "Tracker", 295, 1080, nothing)
    
    cv2.createTrackbar("out_x", "Tracker", 290, 1080, nothing)

    # cv2.createTrackbar("in_ymax", "Tracker", 855, 1080, nothing)
    # cv2.createTrackbar("in_ymin", "Tracker", 537, 1080, nothing)

    # cv2.createTrackbar("out_ymax", "Tracker", 534, 1080, nothing)
    # cv2.createTrackbar("out_ymin", "Tracker", 150, 1080, nothing)

    # cv2.createTrackbar("in_x", "Tracker", 831, 1080, nothing)
    # cv2.createTrackbar("out_x", "Tracker", 831, 1080, nothing)
    
    # for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
    no_detection_count = 0
    print('Creating server...')
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(ADDR)
    server.close()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(ADDR)
    server.listen()
    print(f"The server has been created with address {ADDR}")

    frame_num = 1

    while True:
        print("Waiting for connection...")
        conn, addr = server.accept()
        print(f'New connection established with {addr}!')

        id_msg = conn.recv(8)
        if id_msg:
            cam_id = id_msg.decode('utf-8')
            
        id_ok = 'OKOK'
        id_ok = str(id_ok).encode('utf-8')
        conn.sendall(id_ok)
            
        print(f'The camera ID is {cam_id}!')
        
        
        ref = db.reference(f'root/cameras/{cam_id}/')
            
        ref.update({
            'association': 'Florida Atlantic University',
            'count': 0,
            'location': 'Library - Main Entrance',
            'sleeping': False,
        });
        
        
        batt_ind_msg = conn.recv(8)
        print(batt_ind_msg)
        if batt_ind_msg == BATTERY:
            batt_len_msg = conn.recv(8)
            batt_info_len = int(batt_len_msg.decode('utf-8'))
            batt_info_len = int(batt_len_msg.decode('utf-8'))
                    
            batt_info = handle_msg(conn, batt_info_len)
            batt_info = ujson.loads(batt_info)
            batt_info = batt_info['batt']
            # batt_percent = floor(batt_info[1]*100)
            batt_percent = (batt_info[0] - 800)/32.95
            if batt_percent < 0:
                batt_percent = 0
                batt_disconnected = True
            else:
                batt_disconnected = False
            short_err = 'YES' if batt_info[2] else 'NO'
            disc_err = 'YES' if batt_info[3] else 'NO'
                
            print(f'ESP32-CAM with address {addr} battery info: Voltage = {batt_info[0]}mV, Battery% = {batt_percent:.3f}%, Short Error = {short_err}, Battery Disconnect = {disc_err}')
            print('Sending battery information to Firebase...')    

            
            ref.update({
                'battery_disconnected': batt_disconnected,
                'battery_short': bool(batt_info[3]),
                'battery': int(batt_percent),
            });
            
                    
            print('Battery info sent to Firebase!')

        fps_counter_old = time_sync()

        while True: # when this breaks, client has disconnected, no more detections
            start_of_while = time_sync()
            # get thse current values of the trackbars
            in_ymax = cv2.getTrackbarPos("in_ymax", "Tracker")
            in_ymin = cv2.getTrackbarPos("in_ymin", "Tracker")
            out_ymax = cv2.getTrackbarPos("out_ymax", "Tracker")
            out_ymin = cv2.getTrackbarPos("out_ymin", "Tracker")
            in_x = cv2.getTrackbarPos("in_x", "Tracker")
            out_x = cv2.getTrackbarPos("out_x", "Tracker")

            init_msg = conn.recv(8)

            if init_msg:
          
                if init_msg != BATTERY and init_msg != DISCONNECT:
                    t1 = time_sync()
                    img_len = int(init_msg.decode('utf-8'))
                
                    # print(f"image recieved ({time.time() - t1:.3f}s)")

                    frame_num += 1 
                    s = f'image {frame_num}: '
                    # print(f"frame num {frame_num}")
                    if frame_num == 200:
                        frame_num = 1
                        if no_detection_count > 100:
                            print("No people detected for more than 100 frames.  Sending cam to sleep.")
                            det_msg = 'N'
                            det_msg = str(det_msg).encode('utf-8')
                            conn.sendall(det_msg)
                            no_detection_count = 0
                            frame_num += 1 
                        else:
                            
                            det_msg = 'D'
                            det_msg = str(det_msg).encode('utf-8')
                            conn.sendall(det_msg)
            
                    img_msg = handle_msg(conn, img_len)
                    # print(f"top to handle message ({time.time() - t1:.3f}s)")
                    img0 = Image.open(io.BytesIO(img_msg))
                    img0 = np.array(img0)
                    img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
                    # print(f"top to conver color ({time.time() - t1:.3f}s)")
                        
                    # Padded resize
                    img = letterbox(img0, imgsz, stride, True)[0]
                    # print(f"top to after letterbox ({time.time() - t1:.3f}s)")
                    # Convert
                    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                    img = np.ascontiguousarray(img)
                    
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)
                    t2 = time_sync()
                    dt[0] += t2 - t1

                    # Inference
                    visualize = False
                    pred = model(img, augment=opt.augment, visualize=visualize)
                    t3 = time_sync()
                    dt[1] += t3 - t2

                    # Apply NMS
                    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
                    dt[2] += time_sync() - t3

                    # Process detections
                    for i, det in enumerate(pred):  # detections per image
                        seen += 1

                        im0 = img0.copy()

                        s += '%gx%g ' % img.shape[2:]  # print string

                        annotator = Annotator(im0, line_width=2, pil=not ascii)
                        w, h = im0.shape[1],im0.shape[0]
                        if det is not None and len(det):
                            no_detection_count = 0
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(
                                img.shape[2:], det[:, :4], im0.shape).round()

                            # Print results
                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                            xywhs = xyxy2xywh(det[:, 0:4])
                            confs = det[:, 4]
                            clss = det[:, 5]

                            # pass detections to deepsort
                            t4 = time_sync()
                            outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                            t5 = time_sync()
                            dt[3] += t5 - t4

                            # draw boxes for visualization
                            if len(outputs) > 0:
                                for j, (output, conf) in enumerate(zip(outputs, confs)):

                                    bboxes = output[0:4]
                                    id = output[4]
                                    cls = output[5]
                                    #count
                                    count_obj(bboxes,w,h,id,in_ymax,in_ymin, out_ymax, out_ymin, in_x, out_x)
                                    c = int(cls)  # integer class
                                    label = f'{id} {names[c]} {conf:.2f}'
                                    annotator.box_label(bboxes, label, color=get_color(id))
                            t6 = time_sync()
                            # LOGGER.info(f'{s}Done. Recieving Img ({t1 - start_of_while:.3f}s), Preprocessing ({t2 - t1:.3f}s), YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s), Count:({t6 - t5:.3f}s), Total:({t6-start_of_while:.3f})')

                        else:
                            start_increment_age = time_sync()
                            deepsort.increment_ages()
                            no_detection_count = no_detection_count + 1
                            # print(f"No detection count is {no_detection_count}")

                            # LOGGER.info(f'No detections ({time_sync() - start_increment_age:.3f}s)')

                    start_of_display = time.time()
        
                    fps_counter = time.time()

                    fps = round(1/(fps_counter - fps_counter_old))

                    fps_counter_old = fps_counter
                    # Stream results
                    im0 = annotator.result()

                    global in_count, out_count
                    net_count = init_count + (in_count-out_count)
                    in_color=(0,255,0)

                    in_start_point = (w-in_x, h - in_ymin)
                    in_end_point = (w-in_x, h - in_ymax)

                    out_color=(0,0,255)

                    out_start_point = (w-out_x, h - out_ymin)
                    out_end_point = (w-out_x, h - out_ymax)

                    net_color=(0, 153, 252)
                    
                    cv2.line(im0, in_start_point, in_end_point, in_color, thickness=2)
                    cv2.line(im0, out_start_point, out_end_point, out_color, thickness=2)
                    thickness = 3
                    fps_text_placement = (50, 50)
                    battery_text_placement = (220, 50)
                    in_text_placement = (50, 150)
                    out_text_placement = (50, 250)
                    net_text_placement = (50, 350)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 2
                    count_img = np.zeros((400, 400, 3), np.uint8)
            
                    cv2.putText(count_img, f"BAT: {batt_percent:.0f}%", battery_text_placement, font, 
                        1, (255,255,255), 1, cv2.LINE_AA)
                    cv2.putText(count_img, "FPS: " + str(fps), fps_text_placement, font, 
                        1, (255,255,255), 1, cv2.LINE_AA)
                    cv2.putText(count_img, "In: " + str(in_count), in_text_placement, font, 
                        fontScale, in_color, thickness, cv2.LINE_AA)
                    cv2.putText(count_img, "Out: " + str(out_count), out_text_placement, font, 
                        fontScale, out_color, thickness, cv2.LINE_AA)
                    cv2.putText(count_img, "Net: " + str(net_count), net_text_placement, font, 
                        fontScale, net_color, thickness, cv2.LINE_AA)

                    cv2.imshow("Frame", im0)
                    cv2.imshow("Count", count_img)
                    
                    # Only update the db (which takes a lot of time) every 20 frames and if the netcount has changed
                    if frame_num % 20 == 0 and net_count != last_net_count:
                        update_count(cameras_ref, net_count)
                        # print(f"Updated DB net count from {last_net_count} to {net_count} in ({time.time() - update_count_timer:.3f}s)")
                        last_net_count = net_count

                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration
                    # print(f"Display Time ({time.time() - start_of_display:.3f}s)")
                    # print(f"Total While loop ({time.time() - start_of_while:.3f}s)")

                elif init_msg == DISCONNECT:
                    msg_ok = 'OKOK'
                    msg_ok = str(msg_ok).encode('utf-8')
                    conn.sendall(msg_ok)
                    
                    ref.update({
                        'sleeping': True,
                    });
                    

                    print(f'Client {addr} disconnected!')
                    conn.close() # might throw hands ongawd
                    
                    break

                else:
                    batt_len_msg = conn.recv(8)
                    batt_info_len = int(batt_len_msg.decode('utf-8'))

                    batt_info = handle_msg(conn, batt_info_len)
                    batt_info = ujson.loads(batt_info)
                    batt_info = batt_info['batt']
                    print(batt_info)
                    batt_percent = (batt_info[0] - 800)/32.95
                    if batt_percent < 0:
                       batt_percent = 0
                       batt_disconnected = True
                    else:
                       batt_disconnected = False
                    short_err = 'YES' if batt_info[2] else 'NO'
                    disc_err = 'YES' if batt_info[3] else 'NO'

                    print(f'ESP32-CAM with address {addr} battery info: Voltage = {batt_info[0]}mV, Battery% = {batt_percent:.3f}%, Short Error = {short_err}, Battery Disconnect = {disc_err}')
                    print('Sending battery information to Firebase...')


                    ref.update({
                        'battery_disconnected': batt_disconnected,
                        'battery_short': bool(batt_info[3]),
                        'battery': int(batt_percent),
                        });

                    print('Battery info sent to Firebase!')

                    continue


# this function counts the number of people entering and exiting the library using thresholding lines
def count_obj(box, w, h, id, in_ymax, in_ymin, out_ymax, out_ymin, in_x, out_x):
    # use global variables to store the counts and the ids of the people
    global in_count, out_count, net_count ,in_data, out_data, about_to_enter, about_to_exit
    
    # calculate the center coordinates of the bounding box
    center_coordinates = (int(box[0]+(box[2]-box[0])/2) , int(box[1]+(box[3]-box[1])/2))
    
    # check if the center coordinates are within the in threshold line
    if (center_coordinates[0] > (w -in_x) and (h - in_ymax) < center_coordinates[1] < (h - in_ymin)):
        # add the id to the list of people who are about to enter
        if  id not in about_to_enter:
            about_to_enter.append(id)

    # check if the center coordinates are past the in threshold line
    if (center_coordinates[0] < (w -in_x)):
        # increment the in count and add the id to the list of people who have entered
        if  id in about_to_enter and id not in in_data:
            in_count += 1
            in_data.append(id)

    # check if the center coordinates are within the out threshold line
    if (center_coordinates[0] < (w -out_x) and (h - out_ymax) < center_coordinates[1] < (h - out_ymin)):
        # add the id to the list of people who are about to exit
        if  id not in about_to_exit:
            about_to_exit.append(id)

    # check if the center coordinates are past the out threshold line
    if (center_coordinates[0] > (w -out_x)):
        # increment the out count and add the id to the list of people who have exited
        if  id in about_to_exit and id not in out_data:
            out_count += 1
            out_data.append(id)

def get_color(number):
    h = int(180.0 * pow(number, 0.5) % 180)
    s = 255
    v = 255
    hsv = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return tuple(bgr[0][0].tolist())


def cvt_data_to_img(data):
    load_img = io.BytesIO(data)
    img = np.load(load_img, allow_pickle=True)
    return img

def handle_msg(conn, length):
    msg = b''
    i = 0
    while length > 0:
        yuhh = time.time()
        temp = conn.recv(length)
        
        if temp:
            msg += temp
        length -= len(temp)
        i += 1

    
    
    return msg

def handle_msg_simple(conn, length):
    msg = b''
    msg = conn.recv(length)
    print(f"Value of length is {length} and Message is {len(msg)} long and is {msg}")
    return msg

def handle_msg_better(conn, length):
    msg_parts = []
    while length > 0:
        temp = conn.recv(min(length, 4096))
        if temp:
            msg_parts.append(temp)
            length -= len(temp)
    
    return b''.join(msg_parts)

def update_count(cameras_ref, net_count):
    cameras_ref.update({"cam0/count": net_count})
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reset_count', type=bool, default=False, help='set True to reset the count to 0 at the start of the program')
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5n.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='videos/Traffic.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[480], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_false', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    detect(opt)

