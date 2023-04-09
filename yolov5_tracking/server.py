# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 02:39:56 2022

@author: nikit
"""

import socket
import io
import numpy as np
from PIL import Image
import threading
import time
import ujson
from math import floor
import firebase_admin as fa
from firebase_admin import credentials 
from firebase_admin import db
import multiprocessing as mp
import struct

from pathlib import Path

import cv2

HOST = socket.gethostbyname(socket.gethostname())
PORT = 50001
ADDR = (HOST, PORT)
DISCONNECT = str('BYEBYE').encode('utf-8')
BATTERY = str('BATTERY!').encode('utf-8')
SKIP = 1


def cvt_data_to_img(data):
    load_img = io.BytesIO(data)
    img = np.load(load_img, allow_pickle=True)
    return img

def handle_msg(conn, length):
    msg = b''
    while length > 0:
        temp = conn.recv(length)
        if temp:
            msg += temp
        length -= len(temp)
    
    return msg

def handle_client(conn, addr):
    frame_num = 0

    '''
    cred = credentials.Certificate('firebase_sdk.json')
    fa.initialize_app(cred, {
        'databaseURL': 'https://group-11-fall2022-spring2023-default-rtdb.firebaseio.com/'
    })
    '''

    id_msg = conn.recv(8)
    if id_msg:
        cam_id = id_msg.decode('utf-8')
        
    id_ok = 'OKOK'
    id_ok = str(id_ok).encode('utf-8')
    conn.sendall(id_ok)
        
    print(f'The camera ID is {cam_id}!')
    
    '''
    ref = db.reference(f'root/cameras/{cam_id}/')
        
    ref.update({
        'association': 'Florida Alantic University',
        'count': 0,
        'location': 'Library - Main Entrance',
        'sleeping': False,
    });
    '''

    frame_num = 0
    
    batt_len_msg = conn.recv(8)
    print(batt_len_msg)
    if batt_len_msg:
        batt_info_len = int(batt_len_msg.decode('utf-8'))
                
        batt_info = handle_msg(conn, batt_info_len)
        batt_info = ujson.loads(batt_info)
        batt_info = batt_info['batt']
        batt_percent = floor(batt_info[1]*100)
        short_err = 'YES' if batt_info[2] else 'NO'
        disc_err = 'YES' if batt_info[3] else 'NO'
            
        print(f'ESP32-CAM with address {addr} battery info: Voltage = {batt_info[0]}mV, Battery% = {batt_percent}%, Short Error = {short_err}, Battery Disconnect = {disc_err}')
        print('Sending battery information to Firebase...')    

        '''
        ref.update({
            'battery_disconnected': bool(batt_info[2]),
            'battery_short': bool(batt_info[3]),
            'battery': batt_percent,
        });
        '''
                
        print('Battery info sent to Firebase!')
             
    t_start = time.perf_counter()
    while True:
        #if frame_num%30 == 0:
            #t_start = t_fin
        init_msg = conn.recv(8)
        #print(f'Frame {frame_num+1} received!')
        if init_msg:
            if init_msg == DISCONNECT:
                msg_ok = 'OKOK'
                msg_ok = str(msg_ok).encode('utf-8')
                conn.sendall(msg_ok)
                
                '''
                ref.update({
                    'sleeping': True,
                });
                '''

                print(f'Client {addr} disconnected!')
                
                break
            
            else:
                img_len = int(init_msg.decode('utf-8'))
        
        img_msg = handle_msg(conn, img_len)
        
        
        #print(check_e-check_s)
        #check_s = time.perf_counter()
        img = Image.open(io.BytesIO(img_msg))
        #check_e = time.perf_counter()
        #print(check_e-check_s)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        #DO DETECTION HERE
        #SEND FLOW DATA TO FIREBASE HERE
        zfilled = str(frame_num).zfill(10)
        
        
        #detect(source=img_jpeg)
        if frame_num%SKIP == 0:
            cv2.imwrite(f'C:/Users/vmacr/OneDrive/Documents/fau/ed2/tracking/yolov5/Yolov5_DeepSort_Pytorch/frames/img{zfilled}.jpg', img)
        #print('Saved new image!')
        
        #if frame_num%30 == 0:
            #t_fin = time.perf_counter()
            #fps = 30/(t_fin-t_start)
            #print(fps)
            
        frame_num += 1 

    t_fin = time.perf_counter()

    fps = round(frame_num/(t_fin-t_start))
    
    '''
    ref.update({
        'fps': fps,
    });
    '''

    print(fps)
    
     
    return
        
def server_main():
    server.listen()
     
    while True:
        conn, addr = server.accept()
        print(f'New connection established with {addr}!')


        process = mp.Process(target=handle_client, args = [conn, addr])
        process.start()

        if cv2.waitKey(1) == ord('q'):  # q to quit
            break
    server.close()

    
if __name__ == '__main__':    
    print('Creating server...')
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(ADDR)
    server.close()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(ADDR)

    print(f'Running server with address {ADDR}!')
    server_main() 