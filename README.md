# CamCount_ML_Scripts
AUTHOR(S): VINCENZO MACRI // This repo contains all ML related scripts for the CamCount project.

### pose_model_old
* old pose estimation model used in earlier demos (ed1)

### tracking_model
* new yolov5 with deepsort tracking currently being used (ed2)

### How to run
#### Environment Setup
1. Download [this](https://drive.google.com/file/d/1b7Ju3vAtfMuGh9zdDMigS71z0I4d-k2S/view?usp=share_link) zip file and unzip it
2. Download Anaconda from [here](https://www.anaconda.com/) if you dont have it already to help with python version control
3. Open the Anaconda terminal and create a new python environment using the following command
```bash
conda create -n "env" python=3.9
```
4. Enter your environment
```bash
conda activate env 
```
5. To install all of the required python libraries inside of your environment, run 
```bash
pip install -r requirements.txt
```
6. To run using GPU you will need to download the GPU torch libraries from their downloads page [here](https://pytorch.org/)
#### Running Project
1. First run the tracking script by running the following command in the home directory
```bash
python track.py --source frames/ --yolo_model yolov5/weights/crowdhuman_yolov5m.pt --classes 0
```
2. Next open a second anaconda prompt and navigate to the /server folder and run the server.py script
```bash
python server.py
```
3. Power on the ESP32, once it connects to the server.py script a message will be sent to the screen ensuring the connection and frames will start comming in and being interpreted by the tracking script
