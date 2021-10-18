import asyncio
import websockets
import base64
import json
import pickle
import time
import cv2
import numpy as np
from os import path
from subprocess import call
import pickle
import sys
import torch
import os
sys.path.append(os.pardir)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import warnings
warnings.filterwarnings("ignore")

from demo.monitor2 import monitor
from new_frame_processor import frame_processer
from demo.camera import cam_calibrate
from demo.person_calibration import collect_data, fine_tune


cam_calib = pickle.load(open("calib_cam0.pkl", "rb"))
mon = monitor()
frame_processer = frame_processer(cam_calib)

#################################
# Load gaze network
#################################
ted_parameters_path = '5000_gaze_network.pth.tar'
k = 9

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create network
sys.path.append("../src")
from src.models import DTED
gaze_network = DTED(
    growth_rate=32,
    z_dim_app=64,
    z_dim_gaze=2,
    z_dim_head=16,
    decoder_input_c=32,
    normalize_3d_codes=True,
    normalize_3d_codes_axis=1,
    backprop_gaze_to_encoder=False,
).to(device)

#################################

# Load DT-ED weights if available
assert os.path.isfile(ted_parameters_path)
print('> Loading: %s' % ted_parameters_path)
ted_weights = torch.load(ted_parameters_path)
if torch.cuda.device_count() == 1:
    if next(iter(ted_weights.keys())).startswith('module.'):
        ted_weights = dict([(k[7:], v) for k, v in ted_weights.items()])
#####################################
print('> Loading Finished!')

async def accept(websocket, path):
    while True:
        data = await websocket.recv()
        jd = json.loads(data)
        type = jd['type']
        if type == 'Stop':
            cv2.destroyAllWindows()
            continue
        else:
            data = jd['data']
            frame = cv2.imdecode(np.frombuffer(base64.b64decode(data.split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
            x_hat, y_hat = frame_processer.process('gang', frame, mon, device, gaze_network, show=True)
            # show point of regard on screen

            display = np.ones((mon.h_pixels, mon.w_pixels, 3), np.float32)
            h, w, c = patch.shape
            display[0:h, int(mon.w_pixels / 2 - w / 2):int(mon.w_pixels / 2 + w / 2), :] = 1.0 * patch / 255.0
            font = cv2.FONT_HERSHEY_SIMPLEX
            if type(g_n) is np.ndarray:
                cv2.putText(display, '.', (x_pixel_gt, y_pixel_gt), font, 0.5, (0, 0, 0), 10, cv2.LINE_AA)
            cv2.putText(display, '.', (int(x_pixel_hat), int(y_pixel_hat)), font, 0.5, (0, 0, 255), 10, cv2.LINE_AA)
            cv2.namedWindow("por", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("por", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('por', display)

            print(x_hat, y_hat)
            mon_x, mon_y, _ = mon.monitor_to_camera(x_hat, y_hat)
            print(mon.monitor_to_camera(x_hat, y_hat))
        await websocket.send(str(mon_x) + ',' + str(mon_y))


start_server = websockets.serve(accept, '192.168.0.2', 443)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
