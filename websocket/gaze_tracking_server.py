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
ted_parameters_path = '../demo/18_gaze_network.pth.tar'
maml_parameters_path = '../demo/demo_weights/weights_maml'
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
'''
print(ted_weights.keys())
if torch.cuda.device_count() == 1:
    if next(iter(ted_weights.keys())).startswith('module.'):
        ted_weights = dict([(k[7:], v) for k, v in ted_weights.items()])
#####################################

# Load MAML MLP weights if available
full_maml_parameters_path = maml_parameters_path +'/%02d.pth.tar' % k
assert os.path.isfile(full_maml_parameters_path)
print('> Loading: %s' % full_maml_parameters_path)
maml_weights = torch.load(full_maml_parameters_path)
ted_weights.update({  # rename to fit
    'gaze1.weight': maml_weights['layer01.weights'],
    'gaze1.bias':   maml_weights['layer01.bias'],
    'gaze2.weight': maml_weights['layer02.weights'],
    'gaze2.bias':   maml_weights['layer02.bias'],
})
'''
gaze_network.load_state_dict(ted_weights)
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
            try:
                x_hat, y_hat = frame_processer.process('gang', frame, mon, device, gaze_network, show=True)
                print(x_hat, y_hat)
                await websocket.send(str(x_hat) + ', ' + str(y_hat))
            except:
                print('Error')

start_server = websockets.serve(accept, 'localhost', 443)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
