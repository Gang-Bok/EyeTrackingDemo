import asyncio
import websockets
import torch
import cv2
import numpy as np
import base64
import json
import pickle
import os
import sys
import random
sys.path.append(os.pardir)

from demo.monitor2 import monitor
from new_frame_processor import frame_processer
from src.losses import GazeAngularLoss
#################################
# Load gaze network
#################################
ted_parameters_path = '../demo/demo_weights/weights_ted.pth.tar'
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
gaze_network.load_state_dict(ted_weights)

mon = monitor()
lr = 1e-5
steps = 5000
cnt = 0
print("----------------------Loading Finish---------------------")

async def accept(websocket, path):
    global cnt
    cam_calib = pickle.load(open("calib_cam.pkl", "rb"))
    frame_processor = frame_processer(cam_calib)
    subject = 'gang'
    data = {'image_a': [], 'gaze_a': [], 'head_a': [], 'R_gaze_a': [], 'R_head_a': []}
    while True:
        ws_data = await websocket.recv()
        json_data = json.loads(ws_data)
        frame_data = json_data['frame']
        g_x = json_data['x']
        g_y = json_data['y']
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(frame_data.split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
        try:
            processed_patch, g_n, h_n, R_gaze_a, R_head_a = frame_processor.process('gang', frame, mon,
                                                                                    device, gaze_network,
                                                                                    por_available=True, show=False,
                                                                                    target=(g_x, g_y))
            cnt += 1
        except:
            await websocket.send('failed to make image')
        data['image_a'].append(processed_patch)
        data['gaze_a'].append(g_n)
        data['head_a'].append(h_n)
        data['R_gaze_a'].append(R_gaze_a)
        data['R_head_a'].append(R_head_a)

        if cnt == 14:
            n = len(data['image_a'])
            assert n == 130, "Face not detected correctly. Collect calibration data again."
            _, c, h, w = data['image_a'][0].shape
            img = np.zeros((n, c, h, w))
            gaze_a = np.zeros((n, 2))
            head_a = np.zeros((n, 2))
            R_gaze_a = np.zeros((n, 3, 3))
            R_head_a = np.zeros((n, 3, 3))
            for i in range(n):
                img[i, :, :, :] = data['image_a'][i]
                gaze_a[i, :] = data['gaze_a'][i]
                head_a[i, :] = data['head_a'][i]
                R_gaze_a[i, :, :] = data['R_gaze_a'][i]
                R_head_a[i, :, :] = data['R_head_a'][i]

            # create data subsets
            train_indices = []
            for i in range(0, k * 10, 10):
                train_indices.append(random.sample(range(i, i + 10), 3))
            train_indices = sum(train_indices, [])

            valid_indices = []
            for i in range(k * 10, n, 10):
                valid_indices.append(random.sample(range(i, i + 10), 1))
            valid_indices = sum(valid_indices, [])

            input_dict_train = {
                'image_a': img[train_indices, :, :, :],
                'gaze_a': gaze_a[train_indices, :],
                'head_a': head_a[train_indices, :],
                'R_gaze_a': R_gaze_a[train_indices, :, :],
                'R_head_a': R_head_a[train_indices, :, :],
            }

            input_dict_valid = {
                'image_a': img[valid_indices, :, :, :],
                'gaze_a': gaze_a[valid_indices, :],
                'head_a': head_a[valid_indices, :],
                'R_gaze_a': R_gaze_a[valid_indices, :, :],
                'R_head_a': R_head_a[valid_indices, :, :],
            }

            for d in (input_dict_train, input_dict_valid):
                for k, v in d.items():
                    d[k] = torch.FloatTensor(v).to(device).detach()

            #############
            # Finetuning
            #################

            loss = GazeAngularLoss()
            optimizer = torch.optim.SGD(
                [p for n, p in gaze_network.named_parameters() if n.startswith('gaze')],
                lr=lr,
            )

            gaze_network.eval()
            output_dict = gaze_network(input_dict_valid)
            valid_loss = loss(input_dict_valid, output_dict).cpu()
            print('%04d> , Validation: %.2f' % (0, valid_loss.item()))

            for i in range(steps):
                # zero the parameter gradient
                gaze_network.train()
                optimizer.zero_grad()

                # forward + backward + optimize
                output_dict = gaze_network(input_dict_train)
                train_loss = loss(input_dict_train, output_dict)
                train_loss.backward()
                optimizer.step()

                if i % 100 == 99:
                    gaze_network.eval()
                    output_dict = gaze_network(input_dict_valid)
                    valid_loss = loss(input_dict_valid, output_dict).cpu()
                    message = '%04d> Train: %.2f, Validation: %.2f' % (i + 1, train_loss.item(), valid_loss.item())
                    websocket.send(message)
                    print(message)
            torch.save(gaze_network.state_dict(), '%s_gaze_network.pth.tar' % subject)
            torch.cuda.empty_cache()
        else:
            await websocket.send('echo : image get')


start_server = websockets.serve(accept, 'localhost', 9898)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
