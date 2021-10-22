import asyncio
import websockets
import cv2
import numpy as np
import base64
import json
import pickle


async def accept(websocket, path):
    '''
    Calibrate Initialization
    get ChessboardImage and get Camera Parameter
    '''
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    pts = np.zeros((6 * 9, 3), np.float32)
    pts[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    # capture calibration frames
    obj_points = []  # 3d point in real world space
    img_points = []  # 2d points in image plane.
    frames = []
    cam_calib = {'mtx': np.eye(3), 'dist': np.zeros((1, 5))}
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
            if type == 'Calibrate':
                frame_copy = frame.copy()
                gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
                retc, corners = cv2.findChessboardCorners(gray, (9, 6), None)
                if retc:
                    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    # Draw and display the corners
                    cv2.drawChessboardCorners(frame_copy, (9, 6), corners, True)
                    '''
                    cv2.imshow('points', frame_copy)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    '''
                    # s to save, c to continue, q to quit
                    if len(frames) < 20:
                        img_points.append(corners)
                        obj_points.append(pts)
                        frames.append(frame)
                        await websocket.send('echo : image get')
                        continue
                    else:
                        # compute calibration matrices
                        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, frames[0].shape[0:2], None, None)
                        # check
                        error = 0.0
                        for i in range(len(frames)):
                            proj_imgpoints, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
                            error += (cv2.norm(img_points[i], proj_imgpoints, cv2.NORM_L2) / len(proj_imgpoints))
                        print("Camera calibrated successfully, total re-projection error: %f" % (error / len(frames)))

                        cam_calib['mtx'] = mtx
                        cam_calib['dist'] = dist
                        print("Camera parameters:")
                        print(cam_calib)
                        pickle.dump(cam_calib, open("calib_cam.pkl", "wb"))
                        await websocket.send('echo : finish calibrate')


start_server = websockets.serve(accept, 'localhost', 9897)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
