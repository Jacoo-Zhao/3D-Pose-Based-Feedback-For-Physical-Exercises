import pickle
import os
import numpy as np


def pickle_read(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def pickle_write(filename, data):
    directory = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def reshape_2d(k2d, bboxes):

    def scaling(point, bbox=None):
        if bbox is not None:
            p0 = (point[0] - bbox['xmin']) / (bbox['xmax'] - bbox['xmin'])
            p1 = (point[1] - bbox['ymin']) / (bbox['ymax'] - bbox['ymin'])
        else:
            p0 = point[0]
            p1 = point[1]
        return np.array([p0, p1])

    k2d_bis = {}
    for cam, bbox in bboxes.items():
        k2d_bis[cam] = {'p': {}, 'c': {}}
        for i in range(25):
            point = k2d[cam].get(i, None)
            if point is not None:
                k2d_bis[cam]['p'][i] = scaling(point)
                k2d_bis[cam]['c'][i] = 1
            else:
                k2d_bis[cam]['p'][i] = np.array([0, 0])
                k2d_bis[cam]['c'][i] = 0
    return k2d_bis


def get_3d_camera(k2d, k3d, params, bboxes):
    k3d_bis = {}
    for cam, bbox in bboxes.items():
        t = params[cam]['extrinsics']['t']
        k3d_bis[cam] = {}
        if not np.isnan(k3d[8]).any():
            r = 0.5 / np.linalg.norm(k3d[8] - t)
        elif not (np.isnan(k3d[9]).any() or np.isnan(k3d[12]).any()):
            r = 0.5 / np.linalg.norm(np.mean([k3d[9], k3d[12]], axis=0) - t)
        for i in range(25):
            p0 = k2d[cam]['p'][i][0]
            p1 = k2d[cam]['p'][i][1]
            p2 = np.linalg.norm(k3d[i] - t) * r
            k3d_bis[cam][i] = np.array([p0, p1, p2])
    return k3d_bis


def get_3d_world(k3d):
    ref = k3d[8]
    k3d_bis = {}
    for i in k3d.keys():
        k3d_bis[i] = k3d[i] - ref
    return k3d_bis


print('Loading data...')
data = pickle_read('../Data/010920/data.pickle')
data_reshaped = {}
bboxes = {'6_1': {'xmin': 278.0, 'xmax': 1141.0, 'ymin': 34.0, 'ymax': 924.0},
          '6_2': {'xmin': 632.0, 'xmax': 1557.0, 'ymin': 171.0, 'ymax': 1076.0},
          '6_3': {'xmin': 782.0, 'xmax': 1412.0, 'ymin': 126.0, 'ymax': 1075.0},
          '6_4': {'xmin': 426.0, 'xmax': 896.0, 'ymin': 256.0, 'ymax': 1074.0}}

print('Reshaping data...')
for act in data['frames'].keys():
    print(act)
    data_reshaped[act] = {}
    for sub in data['frames'][act].keys():
        print(f'--{sub}')
        data_reshaped[act][sub] = {}
        for lab in data['frames'][act][sub].keys():
            data_reshaped[act][sub][lab] = {}
            for frame, value in data['frames'][act][sub][lab].items():
                if value['3D_gt']:
                    op2d = reshape_2d(value['2D_op'], bboxes)
                    gt2d = reshape_2d(value['2D_gt'], bboxes)
                    gt3d_c = get_3d_camera(gt2d, value['3D_gt'], data['params'], bboxes)
                    gt3d_w = get_3d_world(value['3D_gt'])
                    data_reshaped[act][sub][lab][frame] = {'2D_op': op2d, '2D_gt': gt2d,
                                                           '3D_gt': {'world': gt3d_w, 'cam': gt3d_c}}

print('Saving data...')
pickle_write(f'../Data/010920/data_reshaped.pickle', data_reshaped)
