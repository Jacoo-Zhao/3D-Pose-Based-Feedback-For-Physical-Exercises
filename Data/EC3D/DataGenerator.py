import pandas as pd
import itertools
import numpy as np
import cv2
import os
import json
import pickle

from scipy.signal import find_peaks


def json_read(filename):
    try:
        with open(os.path.abspath(filename)) as f:
            data = json.load(f)
        return data
    except:
        raise ValueError("Unable to read JSON {}".format(filename))


def pickle_write(filename, data):
    directory = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def pickle_read(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def compute_op2d(op2d, K, d, bbox, threshold: float = 0):
    op2d_dict = {}
    for i, point in enumerate(op2d):
        if point[2] > threshold and bbox['xmin'] < point[0] < bbox['xmax'] and bbox['ymin'] < point[1] < bbox['ymax']:
            point = np.array([point[0], point[1]])
            try:
                undist_point = cv2.undistortPoints(point, K, d, P=K)[0][0]
                op2d_dict[i] = undist_point
            except cv2.error:
                op2d_dict[i] = point
    return op2d_dict


def compute_gt3d(data_op2d, data_params):
    def triangulate(K1, R1, t1, pt1, K2, R2, t2, pt2):
        P1 = np.dot(K1, np.hstack([R1, t1.reshape(3, 1)]))
        P2 = np.dot(K2, np.hstack([R2, t2.reshape(3, 1)]))
        pt3d = cv2.triangulatePoints(P1, P2, pt1.T, pt2.T)
        if pt3d[3] != 0:
            pt3d = pt3d / pt3d[3]
            return pt3d[:3].T[0]
        else:
            return None

    all_op2d = {}
    gt3d = {}
    for cam in data_op2d.keys():
        for i, point in data_op2d[cam].items():
            if all_op2d.get(i, []):
                all_op2d[i][cam] = point
            else:
                all_op2d[i] = {cam: point}

    for i in range(25):
        if all_op2d.get(i, []) and len(all_op2d[i].keys()) > 1:
            loc3d = {}
            cameras = list(all_op2d[i].keys())
            camwise = {}
            for pair in itertools.combinations(cameras, 2):
                K1 = data_params[pair[0]]['intrinsics']['K']
                K2 = data_params[pair[1]]['intrinsics']['K']
                R1 = data_params[pair[0]]['extrinsics']['R']
                R2 = data_params[pair[1]]['extrinsics']['R']
                t1 = data_params[pair[0]]['extrinsics']['t']
                t2 = data_params[pair[1]]['extrinsics']['t']
                pt1 = all_op2d[i][pair[0]]
                pt2 = all_op2d[i][pair[1]]
                tri = triangulate(K1, R1, t1, pt1, K2, R2, t2, pt2)
                if tri is not None:
                    loc3d[pair] = tri
            for cam in cameras:
                camwise[cam] = [v for k, v in loc3d.items() if cam in k]
            means = [np.mean(camwise[cam], axis=0) for cam in camwise.keys()]
            up = np.mean(means, axis=0) + np.std(means, axis=0)
            down = np.mean(means, axis=0) - np.std(means, axis=0)
            args = list(np.where((np.array(means) < down).any() | (np.array(means) > up).any())[0])
            out = [list(camwise.keys())[el] for el in args]
            for el in out:
                loc3d = {i: loc3d[i] for i in loc3d if el not in i}
            gt3d[i] = np.mean(list(loc3d.values()), axis=0)
        else:
            gt3d[i] = np.array([np.nan, np.nan, np.nan])

    return gt3d


def compute_gt2d(data_gt3d, data_params):
    gt2d_all = {}
    for cam in cameras:
        gt2d_all[cam] = {}
        R = data_params[cam]['extrinsics']['R']
        rvec = cv2.Rodrigues(R)[0]
        t = data_params[cam]['extrinsics']['t']
        K = data_params[cam]['intrinsics']['K']
        d = data_params[cam]['intrinsics']['distCoeffs']
        for i, point in data_gt3d.items():
            if not np.isnan(point).all():
                gt2d_all[cam][i] = cv2.projectPoints(point, rvec, t, K, d)[0].reshape(-1, 2)[0]
    return gt2d_all


def outlier_correction(op2d_0, op2d_1, op2d_2, threshold=40):
    diff0 = np.linalg.norm(op2d_0 - op2d_1)
    diff1 = np.linalg.norm(op2d_2 - op2d_1)
    if diff0 > threshold and diff1 > threshold:
        return np.mean([op2d_0, op2d_2], axis=0)
    else:
        return None


def temporal_smoothing(s, filter):
    n_frames = len(s[0][0])
    for point in s.keys():
        seq = [np.array([s[point][0][i], s[point][1][i], s[point][2][i]]) for i in range(n_frames)]
        diff = [np.linalg.norm(seq[i] - seq[i - 1]) for i in range(len(seq)) if i > 0]
        if np.isnan(diff).all():
            print(point)
            continue
        else:
            up = np.nanmean(diff) + 3 * np.nanstd(diff)
            for k, v in s[point].items():
                for i, el in enumerate(v):
                    if np.isnan(el):
                        if i == 0:
                            j = i + 1
                            while np.isnan(v[j]):
                                j = j + 1
                                if j == len(v)-1:
                                    break
                            v[i] = v[j]
                        elif i == len(v) - 1:
                            v[i] = v[i - 1]
                        else:
                            v[i] = np.nanmean([v[i - 1], v[i + 1]])
            for i in range(n_frames):
                if 0 < i < n_frames-1:
                    p0 = np.array([s[point][0][i-1], s[point][1][i-1], s[point][2][i-1]])
                    p1 = np.array([s[point][0][i], s[point][1][i], s[point][2][i]])
                    p2 = np.array([s[point][0][i+1], s[point][1][i+1], s[point][2][i+1]])
                    if np.linalg.norm(p1-p0) > up:
                        if np.linalg.norm(p1-p2) > up:
                            for k in range(3):
                                s[point][k][i] = np.mean([p0[k], p2[k]])
                        else:
                            for j in range(1, 3):
                                try:
                                    p1 = np.array([s[point][0][i + j], s[point][1][i + j], s[point][2][i + j]])
                                    p2 = np.array([s[point][0][i + j + 1], s[point][1][i + j + 1], s[point][2][i + j + 1]])
                                    if np.linalg.norm(p1 - p2) > up:
                                        d = i + j + 2
                                        for k in range(3):
                                            s[point][k][i] = (d-1)/d * p0[k] + 1/d * p2[k]
                                        break
                                except IndexError:
                                    break
            for k, v in s[point].items():
                s[point][k] = np.convolve(filter, np.array(v), mode='valid')
    return s


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


def extract_data(cameras, labels, data_path, filter, bboxes, peaks_params):
    print('Getting 2D OpenPose keypoints...')
    data = {'frames': {}, 'params': {}}
    for cam in cameras:
        intrinsics = pickle_read(f'{data_path}/{cam}/intrinsics.pickle')
        extrinsics = pickle_read(f'{data_path}/{cam}/extrinsics.pickle')
        data['params'][cam] = {'intrinsics': intrinsics, 'extrinsics': extrinsics}
    for row in labels.iterrows():
        act = row[1]['Activity']
        sub = row[1]['Subject']
        lab = row[1]['Label']
        if data['frames'].get(act, []):
            if data['frames'][act].get(sub, []):
                data['frames'][act][sub][lab] = {}
            else:
                print(f'--{sub}')
                data['frames'][act][sub] = {lab: {}}
        else:
            print(act)
            print(f'--{sub}')
            data['frames'][act] = {sub: {lab: {}}}

        start = int(row[1]['Start Frame'])
        stop = int(row[1]['Stop Frame'])
        for f in range(start, stop + 1):
            frame = f'frame_{str(f).zfill(6)}'
            data['frames'][act][sub][lab][frame] = {'path': {}, '2D_op': {}, '2D_gt': {}, '3D_gt': {}}
            for cam in cameras:
                path = f'/cvlabdata2/home/vinzant/Data/010920/{cam}/Frames/{frame}.png'
                data['frames'][act][sub][lab][frame]['path'][cam] = path
                op2d = json_read(f'{data_path}/{cam}/Keypoints/{frame}_keypoints.json')
                if cam == '6_4':
                    keypoints = []
                    minimums = []
                    for i, subject in enumerate(op2d['people']):
                        it = [iter(subject['pose_keypoints_2d'])] * 3
                        keypoints.append([k for k in zip(*it)])
                        minimums.append(min([j[0] for j in keypoints[i] if j[2] > 0]))
                    op2d = keypoints[minimums.index(min(minimums))]
                else:
                    keypoints = []
                    maximums = []
                    for i, subject in enumerate(op2d['people']):
                        it = [iter(subject['pose_keypoints_2d'])] * 3
                        keypoints.append([k for k in zip(*it)])
                        maximums.append(max([j[1] for j in keypoints[i]]))
                    op2d = keypoints[maximums.index(max(maximums))]
                op2d = compute_op2d(op2d, data['params'][cam]['intrinsics']['K'],
                                    data['params'][cam]['intrinsics']['distCoeffs'], bboxes[cam], 0.5)
                data['frames'][act][sub][lab][frame]['2D_op'][cam] = op2d

                if f > start + 1:
                    frame_0 = f'frame_{str(f - 2).zfill(6)}'
                    frame_1 = f'frame_{str(f - 1).zfill(6)}'
                    op2d_0 = data['frames'][act][sub][lab][frame_0]['2D_op'][cam]
                    op2d_1 = data['frames'][act][sub][lab][frame_1]['2D_op'][cam]
                    for point in op2d_1.keys():
                        if point in op2d_0.keys() and point in op2d.keys():
                            out = outlier_correction(op2d_0[point], op2d_1[point], op2d[point])
                            if out is not None:
                                data['frames'][act][sub][lab][frame_1]['2D_op'][cam][point] = out

    print('')
    print('Getting 3D and temporal smoothing...')
    signals = {}
    for act in data['frames'].keys():
        print(act)
        signals[act] = {}
        for sub in data['frames'][act].keys():
            print(f'--{sub}')
            signals[act][sub] = {}
            for lab in data['frames'][act][sub].keys():
                signals[act][sub][lab] = {}
                for frame in data['frames'][act][sub][lab].keys():
                    gt3d = compute_gt3d(data['frames'][act][sub][lab][frame]['2D_op'], data['params'])
                    for i, point in gt3d.items():
                        if signals[act][sub][lab].get(i, []):
                            for j in [0, 1, 2]:
                                if signals[act][sub][lab][i].get(j, []):
                                    signals[act][sub][lab][i][j].append(point[j])
                                else:
                                    signals[act][sub][lab][i][j] = [point[j]]
                        else:
                            signals[act][sub][lab][i] = {0: [point[0]], 1: [point[1]], 2: [point[2]]}

                signals[act][sub][lab] = temporal_smoothing(signals[act][sub][lab], filter)

    print('')
    print('Getting 2D projections...')
    for act in data['frames'].keys():
        print(f'{act}')
        for sub in data['frames'][act].keys():
            print(f'--{sub}')
            for lab in data['frames'][act][sub].keys():
                for f, frame in enumerate(data['frames'][act][sub][lab].keys()):
                    l = int((len(filter) - 1) / 2)
                    if l <= f < len(data['frames'][act][sub][lab].keys()) - l:
                        for i, value in signals[act][sub][lab].items():
                            pt3d = np.array([value[0][f-l], value[1][f-l], value[2][f-l]])
                            data['frames'][act][sub][lab][frame]['3D_gt'][i] = pt3d
                        gt2d = compute_gt2d(data['frames'][act][sub][lab][frame]['3D_gt'], data['params'])
                        data['frames'][act][sub][lab][frame]['2D_gt'] = gt2d

    print('')
    print('Getting individual movements...')
    data_new = {'params': data['params'], 'frames': {}}
    default_cam = '6_2'
    for act in data['frames'].keys():
        print(f'{act}')
        if act in peaks_params.keys():
            data_new['frames'][act] = {}
            for sub in data['frames'][act].keys():
                print(f'--{sub}')
                data_new['frames'][act][sub] = {}
                for lab in data['frames'][act][sub].keys():
                    data_new['frames'][act][sub][lab] = {}
                    minmax = []
                    for i, frame in enumerate(data['frames'][act][sub][lab].keys()):
                        if i == 0:
                            start = int(frame[6:])
                        elif i == len(data['frames'][act][sub][lab].keys()) - 1:
                            stop = int(frame[6:])
                        ymax = 0
                        ymin = np.inf
                        for i, point in data['frames'][act][sub][lab][frame]['2D_op'][default_cam].items():
                            if point[1] > ymax:
                                ymax = point[1]
                            if point[1] < ymin:
                                ymin = point[1]
                        minmax.append(ymax - ymin)
                    xs = moving_average(minmax, peaks_params[act]['w'])
                    peaks, _ = find_peaks(xs, prominence=1, distance=peaks_params[act]['dist'], height=550)
                    for i in range(len(peaks)):
                        if i < len(peaks) - 1:
                            data_new['frames'][act][sub][lab][i + 1] = {}
                            for n in range(start+peaks[i], start+peaks[i + 1]):
                                name = f'frame_{str(n).zfill(6)}'
                                data_new['frames'][act][sub][lab][i + 1][name] = data['frames'][act][sub][lab][name]
        else:
            # data_new['frames'][act] = data['frames'][act].copy()
            continue

    return data, data_new


data_path = '../Data/010920'
labels = pd.read_csv(f'{data_path}/videoLabelling.csv', sep=',')
cameras = ['6_1', '6_2', '6_3', '6_4']
filter = np.hamming(11) / sum(np.hamming(11))
bboxes = {'6_1': {'xmin': 278.0, 'xmax': 1141.0, 'ymin': 34.0, 'ymax': 924.0},
          '6_2': {'xmin': 632.0, 'xmax': 1557.0, 'ymin': 171.0, 'ymax': 1076.0},
          '6_3': {'xmin': 782.0, 'xmax': 1412.0, 'ymin': 126.0, 'ymax': 1075.0},
          '6_4': {'xmin': 426.0, 'xmax': 896.0, 'ymin': 256.0, 'ymax': 1074.0}}
peaks_params = {'SQUAT': {'w': 15, 'dist': 45}, 'Lunges': {'w': 20, 'dist': 60}}
data, data_new = extract_data(cameras, labels, data_path, filter, bboxes, peaks_params)
print('')
print('Writing pickle file...')
pickle_write(f'{data_path}/data.pickle', data)
pickle_write(f'{data_path}/data_act.pickle', data)
