import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pdb
import sys
import torch
import time
import random
from progress.bar import *
from utils.skeleton_uniform import centralize_normalize_rotate_poses

def ntu_uniform():
    uniform = True

    ## Get joints data of 56578 skeletons sequences ,Get action class and subject ID
    raw_joints_path = 'ntu/SGN_output/denoised_data_for_first_actor/raw_denoised_joints.pkl'
    uniformd_savepath = '../data/ntu/3D_PC_output/ntu_uniformed.pickle'
    raw_savepath  = '../data/ntu/3D_PC_output/ntu_raw.pickle'

    action_class = np.loadtxt('../data/ntu/SGN_output/statistics/label.txt')
    subject_id = np.loadtxt('../data/ntu/SGN_output/statistics/performer.txt')
    with open(raw_joints_path, "rb") as f:
        data = pickle.load(f)  # here: data = list[np.array:(k,75),(k,75)...,len=56578] skeletons

    dataset = np.stack((data,action_class,subject_id),1)
    labels = {}
    labels['action_class'] = dataset[:,1:2].astype(int)
    labels['subject_id'] = dataset[:,2:3].astype(int)

    if uniform & os.path.exists(uniformd_savepath)==False:
        ## prepare params for uniform func
        # 1.pose_dict_ntu
        print('---Loading pose_dict_ntu.pkl---')
        pose_dict_ntu_path = 'ntu/3D_PC_output/pose_dict_ntu.pickle'
        if(not os.path.exists(pose_dict_ntu_path)):
            joints = {'MidHip':0, "Neck":20, "LShoulder":4, "RShoulder":8}
            joint_names = {v: k for k, v in joints.items()}

            subject_1_pose = data[0][0,:] #75
            subject_1_pose = subject_1_pose.reshape([1,25,3]).transpose((0,2,1))*1000
            subject_1_pose -= subject_1_pose[:,:,0:1]

            bone_connections = [[0, 1], [1, 20], [20, 2], [2, 3], [20, 4], [4, 5], [5, 6], [6, 7], [7, 22], [7, 21], [20, 8],
                                [8, 9], [9, 10], [10, 11], [11, 24], [11, 23], [0, 12], [12, 13], [13, 14], [14, 15], [0, 16],
                                [16, 17], [17, 18], [18, 19]]

            pose_dict_ntu_path  = 'ntu/3D_PC_output/pose_dict_ntu.pickle'
            with open(pose_dict_ntu_path, 'wb') as f:
                pickle.dump({'joints': joints, 'default': subject_1_pose, 'links':bone_connections}, f)

        # 2.poses: type(torch.Tensor),shape(k,3,25)
        #procedure:data = list[np.array:(k,75),(k,75)...,len=56578]——>(k*3*25)——>56578 cycles
        # poses_reshape
        print('---Loading poses_reshape_first_actor.pkl---')
        poses_reshape_path  = 'ntu/3D_PC_output/poses_reshape_first_actor.pickle'
        if(not os.path.exists(poses_reshape_path)):
            try:
                poses_reshape = []
                for i in range(len(data)):
                    for j in range(data[i].shape[0]):
                        subject_1_pose = data[i][j,:] #75
                        subject_1_pose = subject_1_pose.reshape([1,25,3]).transpose((0,2,1))*1000
                        if j == 0: 
                            poses = subject_1_pose
                        else:
                            poses = np.vstack((poses,subject_1_pose))
                    poses = torch.from_numpy(poses)
                    poses_reshape.append(poses)
                with open(poses_reshape_path, 'wb') as f:
                    pickle.dump(poses_reshape, f)
            except ValueError:
                print("!!! ValueError Found !!!")
        # pdb.set_trace()

        ## Skes Uniform
        print('---Start to perform uniform operations---')
        
        with open(pose_dict_ntu_path, 'rb') as f:
            pose_dict = pickle.load(f)
        with open(poses_reshape_path, 'rb') as f:
            skes =  pickle.load(f)

        bar = IncrementalBar('Processing', max=len(skes)-1)

        for (i,poses) in enumerate(skes):
            mask = torch.all(torch.sum(poses, dim=1)!=0, dim=1)
            poses=poses[mask]
            if len(poses)==0:
                continue
            try:
                skes_uniform = centralize_normalize_rotate_poses(poses, pose_dict)
                skes[i] = skes_uniform
            except AssertionError:
                pdb.set_trace()
            bar.next()
        bar.finish()

        # save the skes_uniform data!     
        with open(uniformd_savepath, 'wb') as f:
            pickle.dump({'labels': labels, 'skes': skes}, f)
        print("---All sequesces unfiormed---")
    elif uniform==False & os.path.exists(raw_savepath)==False:
        print('---Process dataset without poses unifrom---')
        skes = dataset[:,0]*1000
        # Save the raw data and corresponding labels
        with open(raw_savepath, 'wb') as f:
            pickle.dump({'labels': labels, 'skes': skes}, f)
        print("---All sequesces dumped without uniformed---")
    else:
        print('~~~ No code excuted ~~~')


if __name__ == '__main__':
    ntu_uniform()
    # add random noise to the coodinates of the joints
    # uniformed_path = '../data/ntu/3D_PC_output/ntu_uniformed.pickle'
    # with open('../data/ntu/3D_PC_output/ntu_uniformed.pickle', "rb") as f: 
    #     data = pickle.load(f)
    # skes = data['skes']
    # labels = data['labels']

    # bar = IncrementalBar('Processing', max=len(skes)-1, suffix = '%(percent)d%%')
    # mu = 0
    # sigma = 100
    # for i in range(len(skes)):
    #     dim0, dim1, dim2 = skes[i].shape
    #     for l in range(dim0):
    #         for n in range(dim2):
    #             skes[i][l,:,n] += random.gauss(mu,sigma) 
    #     bar.next()
    

    # with open('../data/ntu/3D_PC_output/ntu_uniformed.pickle', 'wb') as f:
    #     pickle.dump({'labels': labels, 'skes': skes}, f)
    
    # bar.finish()
