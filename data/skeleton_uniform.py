import numpy as np
import torch
import pickle
import os
from scipy.signal import find_peaks


def params(data_3D):
    joints = {0: 'Nose', 1: 'Neck', 2: 'RShoulder', 3: 'RElbow', 4: 'RWrist', 5: 'LShoulder',
              6: 'LElbow', 7: 'LWrist', 8: 'MidHip', 9: 'RHip', 10: 'RKnee', 11: 'RAnkle', 12: 'LHip',
              13: 'LKnee', 14: 'LAnkle', 15: 'REye', 16: 'LEye', 17: 'REar', 18: 'LEar', 19: 'LBigToe',
              20: 'LSmallToe', 21: 'LHeel', 22: 'RBigToe', 23: 'RSmallToe', 24: 'RHeel'}
    joint_names = {v: k for k, v in joints.items()}
    subject_1_pose = data_3D['poses'][np.where(data_3D['labels'][:, -1] == '123')]
    bone_connections = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [10, 11],
                        [8, 12], [12, 13], [13, 14], [1, 15], [1, 16], [1, 17], [1, 18], [11, 24], [14, 21], [14, 19],
                        [14, 20], [11, 22], [11, 23]]
    return joint_names, subject_1_pose, bone_connections


def process_bone(poses, start_joint, end_joint, old_val, goal_bone_len):
    bone_vec = poses[:, :, end_joint] - old_val
    new_old_val = poses[:, :, end_joint].copy()
    new_bone_vector = goal_bone_len * bone_vec / np.linalg.norm(bone_vec, axis=1, keepdims=True)
    poses[:, :, end_joint] = poses[:, :, start_joint] + new_bone_vector
    return poses, new_old_val


def traverse_body(poses, start_joint, old_val, bone_connections, visited_list, goal_bone_len):
    for count in range(len(bone_connections)):
        list_start_joint, list_end_joint = bone_connections[count]
        if start_joint == list_start_joint and not visited_list[count]:
            visited_list[count] = True
            poses, new_old_val = process_bone(poses, list_start_joint, list_end_joint, old_val, goal_bone_len[0][count])
            traverse_body(poses, list_end_joint, new_old_val, bone_connections, visited_list, goal_bone_len)
    return poses


def find_bone_lengths(poses, bone_connections):
    ##poses are of format (N,3,D)
    bone_len = np.zeros([poses.shape[0], len(bone_connections)])
    bone_connections = np.array(bone_connections)
    i = bone_connections[:, 0]
    j = bone_connections[:, 1]
    bone_len = np.linalg.norm(poses[:, :, i] - poses[:, :, j], axis=1)
    assert bone_len.shape == (poses.shape[0], len(bone_connections))
    return bone_len


def convert_to_skeleton(poses, goal_bone_len, bone_connections):
    initial_poses = poses.copy()

    # make skeleton independent
    visited_list = [False] * len(bone_connections)

    # convert poses (this is a recursive function)
    converted_poses = traverse_body(initial_poses, bone_connections[0][0], initial_poses[:, :, bone_connections[0][0]],
                                    bone_connections, visited_list, goal_bone_len)
    fail_msg = str(np.mean(
        np.mean(np.array(find_bone_lengths(converted_poses, bone_connections)), axis=0) - np.array(goal_bone_len)))
    assert np.allclose(np.mean(np.array(find_bone_lengths(converted_poses, bone_connections)), axis=0),
                       np.array(goal_bone_len), atol=1e-7), fail_msg
    return converted_poses


def centralize_normalize_rotate_poses(poses, pose_dict):

    joint_names = pose_dict['joints']
    subject_1_pose = pose_dict['default']
    bone_connections = pose_dict['links']

    hip_index = joint_names['MidHip']

    # centralize
    hip_pose = poses[:, :, hip_index]
    normalized_poses = poses - hip_pose.unsqueeze(2)
    num_of_poses = poses.shape[0]

    fail_msg = "normalization created nans"
    assert not torch.isnan(normalized_poses).any(), fail_msg

    # make skeleton indep
    visited_list = [False] * len(bone_connections)
    subject_1_bone_len = find_bone_lengths(subject_1_pose, bone_connections)
    normalized_poses = torch.from_numpy(
        convert_to_skeleton(normalized_poses.cpu().numpy(), subject_1_bone_len, bone_connections))

    hip_pose = normalized_poses[:, :, hip_index]
    normalized_poses = normalized_poses - hip_pose.unsqueeze(2)

    # assert torch.allclose(torch.FloatTensor(find_bone_lengths(normalized_poses[:, :, :], bone_connections)),
    #                       torch.FloatTensor(subject_1_bone_len))

    # first rotation: make everyone's shoulder vector [0, 1]
    shoulder_vector = normalized_poses[:, :, joint_names['LShoulder']] - normalized_poses[:, :,
                                                                         joint_names['RShoulder']]
    spine_vector = normalized_poses[:, :, joint_names['Neck']] - normalized_poses[:, :, joint_names['MidHip']]

    shoulder_vector = shoulder_vector / torch.norm(shoulder_vector, dim=1, keepdim=True)
    spine_vector = spine_vector / torch.norm(spine_vector, dim=1, keepdim=True)

    normal_vector = torch.cross(shoulder_vector, spine_vector, dim=1)
    spine_vector = torch.cross(normal_vector, shoulder_vector, dim=1)
    assert normal_vector.shape == shoulder_vector.shape

    inv_rotation_matrix = torch.inverse(
        torch.cat([shoulder_vector.unsqueeze(2), normal_vector.unsqueeze(2), spine_vector.unsqueeze(2)], dim=2))

    rotated_poses = torch.bmm(inv_rotation_matrix, normalized_poses)

    fail_msg = "first rotation created nans"
    assert not torch.isnan(rotated_poses).any(), fail_msg

    # second rotation: make everyone's shoulder vector [0, 1]
    new_shoulder_vector = rotated_poses[:, :, joint_names['LShoulder']] - rotated_poses[:, :, joint_names['RShoulder']]
    new_shoulder_vector = new_shoulder_vector / torch.norm(new_shoulder_vector, dim=1, keepdim=True)
    new_spine_vector = rotated_poses[:, :, joint_names['Neck']] - rotated_poses[:, :, joint_names['MidHip']]
    new_spine_vector = new_spine_vector / torch.norm(new_spine_vector, dim=1, keepdim=True)
    new_normal_vector = torch.cross(new_shoulder_vector, new_spine_vector, dim=1)
    new_spine_vector = torch.cross(new_normal_vector, new_shoulder_vector, dim=1)

    # assert (torch.allclose(torch.mean(new_shoulder_vector[:, 1:], dim=0).type(torch.FloatTensor),
    #                        torch.FloatTensor([0, 0])))
    # assert (torch.allclose(torch.mean(new_normal_vector[:, [0, 2]], dim=0).type(torch.FloatTensor),
    #                        torch.FloatTensor([0, 0])))
    # assert (
    #     torch.allclose(torch.mean(new_spine_vector[:, :-1], dim=0).type(torch.FloatTensor), torch.FloatTensor([0, 0])))
    # assert (torch.allclose(torch.mean(rotated_poses[:, :, hip_index]).type(torch.FloatTensor), torch.FloatTensor([0])))

    return rotated_poses
