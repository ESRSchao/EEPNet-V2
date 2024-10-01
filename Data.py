import os
import math
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import random
from scipy.spatial.transform import Rotation
import cv2
import matplotlib.pyplot as plt
import pywt
from nuscenes import NuScenes
from pyquaternion import Quaternion

test_night_scene_tokens = ['e59a4d0cc6a84ed59f78fb21a45cdcb4',
                           '7209495d06f24712a063ac6c4a9b403b',
                           '3d776ea805f240bb925bd9b50b258416',
                           '48f81c548d0148fc8010a73d70b2ef9c',
                           '2ab683f384234dce89800049dec19a30',
                           '7edca4cdfadafdafdafdvccxx4e56b7e',
                           '81c939ce8c0d4cc7b159cb5ed4c4e712',
                           '24e6e64ecf794be4a51f7454c8b6d0b2',
                           '828ed34a5e0c456fbf0751cabbab3341',
                           'edfd6cfd1805477fbeadbd29f39ed599',
                           '7692a3e112b4dfsdfdcxde45954a813c',
                           '58d27a9f83294d99a4ff451dcad5f4d2',
                           'a1573aef0bf74324b373dd8a22b4dd68',
                           'ba06095d4e2e425b8e398668abc301d8',
                           '7c315a1db2ac49439d281605f3cca6be',
                           '732d7a84353f4ada803a9a115728496c',
                           '1630a1d9cf8a46b3843662a23126e3f6',
                           'f437809584344859882bdff7f8784c43']

def get_scene_lidar_token(nusc, scene_token, frame_skip=2):
    sensor = 'LIDAR_TOP'
    scene = nusc.get('scene', scene_token)
    first_sample = nusc.get('sample', scene['first_sample_token'])
    lidar = nusc.get('sample_data', first_sample['data'][sensor])

    lidar_token_list = [lidar['token']]
    counter = 1
    while lidar['next'] != '':
        lidar = nusc.get('sample_data', lidar['next'])
        counter += 1
        if counter % frame_skip == 0:
            lidar_token_list.append(lidar['token'])
    return lidar_token_list


def get_lidar_token_list(nusc, frame_skip):
    daytime_scene_list = []
    for scene in nusc.scene:
        if 'night' in scene['description'] \
                or 'Night' in scene['description'] \
                or scene['token'] in test_night_scene_tokens:
            continue
        else:
            daytime_scene_list.append(scene['token'])

    lidar_token_list = []
    for scene_token in daytime_scene_list:
        lidar_token_list += get_scene_lidar_token(nusc, scene_token, frame_skip=frame_skip)
    return lidar_token_list


def downsample(matrix):
    H = matrix.shape[0]
    W = matrix.shape[1]
    if H % 4 != 0 or W % 4 != 0:
        raise ValueError("The height and width of the input matrix must be divisible by 4.")
    if len(matrix.shape) == 3:
        downsampled_matrix = matrix[2::4, 2::4, :]
    elif len(matrix.shape) == 2:
        downsampled_matrix = matrix[2::4, 2::4]
    else: raise ValueError("Wrong shape.")
    return downsampled_matrix

def cartesian_to_spherical(cartesian_points):
    x, y, z = cartesian_points[:, 0], cartesian_points[:, 1], cartesian_points[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x) 
    theta = np.arccos(z / r)  
    return r, phi, theta


def angles2rotation_matrix(self, angles):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R

def get_P_from_Rt(R, t):
    P = np.identity(4)
    P[0:3, 0:3] = R
    P[0:3, 3] = t
    return P

def random_translation(points, max_translation=10):
    translation = np.random.uniform(-max_translation, max_translation, size=(3,))
    points[:, :2] += translation[:2]
    return points, translation

def random_rotation_z(points, max_angle=np.pi):
    angle = np.random.uniform(-max_angle, max_angle)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle), 0],
                                [0, 0, 1]])
    points = np.dot(points, rotation_matrix.T)
    return points, rotation_matrix

def tr3d2d(corrd_map, image, inter_matrix, transform_matrix, T, H, W, scale=False):
    ones_array = np.ones((corrd_map.shape[0], corrd_map.shape[1], 1))  
    corrds = np.concatenate((corrd_map, ones_array), axis=2)
    corrds_flat = corrds.reshape(-1, 4) 
    inter_matrix = inter_matrix[:, :3]  
    transform_matrix = np.matmul(transform_matrix.astype(np.float32), T.astype(np.float32))
    corrds = np.matmul(transform_matrix, corrds_flat.astype(np.float32).T)
    corrds = np.matmul(inter_matrix, corrds).T
    corrds = corrds.reshape(corrd_map.shape[0], corrd_map.shape[1], 3)
    z_coords = corrds[:, :, 2:3]
    
    positive_mask = (corrds >= 0).all(axis=2)
    positive_mask_z = (z_coords >= 0).all(axis=2)
    corrds[~positive_mask] = 0
    z_coords[~positive_mask_z] = 0

    z_coords[z_coords == 0] = 1e-9

    corrds = corrds / z_coords

    xy_points = corrds[:, :, :2]
    
    org_mask = ((corrd_map != [0, 0, 0]).any(axis=2))
    non_zero_mask = (xy_points != [0, 0]).any(axis=2)
    fov_mask = (xy_points[:, :, 0] <= W) & (xy_points[:, :, 1] <= H)
    final_mask = org_mask & non_zero_mask & fov_mask

    indices = np.argwhere(final_mask)
    good_values = xy_points[final_mask]
    good_values_int = good_values.astype(np.int64)

    binary_image1 = np.zeros((corrd_map.shape[0], corrd_map.shape[1]), dtype=np.uint8)
    binary_image2 = np.zeros((H, W), dtype=np.uint8)
    binary_image1[final_mask] = 1
    
    for value in good_values_int:
        binary_image2[value[1], value[0]] = 1
        image[value[1], value[0]] = [255,0,0]
    plt.imshow(binary_image2, cmap='gray')
    plt.title("Non-Zero Points Visualization")
    plt.axis('off')  
    plt.savefig("./KITTI_Images/scale123.png")
    
    plt.imshow(image)                              
    plt.title("Non-Zero Points Visualization")
    plt.axis('off')  
    plt.savefig("./KITTI_Images/rgb.png")
    if scale == True:
        indices = indices // 4
        good_values_int = good_values_int // 4
        binary_image1 = np.zeros((corrd_map.shape[0]//4, corrd_map.shape[1]//4), dtype=np.uint8)
        binary_image2 = np.zeros((H//4, W//4), dtype=np.uint8)
        for value in indices:
            binary_image1[value[0], value[1]] = 1
        for value in good_values_int:
            binary_image2[value[1], value[0]] = 1
        plt.imshow(binary_image1, cmap='gray')
        plt.title("Non-Zero Points Visualization")
        plt.axis('off')  
        plt.savefig("./nuScenes_Images/dw_scale909.png")

    return indices, good_values_int


class MOD_Dataset(Dataset):
    def __init__(self, mode, shuffle_data=True):
        self.mode = mode
        if (self.mode == 0):     # KITTI_Train
            self.data_root = '../DataSets/kto/data_odometry_calib/dataset'
            self.sequence_range = range(9)
            self.data = self.load_data_KITTI()
            self.proj_H = 64
            self.proj_W = 1024
            self.img_H = 160
            self.img_W = 512
            self.trans_T = 10
            self.trans_R = 2 * np.pi
        if (self.mode == 1):     # KITTI_Test
            self.data_root = '../DataSets/kto/data_odometry_calib/dataset'
            self.sequence_range = [9, 10]  
            self.data = self.load_data_KITTI()
            self.proj_H = 64
            self.proj_W = 1024
            self.img_H = 160
            self.img_W = 512
            self.trans_T = 10
            self.trans_R = 2 * np.pi
        if (self.mode == 2):     # nuScenes_Train
            self.data_root = '../' + 'v1.0-trainval'    # dataroot+version
            self.nusc = NuScenes(version='v1.0-trainval', dataroot='../', verbose=True)
            self.data = self.load_data_nuScenes()
            self.proj_H = 32
            self.proj_W = 1024
            self.img_H = 160
            self.img_W = 320
            self.trans_T = 10
            self.trans_R = 2 * np.pi
        if (self.mode == 3):     # nuScenes_Train
            self.data_root = '../' + 'v1.0-test'    # dataroot+version
            self.nusc = NuScenes(version='v1.0-test', dataroot='../', verbose=True)
            self.data = self.load_data_nuScenes()
            self.proj_H = 32
            self.proj_W = 1024
            self.img_H = 160
            self.img_W = 320
            self.trans_T = 10
            self.trans_R = 2 * np.pi
        if (self.mode == 4):     # KITTI_Train
            self.data_root = '../DataSets/kto/data_odometry_calib/dataset'
            self.sequence_range = range(9)
            self.data = self.load_data_KITTI()
            self.proj_H = 64
            self.proj_W = 1024
            self.img_H = 160
            self.img_W = 512
            self.trans_T = 10
            self.trans_R = 2 * np.pi
        if (self.mode == 5):     # KITTI_Test
            self.data_root = '../DataSets/kto/data_odometry_calib/dataset'
            self.sequence_range = [9, 10]  
            self.data = self.load_data_KITTI()
            self.proj_H = 64
            self.proj_W = 1024
            self.img_H = 160
            self.img_W = 512
            self.trans_T = 10
            self.trans_R = 2 * np.pi
        if (self.mode == 6):     # nuScenes_Train
            self.data_root = '../' + 'v1.0-trainval'    # dataroot+version
            self.nusc = NuScenes(version='v1.0-trainval', dataroot='../', verbose=True)
            self.data = self.load_data_nuScenes()
            self.proj_H = 32
            self.proj_W = 1024
            self.img_H = 160
            self.img_W = 320
            self.trans_T = 10
            self.trans_R = 2 * np.pi
        if (self.mode == 7):     # nuScenes_Train
            self.data_root = '../' + 'v1.0-test'    # dataroot+version
            self.nusc = NuScenes(version='v1.0-test', dataroot='../', verbose=True)
            self.data = self.load_data_nuScenes()
            self.proj_H = 32
            self.proj_W = 1024
            self.img_H = 160
            self.img_W = 320
            self.trans_T = 10
            self.trans_R = 2 * np.pi
        if shuffle_data:
            random.shuffle(self.data)  # 打乱数据

    def load_data_KITTI(self):
        data = []
        for sequence_num in self.sequence_range:
            sequence_dir = os.path.join(self.data_root, f"sequences/{sequence_num:02d}")
            image2_dir = os.path.join(sequence_dir, "image_2")
            image3_dir = os.path.join(sequence_dir, "image_3")
            pointcloud_dir = os.path.join(sequence_dir, "velodyne")
            pose_dir = os.path.join(sequence_dir, 'calib.txt')

            pointcloud_files = sorted(os.listdir(pointcloud_dir))
            image2_files = sorted(os.listdir(image2_dir))
            image3_files = sorted(os.listdir(image3_dir))

            for pointcloud_file, image2_file, image3_file in zip(pointcloud_files, image2_files, image3_files):
                pointcloud_path = os.path.join(pointcloud_dir, pointcloud_file)
                image2_path = os.path.join(image2_dir, image2_file)
                image3_path = os.path.join(image3_dir, image3_file)
                data.append((pointcloud_path, image2_path, pose_dir, 'P2'))
                data.append((pointcloud_path, image3_path, pose_dir, 'P3'))
        return data
    
    def load_data_nuScenes(self):
        data = []
        # lidar_path_list = []
        # camera_path_list = []
        lidar_token_list = get_lidar_token_list(self.nusc,
                                            frame_skip=1)
        for i, lidar_token in enumerate(lidar_token_list):
            lidar = self.nusc.get('sample_data', lidar_token)
            lidar_sample_token = lidar['sample_token']
            lidar_sample = self.nusc.get('sample', lidar_sample_token)

            init_camera_token = lidar_sample['data']['CAM_FRONT']
            init_camera = self.nusc.get('sample_data', init_camera_token)

            # lidar_path_list.append(lidar['filename'])
            # camera_path_list.append(init_camera['filename'])
            calib_data_cam = self.nusc.get('calibrated_sensor', init_camera['calibrated_sensor_token'])
            inter_matrix = np.array(calib_data_cam['camera_intrinsic'])

            # step1: lidar frame -> ego frame
            calib_data_lidar = self.nusc.get('calibrated_sensor', lidar['calibrated_sensor_token'])
            rot_matrix1 = np.asarray(Quaternion(calib_data_lidar['rotation']).rotation_matrix).astype(np.float32)
            tran_matrix1 = np.asarray(calib_data_lidar['translation']).astype(np.float32)
            P1 = get_P_from_Rt(rot_matrix1, tran_matrix1)

            # step2: ego frame -> global frame
            ego_data_lidar = self.nusc.get('ego_pose', lidar['ego_pose_token'])
            rot_matrix2 = np.asarray(Quaternion(ego_data_lidar['rotation']).rotation_matrix).astype(np.float32)
            tran_matrix2 = np.asarray(ego_data_lidar['translation']).astype(np.float32)
            P2 = get_P_from_Rt(rot_matrix2, tran_matrix2)
            # rot_matrix = np.dot(rot_matrix_ego_global, rot_matrix)
            # tran_matrix = tran_matrix + translation_ego_global 

            # step3: global frame -> ego frame
            ego_data_cam = self.nusc.get('ego_pose', init_camera['ego_pose_token'])
            tran_matrix3 = np.asarray(ego_data_cam['translation']).astype(np.float32)
            rot_matrix3 = np.asarray(Quaternion(ego_data_cam['rotation']).rotation_matrix).astype(np.float32)
            P3 = get_P_from_Rt(rot_matrix3, tran_matrix3)
            P3 = np.linalg.inv(P3)
            # rot_matrix = np.dot(rot_matrix_ego_cam, rot_matrix)
            # tran_matrix = tran_matrix - translation_cam_ego

            # step4: ego frame -> cam frame
            calib_data_cam = self.nusc.get('calibrated_sensor', init_camera['calibrated_sensor_token'])
            tran_matrix4 = np.asarray(calib_data_cam['translation']).astype(np.float32)
            rot_matrix4 = np.asarray(Quaternion(calib_data_cam['rotation']).rotation_matrix).astype(np.float32)
            P4 = get_P_from_Rt(rot_matrix4, tran_matrix4)
            P4 = np.linalg.inv(P4)

            transform_matrix = np.dot(P4, np.dot(P3, np.dot(P2, P1)))[:3, :]
            pointcloud_file = os.path.join(self.data_root, lidar['filename'])
            image = os.path.join(self.data_root, init_camera['filename'])
            data.append((pointcloud_file, image, transform_matrix, inter_matrix))
        return data
    def augment_img(self, img_np):
        brightness = (0.8, 1.2)
        contrast = (0.8, 1.2)
        saturation = (0.8, 1.2)
        hue = (-0.1, 0.1)
        color_aug = transforms.ColorJitter(
            brightness, contrast, saturation, hue)
        img_color_aug_np = np.array(color_aug(Image.fromarray(img_np)))

        return img_color_aug_np
    
    def camera_matrix_scaling(self, K: np.ndarray, s: float):
        K_scale = s * K
        K_scale[2, 2] = 1
        return K_scale
    
    def camera_matrix_cropping(self, K: np.ndarray, dx: float, dy: float):
        K_crop = np.copy(K)
        K_crop[0, 2] -= dx
        K_crop[1, 2] -= dy
        return K_crop

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if (self.mode in (0, 1)):
            pointcloud_path, image_path, pose_dir, Kidx = self.data[idx]
        
            # pc
            pointcloud_full = np.fromfile(pointcloud_path, dtype=np.float32)
            pointcloud_full = pointcloud_full.reshape(-1, 4)        
            pointcloud_i = pointcloud_full[:, 3]  
            pointcloud  = pointcloud_full[:, :3]  
            
            #img
            img = cv2.imread(image_path)
            img = cv2.resize(img,
                            (int(round(img.shape[1] * 0.5)),
                            int(round((img.shape[0] * 0.5)))),
                            interpolation=cv2.INTER_LINEAR)

            # if self.mode == 0:
            #     img_crop_dx = random.randint(0, img.shape[1] - self.img_W)
            #     img_crop_dy = random.randint(0, img.shape[0] - self.img_H)
            # else:
            img_crop_dx = int((img.shape[1] - self.img_W) / 2)
            img_crop_dy = int((img.shape[0] - self.img_H) / 2)
            img = img[img_crop_dy:img_crop_dy + self.img_H,
                img_crop_dx:img_crop_dx + self.img_W  , :]

            if self.mode == 0:
                img = self.augment_img(img)

            # calib
            with open(pose_dir, "r") as calib_file:
                lines = calib_file.readlines()
            if (Kidx == 'P2'): 
                inter_matrix_str = lines[2].strip().split(" ")[1:]  
            elif (Kidx == 'P3'): 
                inter_matrix_str = lines[3].strip().split(" ")[1:]  
            transform_matrix_str = lines[-1].strip().split(" ")[1:] 
            inter_matrix = [float(x) for x in inter_matrix_str]
            transform_matrix = [float(y) for y in transform_matrix_str]
            inter_matrix = np.array(inter_matrix).reshape(3, 4)
            K = inter_matrix[:3, :3]
            K = self.camera_matrix_scaling(K, 0.5)
            K = self.camera_matrix_cropping(K, dx=img_crop_dx, dy=img_crop_dy)
        
            transform_matrix = np.array(transform_matrix).reshape(3, 4)

            fx = inter_matrix[0, 0]
            fy = inter_matrix[1, 1]
            cx = inter_matrix[0, 2]
            cy = inter_matrix[1, 2]
            tz = inter_matrix[2, 3]
            tx = (inter_matrix[0, 3] - cx * tz) / fx
            ty = (inter_matrix[1, 3] - cy * tz) / fy

            transform_matrix[0, 3] += tx
            transform_matrix[1, 3] += ty
            transform_matrix[2, 3] += tz
            inter_matrix[:3, :3] = K

            pointcloud, rotation_mat = random_rotation_z(pointcloud, max_angle=self.trans_R)

            pointcloud_xyz, translation_mat = random_translation(pointcloud, max_translation=self.trans_T)
            # pointcloud = np.concatenate((pointcloud, np.expand_dims(pointcloud_i, axis=1)), axis=1)

            proj_x = 0.5 * (phi / np.pi + 1.0)          # in [0.0, 1.0]
            proj_y = change 
            proj_x *= self.proj_W                              # in [0.0, W]

            # round and clamp for use as index
            proj_x = np.floor(proj_x)
            proj_x = np.minimum(self.proj_W - 1, proj_x)
            proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]

            proj_y = np.floor(proj_y)
            proj_y = np.minimum(self.proj_H - 1, proj_y)
            proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]

            indices = np.arange(depth.shape[0])
            order = np.argsort(depth)[::-1]

            reflect = pointcloud_i + 1/255
            corrd = pointcloud_xyz
            
            reflect = reflect[order]
            corrd = corrd[order]
            depth = depth[order]
            indices = indices[order]
            proj_y = proj_y[order]
            proj_x = proj_x[order]

            proj_range[proj_y, proj_x] = reflect
            proj_depth[proj_y, proj_x] = depth
            corrd_map[proj_y, proj_x] = corrd
            image3d_R = (proj_range / proj_range.max() * 255).astype(np.uint8)
            image3d_D = (proj_depth / proj_depth.max() * 255).astype(np.uint8)
            # proj_range_down = downsample(proj_range)
            # proj_map_down = downsample(proj_depth)
            corrd_map_down = downsample(corrd_map)


            T = np.eye(4)
            T[:3, :3] = rotation_mat
            T[:2, 3] = translation_mat[:2]

            T_inv = np.linalg.inv(T).astype(np.float32)
            # pair_3d, pair_2d = tr3d2d(corrd_map, img, inter_matrix, transform_matrix, T_inv, self.img_H, self.img_W, scale=False)
            # pair_3d_down, pair_2d_down = tr3d2d(corrd_map_down, inter_matrix, transform_matrix, T_inv, self.img_H, self.img_W, scale=True)

            # # 将图像、点云数据和反射率数据作为输入返回
            # if (Kidx == 'P2'):
            #     filename = pointcloud_path.replace('velodyne', 'test_P2_ACCD')
            # elif (Kidx == 'P3'):
            #     filename = pointcloud_path.replace('velodyne', 'test_P3_ACCD')
            # with open(filename, 'wb') as fout:
            #     fout.write(np.array(pc_3d, dtype=np.float32).tobytes())
            #     fout.write(np.array(img[:, :, 2], dtype=np.int16).tobytes())
            #     fout.write(np.array(image3d, dtype=np.int16).tobytes())
            #     fout.write(np.array(kp2d, dtype=np.int16).tobytes())
            #     fout.write(np.array(kp3d, dtype=np.int16).tobytes())
            #     fout.write(np.array(inter_matrix, dtype=np.float32).tobytes())
            #     fout.write(np.array(transform_matrix, dtype=np.float32).tobytes())
            #     fout.write(np.array(T_inv, dtype=np.float32).tobytes())


            return {
                'image_2d': torch.from_numpy(img.astype(np.float32) / 255.),
                'image_3d_R': torch.from_numpy(image3d_R.astype(np.float32) / 255.),
                'image_3d_D': torch.from_numpy(image3d_D.astype(np.float32) / 255.),
                'pc_2d': corrd_map,
                'pc_2d_scaled': corrd_map_down,
                # 'pair_3d': pair_3d,
                # 'pair_2d': pair_2d,
                # 'pair_3d_down': pair_3d_down,
                # 'pair_2d_down': pair_2d_down,
                'inter_matrix': inter_matrix,
                'transform_matrix': transform_matrix,
                'T_inv': T_inv,
                }

#############################################         nuScenes         #################################################
        
        if (self.mode in (2, 3)):
            pointcloud_path, image_path, transform_matrix, inter_matrix = self.data[idx]
            # pc
            pointcloud_full = np.fromfile(pointcloud_path, dtype=np.float32)
            lidar_pt_list = pointcloud_full.reshape((-1, 5))[:, :4]       
            lidar_pt_index = pointcloud_full.reshape((-1, 5))[:, 4] 
            
            #img
            img = cv2.imread(image_path)
            img = cv2.resize(img,
                            (int(round(img.shape[1] * 0.25)),
                            int(round((img.shape[0] * 0.25)))),
                            interpolation=cv2.INTER_LINEAR)

            # if self.mode == 2:
            #     img_crop_dx = random.randint(0, img.shape[1] - self.img_W)
            #     img_crop_dy = random.randint(0, img.shape[0] - self.img_H)
            # else:
            img_crop_dx = int((img.shape[1] - self.img_W) / 2)
            img_crop_dy = int((img.shape[0] - self.img_H) / 2)
            img = img[img_crop_dy:img_crop_dy + self.img_H,
                img_crop_dx:img_crop_dx + self.img_W  , :]

            if self.mode == 2:
                img = self.augment_img(img)

            # calib
            K = inter_matrix
            K = self.camera_matrix_scaling(K, 0.25)
            K = self.camera_matrix_cropping(K, dx=img_crop_dx, dy=img_crop_dy)
            inter_matrix = K

            unique_scan_lines = np.unique(lidar_pt_index)
            grouped_points = [lidar_pt_list[lidar_pt_index == line] for line in unique_scan_lines]
            grouped_idx = [lidar_pt_index[lidar_pt_index == line] for line in unique_scan_lines]    
            grouped_points = np.array(grouped_points)
            grouped_idx = np.array(grouped_idx)
            proj_range = np.full((self.proj_H, self.proj_W), 0, dtype=np.float32)
            proj_depth = np.full((self.proj_H, self.proj_W), 0, dtype=np.float32)
            corrd_map = np.full((self.proj_H, self.proj_W, 3), [0, 0, 0], dtype=np.float32)

            change = 31 - grouped_idx.reshape((-1))


            scan = grouped_points
            scan = scan.reshape((-1, 4))
            pointcloud = scan[:, 0:3]    # get xyz
            
            pointcloud, rotation_mat = random_rotation_z(pointcloud, max_angle=self.trans_R)
            pointcloud_i = scan[:, 3]  # get remission

            depth, _, _ = cartesian_to_spherical(pointcloud)
            change = 31 - grouped_idx.reshape((-1))

            # get scan components
            scan_x = pointcloud[:, 0]
            scan_y = pointcloud[:, 1]
            scan_z = pointcloud[:, 2]

            # get angles of all points
            yaw = - np.arctan2(scan_y, scan_x)
            pitch = np.arcsin(scan_z / depth)

            pointcloud_xyz, translation_mat = random_translation(pointcloud, max_translation=self.trans_T)
            # pointcloud = np.concatenate((pointcloud, np.expand_dims(pointcloud_i, axis=1)), axis=1)

            # get projections in image coords
            proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
            proj_y = change / self.proj_H

            # scale to image size using angular resolution
            proj_x *= self.proj_W                              # in [0.0, W]
            proj_y *= self.proj_H                              # in [0.0, H]
            # round and clamp for use as index
            proj_x = np.floor(proj_x)
            proj_x = np.minimum(self.proj_W - 1, proj_x)
            proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]

            proj_y = np.floor(proj_y)
            proj_y = np.minimum(self.proj_H - 1, proj_y)
            proj_y = np.maximum(0, proj_y).astype(np.int32)


            corrd_map_down = downsample(corrd_map)


            T = np.eye(4)
            T[:3, :3] = rotation_mat
            T[:2, 3] = translation_mat[:2]

            T_inv = np.linalg.inv(T).astype(np.float32)
            # pair_3d, pair_2d = tr3d2d(corrd_map, img, inter_matrix, transform_matrix, T_inv, self.img_H, self.img_W, scale=False)
            # pair_3d_down, pair_2d_down = tr3d2d(corrd_map_down, inter_matrix, transform_matrix, T_inv, self.img_H, self.img_W, scale=True)

            # # 将图像、点云数据和反射率数据作为输入返回
            # if (Kidx == 'P2'):
            #     filename = pointcloud_path.replace('velodyne', 'test_P2_ACCD')
            # elif (Kidx == 'P3'):
            #     filename = pointcloud_path.replace('velodyne', 'test_P3_ACCD')
            # with open(filename, 'wb') as fout:
            #     fout.write(np.array(pc_3d, dtype=np.float32).tobytes())
            #     fout.write(np.array(img[:, :, 2], dtype=np.int16).tobytes())
            #     fout.write(np.array(image3d, dtype=np.int16).tobytes())
            #     fout.write(np.array(kp2d, dtype=np.int16).tobytes())
            #     fout.write(np.array(kp3d, dtype=np.int16).tobytes())
            #     fout.write(np.array(inter_matrix, dtype=np.float32).tobytes())
            #     fout.write(np.array(transform_matrix, dtype=np.float32).tobytes())
            #     fout.write(np.array(T_inv, dtype=np.float32).tobytes())


            return {
                'image_2d': torch.from_numpy(img.astype(np.float32) / 255.),
                'image_3d_R': torch.from_numpy(image3d_R.astype(np.float32) / 255.),
                'image_3d_D': torch.from_numpy(image3d_D.astype(np.float32) / 255.),
                'pc_2d': corrd_map,
                'pc_2d_scaled': corrd_map_down,
                # 'pair_3d': pair_3d,
                # 'pair_2d': pair_2d,
                # 'pair_3d_down': pair_3d_down,
                # 'pair_2d_down': pair_2d_down,
                'inter_matrix': inter_matrix,
                'transform_matrix': transform_matrix,
                'T_inv': T_inv,
                }
############################################ KITTI  ###########################################

        if (self.mode in (4, 5)):
            pointcloud_path, image_path, pose_dir, Kidx = self.data[idx]
        
            # pc
            pointcloud_full = np.fromfile(pointcloud_path, dtype=np.float32)
            pointcloud_full = pointcloud_full.reshape(-1, 4)        
            pointcloud_i = pointcloud_full[:, 3]  
            pointcloud  = pointcloud_full[:, :3]  
            
            #img
            img = cv2.imread(image_path)
            img = cv2.resize(img,
                            (int(round(img.shape[1] * 0.5)),
                            int(round((img.shape[0] * 0.5)))),
                            interpolation=cv2.INTER_LINEAR)
            img=img[:, :, 2]

            img_crop_dx = int((img.shape[1] - self.img_W) / 2)
            img_crop_dy = int((img.shape[0] - self.img_H) / 2)
            img = img[img_crop_dy:img_crop_dy + self.img_H,
                img_crop_dx:img_crop_dx + self.img_W]

            if self.mode == 4:
                img = self.augment_img(img)
            img = np.expand_dims(img, axis=-1)
            # calib
            with open(pose_dir, "r") as calib_file:
                lines = calib_file.readlines()
            if (Kidx == 'P2'): 
                inter_matrix_str = lines[2].strip().split(" ")[1:]  
            elif (Kidx == 'P3'): 
                inter_matrix_str = lines[3].strip().split(" ")[1:]  
            transform_matrix_str = lines[-1].strip().split(" ")[1:] 
            inter_matrix = [float(x) for x in inter_matrix_str]
            transform_matrix = [float(y) for y in transform_matrix_str]
            inter_matrix = np.array(inter_matrix).reshape(3, 4)
            K = inter_matrix[:3, :3]
            K = self.camera_matrix_scaling(K, 0.5)
            K = self.camera_matrix_cropping(K, dx=img_crop_dx, dy=img_crop_dy)
        
            transform_matrix = np.array(transform_matrix).reshape(3, 4)

            fx = inter_matrix[0, 0]
            fy = inter_matrix[1, 1]
            cx = inter_matrix[0, 2]
            cy = inter_matrix[1, 2]
            tz = inter_matrix[2, 3]
            tx = (inter_matrix[0, 3] - cx * tz) / fx
            ty = (inter_matrix[1, 3] - cy * tz) / fy

            transform_matrix[0, 3] += tx
            transform_matrix[1, 3] += ty
            transform_matrix[2, 3] += tz
            inter_matrix[:3, :3] = K

            pointcloud, rotation_mat = random_rotation_z(pointcloud, max_angle=self.trans_R)

            pointcloud_xyz, translation_mat = random_translation(pointcloud, max_translation=self.trans_T)
            # pointcloud = np.concatenate((pointcloud, np.expand_dims(pointcloud_i, axis=1)), axis=1)

            proj_x = 0.5 * (phi / np.pi + 1.0)          # in [0.0, 1.0]
            proj_y = change 
            proj_x *= self.proj_W                              # in [0.0, W]

            # round and clamp for use as index
            proj_x = np.floor(proj_x)
            proj_x = np.minimum(self.proj_W - 1, proj_x)
            proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]

            proj_y = np.floor(proj_y)
            proj_y = np.minimum(self.proj_H - 1, proj_y)
            proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]

            indices = np.arange(depth.shape[0])
            order = np.argsort(depth)[::-1]

            reflect = pointcloud_i + 1/255
            corrd = pointcloud_xyz
            
            reflect = reflect[order]
            corrd = corrd[order]
            depth = depth[order]
            indices = indices[order]
            proj_y = proj_y[order]
            proj_x = proj_x[order]

            proj_range[proj_y, proj_x] = reflect
            proj_depth[proj_y, proj_x] = depth
            corrd_map[proj_y, proj_x] = corrd
            image3d_R = (proj_range / proj_range.max() * 255).astype(np.uint8)
            image3d_D = (proj_depth / proj_depth.max() * 255).astype(np.uint8)
            # proj_range_down = downsample(proj_range)
            # proj_map_down = downsample(proj_depth)
            corrd_map_down = downsample(corrd_map)


            T = np.eye(4)
            T[:3, :3] = rotation_mat
            T[:2, 3] = translation_mat[:2]

            T_inv = np.linalg.inv(T).astype(np.float32)
            # pair_3d, pair_2d = tr3d2d(corrd_map, img, inter_matrix, transform_matrix, T_inv, self.img_H, self.img_W, scale=False)
            # pair_3d_down, pair_2d_down = tr3d2d(corrd_map_down, inter_matrix, transform_matrix, T_inv, self.img_H, self.img_W, scale=True)

            # # 将图像、点云数据和反射率数据作为输入返回
            # if (Kidx == 'P2'):
            #     filename = pointcloud_path.replace('velodyne', 'test_P2_ACCD')
            # elif (Kidx == 'P3'):
            #     filename = pointcloud_path.replace('velodyne', 'test_P3_ACCD')
            # with open(filename, 'wb') as fout:
            #     fout.write(np.array(pc_3d, dtype=np.float32).tobytes())
            #     fout.write(np.array(img[:, :, 2], dtype=np.int16).tobytes())
            #     fout.write(np.array(image3d, dtype=np.int16).tobytes())
            #     fout.write(np.array(kp2d, dtype=np.int16).tobytes())
            #     fout.write(np.array(kp3d, dtype=np.int16).tobytes())
            #     fout.write(np.array(inter_matrix, dtype=np.float32).tobytes())
            #     fout.write(np.array(transform_matrix, dtype=np.float32).tobytes())
            #     fout.write(np.array(T_inv, dtype=np.float32).tobytes())


            return {
                'image_2d': torch.from_numpy(img.astype(np.float32) / 255.),
                'image_3d_R': torch.from_numpy(image3d_R.astype(np.float32) / 255.),
                'image_3d_D': torch.from_numpy(image3d_D.astype(np.float32) / 255.),
                'pc_2d': corrd_map,
                'pc_2d_scaled': corrd_map_down,
                # 'pair_3d': pair_3d,
                # 'pair_2d': pair_2d,
                # 'pair_3d_down': pair_3d_down,
                # 'pair_2d_down': pair_2d_down,
                'inter_matrix': inter_matrix,
                'transform_matrix': transform_matrix,
                'T_inv': T_inv,
                }
        

#############################################         nuScenes         #################################################
        
        if (self.mode in (6, 7)):
            pointcloud_path, image_path, transform_matrix, inter_matrix = self.data[idx]
            # pc
            pointcloud_full = np.fromfile(pointcloud_path, dtype=np.float32)
            lidar_pt_list = pointcloud_full.reshape((-1, 5))[:, :4]       
            lidar_pt_index = pointcloud_full.reshape((-1, 5))[:, 4] 
            
                        #img
            img = cv2.imread(image_path)
            img = cv2.resize(img,
                            (int(round(img.shape[1] * 0.5)),
                            int(round((img.shape[0] * 0.5)))),
                            interpolation=cv2.INTER_LINEAR)
            img=img[:, :, 2]
            # if self.mode == 0:
            #     img_crop_dx = random.randint(0, img.shape[1] - self.img_W)
            #     img_crop_dy = random.randint(0, img.shape[0] - self.img_H)
            # else:
            img_crop_dx = int((img.shape[1] - self.img_W) / 2)
            img_crop_dy = int((img.shape[0] - self.img_H) / 2)
            img = img[img_crop_dy:img_crop_dy + self.img_H,
                img_crop_dx:img_crop_dx + self.img_W]

            if self.mode == 6:
                img = self.augment_img(img)
            img = np.expand_dims(img, axis=-1)
            
            # calib
            K = inter_matrix
            K = self.camera_matrix_scaling(K, 0.25)
            K = self.camera_matrix_cropping(K, dx=img_crop_dx, dy=img_crop_dy)
            inter_matrix = K

            unique_scan_lines = np.unique(lidar_pt_index)
            grouped_points = [lidar_pt_list[lidar_pt_index == line] for line in unique_scan_lines]
            grouped_idx = [lidar_pt_index[lidar_pt_index == line] for line in unique_scan_lines]    
            grouped_points = np.array(grouped_points)
            grouped_idx = np.array(grouped_idx)
            proj_range = np.full((self.proj_H, self.proj_W), 0, dtype=np.float32)
            proj_depth = np.full((self.proj_H, self.proj_W), 0, dtype=np.float32)
            corrd_map = np.full((self.proj_H, self.proj_W, 3), [0, 0, 0], dtype=np.float32)

            change = 31 - grouped_idx.reshape((-1))


            scan = grouped_points
            scan = scan.reshape((-1, 4))
            pointcloud = scan[:, 0:3]    # get xyz
            
            pointcloud, rotation_mat = random_rotation_z(pointcloud, max_angle=self.trans_R)
            pointcloud_i = scan[:, 3]  # get remission

            depth, _, _ = cartesian_to_spherical(pointcloud)
            change = 31 - grouped_idx.reshape((-1))

            # get scan components
            scan_x = pointcloud[:, 0]
            scan_y = pointcloud[:, 1]
            scan_z = pointcloud[:, 2]

            # get angles of all points
            yaw = - np.arctan2(scan_y, scan_x)
            pitch = np.arcsin(scan_z / depth)

            pointcloud_xyz, translation_mat = random_translation(pointcloud, max_translation=self.trans_T)
            # pointcloud = np.concatenate((pointcloud, np.expand_dims(pointcloud_i, axis=1)), axis=1)

            # get projections in image coords
            proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
            proj_y = change / self.proj_H

            # scale to image size using angular resolution
            proj_x *= self.proj_W                              # in [0.0, W]
            proj_y *= self.proj_H                              # in [0.0, H]
            # round and clamp for use as index
            proj_x = np.floor(proj_x)
            proj_x = np.minimum(self.proj_W - 1, proj_x)
            proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]

            proj_y = np.floor(proj_y)
            proj_y = np.minimum(self.proj_H - 1, proj_y)
            proj_y = np.maximum(0, proj_y).astype(np.int32)


            # proj_range_down = downsample(proj_range)
            # proj_map_down = downsample(proj_depth)
            corrd_map_down = downsample(corrd_map)


            T = np.eye(4)
            T[:3, :3] = rotation_mat
            T[:2, 3] = translation_mat[:2]

            T_inv = np.linalg.inv(T).astype(np.float32)
            # pair_3d, pair_2d = tr3d2d(corrd_map, img, inter_matrix, transform_matrix, T_inv, self.img_H, self.img_W, scale=False)
            # pair_3d_down, pair_2d_down = tr3d2d(corrd_map_down, inter_matrix, transform_matrix, T_inv, self.img_H, self.img_W, scale=True)

            # # 将图像、点云数据和反射率数据作为输入返回
            # if (Kidx == 'P2'):
            #     filename = pointcloud_path.replace('velodyne', 'test_P2_ACCD')
            # elif (Kidx == 'P3'):
            #     filename = pointcloud_path.replace('velodyne', 'test_P3_ACCD')
            # with open(filename, 'wb') as fout:
            #     fout.write(np.array(pc_3d, dtype=np.float32).tobytes())
            #     fout.write(np.array(img[:, :, 2], dtype=np.int16).tobytes())
            #     fout.write(np.array(image3d, dtype=np.int16).tobytes())
            #     fout.write(np.array(kp2d, dtype=np.int16).tobytes())
            #     fout.write(np.array(kp3d, dtype=np.int16).tobytes())
            #     fout.write(np.array(inter_matrix, dtype=np.float32).tobytes())
            #     fout.write(np.array(transform_matrix, dtype=np.float32).tobytes())
            #     fout.write(np.array(T_inv, dtype=np.float32).tobytes())


            return {
                'image_2d': torch.from_numpy(img.astype(np.float32) / 255.),
                'image_3d_R': torch.from_numpy(image3d_R.astype(np.float32) / 255.),
                'image_3d_D': torch.from_numpy(image3d_D.astype(np.float32) / 255.),
                'pc_2d': corrd_map,
                'pc_2d_scaled': corrd_map_down,
                # 'pair_3d': pair_3d,
                # 'pair_2d': pair_2d,
                # 'pair_3d_down': pair_3d_down,
                # 'pair_2d_down': pair_2d_down,
                'inter_matrix': inter_matrix,
                'transform_matrix': transform_matrix,
                'T_inv': T_inv,
                }


# train_data = MOD_Dataset(mode=0)
# train_loader = DataLoader(
#                         dataset=train_data, 
#                         batch_size=4,
#                         shuffle=True, 
#                         drop_last=True,
#                         # num_workers=4,  
#                         pin_memory=True 
#                         )
# for batch_idx, batch in enumerate(train_loader):
#     data = batch
#     print(data)