import gc
import os
import pickle
import random
import joblib
import pandas as pd
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from Data import MOD_Dataset
import cv2
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

class CFG:
    PREPROCESS = False
    EPOCHS = 30 #20
    LR = 1e-3
    WD = 0.05
    BATCH_SIZE = 8
    NBR_FOLDS = 15
    SELECTED_FOLDS = [0]
    DEVICE = 3
    SEED = 2024
    Mode = 5
    XPATH = './K_R.xlsx'
    TOPK = 500
    MODPATH = "./ck_K_R/best_model-v5.ckpt"

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    #tf.random.set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seeds(seed=CFG.SEED)

def sigmoid_log_double_softmax(
    sim: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor
) -> torch.Tensor:
    """create the log assignment matrix from logits and similarity"""
    b, m, n = sim.shape
    oa = F.logsigmoid(z0)
    ob = F.logsigmoid(z1)
    certainties = oa + ob.transpose(1, 2)
    scores0 = F.log_softmax(sim, 2)
    scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    scores = sim.new_full((b, m + 1, n + 1), 0)
    scores[:, :m, :n] = scores0 + scores1 + certainties
    scores[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))
    scores[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))
    return scores, oa, ob

def filter_matches(scores: torch.Tensor, th: float):
    """obtain matches from a log assignment matrix [Bx M+1 x N+1]"""
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    m0, m1 = max0.indices, max1.indices
    indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
    indices1 = torch.arange(m1.shape[1], device=m1.device)[None]
    mutual0 = indices0 == m1.gather(1, m0)
    mutual1 = indices1 == m0.gather(1, m1)
    max0_exp = max0.values.exp()
    zero = max0_exp.new_tensor(0)
    mscores0 = torch.where(mutual0, max0_exp, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
    valid0 = mutual0 & (mscores0 > th)
    valid1 = mutual1 & valid0.gather(1, m1)
    m0 = torch.where(valid0, m0, -1)
    m1 = torch.where(valid1, m1, -1)
    return m0, m1, mscores0, mscores1

class MatchAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """build assignment matrix from descriptors"""
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**0.25, mdesc1 / d**0.25
        sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores, oa, ob = sigmoid_log_double_softmax(sim, z0, z1)
        return scores, sim, oa, ob

    def get_matchability(self, desc: torch.Tensor):
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)

def tr3d2d(corrd_map, inter_matrix, transform_matrix, T, H, W, scale=False):
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
    good_values_int = good_values.astype(np.int64)[:, [1, 0]]
    good_values_int_scale = good_values_int // 4
    return indices, good_values_int, good_values_int_scale

def batch_convert_to_reshape_indices(arrays, original_shape):
    height, width = original_shape
    results = []
    for positions in arrays:
        a1 = positions[0]  
        a2 = positions[1]  
        k = a1 * width + a2
        results.append(k)
    
    return results

def batch_convert_from_reshape_indices(indices, original_shape):
    height, width = original_shape
    results = []
    for k in indices:
        a1 = k // width  
        a2 = k % width 
        results.append([a1, a2])   
    return results

# def corrd_grouping(downsampled_coords, scale_factor=4):
#     N = downsampled_coords.shape[0]
#     patch_size = scale_factor ** 2  # 对于 4x 下采样，patch_size 是 16

#     # 初始化输出的原图大小的坐标列表
#     original_coords = np.zeros((N * patch_size, 2), dtype=int)

#     # 将下采样后的坐标映射回原图大小的patch
#     idx = 0
#     for coord in downsampled_coords:
#         x, y = coord
#         for m in range(scale_factor):
#             for n in range(scale_factor):
#                 original_coords[idx] = np.array([x * scale_factor + m, y * scale_factor + n])
#                 idx += 1

#     return original_coords


def F_grouping(F2f, F3f, good_values_int_scale, indices_c):
    F_2D_16 = []
    F_3D_16 = []
    for I_2D, p_2d in enumerate(good_values_int_scale):
        group_F_2D = F2f[:, p_2d[0] * 4:(p_2d[0] + 1) * 4, p_2d[1] * 4:(p_2d[1] + 1) * 4]
        F_2D_16.append(group_F_2D.reshape(group_F_2D.shape[0], 16))
        p_3d = indices_c[I_2D]
        group_F_3D = F3f[:, p_3d[0] * 4:(p_3d[0] + 1) * 4, p_3d[1] * 4:(p_3d[1] + 1) * 4]
        F_3D_16.append(group_F_3D.reshape(group_F_3D.shape[0], 16))
    return F_2D_16, F_3D_16
class MatchAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """build assignment matrix from descriptors"""
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**0.25, mdesc1 / d**0.25
        sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores, oa, ob = sigmoid_log_double_softmax(sim, z0, z1)
        return scores, sim, oa, ob

    def get_matchability(self, desc: torch.Tensor):
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)            

class Match_C(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, top_k : int) -> None:
        super().__init__()
        self.mlpc0 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.mlpc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.top_k = top_k

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """build assignment matrix from descriptors"""
        mdesc0, mdesc1 = self.mlpc0(desc0), self.mlpc1(desc1)
        similarity_matrix = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        _, top_k_indices = torch.topk(similarity_matrix.squeeze(0).view(-1), self.top_k, dim=-1)
        # top_k_indices_2d = torch.stack(torch.unravel_index(top_k_indices, similarity_matrix.squeeze(0).shape)).T
            # 将一维索引转换回二维索引
        top_k_indices_2d = torch.unravel_index(top_k_indices, similarity_matrix.squeeze(0).shape)
        
        # 将结果堆叠并转置，转换成形状为 [500, 2]
        top_k_indices_2d = torch.stack(top_k_indices_2d, dim=1)

        return top_k_indices_2d, similarity_matrix
    
class Match_F(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, top_k : int) -> None:
        super().__init__()
        self.mlp0 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.top_k = top_k

    def forward(self, F_2D_16, F_3D_16):
        top1 = []
        pos_value = 0
        for F_2d, F_3d in zip(F_2D_16, F_3D_16):
            mdesc0, mdesc1 = self.mlp0(F_2d.transpose(1, 0).unsqueeze(0)), self.mlp1(F_3d.transpose(1, 0).unsqueeze(0))
            
            similarity_matrix = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
            _, top_k_indices = torch.topk(similarity_matrix.squeeze(0).view(-1), self.top_k, dim=-1)
            top1.append(top_k_indices.cpu().numpy().squeeze())
        #     scores0 = F.log_softmax(similarity_matrix, 2)
        #     scores1 = F.log_softmax(similarity_matrix.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
        #     mat_norm = scores0 + scores1
        #     pos_match = mat_norm.squeeze(0)[id2, id3]
        #     pos_value += pos_match
        # pos_match = - pos_value / len(F_2D_16)
        return top1

def c_loss(mat, c_2d, c_3d):
    scores0 = F.log_softmax(mat, 2)
    scores1 = F.log_softmax(mat.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    mat_norm = scores0 + scores1
    # ground_truth = create_ground_truth(c_2d, c_3d, mat.shape[0], mat.shape[1])
    pos_match = mat_norm.squeeze(0)[c_2d, c_3d]
    return -pos_match.mean()

class MyModel(pl.LightningModule):
    def __init__(self, input_dim_A=1, input_dim_B=1, input_dim_C=1, lr=0.1, weight_decay=0.1):
        super(MyModel, self).__init__()
        self.save_hyperparameters()

        self.encoder_2D_A = nn.Sequential(
            nn.Conv2d(self.hparams.input_dim_A, 64, kernel_size=4, stride=2, padding=1),  # 输出形状 (B, 64, H/2, W/2)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 输出形状 (B, 128, H/4, W/4)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 输出形状 (B, 256, H/8, W/8)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 输出形状 (B, 512, H/16, W/16)
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 128, kernel_size=4, stride=2, padding=1),  # 输出形状 (B, 128, H/32, W/32)
            nn.ReLU(inplace=True)
        )

        self.decoder_2D_A = nn.Sequential(
            nn.ConvTranspose2d(128, 512, kernel_size=4, stride=2, padding=1),  # 输出形状 (B, 512, H/16, W/16)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512 + 512, 256, kernel_size=4, stride=2, padding=1),  # 输出形状 (B, 256, H/8, W/8)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256 + 256, 256, kernel_size=4, stride=2, padding=1),  # 输出形状 (B, 256, H/4, W/4)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256 + 128, 256, kernel_size=4, stride=2, padding=1),  # 输出形状 (B, 256, H/2, W/2)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256 + 64, 256, kernel_size=4, stride=2, padding=1),  # 输出形状 (B, 256, H, W)
            nn.ReLU(inplace=True)
        )

        self.encoder_3D_B = nn.Sequential(
            nn.Conv2d(self.hparams.input_dim_B, 64, kernel_size=4, stride=2, padding=1),  # 输出形状 (B, 64, H/2, W/2)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 输出形状 (B, 128, H/4, W/4)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 输出形状 (B, 256, H/8, W/8)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 输出形状 (B, 512, H/16, W/16)
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 128, kernel_size=4, stride=2, padding=1),  # 输出形状 (B, 128, H/32, W/32)
            nn.ReLU(inplace=True)
        )
        self.encoder_3D_C = nn.Sequential(
            nn.Conv2d(self.hparams.input_dim_C, 64, kernel_size=4, stride=2, padding=1),  # 输出形状 (B, 64, H/2, W/2)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 输出形状 (B, 128, H/4, W/4)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 输出形状 (B, 256, H/8, W/8)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 输出形状 (B, 512, H/16, W/16)
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 128, kernel_size=4, stride=2, padding=1),  # 输出形状 (B, 128, H/32, W/32)
            nn.ReLU(inplace=True)
        )

        self.merge_conv = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)  # (B, 128, H/32, W/32)
        self.relu = nn.ReLU(inplace=True)

        self.decoder_3D_BC = nn.Sequential(
            nn.ConvTranspose2d(128, 512, kernel_size=4, stride=2, padding=1),  # 输出形状 (B, 512, H/16, W/16)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512 + 512 + 512, 256, kernel_size=4, stride=2, padding=1),  # 输出形状 (B, 256, H/8, W/8)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256 + 256 + 256, 256, kernel_size=4, stride=2, padding=1),  # 输出形状 (B, 256, H/4, W/4)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256 + 128 + 128, 256, kernel_size=4, stride=2, padding=1),  # 输出形状 (B, 256, H/2, W/2)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256 + 64 + 64, 256, kernel_size=4, stride=2, padding=1),  # 输出形状 (B, 256, H, W)
            nn.ReLU(inplace=True)
        )
        self.log_assignment = MatchAssignment(256)
        self.match_c = Match_C(256, 256, CFG.TOPK)
        self.match_f = Match_F(256, 256, 1)
    def forward(self, A, B, C):
        A1 = self.encoder_2D_A[0:2](A)  # First layer output (B, 64, H/2, W/2)
        A2 = self.encoder_2D_A[2:4](A1)  # Second layer output (B, 128, H/4, W/4)
        A3 = self.encoder_2D_A[4:6](A2)  # Third layer output (B, 256, H/8, W/8)
        A4 = self.encoder_2D_A[6:8](A3)  # Forth encoder output (B, 512, H/16, W/16)
        A5 = self.encoder_2D_A[8:](A4)  # Fifth encoder output (B, 128, H/16, W/16)

        B1 = self.encoder_3D_B[0:2](B)  # First layer output for depth (B, 64, H/2, W/2)
        B2 = self.encoder_3D_B[2:4](B1)  # Second layer (B, 128, H/4, W/4)
        B3 = self.encoder_3D_B[4:6](B2)  # Third layer (B, 256, H/8, W/8)
        B4 = self.encoder_3D_B[6:8](B3)  # Final encoder output (B, 512, H/16, W/16)
        B5 = self.encoder_3D_B[8:](B4)  # Fifth encoder output (B, 128, H/16, W/16)

        C1 = self.encoder_3D_C[0:2](C)  # First layer output for reflectivity (B, 64, H/2, W/2)
        C2 = self.encoder_3D_C[2:4](C1)  # Second layer (B, 128, H/4, W/4)
        C3 = self.encoder_3D_C[4:6](C2)  # Third layer (B, 256, H/8, W/8)
        C4 = self.encoder_3D_C[6:8](C3)  # Final encoder output (B, 512, H/16, W/16)
        C5 = self.encoder_3D_C[8:](C4)  # Fifth encoder output (B, 128, H/16, W/16)

        merged = torch.cat((B5, C5), dim=1)  # (B, 256, H/32, W/32)
        merged_feature = self.merge_conv(merged)  # (B, 128, H/32, W/32)
        Gol_3D = self.relu(merged_feature)

        LOC_2D_A = self.decoder_2D_A[0:2](A5)
        LOC_2D_A = torch.cat((LOC_2D_A, A4), dim=1)  # Skip connection
        LOC_2D_A = self.decoder_2D_A[2:4](LOC_2D_A)
        LOC_2D_A = torch.cat((LOC_2D_A, A3), dim=1)  # Skip connection
        LOC_2D_A = self.decoder_2D_A[4:6](LOC_2D_A)  # 输出形状 (B, 256, H/4, W/4)
        LOC_2D_B = torch.cat((LOC_2D_A, A2), dim=1)  # Skip connection 
        LOC_2D_B = self.decoder_2D_A[6:8](LOC_2D_B)
        LOC_2D_B = torch.cat((LOC_2D_B, A1), dim=1)  # Skip connection
        LOC_2D_B = self.decoder_2D_A[8: ](LOC_2D_B)  # 输出形状 (B, 256, H, W)


        LOC_3D_A = self.decoder_3D_BC[0:2](Gol_3D)  
        LOC_3D_A = torch.cat((LOC_3D_A, B4), dim=1)  # Skip connection
        LOC_3D_A = torch.cat((LOC_3D_A, C4), dim=1)  # Skip connection
        LOC_3D_A = self.decoder_3D_BC[2:4](LOC_3D_A)
        LOC_3D_A = torch.cat((LOC_3D_A, B3), dim=1)  # Skip connection
        LOC_3D_A = torch.cat((LOC_3D_A, C3), dim=1)  # Skip connection  
        LOC_3D_A = self.decoder_3D_BC[4:6](LOC_3D_A) # 输出形状 (B, 256, H/4, W/4)
        LOC_3D_B = torch.cat((LOC_3D_A, B2), dim=1)  # Skip connection
        LOC_3D_B = torch.cat((LOC_3D_B, C2), dim=1)  # Skip connection
        LOC_3D_B = self.decoder_3D_BC[6:8](LOC_3D_B)
        LOC_3D_B = torch.cat((LOC_3D_B, B1), dim=1)  # Skip connection
        LOC_3D_B = torch.cat((LOC_3D_B, C1), dim=1)  # Skip connection
        LOC_3D_B = self.decoder_3D_BC[8:](LOC_3D_B)  # 输出形状 (B, 256, H, W)
        return LOC_2D_A, LOC_2D_B, LOC_3D_A, LOC_3D_B
    
if __name__ == '__main__':
    test_data = MOD_Dataset(mode=CFG.Mode)
    test_loader = DataLoader(
                        dataset=test_data, 
                        batch_size=CFG.BATCH_SIZE, 
                        shuffle=True, 
                        drop_last=True,
                        num_workers=4,  
                        pin_memory=True 
                        )
    
    checkpoint_path = CFG.MODPATH
    model = MyModel.load_from_checkpoint(checkpoint_path)
    model.to(CFG.DEVICE)
    model.eval()

    # 禁用梯度计算，以减少内存消耗并加快测试过程
    with torch.no_grad():
        data = dict()
        data['RRE'] = []
        data['RTE'] = []
        test_loss = 0
        for batch in test_loader:
            A = batch['image_2d'].permute(0, 3, 1, 2).to(CFG.DEVICE)
            B = batch['image_3d_R'].unsqueeze(-1).permute(0, 3, 1, 2).to(CFG.DEVICE)
            C = batch['image_3d_D'].unsqueeze(-1).permute(0, 3, 1, 2).to(CFG.DEVICE)
            PC_IM = batch['pc_2d'].cpu().numpy()
            PC_IM_DW = batch['pc_2d_scaled'].cpu().numpy()
            inter_matrix = batch['inter_matrix'].cpu().numpy()
            transform_matrix = batch['transform_matrix'].cpu().numpy()
            T = batch['T_inv'].cpu().numpy()
            H = A.shape[-2]
            W = A.shape[-1]

            # 前向传播
            F_2D_c, F_2D_f, F_3D_c, F_3D_f = model(A, B, C)
            T_real = np.matmul(transform_matrix, T)
            for i in range(PC_IM.shape[0]):
                # indices_c, good_values_int_f, good_values_int_scale = tr3d2d(PC_IM_DW[i], inter_matrix[i], transform_matrix[i], T[i], H, W, scale=True)
                # c_2d = batch_convert_to_reshape_indices(good_values_int_scale, [A.shape[-1]//4, A.shape[-2]//4])
                # c_3d = batch_convert_to_reshape_indices(indices_c, [B.shape[-1]//4, B.shape[-2]//4])

                F3c = F_3D_c[i].reshape(F_3D_c.shape[1], F_3D_c.shape[2] * F_3D_c.shape[3]).transpose(1, 0).unsqueeze(0)
                F2c = F_2D_c[i].reshape(F_2D_c.shape[1], F_2D_c.shape[2] * F_2D_c.shape[3]).transpose(1, 0).unsqueeze(0)
                F3f = F_3D_f[i]
                F2f = F_2D_f[i]

                # top_2D.cpu().numpy()
                top_ind, mat = model.match_c(F2c, F3c)
                top_2D, top_3D = torch.split(top_ind, 1, dim=1)
                top_2D = top_2D.cpu().numpy().squeeze()  # 变成 (500,) 的 NumPy 数组
                top_3D = top_3D.cpu().numpy().squeeze()  # 变成 (500,) 的 NumPy 数组
                converted_x = batch_convert_from_reshape_indices(top_2D, (int(H / 4), int(W / 4)))  # 变成 (500, 2)
                converted_y = batch_convert_from_reshape_indices(top_3D, (int(B.shape[-2] / 4), int(B.shape[-1] / 4)))  # 变成 (500, 2)
                F_2D_16, F_3D_16 = F_grouping(F2f, F3f, converted_x, converted_y)

                top_16 = model.match_f(F_2D_16, F_3D_16)
                top_16 = batch_convert_from_reshape_indices(top_16, (16, 16))
                top_16 = np.array(top_16)
                top_2D_16 = batch_convert_from_reshape_indices(top_16[:, 0], (4, 4))
                top_3D_16 = batch_convert_from_reshape_indices(top_16[:, 1], (4, 4))
                top_F_2D = np.array(converted_x) * 4 + np.array(top_2D_16)
                top_F_3D = np.array(converted_y) * 4 + np.array(top_3D_16)
                pc_map = PC_IM[i]
                top_F_PC = pc_map[top_F_3D[:, 0], top_F_3D[:, 1]]

                # binary_image1 = np.zeros((PC_IM.shape[1], PC_IM.shape[2]), dtype=np.uint8)
                # binary_image2 = np.zeros((H, W), dtype=np.uint8)
                # binary_image1[top_F_3D[:, 0], top_F_3D[:, 1]] = 1
                # binary_image2[top_F_2D[:, 0], top_F_2D[:, 1]] = 1
                # plt.imshow(binary_image1, cmap='gray')
                # plt.title("Non-Zero Points Visualization")
                # plt.axis('off')  
                # plt.savefig("./test/scale1.png")
                # plt.imshow(binary_image2, cmap='gray')
                # plt.title("Non-Zero Points Visualization")
                # plt.axis('off')  
                # plt.savefig("./test/scale2.png")

                try:
                    success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(top_F_PC, top_F_2D.astype(np.float32)[:, [1, 0]],
                                                                                           inter_matrix[i, :, :3], None, iterationsCount=5000, reprojectionError=6.5, flags=cv2.SOLVEPNP_EPNP)
                except:
                    success = False
                if success:
                    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                    rotation_matrix = np.concatenate((rotation_matrix, translation_vector), axis=1)
                    extra_row = np.array([0, 0, 0, 1])
                    GT_Tr = T_real[i]
                    GT_Tr = np.vstack([GT_Tr, extra_row])
                    R,_=cv2.Rodrigues(rotation_vector)
                    T_pred=np.eye(4)
                    T_pred[0:3,0:3] = R
                    T_pred[0:3,3:] = translation_vector
                    P_diff=np.dot(np.linalg.inv(T_pred),GT_Tr)
                    t_diff=np.linalg.norm(P_diff[0:3,3])
                    r_diff=P_diff[0:3,0:3]
                    R_diff=Rotation.from_matrix(r_diff)
                    angles_diff=np.sum(np.abs(R_diff.as_euler('xzy',degrees=True)))
                    rte = t_diff
                    rre = angles_diff
                    print(rre, rte)
                    print('success')
                else:
                    rre = np.inf
                    rte = np.inf
                    print('G')
                data['RRE'].append(rre)
                data['RTE'].append(rte)
        df = pd.DataFrame(data)
        rre_mean = df['RRE'].mean()
        rre_std = df['RRE'].std()
        rte_mean = df['RTE'].mean()
        rte_std = df['RTE'].std()
        total_count = len(df)
        satisfied_count = len(df[(df['RRE'] < 5) & (df['RTE'] < 2)])
        acc = satisfied_count / total_count
        summary_data = {
            'RRE': ['Mean', rre_mean, 'Std', rre_std],
            'RTE': ['Mean', rte_mean, 'Std', rte_std],
            'ACC': ['ACC', acc, '', '']
        }
        # 指定 Excel 文件路径
        excel_file_path = CFG.XPATH
        summary_df = pd.DataFrame(summary_data)

        # 将 summary_df 的空值补成 NaN
        summary_df.fillna('', inplace=True)

        # 将 summary 数据附加到原始的 df 后面
        df_with_summary = pd.concat([df, summary_df], ignore_index=True)
        # 将 DataFrame 写入 Excel 文件
        df_with_summary.to_excel(excel_file_path, index=False)        

                # pil_image = Image.fromarray(np.zeros((160,512,3)), "RGB")

                # # 在PIL图像上绘制特征点
                # draw = ImageDraw.Draw(pil_image)
                # ones_array = np.ones((top_F_PC.shape[0], 1))  
                # corrds = np.concatenate((top_F_PC, ones_array), axis=1)
                # inter_matrix2 = inter_matrix[i, :, :3]  
                # transform_matrix2 = np.matmul(transform_matrix[i].astype(np.float32), T[i].astype(np.float32))
                # corrds = np.matmul(T_pred[:3, :], corrds.astype(np.float32).T)
                # corrds = np.matmul(inter_matrix2, corrds).T
                # z_coords = corrds[:, 2:3]
                
                # positive_mask = (corrds >= 0).all(axis=1)
                # positive_mask_z = (z_coords >= 0).all(axis=1)
                # corrds[~positive_mask] = 0
                # z_coords[~positive_mask_z] = 0

                # z_coords[z_coords == 0] = 1e-9

                # corrds = corrds / z_coords

                # xy_points = corrds[:, :2]
            

                # for n in range(top_F_2D.shape[0]):
                #     if (xy_points[n].all() < 0):
                #         continue
                #     x1 = int(xy_points[n][0])
                #     y1 = int(xy_points[n][1])
                #     x2 = int(top_F_2D[n][1])
                #     y2 = int(top_F_2D[n][0])
                #     draw.ellipse([x1 - 3, y1 - 3, x1 + 3, y1 + 3], outline='red', width=3)  # 绘制特征点
                #     draw.ellipse([x2 - 3, y2 - 3, x2 + 3, y2 + 3], outline='blue', width=3)  # 绘制特征点
                #     draw.line([(x1, y1), (x2, y2)], fill='green', width=3)
                # pil_image.save('./test/RGB.png')