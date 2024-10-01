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

class CFG:
    PREPROCESS = False
    EPOCHS = 30 #20
    LR = 1e-3
    WD = 0.05
    BATCH_SIZE = 8
    NBR_FOLDS = 15
    SELECTED_FOLDS = [0]
    DEVICE = 0
    SEED = 2024
    Mode = 6
    CK_FOLDER = './ck_n_R/'

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

# def batch_convert_from_reshape_indices(indices, original_shape):
#     height, width = original_shape
#     results = []
#     for k in indices:
#         a1 = k // width  
#         a2 = k % width 
#         results.append([a1, a2])   
#     return results

# def corrd_grouping(downsampled_coords, scale_factor=4):
#     N = downsampled_coords.shape[0]
#     patch_size = scale_factor ** 2  

#     original_coords = np.zeros((N * patch_size, 2), dtype=int)

#     idx = 0
#     for coord in downsampled_coords:
#         x, y = coord
#         for m in range(scale_factor):
#             for n in range(scale_factor):
#                 original_coords[idx] = np.array([x * scale_factor + m, y * scale_factor + n])
#                 idx += 1

#     return original_coords


def F_grouping(F3f, F2f, indices_c, good_values_int_f, good_values_int_scale):
    index_2D = []
    index_3D = []
    F_2D_16 = []
    F_3D_16 = []
    for I_2D, p_2d in enumerate(good_values_int_scale):
        group_F_2D = F2f[:, p_2d[0] * 4:(p_2d[0] + 1) * 4, p_2d[1] * 4:(p_2d[1] + 1) * 4]
        F_2D_16.append(group_F_2D.reshape(group_F_2D.shape[0], 16))
        index_2D_f = good_values_int_f[I_2D]
        index_2D_f = (index_2D_f[0] % 4) * 4 + index_2D_f[1] % 4
        index_2D.append(index_2D_f)
        index_3D.append(10)
        p_3d = indices_c[I_2D]
        group_F_3D = F3f[:, p_3d[0] * 4:(p_3d[0] + 1) * 4, p_3d[1] * 4:(p_3d[1] + 1) * 4]
        F_3D_16.append(group_F_3D.reshape(group_F_3D.shape[0], 16))


    return F_2D_16, F_3D_16, index_2D, index_3D
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
    #This Part has been hidden#
    
class Match_F(nn.Module):
    #This Part has been hidden#

def c_loss(mat, c_2d, c_3d):
    #This Part has been hidden#

class MyModel(pl.LightningModule):
    def __init__(self, input_dim_A=3, input_dim_B=1, input_dim_C=1, lr=0.1, weight_decay=0.1):
        super(MyModel, self).__init__()
        self.save_hyperparameters()

        self.encoder_2D_A = nn.Sequential(
            nn.Conv2d(self.hparams.input_dim_A, 64, kernel_size=4, stride=2, padding=1),  # Output shape (B, 64, H/2, W/2)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output shape (B, 128, H/4, W/4)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output shape (B, 256, H/8, W/8)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # Output shape (B, 512, H/16, W/16)
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 128, kernel_size=4, stride=2, padding=1),  # Output shape (B, 128, H/32, W/32)
            nn.ReLU(inplace=True)
        )

        self.decoder_2D_A = nn.Sequential(
            nn.ConvTranspose2d(128, 512, kernel_size=4, stride=2, padding=1),  # Output shape (B, 512, H/16, W/16)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512 + 512, 256, kernel_size=4, stride=2, padding=1),  # Output shape (B, 256, H/8, W/8)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256 + 256, 256, kernel_size=4, stride=2, padding=1),  # Output shape (B, 256, H/4, W/4)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256 + 128, 256, kernel_size=4, stride=2, padding=1),  # Output shape (B, 256, H/2, W/2)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256 + 64, 256, kernel_size=4, stride=2, padding=1),  # Output shape (B, 256, H, W)
            nn.ReLU(inplace=True)
        )

        self.encoder_3D_B = nn.Sequential(
            nn.Conv2d(self.hparams.input_dim_B, 64, kernel_size=4, stride=2, padding=1),  # Output shape (B, 64, H/2, W/2)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output shape (B, 128, H/4, W/4)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output shape (B, 256, H/8, W/8)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # Output shape (B, 512, H/16, W/16)
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 128, kernel_size=4, stride=2, padding=1),  # Output shape (B, 128, H/32, W/32)
            nn.ReLU(inplace=True)
        )
        self.encoder_3D_C = nn.Sequential(
            nn.Conv2d(self.hparams.input_dim_C, 64, kernel_size=4, stride=2, padding=1),  # Output shape (B, 64, H/2, W/2)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output shape (B, 128, H/4, W/4)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output shape (B, 256, H/8, W/8)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # Output shape (B, 512, H/16, W/16)
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 128, kernel_size=4, stride=2, padding=1),  # Output shape (B, 128, H/32, W/32)
            nn.ReLU(inplace=True)
        )

        self.merge_conv = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)  # (B, 128, H/32, W/32)
        self.relu = nn.ReLU(inplace=True)

        self.decoder_3D_BC = nn.Sequential(
            nn.ConvTranspose2d(128, 512, kernel_size=4, stride=2, padding=1),  # Output shape (B, 512, H/16, W/16)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512 + 512 + 512, 256, kernel_size=4, stride=2, padding=1),  # Output shape (B, 256, H/8, W/8)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256 + 256 + 256, 256, kernel_size=4, stride=2, padding=1),  # Output shape (B, 256, H/4, W/4)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256 + 128 + 128, 256, kernel_size=4, stride=2, padding=1),  # Output shape (B, 256, H/2, W/2)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256 + 64 + 64, 256, kernel_size=4, stride=2, padding=1),  # Output shape (B, 256, H, W)
            nn.ReLU(inplace=True)
        )
        self.log_assignment = MatchAssignment(256)
        self.match_c = Match_C(256, 256, 500)
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
        LOC_2D_A = self.decoder_2D_A[4:6](LOC_2D_A)  # Output shape (B, 256, H/4, W/4)
        LOC_2D_B = torch.cat((LOC_2D_A, A2), dim=1)  # Skip connection 
        LOC_2D_B = self.decoder_2D_A[6:8](LOC_2D_B)
        LOC_2D_B = torch.cat((LOC_2D_B, A1), dim=1)  # Skip connection
        LOC_2D_B = self.decoder_2D_A[8: ](LOC_2D_B)  # Output shape (B, 256, H, W)


        LOC_3D_A = self.decoder_3D_BC[0:2](Gol_3D)  
        LOC_3D_A = torch.cat((LOC_3D_A, B4), dim=1)  # Skip connection
        LOC_3D_A = torch.cat((LOC_3D_A, C4), dim=1)  # Skip connection
        LOC_3D_A = self.decoder_3D_BC[2:4](LOC_3D_A)
        LOC_3D_A = torch.cat((LOC_3D_A, B3), dim=1)  # Skip connection
        LOC_3D_A = torch.cat((LOC_3D_A, C3), dim=1)  # Skip connection  
        LOC_3D_A = self.decoder_3D_BC[4:6](LOC_3D_A) # Output shape (B, 256, H/4, W/4)
        LOC_3D_B = torch.cat((LOC_3D_A, B2), dim=1)  # Skip connection
        LOC_3D_B = torch.cat((LOC_3D_B, C2), dim=1)  # Skip connection
        LOC_3D_B = self.decoder_3D_BC[6:8](LOC_3D_B)
        LOC_3D_B = torch.cat((LOC_3D_B, B1), dim=1)  # Skip connection
        LOC_3D_B = torch.cat((LOC_3D_B, C1), dim=1)  # Skip connection
        LOC_3D_B = self.decoder_3D_BC[8:](LOC_3D_B)  # Output shape (B, 256, H, W)
        return LOC_2D_A, LOC_2D_B, LOC_3D_A, LOC_3D_B

    def training_step(self, batch, batch_idx):
        A = batch['image_2d'].permute(0, 3, 1, 2)
        B = batch['image_3d_R'].unsqueeze(-1).permute(0, 3, 1, 2)
        C = batch['image_3d_D'].unsqueeze(-1).permute(0, 3, 1, 2)
        PC_IM = batch['pc_2d'].cpu().numpy()
        PC_IM_DW = batch['pc_2d_scaled'].cpu().numpy()
        inter_matrix = batch['inter_matrix'].cpu().numpy()
        transform_matrix = batch['transform_matrix'].cpu().numpy()
        T = batch['T_inv'].cpu().numpy()
        H = A.shape[-2]
        W = A.shape[-1]
        F_2D_c, F_2D_f, F_3D_c, F_3D_f = self(A, B, C)

        match_list = []
        sim_list = []
        scores0 = []
        scores1 = []
        loss1 = 0
        for i in range(PC_IM.shape[0]):
            indices_c, good_values_int_f, good_values_int_scale = tr3d2d(PC_IM_DW[i], inter_matrix[i], transform_matrix[i], T[i], H, W, scale=True)

            c_2d = batch_convert_to_reshape_indices(good_values_int_scale, [A.shape[-2]//4, A.shape[-1]//4] )
            c_3d = batch_convert_to_reshape_indices(indices_c, [B.shape[-2]//4, B.shape[-1]//4])


            # f_2d_group_xy = corrd_grouping(good_values_int_scale, scale_factor=4)
            # f_3d_group_xy = corrd_grouping(indices_c, scale_factor=4)
            # f_2d_group = batch_convert_to_reshape_indices(f_2d_group_xy, [A.shape[-1], A.shape[-2]] )
            # f_3d_group = batch_convert_to_reshape_indices(f_3d_group_xy, [B.shape[-1], B.shape[-2]])

            F3c = F_3D_c[i].reshape(F_3D_c.shape[1], F_3D_c.shape[2] * F_3D_c.shape[3]).transpose(1, 0).unsqueeze(0)
            F2c = F_2D_c[i].reshape(F_2D_c.shape[1], F_2D_c.shape[2] * F_2D_c.shape[3]).transpose(1, 0).unsqueeze(0)
            # F3f = F_3D_f[i].reshape(F_3D_f.shape[1], F_3D_f.shape[2] * F_3D_f.shape[3]).transpose(1, 0).unsqueeze(0)
            # F2f = F_2D_f[i].reshape(F_2D_f.shape[1], F_2D_f.shape[2] * F_2D_f.shape[3]).transpose(1, 0).unsqueeze(0)
            F3f = F_3D_f[i]
            F2f = F_2D_f[i]
            # for c_i in range(len(c_2d)):
            #     F3f_grouped, Acc3_idx = F_grouping(indices_c[c_i], F_3D_f[i])
            #     F2f_grouped, Acc2_idx = F_grouping(good_values_int_scale[c_i], F_2D_f[i], good_values_int_c[c_i])
            F_2D_16, F_3D_16, index_2D, index_3D = F_grouping(F3f, F2f, indices_c, good_values_int_f, good_values_int_scale)
            # F2f_f_group = F2f[:, f_2d_group, :]
            # F3f_f_group = F3f[:, f_3d_group, :]

            top_ind, mat = self.match_c(F2c, F3c)
            top_16, loss2 = self.match_f(F_2D_16, F_3D_16, index_2D, index_3D)
            
            loss1 += c_loss(mat, c_2d, c_3d)
            
            # mat = mat.squeeze(0)
            # scores_c, sim_c, oa_c, ob_c = self.log_assignment(f3c, f2c) 

            # scores_f, sim_f, oa_f, ob_f = self.log_assignment(F2f_f_group, F3f_f_group)  
            # scores0.append(oa_f)
            # scores1.append(ob_f)
            
            # m0, m1, mscores0, mscores1 = filter_matches(scores_f, 0.0)
            # valid = m0[0] > -1
            # m_indices_0 = torch.where(valid)[0]
            # m_indices_1 = m0[0][valid]
            # match_list.append(torch.stack([m_indices_0, m_indices_1], -1))
            # sim_list.append(scores_f.squeeze(0)) 
        loss1 = loss1 / PC_IM.shape[0]
        loss = loss1 + loss2
        self.log('train_loss1', loss1)
        self.log('train_loss2', loss2)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer
    
if __name__ == '__main__':
    train_data = MOD_Dataset(mode=CFG.Mode)
    train_loader = DataLoader(
                        dataset=train_data, 
                        batch_size=CFG.BATCH_SIZE, 
                        shuffle=True, 
                        drop_last=True,
                        num_workers=4,  
                        pin_memory=True 
                        )
    model = MyModel(lr=CFG.LR, weight_decay=CFG.WD)

    early_stop_callback = EarlyStopping(monitor="train_loss", mode="min", patience=5, verbose=True)
    checkpoint_callback = ModelCheckpoint(monitor="train_loss", dirpath=CFG.CK_FOLDER, filename=f"best_model", save_top_k=4, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        max_epochs=CFG.EPOCHS,
        callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
        devices=[CFG.DEVICE],
        accelerator="gpu",  # Adjust based on your hardware
        enable_progress_bar=True,
    )

    trainer.fit(model, train_dataloaders=train_loader)

#     model = MyModel.load_from_checkpoint(checkpoint_callback.best_model_path)
#     model.to(CFG.DEVICE)
#     model.eval()
#     preds = model(test)
#     all_preds.append(preds)

# preds = np.mean(all_preds, 0)
