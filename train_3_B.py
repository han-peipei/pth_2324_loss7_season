import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"   # 或 ":4096:8"
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd 
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm,BoundaryNorm
from matplotlib.colors import TwoSlopeNorm
# import cmaps

from model import Direct_Conv3D_GRU
# from data_3_B import standardize
# from data_3_B import build_dataset_batched, standardize

import random
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
kaggle_dir='/kaggle/working/pth_2324_season/'
##########################################################################################
def standardize(data):
    mean = data.mean()
    std = data.std()
    norm = (data - mean) / std
    return norm, mean, std
def plot_delta_mae_heatmap(delta, xlabels, ylabels, outfile,
                           cmap="seismic", annotate=None,
                           min_n_mask=None, cnt=None,
                           vlim=None):
    data = np.asarray(delta, dtype=float)

    if vlim is None:
        vmax = np.nanmax(np.abs(data))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1e-6
    else:
        vmax = float(vlim)

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig_h = 2.2 + 0.35 * data.shape[0]
    fig, ax = plt.subplots(figsize=(12, fig_h))
    im = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap, norm=norm)

    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels)
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)

    ax.set_xlabel("Forecast lead (hour)")
    ax.set_ylabel("Wind level")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("ΔMAE = MAE(pred,true) − MAE(nwp,true)  [m/s]")

    # 可选：标注
    if annotate is not None:
        ann = np.asarray(annotate)
        for j in range(data.shape[0]):
            for i in range(data.shape[1]):
                if np.isfinite(data[j, i]):
                    ax.text(i, j, str(ann[j, i]), ha="center", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(outfile, dpi=160)
    plt.close(fig)

##########################################################################################
def rmse_1d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((a[m] - b[m])**2)))

def corr_1d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2:
        return np.nan
    aa = a[m] - np.mean(a[m])
    bb = b[m] - np.mean(b[m])
    denom = (np.sqrt(np.mean(aa**2)) * np.sqrt(np.mean(bb**2)))
    if denom == 0:
        return np.nan
    return float(np.mean(aa * bb) / denom)

def plot_rmse_r_by_lead(trues, preds, outfile="rmse_r_by_lead.png", unit="m/s"):
    trues = np.asarray(trues, dtype=float)
    preds = np.asarray(preds, dtype=float)
    assert trues.shape == preds.shape, f"shape mismatch: {trues.shape} vs {preds.shape}"

    B, F = trues.shape
    leads = np.arange(1, F+1)

    rmse_lead = np.array([rmse_1d(trues[:, i], preds[:, i]) for i in range(F)], dtype=float)
    r_lead    = np.array([corr_1d(trues[:, i], preds[:, i]) for i in range(F)], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    ax = axes[0]
    ax.plot(leads, rmse_lead, marker='.')
    ax.set_xlabel("Forecast lead (hour)")
    ax.set_ylabel(f"RMSE [{unit}]")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(leads, r_lead, marker='.')
    ax.set_xlabel("Forecast lead (hour)")
    ax.set_ylabel("Correlation (R)")
    ax.set_ylim(-1, 1)  # 常用显示范围
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outfile, dpi=160)
    plt.close(fig)

    return rmse_lead, r_lead

#########################################################################################
def stitch_overlapping_forecasts(y_windows):
    """
    将形如 [N, F] 的滑动窗口序列拼接成一条连续序列（对重叠处做平均）。
    返回:
      y_stitched: [N+F-1]
      counts    : [N+F-1] 每个时刻被多少个窗口覆盖
    """
    N, F = y_windows.shape
    T = N + F - 1
    acc = np.zeros(T, dtype=float)
    cnt = np.zeros(T, dtype=int)
    for i in range(N):
        acc[i:i+F] += y_windows[i]
        cnt[i:i+F] += 1
    y_stitched = acc / np.maximum(cnt, 1)
    return y_stitched, cnt
def plot_timeseries_stitched(y_true_windows, y_pred_windows, wind_speed, time, unit='m/s', tag='direct'):

    yt_st, _ = stitch_overlapping_forecasts(y_true_windows)
    yp_st, cnt = stitch_overlapping_forecasts(y_pred_windows)
    T = len(yt_st)

    # if wind_speed is not None:
    ws_st, _ = stitch_overlapping_forecasts(wind_speed)
        
    # if time is None:
    #     x = np.arange(T) #if time is None else time[:T]
    # else:
    time = np.asarray(time)
    x = time[24:24+T]
    fig, ax = plt.subplots(figsize=(12,4))
    # ax.spines['left'].set_position('zero')    # y轴穿过x=0
    ax.spines['bottom'].set_position('zero')  # x轴穿过y=0
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # 让坐标轴箭头更明显（可选）
    # ax.spines['left'].set_linewidth(1.2)
    # ax.spines['bottom'].set_linewidth(1.2)
    # plt.figure(figsize=(12,4))
    ax.plot(x[:100], yt_st[:100], label='True', color='black',lw=1.2)
    # if wind_speed is not None:
    ax.plot(x[:100], ws_st[:100], '--',label='model', color='tab:blue', lw=1.2)
    ax.plot(x[:100], yp_st[:100], '--',label='Pred', color='tab:red', lw=1.0)
    ax.set_xlabel('Time') 
    ax.set_ylabel(f'Wind speed [{unit}]')
    # ax.title('Sliding-window Forecast')
    ax.legend(loc='upper right') 
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{kaggle_dir}timeseries_stitched.png")
    plt.close()
##########################################################################################
def chain_rows_by_step(y_windows, time_full, auto_time=True):
    y_windows = np.asarray(y_windows)
    N, F0 = y_windows.shape
    print(y_windows.shape)
    # print(F0)
    # F0=24
    idx_list = list(range(0, N, F0))
    chunks, tchunks = [], []
    for i in idx_list:
        seg = y_windows[i, :F0]  # 整段窗口
        # if time is None:
        #     chunks.append(seg)
        # else:
        gidx = i + np.arange(F0)      # 该窗口映射到全局时间的索引
        mask = gidx < len(time_full[24:])       # 越界保护
        chunks.append(seg[mask])
        tchunks.append(np.asarray(time_full[24:])[gidx[mask]])

    y_chain = np.concatenate(chunks, axis=0) if chunks else np.array([], dtype=float)
    t_chain = (np.concatenate(tchunks, axis=0) if (time_full is not None and tchunks) else None)
    return y_chain, t_chain, np.asarray(idx_list)
def plot_chain_rows_by_step(y_true_windows, y_pred_windows, spd_train_all, time_full,
                            unit='m/s', tag='direct', auto_time=True):
    yt, tt, idx = chain_rows_by_step(y_true_windows, time_full,  auto_time=auto_time)
    yp, tp, _   = chain_rows_by_step(y_pred_windows, time_full,  auto_time=auto_time)
    # if spd_train_all is not None:
    ws_st, tw ,_= chain_rows_by_step(spd_train_all, time_full,  auto_time=auto_time)

    fig, ax = plt.subplots(figsize=(12,4))
    # if tt is not None:
    #     x = np.arange(len(yt))
    # else:
    x = tt

    ax.plot(x[:100], yt[:100], label='True', color='black',lw=1.2)
    # if spd_train_all is not None:
    ax.plot(x[:100], ws_st[:100], label='model', linestyle='--',color='tab:blue', lw=1.2)
    ax.plot(x[:100], yp[:100], label='Pred', linestyle='--',color='tab:red',  lw=1.0)
    ax.set_xlabel('Time'); ax.set_ylabel(f'Wind speed [{unit}]')
    # ax.spines['left'].set_position('zero')     # y轴穿过 x=0
    ax.spines['bottom'].set_position('zero')   # x轴穿过 y=0
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    # plt.title(f'Chained series: rows 1, 1+{step}, 1+2×{step}, ... (full window)')
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(f'{kaggle_dir}chain_rows_by_step_{tag}.png'); plt.close()

    # return yt, yp, (tt if tt is not None else None), idx

###########################################################################################
def plot_scatter_by_leads(y_true_windows, y_pred_windows,  unit='m/s', tag='direct'):
    yt = y_true_windows.reshape(-1)
    yp = y_pred_windows.reshape(-1)

    # 坐标范围
    vmin = float(min(yt.min(), yp.min()))
    vmax = float(max(yt.max()+1.5, yp.max()+1.5))

    # 绘制
    plt.figure(figsize=(6,6))
    plt.scatter(yt, yp, s=3, alpha=0.3)
    plt.plot([vmin, vmax+1.5], [vmin, vmax+1.5], 'k--', lw=1)
    plt.xlim(0, vmax+1.5)
    plt.ylim(0, vmax+1.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel(f'True [{unit}]')
    plt.ylabel(f'Predicted [{unit}]')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{kaggle_dir}scatter_all_{tag}.png")
    plt.close()
#######################################################################################################
def plot_kde2d_full(y_true_windows, y_pred_windows, unit='m/s', tag='direct',
                    nx=200, ny=200, 
                    show_contour=True,
                    bw_method=None):

    yt = y_true_windows.reshape(-1).astype(float)
    yp = y_pred_windows.reshape(-1).astype(float)

    vmin = min(yt.min(), yp.min())
    vmax = max(yt.max(), yp.max())

    # KDE
    kde = gaussian_kde(np.vstack([yt, yp]), bw_method=bw_method)  # 形状 [2, N]
    x = np.linspace(vmin, vmax, nx)
    y = np.linspace(vmin, vmax, ny)
    X, Y = np.meshgrid(x, y)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(ny, nx)  # 概率密度（积分≈1）
    
    fig, ax = plt.subplots(figsize=(6,6))
    # bounds = np.arange(0.1,0.6,0.01)
    # norm = BoundaryNorm(bounds, ncolors=plt.get_cmap(cmap).N, clip=True)
    # cmap = 'gray_r'
    # cmap = 'pink_r'
    cmap = 'Blues'
    pcm = ax.pcolormesh(X, Y, Z, shading='auto', cmap=plt.get_cmap(cmap))
    cbar = fig.colorbar(pcm, ax=ax,fraction=0.04)
    cbar.set_label('Probability density')

    ax.set_xlim(0, vmax)
    ax.set_ylim(0, vmax)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(f'True [{unit}]')
    ax.set_ylabel(f'Predicted [{unit}]')
    ax.grid(True, alpha=0.25)

    # ax.spines['left'].set_position('zero')     # y轴穿过 x=0
    # ax.spines['bottom'].set_position('zero')   # x轴穿过 y=0
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    fig.tight_layout()
    fig.savefig(f"{kaggle_dir}scatter_kde2d_full_{tag}.png", dpi=160)
    plt.close(fig)
###############################################################################################################
def plot_residual_hist_all(y_true_windows, y_pred_windows, bins=40, unit='m/s', tag='direct'):
    yt = y_true_windows.reshape(-1)
    yp = y_pred_windows.reshape(-1)
    err = yp - yt
    weights = np.ones(err.shape, dtype=float) / err.size * 100.0

    plt.figure(figsize=(7,4.5))
    plt.hist(err, bins=bins, alpha=0.85, edgecolor='k',weights=weights)
    plt.xlabel(f'Error (Pred - True) [{unit}]')
    plt.ylabel('Percentage (%)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{kaggle_dir}residual_hist_ln.png")
    plt.close()
##################################################################################################
def bin_percentages(err, edges=(-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12), include_outside=True):
    edges = np.asarray(edges, dtype=int)
    err = np.asarray(err, dtype=int)
    if include_outside:
        bins = np.concatenate(([-np.inf], edges, [np.inf]))
        counts, _ = np.histogram(err, bins=bins)
        labels = [f"< {edges[0]:.0f}"]
        labels += [f"({edges[i-1]:.0f}, {edges[i]:.0f}]" for i in range(1, len(edges))]
        labels += [f"> {edges[-1]:.0f}"]
    else:
        counts, _ = np.histogram(err, bins=edges)
        labels = [f"[{edges[0]:.0f}, {edges[1]:.0f})"] + \
                 [f"[{edges[i]:.0f}, {edges[i+1]:.0f})" for i in range(1, len(edges)-2)] + \
                 [f"[{edges[-2]:.0f}, {edges[-1]:.0f}]"]  # 最后一箱右闭

    # inside_count = counts.sum()
    total = len(err)
    # print(total)
    # print(err)
    # print(inside_count)
    # labels = []
    # for i in range(len(edges)-1):
    #     a, b = edges[i], edges[i+1]
    #     if i == 0:
    #         labels.append(f"[{a},{b}]")
    #     else:
    #         labels.append(f"({a},{b}]")
    percents = (counts / total) * 100.0
    return labels, percents, counts, total
def plot_bin_percentages(labels, percents, tag='direct'):
    plt.figure(figsize=(9,3.5))
    x = np.arange(len(labels))
    plt.bar(x, percents,width=0.4)
    plt.xticks(x, labels, rotation=45, fontsize=9)
    plt.ylabel('Percentage (%)')
    plt.title('error distribution (log scale)')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.yscale('log') 
    plt.savefig(f'{kaggle_dir}error_bin_percent_{tag}.png')
    plt.close()
##########################################################################################
##########################################################################################
def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 若版本支持，强制确定性（某些算子会报错，可改 warn_only=True）
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

set_seed(2025)

seed = 2025
g = torch.Generator()
g.manual_seed(seed)

def _worker_init_fn(wid):
    np.random.seed(seed + wid)
    random.seed(seed + wid)

def train_and_evaluate_from_npy(
    hist_train,        # [B, H_hist]          
    nwp_train,         # [B, F, C, H, W]     
    y_train,           # [B, F]   
               
    # hist_val,          # [Bv, H_hist]
    # nwp_val,           # [Bv, F, C, H, W]
    # y_val,             # [Bv, F]
    coords_tr, # [B, d]  (d=2/3)
    spd_train_all,
    time, 
    # coords_va,   # [Bv, d]
    num_epochs=1000, batch_size=32, patience=3,
    device=torch.device('cuda'),
):
    hist_train = np.asarray(hist_train, dtype=np.float32)   # [B, H]
    nwp_train  = np.asarray(nwp_train,  dtype=np.float32)   # [B, F, C, H, W]
    y_train    = np.asarray(y_train,    dtype=np.float32)   # [B, F]

    # hist_val   = np.asarray(hist_val,   dtype=np.float32)
    # nwp_val    = np.asarray(nwp_val,    dtype=np.float32)
    # y_val      = np.asarray(y_val,      dtype=np.float32)

    B, H_hist = hist_train.shape
    # Bv        = hist_val.shape[0]
    F, C, H, W = nwp_train.shape[1], nwp_train.shape[2], nwp_train.shape[3], nwp_train.shape[4]

    coords_train = np.asarray(coords_tr, dtype=np.float32)
    # coords_val   = np.asarray(coords_va,   dtype=np.float32)
    

    def normalize_coords_trig(coords, elev_mean=None, elev_std=None,
                            fix_lon=True, eps=1e-8, return_stats=False):
        """
        coords: [B,2]或[B,3]，列为 [lat, lon, (elev)]
        输出: [B, 4] 或 [B, 5]  -> [sin(lat), cos(lat), sin(lon), cos(lon), (elev_z)]
        训练集：不传 elev_mean/std，会在内部计算并返回
        验证/测试：把训练集的 elev_mean/std 传入，保持一致
        """
        coords = np.asarray(coords, dtype=np.float32)
        # if coords.ndim != 2 or coords.shape[1] < 2:
        #     raise ValueError(f"coords 形状应为 [B,2] 或 [B,3]，收到 {coords.shape}")

        lat_deg = coords[:, 0].copy()
        lon_deg = coords[:, 1].copy()
        # if fix_lon:
        #     lat_deg, lon_deg = _fix_lat_lon(lat_deg, lon_deg)

        lat_rad = np.deg2rad(lat_deg)
        lon_rad = np.deg2rad(lon_deg)

        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)

        # if coords.shape[1] >= 3:
        elev = coords[:, 2].astype(np.float32)
        if elev_mean is None or elev_std is None:
            # 训练集：拟合
            elev_mean = float(elev.mean())
            elev_std  = float(elev.std()) + eps
        elev_z = (elev - elev_mean) / max(elev_std, eps)
        out = np.stack([sin_lat, cos_lat, sin_lon, cos_lon, elev_z], axis=1)
        if return_stats:
            return out, elev_mean, elev_std
        return out
        # else:
        #     out = np.stack([sin_lat, cos_lat, sin_lon, cos_lon], axis=1)
        #     if return_stats:
        #         # 没有海拔，返回 None
        #         return out, None, None
        #     return out
    y_mean = np.load(os.path.join(kaggle_dir, "y_mean.npy"))
    y_std = np.load(os.path.join(kaggle_dir,"y_std.npy"))

    elev_mean = np.load(os.path.join(kaggle_dir,"coords_elev_mean.npy"))
    elev_std = np.load(os.path.join(kaggle_dir,"coords_elev_std.npy"))
    
    coords_tr_norm = normalize_coords_trig(
        coords_train, elev_mean=elev_mean, elev_std=elev_std
    )
    # 验证/测试复用训练统计量
    # coords_va_norm = normalize_coords_trig(
    #     coords_val, elev_mean=elev_mean, elev_std=elev_std
    # )
    coord_dim    = coords_tr_norm.shape[1]
    print(coords_tr_norm.shape)
    print(hist_train.shape)
    print(nwp_train.shape)
    print(y_train.shape)
    # np.save("coords_elev_mean.npy", np.array([elev_mean], dtype=np.float32))
    # np.save("coords_elev_std.npy",  np.array([elev_std],  dtype=np.float32))


    # y_train, y_mean, y_std = standardize(y_train)
    # y_val = (y_val - y_mean)/y_std
    y_train = (y_train - y_mean)/y_std

########################
    hist_train_3d = hist_train[..., None]   # [B, H_hist, 1]
    # hist_val_3d   = hist_val[...,   None]   # [Bv, H_hist, 1]

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(hist_train_3d).float(),   # [B, H, 1]
            torch.tensor(nwp_train).float(),       # [B, F, C, H, W]
            torch.tensor(coords_tr_norm).float(),    # [B, d]
            torch.tensor(y_train).float(),       # [B, F]
        ),
        batch_size=batch_size, shuffle=False,
        generator=g, worker_init_fn=_worker_init_fn, num_workers=0
    )
    # val_loader = DataLoader(
    #     TensorDataset(
    #         torch.tensor(hist_val_3d).float(),
    #         torch.tensor(nwp_val).float(),
    #         torch.tensor(coords_va_norm).float(),
    #         torch.tensor(y_val).float(),
    #     ),
    #     batch_size=batch_size, shuffle=False, num_workers=0
    # )

    # ===== 3) 建模 =====
    model = Direct_Conv3D_GRU(
        in_channels=C,
        forecast_hours=F,
        coord_dim=coord_dim,
        hist_input_size=1,
        hidden_size=32,
        coord_feat_dim=16,
        kt=3
    ).to(device)

    # === 6. 评估 ===
    model.load_state_dict(torch.load(os.path.join(kaggle_dir,'best_direct_model.pth'), weights_only=True))
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        # for hist_b, nwp_b, coord_b, y_b in val_loader:
        for hist_b, nwp_b, coord_b, y_b in train_loader:
            out = model(coord_b.to(device), hist_b.to(device), nwp_b.to(device))
            pred = (out.cpu().numpy()*y_std + y_mean)
            true = (y_b.numpy()*y_std + y_mean)
            preds.append(pred) 
            trues.append(true)
    preds = np.concatenate(preds,0)
    trues = np.concatenate(trues,0)
    np.save(os.path.join(kaggle_dir, "preds_windows.npy"), preds.astype(np.float32))
    np.save(os.path.join(kaggle_dir, "trues_windows.npy"), trues.astype(np.float32))
#########################################################################
    time_val = None   # 或者用 pandas.date_range(...) 构造

    # 1) 连续时间序列（拼接）
    # time_full = np.load(f"/thfs1/home/qx_hyt/hpp/data/station_AI/test_24_train/54456/train_time.npy", allow_pickle=True)
    # print(time_full[24:])
    time_full=time
    # plot_chain_rows_by_step(trues, preds,  spd_train_all, time_full,unit='m/s',tag='direct', auto_time=True)
    # plot_timeseries_stitched(trues, preds, spd_train_all,time_full, unit='m/s', tag='direct')
    np.save(os.path.join(kaggle_dir, "spd_train_all.npy"), spd_train_all)
    np.save(os.path.join(kaggle_dir, "time_full.npy"), time_full)
    # plot_kde2d_full(trues, preds, unit='m/s', tag='direct',
    #             nx=200, ny=200, 
    #             show_contour=True,
    #             bw_method=None)  
    # plot_scatter_by_leads(trues, preds,  unit='m/s', tag='direct')
    # plot_residual_hist_all(trues, preds,bins=40, unit='m/s', tag='direct')
    yt = trues.reshape(-1)
    yp = preds.reshape(-1)
    err = yp - yt
    labels, perc, cnt, total = bin_percentages(
        err,
        edges=(-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12),
        include_outside=True,   # 同时给出区间外比例
        
    )
    # plot_bin_percentages(labels, perc, tag='direct')
##########################################################################
    edges = np.array([1.5, 3.3, 5.4, 7.9, 10.7], dtype=float)
    labels = [
        "1-level", "2-level", "3-level", "4-level",
        "5-level", ">=6-level"   ]
    # edges = np.array([1.5, 3.3, 5.4, 7.9, 10.7, 13.8, 17.1, 20.7, 24.4, 28.4, 32.6], dtype=float)
    # labels = [
    #     "1-level", "2-level", "3-level", "4-level",
    #     "5-level", "6-level", "7-level", "8-level",
    #     "9-level", "10-level", "11-level", "12-level"
    # ]

    def wind_level_index(speed_1d, edges):
        return np.digitize(speed_1d, edges, right=False)  #right=False左闭右开

###############################################################################################
    # xlabels=np.arange(1,25)
    # nlev = len(labels)
    # delta_mae_lvl = np.full((nlev, F), np.nan, dtype=float)
    # cnt_lvl = np.zeros((nlev, F), dtype=int)

    # for lead in range(F):
    #     t = trues[:, lead].astype(float)
    #     p = preds[:, lead].astype(float)
    #     s = spd_train_all[:, lead].astype(float)

    #     lvl = wind_level_index(t, edges)  # 用 True 风速分级
    #     for k in range(nlev):
    #         m = (lvl == k)
    #         cnt_lvl[k, lead] = int(m.sum())
    #         if m.any():
    #             mae_pred = np.mean(np.abs(p[m] - t[m]))
    #             mae_nwp  = np.mean(np.abs(s[m] - t[m]))
    #             delta_mae_lvl[k, lead] = mae_pred - mae_nwp

    # # 如果你想在格子里标注样本数：用 annotate=cnt_lvl
    # plot_delta_mae_heatmap(
    #     delta_mae_lvl,
    #     xlabels=xlabels,
    #     ylabels=labels,
    #     outfile="delta_mae_heatmap_by_level.png",
    #     cmap="seismic",
    #     annotate=None  # 改成 cnt_lvl 就会每格写 n
    # )
################################################################################################
    # ===== 2) 逐 lead 画 24 张图 =====
    # B, F = trues.shape   # F=24
    # x = np.arange(len(labels))
    # w = 0.22  # 柱宽

    # for lead in range(F):
    #     t = trues[:, lead].astype(float)
    #     p = preds[:, lead].astype(float)
    #     s = spd_train_all[:, lead].astype(float)

    #     lvl = wind_level_index(t, edges)  # 用 True 风速分风级（最合理）

    #     mae_pred = np.full(len(labels), np.nan, dtype=float)
    #     mae_nwp  = np.full(len(labels), np.nan, dtype=float)
    #     cnt      = np.zeros(len(labels), dtype=int)

    #     for k in range(len(labels)):
    #         m = (lvl == k)
    #         cnt[k] = int(m.sum())
    #         if m.any():
    #             mae_pred[k] = np.mean(np.abs(p[m] - t[m]))
    #             mae_nwp[k]  = np.mean(np.abs(s[m] - t[m]))

    #     plt.figure(figsize=(6,4))
    #     plt.bar(x - w/2, mae_pred, width=w, label="DL Pred vs True",color='tab:red')
    #     plt.bar(x + w/2, mae_nwp,  width=w, label="NWP(Interp) vs True",color='tab:blue')

    #     plt.xticks(x, labels)
    #     plt.ylabel("MAE [m/s]")
    #     plt.title(f"Lead {lead+1:02d}h MAE by Wind Level")

    #     # 可选：把样本数写在 x 轴下方（避免你误解某些风级样本很少）
    #     for i in range(len(labels)):
    #         plt.text(x[i], 0, f"n={cnt[i]}", ha="center", va="bottom", fontsize=8, rotation=0)

    #     plt.grid(axis='y', alpha=0.3)
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(f"mae_by_windlevel_lead{lead+1:02d}.png", dpi=160)
    #     plt.close()
###########################################################################################################
    t_all = trues.reshape(-1).astype(float)           # True
    p_all = preds.reshape(-1).astype(float)           # Pred
    s_all = spd_train_all.reshape(-1).astype(float)   # NWP interpolated wind speed (baseline)
    lvl_all = wind_level_index(t_all, edges)
    mae_level_pred = np.full(len(labels), np.nan, dtype=float)
    mae_level_spd  = np.full(len(labels), np.nan, dtype=float)
    cnt_level_all  = np.zeros(len(labels), dtype=int)  ##每级有多少样本
    for k in range(len(labels)):
        m = (lvl_all == k)
        cnt_level_all[k] = int(m.sum())
        if m.any():
            mae_level_pred[k] = np.mean(np.abs(p_all[m] - t_all[m]))
            mae_level_spd[k]  = np.mean(np.abs(s_all[m] - t_all[m]))

    # 画“分组柱状图”
    x = np.arange(len(labels))
    w = 0.22

    plt.figure(figsize=(6,4))
    plt.bar(x - w/2, mae_level_pred, width=w, label="DL Pred vs True", color='tab:red')
    plt.bar(x + w/2, mae_level_spd,  width=w, label="NWP(Interp) vs True", color='tab:blue')
    plt.xticks(x, labels, ha='right')
    plt.ylabel("MAE [m/s]")
    for i in range(len(labels)):
        plt.text(x[i], 0, f"n={cnt_level_all[i]}", ha="center", va="bottom", fontsize=8, rotation=0)
    # plt.title("MAE by Wind Level (all hours pooled)")
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{kaggle_dir}mae_by_wind_level_allhours_pred_vs_nwp.png")
    plt.close()
#############################################################################
# 用法（你已有 trues, preds）：
    # rmse_lead, r_lead = plot_rmse_r_by_lead(trues, preds, outfile="rmse_r_by_lead_allstations.png", unit="m/s")
#############################################################################
    # 每小时MSE
    forecast_hours=24
    hourly_mse = [mean_squared_error(trues[:,i], preds[:,i]) for i in range(forecast_hours)]
    hourly_mae  = [mean_absolute_error(trues[:,i], preds[:,i]) for i in range(forecast_hours)]
    plt.plot(hourly_mse,'o-')
    plt.savefig(f"{kaggle_dir}hourly_mse_direct.png")
    plt.close()
    plt.plot(hourly_mae,'o-')
    plt.savefig(f"{kaggle_dir}hourly_mae_direct.png")
    plt.close()

    hourly_mse = [mean_squared_error(spd_train_all[:, i], preds[:, i]) for i in range(forecast_hours)]
    hourly_mae = [mean_absolute_error(spd_train_all[:, i], preds[:, i]) for i in range(forecast_hours)]
    # 绘制 MSE 曲线
    plt.plot(hourly_mse, 'o-')
    plt.title("Hourly MSE for Wind Speed")
    plt.xlabel("Hour")
    plt.ylabel("MSE")
    plt.savefig(f"{kaggle_dir}hourly_mse_wind_speed.png")
    plt.close()
    # 绘制 MAE 曲线
    plt.plot(hourly_mae, 'o-')
    plt.title("Hourly MAE for Wind Speed")
    plt.xlabel("Hour")
    plt.ylabel("MAE")
    plt.savefig(f"{kaggle_dir}hourly_mae_wind_speed.png")
    plt.close()

    # 单个样本
    plt.figure()
    plt.plot(trues[0],label='True')
    plt.plot(preds[0],label='Pred')
    plt.legend()
    plt.savefig(f"{kaggle_dir}example_direct0.png")
    plt.close()
    plt.plot(trues[1],label='True')
    plt.plot(preds[1],label='Pred')
    plt.legend()
    plt.savefig(f"{kaggle_dir}example_direct1.png")
    plt.close()
    plt.plot(trues[10],label='True')
    plt.plot(preds[10],label='Pred')
    plt.legend()
    plt.savefig(f"{kaggle_dir}example_direct10.png")
    plt.close()

    print("Total MSE:", mean_squared_error(trues.flatten(), preds.flatten()))
    print("Total MAE:", mean_absolute_error(trues.flatten(), preds.flatten()))
    return model
