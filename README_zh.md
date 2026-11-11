# `hilbert-ica`

[English](README.md) | **简体中文**

针对纯幅值 fMRI（magnitude-only fMRI）数据的 Yu 氏**希尔伯特相位复值独立成分分析（Hilbert-phase Complex ICA）**精简自包含 Python 实现，支持多被试队列并行计算与基于 Smith (2009) 模板的静息态脑网络（RSN）成分自动筛选。

## 参考文献

1. M.-C. Yu *et al.*, *"Uncovering abnormal patterns of schizophrenia from fMRI regional source phase"* (`phase_fmri_Hilbert.m`, `complex_fmri_HilbertPhase.m`, `ICA_complex.m`, `energymax_T1.m`, `phase_denoising.m`, `selection_component_complex.m`).
2. X.-L. Li and T. Adali, *"Complex independent component analysis by entropy bound minimization,"* IEEE Trans. Circuits Syst. I, vol. 57, no. 7, pp. 1417-1430, 2010 (`complex_ICA_EBM.m`).
3. S. M. Smith *et al.*, *"Correspondence of the brain's functional architecture during activation and rest,"* PNAS, 2009.

---

## 处理流程概览

针对每个 4D 幅值 BOLD NIfTI 数据体（跨被试并行执行，每个工作进程内部将 BLAS 线程数固定为 1）：

1. **脑内掩膜提取（`hilbert.py`）**：保留至少在一半时间帧中幅值高于全局均值（grand mean）的体素（对应 `Complex_ICA.m`），或通过 `--mask` 指定外部 3D 掩膜。
2. **希尔伯特复值信号构建（`hilbert.py`）**：沿时间轴通过 FFT 解析信号计算瞬时相位，并构建复值 fMRI 信号：
   \[
   z(t, v) = x(t, v) \exp\!\bigl(i \arg(\mathcal{H}\{x\}(t, v))\bigr)
   \]
3. **空间去均值与复值 PCA（`ica.py`）**：逐时间点减去空间均值（对应 `icatb_preproc_data(..., 'remove mean per timepoint')`），并将维度降至模型阶数 \(K\)（默认为 `40`；可配置为多阶，如 `40 50 60`）。
4. **复值 ICA-EBM 与反投影（`ebm.py`, `ica.py`）**：通过熵界最小化算法（`SEA` 初始化 + 随机 `North` 阶段 + 椭圆分布精化）分离复值空间源成分，并通过伪逆投影 \(T_c = Z \, S_e^{+}\) 估计复值时间序列。
5. **相位去模糊、去噪与 Smith-2009 模板筛选（`selection.py`）**：
   - 旋转每个成分的相位以最大化时间序列实部能量（`energymax_T1`），并与 Smith 模板（`ref_threshold=2.5`）进行正符号对齐。
   - 执行相位锥去噪（\(|\theta| \le \pi/4\)）与幅值阈值截断（\(|s| < 0.5 \to 0\)）。
   - 为每个目标静息态网络选出使 \(|\operatorname{corr}(|s_{\text{denoised}}|, \text{ref})|\) 最大的最优成分（当设置 `--num-runs > 1` 或多个 `--orders` 时同时跨运行与跨阶数择优；默认目标网络为 `default_mode`、`auditory`、`executive_control`，传入 `--networks all` 可筛选全部 10 个 Smith RSN）。

---

## 安装

```bash
pip install -e .
```

## 命令行使用（CLI）

可在仓库根目录下直接运行 `python main.py`，或安装后使用 `hilbert-ica` 命令入口：

```bash
# 扁平目录下的 NIfTI 文件（快速默认配置：K=40，1 次运行，论文中的 3 个核心 RSN）
python main.py --data-dir /path/to/niftis --out /path/to/out --n-jobs 8

# BIDS 格式数据集目录树
python main.py --bids-dir /path/to/bids --bids-pattern "*_desc-preproc_bold.nii.gz" \
    --out /path/to/out --n-jobs 8

# 完整论文配置：多模型阶数 x 多次重复运行，覆盖全部 10 个 Smith RSN
python main.py --files sub-001.nii sub-002.nii --out /path/to/out \
    --orders 40 50 60 --num-runs 4 --networks all --max-frames 146
```

对同一 `--out` 输出目录重复运行时，默认会自动跳过已生成的 `<subject>.npz` 文件；如需强制重算，请传入 `--overwrite`。

---

## 输出格式（`<out>/<subject>.npz`）

每个被试的结果以原子写入方式保存为独立的压缩 `.npz` 文件，仅存储紧凑的分解空间图谱与时间序列：

| 键名 | 形状与数据类型 | 说明 |
|---|---|---|
| `selected_maps` | `(M, V)` `complex64` | 经相位校正与去噪（`\|angle\| <= pi/4`, `\|s\| >= 0.5`）后的复值空间图谱 |
| `selected_raw_maps` | `(M, V)` `complex64` | 经相位与符号校正、但未做相位锥/幅值置零前的原始复值空间图谱 |
| `selected_time_courses` | `(T, M)` `complex64` | 经相位与符号校正后的复值时间序列 |
| `network_names` | `(M,)` `str` | 选出的 Smith-2009 脑网络名称 |
| `correlations` | `(M,)` `float32` | 最优成分与模板的相关系数 `\|corr(\|selected_maps[m]\|, ref_m)\|` |
| `selected_components` | `(M,)` `int32` | 最优 ICA 成分索引（另含 `selected_runs`、`selected_orders`） |
| `mask` | `(nx, ny, nz)` `bool` | 3D 脑内体素掩膜（`volume[mask]` 映射至第 `V` 维） |
| `affine` / `tr` | `(4, 4)` / 标量 `float64` | 空间仿射矩阵与重复时间 TR（秒） |

### Python API 与按需即时重建复值 BOLD 信号

```python
import numpy as np
from hilbert_ica import load_subject_npz, reconstruct_bold

sub = load_subject_npz("outputs/sub-001.npz")
maps = sub["selected_maps"]               # (M, V) complex64: 幅值 = np.abs(maps), 相位 = np.angle(maps)
tc = sub["selected_time_courses"]         # (T, M) complex64

# 可选：按需即时重建 (V, T) 复值 fMRI 时间序列
bold_vt = reconstruct_bold(
    maps,
    tc,
    selected_components=sub["selected_components"],
    selected_runs=sub["selected_runs"],
    selected_orders=sub["selected_orders"],
)
```
