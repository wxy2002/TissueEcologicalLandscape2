# landscape3d：三维生态景观指数包

`landscape3d` 用于从三维标签矩阵中计算三维生态景观指数。输入是
`H x W x D` 的整数矩阵：

- `0` 表示背景，不参与类别指标计算。
- `1..n` 表示不同类别。
- 每个样本可以保存为一个 `.npy`、`.npz`、`.nii` 或 `.nii.gz` 文件。
- 输出为宽表 CSV：每行一个样本，每列一个指标。

## 安装

在项目目录下运行：

```bash
pip install -e .
```

依赖包括 `numpy`、`pandas`、`scipy`、`nibabel`。

如果不想安装，也可以在当前目录下直接用：

```bash
python3 -m landscape3d.cli samples metrics.csv
```

## 输入格式

推荐把所有样本放在一个文件夹中，例如：

```text
samples/
  case001.npy
  case002.npy
  case003.npz
  case004.nii.gz
```

`.npy` 文件应直接保存一个三维整数矩阵：

```python
import numpy as np

volume = np.zeros((20, 30, 10), dtype=int)
volume[2:8, 3:10, 1:5] = 1
volume[10:15, 12:20, 3:8] = 2

np.save("samples/case001.npy", volume)
```

`.npz` 文件支持两种情况：

- 包含一个名为 `volume` 的数组。
- 或者文件中只有一个数组。

```python
np.savez("samples/case002.npz", volume=volume)
```

`.nii` / `.nii.gz` 文件会用 `nibabel` 读取。数据仍需是三维标签矩阵：

- `0` 表示背景。
- 正整数表示类别。
- 如果 NIfTI 读出为浮点型，但所有值都是整数值，例如 `0.0`、`1.0`、`2.0`，程序会自动转成整数标签。
- 如果存在非整数值、负值、NaN 或 Inf，会报错。

## 命令行使用

最简单用法：

```bash
landscape3d samples metrics.csv
```

指定类别、体素尺寸和连通性：

```bash
landscape3d samples metrics.csv \
  --classes 1,2,3 \
  --spacing 1,1,2 \
  --connectivity 6
```

过滤矩阵内部太小的三维斑块：

```bash
landscape3d samples metrics.csv \
  --min-patch-voxels 50
```

也可以按斑块真实体积过滤。体积会按 `spacing` 计算：

```bash
landscape3d samples metrics.csv \
  --spacing 1,1,2 \
  --min-patch-volume 100
```

如果需要过滤整个非背景区域太少的样本，也可以使用样本级过滤：

```bash
landscape3d samples metrics.csv \
  --min-non-background-voxels 50
```

也可以按真实体积过滤。体积会按 `spacing` 计算：

```bash
landscape3d samples metrics.csv \
  --spacing 1,1,2 \
  --min-non-background-volume 100
```

运行时会在终端显示进度，例如：

```text
[landscape3d] Loading volumes: 1/10 case001.nii.gz
[landscape3d] Computing metrics: 1/10 case001.nii.gz
[landscape3d] Saved metrics: metrics.csv
```

如果不想显示进度：

```bash
landscape3d samples metrics.csv --no-progress
```

参数说明：

- `input_dir`：输入文件夹，包含 `.npy`、`.npz`、`.nii` 或 `.nii.gz`。
- `output_csv`：输出 CSV 文件路径。
- `--classes`：需要统计的类别，例如 `1,2,3`。不提供时，会自动从所有样本中推断 `1..最大类别值`。
- `--spacing`：体素尺寸，格式为 `sx,sy,sz`，默认 `1,1,1`。
- `--connectivity`：三维斑块连通性，可选 `6`、`18`、`26`，默认 `6`。
- `--min-patch-voxels`：计算指标前过滤小斑块，小于该体素数的连通斑块改为背景 `0`，默认 `0`。
- `--min-patch-volume`：计算指标前过滤小斑块，小于该体积的连通斑块改为背景 `0`，默认 `0`。
- `--min-non-background-voxels`：跳过非背景体素数小于该值的样本，默认 `0`。
- `--min-non-background-volume`：跳过非背景体积小于该值的样本，默认 `0`。
- `--no-progress`：关闭命令行进度显示。

## Python API 使用

### 计算单个三维矩阵

```python
import numpy as np
from landscape3d import compute_metrics

volume = np.zeros((10, 10, 10), dtype=int)
volume[1:4, 1:4, 1:4] = 1
volume[6:8, 6:8, 6:8] = 2

metrics = compute_metrics(
    volume,
    classes=[1, 2, 3],
    spacing=(1, 1, 2),
    connectivity=6,
)

print(metrics["class_1_volume"])
print(metrics["landscape_shannon_diversity"])
```

如果某个样本中缺失某类，例如 `classes=[1,2,3]` 但样本里没有类别 `3`，
那么 `class_3_*` 指标会输出 `NaN`。

### 批量计算并保存 CSV

```python
from landscape3d import compute_batch

df = compute_batch(
    input_dir="samples",
    output_csv="metrics.csv",
    classes=[1, 2, 3],
    spacing=(1, 1, 2),
    connectivity=6,
    min_patch_voxels=50,
    progress=True,
)

print(df.head())
```

`progress=True` 时会显示批处理进度；默认 `False`，适合在脚本或 Notebook 中安静调用。

如果设置了 `min_patch_voxels` 或 `min_patch_volume`，每个样本内部不满足阈值的连通斑块会先被改为背景 `0`，然后再计算指标。

如果设置了 `min_non_background_voxels` 或 `min_non_background_volume`，不满足阈值的是整个样本，会被跳过，不进入最终 CSV。若所有样本都被过滤，会报错提示降低阈值。

## 输出结果

输出 CSV 是宽表格式，例如：

```text
sample_id,source_file,class_1_volume,class_1_patch_count,...,landscape_shannon_diversity,...
case001,samples/case001.npy,120.0,2.0,...,0.68,...
case002,samples/case002.npy,NaN,NaN,...,0.00,...
```

列名规则：

- `class_1_volume`：类别 1 的体积。
- `class_2_patch_count`：类别 2 的斑块数。
- `class_1_to_2_boundary_area`：类别 1 和类别 2 的接触面面积。
- `landscape_*`：整体景观水平指标。

## 已实现指标

### Class-level 指标

面积/体积：

- `voxel_count`
- `volume`
- `pland`
- `patch_count`
- `patch_density`
- `largest_patch_volume`
- `largest_patch_index`
- `patch_volume_mean/sd/cv/range`

边界/表面积：

- `total_surface_area`
- `surface_density`
- `patch_surface_area_mean/sd/cv/range`
- `boundary_to_background_area`
- `boundary_to_other_classes_area`

形状：

- `surface_volume_ratio_mean/sd/cv`
- `shape_index_mean/sd/cv`
- `sphericity_mean/sd/cv`
- `fractal_dimension_3d_mean/sd/cv`
- `radius_gyration_mean/sd/cv`

聚集/连通：

- `like_adjacency_percent`
- `clumpiness`
- `splitting_index`
- `effective_mesh_size`
- `cohesion_3d`
- `nearest_neighbor_mean/sd/cv`

非加权类别接触：

- `unlike_boundary_area`
- `unlike_boundary_percent`
- `class_A_to_B_boundary_area`

### Landscape-level 指标

组成/多样性：

- `total_non_background_volume`
- `class_richness`
- `shannon_diversity`
- `simpson_diversity`
- `modified_simpson_diversity`
- `shannon_evenness`
- `simpson_evenness`

面积/边界/形状：

- `total_surface_area`
- `surface_density`
- `largest_patch_index`
- `patch_count`
- `patch_density`
- `shape_index_mean`
- `sphericity_mean`

聚集/蔓延：

- `contagion`
- `interspersion_juxtaposition`
- `effective_mesh_size`
- `splitting_index`
- `landscape_division`
- `aggregation_index`

类别接触：

- `landscape_class_A_to_B_boundary_area`
- `total_unlike_boundary_area`
- `unlike_boundary_percent`

## 计算约定

- 背景 `0` 不计算 class-level 指标。
- 缺失类别用 `NaN` 表示。
- 体积 = 体素数 × `sx × sy × sz`。
- 表面积和类别接触面积基于共享面的相邻体素计算。
- 类别接触只考虑面邻接，不考虑边或角接触。
- `pland` 等比例指标的分母为非背景体积，而不是整个矩阵体积。
- 第一版不包含 core area/core volume。
- 第一版不包含需要类别差异权重矩阵的 weighted contrast 指标。

## 快速测试

当前环境如果安装了 `pytest`，可以运行：

```bash
pytest tests
```

也可以直接检查 CLI：

```bash
python3 -m landscape3d.cli --help
```

---

# landscape2d：二维生态景观指数包

`landscape2d` 用于从二维标签矩阵中计算二维生态景观指数。输入是
`H x W` 的整数矩阵：

- `0` 表示背景，不参与类别指标计算。
- `1..n` 表示不同类别。
- 每个样本可以保存为一个 `.npy`、`.npz`、`.csv` 或 `.txt` 文件。
- 输出为宽表 CSV：每行一个样本，每列一个指标。

包名和命令推荐使用 `landscape2d`；同时也提供 `landscale2d` 作为兼容别名。

## 命令行使用

```bash
landscape2d samples2d metrics2d.csv
```

指定类别、像元尺寸和连通性：

```bash
landscape2d samples2d metrics2d.csv \
  --classes 1,2,3 \
  --spacing 1,1 \
  --connectivity 4 \
  --search-radius 100 \
  --max-classes 5 \
  --min-patch-cells 5 \
  --plot-dir label_png
```

如果不安装包，也可以在项目目录下直接运行：

```bash
python3 -m landscape2d.cli samples2d metrics2d.csv
```

参数说明：

- `input_dir`：输入文件夹，包含 `.npy`、`.npz`、`.csv` 或 `.txt`。
- `output_csv`：输出 CSV 文件路径。
- `--classes`：需要统计的类别，例如 `1,2,3`。不提供时，会自动从所有样本中推断 `1..最大类别值`。
- `--spacing`：像元尺寸，格式为 `sx,sy`，默认 `1,1`。
- `--connectivity`：二维斑块连通性，可选 `4`、`8`，默认 `4`。
- `--search-radius`：`PROX`、`SIMI`、`CONNECT` 的搜索距离，默认无限距离。
- `--max-classes`：最大可能类别数，用于 `relative_patch_richness`。
- `--min-patch-cells`：计算指标前过滤小斑块，小于该像元数的连通斑块改为背景 `0`。
- `--plot-dir`：可选；为每个样本输出一张 PNG 标签图，背景 `0` 为白色，不同类别使用不同颜色。
- `--no-progress`：关闭命令行进度显示。

## Python API 使用

```python
import numpy as np
from landscape2d import compute_metrics, compute_batch

matrix = np.zeros((10, 10), dtype=int)
matrix[1:4, 1:4] = 1
matrix[6:8, 6:8] = 2

metrics = compute_metrics(
    matrix,
    classes=[1, 2, 3],
    spacing=(1, 1),
    connectivity=4,
    search_radius=100,
    max_classes=5,
)

df = compute_batch(
    input_dir="samples2d",
    output_csv="metrics2d.csv",
    classes=[1, 2, 3],
    spacing=(1, 1),
    connectivity=4,
    search_radius=100,
    max_classes=5,
    progress=True,
)
```

`.npz` 文件优先读取名为 `matrix` 的数组，也支持名为 `array` 的数组；
如果文件里只有一个数组，也会自动读取该数组。

## 病理特征向量降维聚类

对于病理 tile 或小方格特征，`landscape2d-cluster-features` 支持先用
PyTorch 自编码器把 `n x m` 特征矩阵压缩到较短的 latent 向量，再聚类成类别。
这里 `n` 是小方格数量，`m` 通常可以是 512、768、1024 等高维特征。

输入可以是单个文件，也可以是一个文件夹。训练时如果输入是文件夹，会读取文件夹中
所有支持的特征文件，合并全部小方格特征后训练同一个自编码器和聚类模型。

输入文件支持 `.h5`、`.hdf5`、`.npy`、`.npz`、`.csv`、`.txt`：

- `.h5/.hdf5`：默认读取 `feats` 作为 `n x m` 特征矩阵；如果存在 `coords`，
  会在输出 `.npz` 中保留坐标。
- `.npy`：直接保存一个二维特征矩阵。
- `.npz`：优先读取 `features`，也支持 `x`、`array`，或单数组文件。

固定聚类数：

```bash
landscape2d-cluster-features uni-0242c340 cluster_out \
  --n-clusters 6 \
  --latent-dim 32 \
  --epochs 100 \
  --batch-size 512 \
  --device cuda
```

自动选择聚类数：

```bash
landscape2d-cluster-features uni-0242c340 cluster_out \
  --k-range 2,10 \
  --latent-dim 32 \
  --epochs 100 \
  --max-train-tiles 200000 \
  --max-train-tiles-per-file 2000 \
  --min-patch-cells 5 \
  --device cuda
```

默认情况下，自编码器训练和 K 搜索只使用抽样 tile：

- `--max-train-tiles 200000`：训练/选 K 最多使用 200000 个 tile。
- `--max-train-tiles-per-file 2000`：每个输入文件最多先抽 2000 个 tile。
- 参数设为 `0` 表示不限制。

训练和 K 搜索使用抽样数据，但保存输出时会用固定后的模型逐文件、分批推理全部 tile，
因此每个样本的二维标签矩阵仍覆盖全量小方格。

输出文件：

- `all_clusters.npz`：全部输入文件合并后的顺序标签，包含 `tile_labels`、
  `sample_ids`、`source_files`、`sample_counts`，用于追踪每个 tile 的原始顺序。
- `每个样本名_clusters.npz`：每个输入文件各自的二维标签矩阵，`labels` 和
  `matrix` 都是可直接用于 `landscape2d` 指标计算的整数矩阵；背景/空洞为 `0`，
  聚类类别为 `1..K`。H5 输入会额外保留 `coords` 和 `tile_labels`。
- `cluster_metrics.csv`：聚类相关指标，包括 `inertia`、`silhouette_score`、
  `calinski_harabasz_score`、`davies_bouldin_score`、每类数量和比例；
  自动 K 时用 `selected=True` 标记选中的 K。
- `latent_scatter.png`：latent 特征 PCA 到二维后的聚类散点图。
- `cluster_model.pt`：固定后的推理参数，包括标准化参数、自编码器权重、选中的 K 和 KMeans 聚类中心。

对新数据集使用固定参数推理：

```bash
landscape2d-cluster-predict new_h5_folder \
  cluster_out/cluster_model.pt \
  new_cluster_out \
  --min-patch-cells 5 \
  --device cuda
```

推理输出同样包括每个样本的 `*_clusters.npz`、`cluster_metrics.csv` 和
`latent_scatter.png`。其中 `labels`/`matrix` 仍为二维整数矩阵，类别编号沿用训练模型的 `1..K`。

常用参数：

- `--n-clusters`：指定固定 K。
- `--k-range`：自动选择 K 的范围，例如 `2,10`；与 `--n-clusters` 二选一。
- `--latent-dim`：自编码器输出维度，默认 `32`。
- `--hidden-dims`：自定义隐藏层，例如 `512,128`；不指定时自动推断。
- `--max-train-tiles`：训练 AE 和选 K 使用的总 tile 上限，默认 `200000`。
- `--max-train-tiles-per-file`：每个文件参与训练抽样的 tile 上限，默认 `2000`。
- `--min-patch-cells`：保存二维标签矩阵前过滤小斑块；默认 `0` 不过滤。
- `--save-latent`：把 latent 特征也保存进 `.npz`。
- `--device`：默认 `cuda`；也可指定 `cuda:0`、`cpu`、`auto` 等。

## 同患者多 WSI 标签图合并

如果同一个患者有多个 WSI 切片，对应输出了多个 `*_clusters.npz`，可以按文件名前
N 个字符分组，并把同组二维标签矩阵横向合并。合并时不同 WSI 之间插入背景空白列，
高度不足的矩阵下方补 `0`。

先预览分组，不写入也不删除：

```bash
python3 -m landscape2d.cli_merge output/ccRCC/test_map \
  --prefix-length 12 \
  --blank-cols 3 \
  --dry-run
```

正式合并：

```bash
python3 -m landscape2d.cli_merge output/ccRCC/test_map \
  --prefix-length 12 \
  --blank-cols 3
```

默认行为：

- 按文件名 stem 的前 `--prefix-length` 个字符作为患者 ID。
- 只合并同组内文件数大于 1 的样本。
- 输出 `{患者ID}_merged.npz`，其中 `matrix` 和 `labels` 是合并后的二维整数矩阵。
- 合并成功后删除同组原始 `.npz`，只保留合并文件。
- 单文件组不会修改。
- 如果想保留原文件，添加 `--keep-originals`。

## landscape2d 已实现指标

### Class-level 指标

面积：

- `cell_count`
- `area`
- `pland`
- `patch_count`
- `patch_density`
- `largest_patch_area`
- `largest_patch_index`
- `patch_area_mean/am/md/sd/cv/range`

边界/周长：

- `total_edge_length`
- `edge_density`
- `patch_perimeter_mean/am/md/sd/cv/range`
- `boundary_to_background_length`
- `boundary_to_other_classes_length`

形状：

- `perimeter_area_ratio_mean/am/md/sd/cv`
- `shape_index_2d_mean/am/md/sd/cv`
- `compactness_mean/am/md/sd/cv`
- `fractal_dimension_2d_mean/am/md/sd/cv`
- `related_circumscribing_circle_mean/am/md/sd/cv`
- `contiguity_index_mean/am/md/sd/cv`
- `perimeter_area_fractal_dimension`
- `landscape_shape_index`
- `normalized_landscape_shape_index`
- `radius_gyration_mean/am/md/sd/cv`

聚集/连通：

- `like_adjacency_percent`
- `clumpiness`
- `splitting_index`
- `effective_mesh_size`
- `cohesion_2d`
- `connectance_index`
- `proximity_index_mean/am/md/sd/cv`
- `similarity_index_mean/am/md/sd/cv`
- `nearest_neighbor_mean/am/md/sd/cv`

类别接触：

- `unlike_boundary_length`
- `unlike_boundary_percent`
- `edge_contrast_index`
- `contrast_weighted_edge_density`
- `total_edge_contrast_index`
- `class_A_to_B_boundary_length`

### Landscape-level 指标

组成/多样性：

- `total_non_background_area`
- `extent_area`
- `class_richness`
- `patch_richness_density`
- `relative_patch_richness`
- `shannon_diversity`
- `simpson_diversity`
- `modified_simpson_diversity`
- `shannon_evenness`
- `simpson_evenness`
- `modified_simpson_evenness`

面积/边界/形状：

- `total_edge_length`
- `edge_density`
- `largest_patch_index`
- `patch_count`
- `patch_density`
- `patch_area_mean/am/md/sd/cv/range`
- `patch_perimeter_mean/am/md/sd/cv/range`
- `landscape_shape_index`
- `normalized_landscape_shape_index`
- `perimeter_area_fractal_dimension`
- `shape_index_2d_mean/am/md/sd/cv`
- `compactness_mean/am/md/sd/cv`
- `fractal_dimension_2d_mean/am/md/sd/cv`
- `related_circumscribing_circle_mean/am/md/sd/cv`
- `contiguity_index_mean/am/md/sd/cv`
- `radius_gyration_mean/am/md/sd/cv`

聚集/蔓延：

- `contagion`
- `interspersion_juxtaposition`
- `effective_mesh_size`
- `splitting_index`
- `landscape_division`
- `aggregation_index`
- `connectance_index`
- `proximity_index_mean/am/md/sd/cv`
- `similarity_index_mean/am/md/sd/cv`

类别接触：

- `landscape_class_A_to_B_boundary_length`
- `total_unlike_boundary_length`
- `unlike_boundary_percent`
- `contrast_weighted_edge_density`
- `total_edge_contrast_index`

## landscape2d 计算约定

- 背景 `0` 不计算 class-level 指标。
- 缺失类别用 `NaN` 表示。
- 面积 = 像元数 × `sx × sy`。
- 周长和类别接触长度基于共享边的相邻像元计算。
- 类别接触只考虑边邻接，不考虑角接触。
- `pland` 等比例指标的分母为非背景面积，而不是整个矩阵面积。
- 不包含 core area 系列指标。
- Contrast 指标默认把所有异类边界权重视为 `1`、同类权重视为 `0`；Python API 可通过 `contrast_weights` 传入类别差异权重。
- `PROX`、`SIMI`、`CONNECT` 使用 `search_radius`，默认无限距离；命令行可用 `--search-radius` 指定。
- `relative_patch_richness` 需要 `max_classes`，未指定时输出 `NaN`；命令行可用 `--max-classes` 指定。
