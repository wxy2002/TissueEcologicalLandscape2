# 指标定义与空间契约

## 网格、单位与背景

二维数组轴为 `(row, column)`，`spacing=(sx,sy)` 分别是列和行的物理网格间距；三维数组轴与 `spacing=(sx,sy,sz)` 顺序一致。长度、面积、体积的单位由 spacing 决定。背景为 0。`classes` 控制输出类别，必须包含所有实际出现的非背景类别；额外请求的缺失类别输出 NaN，不改变景观指标。

二维 tile 坐标使用 `(x,y)`，必须提供真实 `grid_stride`。索引为 `(coord-origin)/stride`，要求整数对齐并拒绝重复位置。默认原点取最小坐标，也可以显式给出以保留最前面的空行列；不能从有缺失的坐标差分可靠推断真实步长。无坐标或无步长的聚类只输出 tile 标签，不生成空间矩阵。

坐标默认属于 level 0；非零层级必须记录对应的 `coord_downsample`。若 level-0 MPP 已知，`spacing = grid_stride * coord_downsample * mpp`，单位 µm；否则只标为 `coordinate_units`，不能与其他切片做物理尺度汇总。网格步长不是 tile 边长，重叠 tiles 也必须使用实际步长。

## 二维聚集指标

以下定义使用边邻接，连通性 4/8 仅影响斑块划分，不影响邻接计数。邻接次数不按物理边长加权；边界长度单独按 spacing 换算。没有外部 border 数据，PLADJ/CLUMPY 的分母包含内部背景及外部边界。

- **CONTAG**：令 `g_ik` 为有序的双计数邻接次数（同类共享边贡献两个同类邻接，异类共享边贡献两个相反方向的邻接）。`p_i` 为非背景面积比例，`q_ik=p_i*g_ik/sum_k(g_ik)`，结果为 `100*(1+sum(q_ik*ln(q_ik))/(2*ln(m)))`。`m` 仅为实际存在的非背景类别数。背景邻接不进入此熵；少于两类或无有效内部邻接时为 NaN。[官方定义](https://www.fragstats.org/index.php/fragstats-metrics/patch-based-metrics/aggregation-metrics/l1-contagion-index)。
- **IJI**：仅对不同类别的唯一共享边取长度 `e_ik`，令 `r_ik=e_ik/sum(e_ik)`，结果为 `-100*sum(r_ik*ln(r_ik))/ln(m*(m-1)/2)`。背景和外部边界不进入此熵；少于三类或无异类共享边时为 NaN。[官方定义](https://www.fragstats.org/index.php/fragstats-metrics/patch-based-metrics/aggregation-metrics/l2-interspersion-and-juxtaposition-index)。
- **AI**：`100*sum_i(p_i*g_ii_single/max_g_ii)`。最大同类邻接数按紧凑栅格计算：`n=floor(sqrt(a_i))`、`r=a_i-n²`；`max_g_ii=2n(n-1)`（r=0），否则增加 `2r-1`（r≤n）或 `2r-2`（r>n）。单像元类别贡献为 0；全部类别均无可定义最大邻接时为 NaN。[官方定义](https://www.fragstats.org/index.php/fragstats-metrics/patch-based-metrics/aggregation-metrics/l4-aggregation-index)。
- **COHESION**：`100*(1-sum(p_j)/sum(p_j*sqrt(a_j)))/(1-1/sqrt(Z))`。`p_j` 是斑块暴露的像元边数，`a_j` 是像元数，`Z` 是景观非背景像元数；不把物理周长直接代入像元公式。Z=1 时为 0。[官方定义](https://www.fragstats.org/index.php/fragstats-metrics/patch-based-metrics/aggregation-metrics/c7-patch-cohesion-index)。
- **PLADJ**：同类双计数邻接数除以该类全部像元边数 `4*a_i`，乘 100。单像元类别为 NaN。[官方定义](https://www.fragstats.org/index.php/fragstats-metrics/patch-based-metrics/aggregation-metrics/c2-percentage-of-like-adjacencies)。
- **CLUMPY**：`G=g_ii_double/(4*a_i)`，`P=a_i/景观范围像元数`（此处包含真实内部背景）。当 G<P 且 P<0.5 时为 `(G-P)/P`，其余为 `(G-P)/(1-P)`。P=1 且 G<1 时公式未定义，返回 NaN；无外部同类 border 时不能强行置为 1。[官方定义](https://www.fragstats.org/index.php/fragstats-metrics/patch-based-metrics/aggregation-metrics/c4-clumpiness-index)。

## 斑块距离与邻域

距离 `d(j,k)` 是两个斑块边缘像元**中心**的最短欧氏物理距离，包含各轴 spacing，不是质心距离，也不是像元外轮廓之间的净空隙。

- **ENN**：同类斑块中最小的 `d`；没有同类邻居时 NaN。[官方定义](https://www.fragstats.org/index.php/fragstats-metrics/patch-based-metrics/aggregation-metrics/p1-euclidean-nearest-neighbor-distance)。
- **PROX**：对搜索半径内同类邻居累加 `area_k/d(j,k)²`；无邻居为 0。[官方定义](https://www.fragstats.org/index.php/fragstats-metrics/patch-based-metrics/aggregation-metrics/p2-proximity-index)。
- **SIMI**：累加 `similarity(i,k)*area_k/d(j,k)²`，可跨类别。`similarity_weights` 为对称的 `[0,1]` 系数；未指定的同类为 1、异类为 0，因此默认 SIMI=PROX。显式系数由分析设计给出，禁止再用类别总面积的 min/max 代替相似性。[官方定义](https://www.fragstats.org/index.php/fragstats-metrics/patch-based-metrics/aggregation-metrics/p3-similarity-index)。
- **CONNECT**：半径内的同类斑块对数除以同类可能斑块对数，乘 100；距离恰等于阈值时计为连接。不存在可能的斑块对时为 0。[官方定义](https://fragstats.org/index.php/fragstats-metrics/patch-based-metrics/aggregation-metrics/c13-connectance-index)。

使用 cKDTree 索引质心包围球以筛选候选，再用边缘点树求精确距离；质心不作为最终距离。ENN 不建立 P×P 稠密矩阵。有限半径可减少候选；无限半径的精确 PROX/SIMI 仍可能需要 O(P²) 时间，但不会存储成对距离矩阵。

## 患者多切片

`region_ids` 与 matrix 同形状，正整数标识独立切片区域（含各自真实背景），0 表示排版留白。任何非背景像元都必须属于正区域。斑块划分、过滤、邻接和所有距离搜索均限制在单个区域；人工间隔为 0 时仍不能跨切片连接。

患者组成先合计类别面积再计算 PLAND/多样性；斑块分布汇集原始记录再计算 mean/中位数/SD/面积加权均值。ENN 的缺失值不参与汇总。CONNECT 汇集每类每切片的已连接对数和可能对数，分母中排除跨切片对。AI 每类的同类邻接数和各切片最大可能邻接数分别求和，再按患者类别面积比例加权。CONTAG 汇集切片内邻接；COHESION 的 Z 为所有切片非背景像元之和。患者范围是各原图真实区域的总面积，不是拼接画布面积。

必须校验同一个固定模型 `model_id`、相同物理 spacing 及单位；患者映射由显式清单或经过核验的专用前缀规则提供。不能从文件名截断自动证明患者身份。来源文件、切片范围、MPP/坐标元数据均保存在合并图中。旧拼接图没有边界时拒绝进入指标流程。

## 三维扩展和输出迁移

三维的 CONTAG 使用六面邻接及相同的类别比例条件概率；IJI 使用异类共享**面面积**的熵。这是本项目明确约定的三维扩展，不是声称存在三维 FRAGSTATS 标准。`like_adjacency_percent` 使用 `6*voxel_count` 为分母，CLUMPY 使用真实体积范围比例；ENN 使用边缘体素中心间的最短物理距离。

原三维 `landscape_aggregation_index` 删除，替换为 `landscape_like_face_adjacency_percent`（同类双计数面邻接/全部组织体素面数 ×100），不再冒充标准 AI。原 `class_*_cohesion_3d` 改名为 `class_*_patch_volume_concentration`，明确为自定义的 `100*sqrt(sum(V_patch²))/sum(V_patch)`。没有得到验证的 `fractal_dimension_3d_*` 字段移除；不再把体积与表面积的单位相关对数比称作分形维数。

二维 `landscape_total_edge_length` 和三维 `landscape_total_surface_area` 均按唯一边界计数，异类共享边/面只计一次，背景/外部边界全部计入。对应 ED/表面密度、LSI、接触比例与 contrast 分母采用该口径。旧的各类边界之和另存为 `landscape_sum_class_edge_length` / `landscape_sum_class_surface_area`。

其他约定：比例 PLAND、斑块密度、ED 的分母为非背景面积/体积；范围密度用真实范围（包含真实背景）。所有 summary SD 使用样本标准差 ddof=1，少于两个有效值时为 NaN；AM 使用斑块面积加权。二维部分形状指标仍是当前物理栅格定义，不能把每个字段都视作对任意分辨率、长宽比均严格等价的 FRAGSTATS 输出。

`load_volume()` 不自动应用 NIfTI header/affine 的尺度。调用者需按数组轴显式传入 spacing；斜切等非正交情况必须先处理数据。二维不能把独立 WSI 当成已配准三维切片。
