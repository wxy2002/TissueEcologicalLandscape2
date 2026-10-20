# 2026-10-20 代码审查问题修复清单

对应审查基线 `78a86e8351d8ef4a1a1b1cd7e940cc724ce77fe3`。下表按原报告编号逐项记录；涉及 API/科学定义的迁移约定见 [指标定义](metric-definitions.md)。

| 编号 | 修复内容 | 回归证据 |
| --- | --- | --- |
| 1 | 按明确 stride/origin 栅格化，保留整行整列空白；拒绝重复/不对齐坐标；无空间信息只输出 tile 序列 | `test_grid_preserves_gaps_and_origin`、`test_grid_rejects_unknown_or_ambiguous_coordinates` |
| 2 | 先收集所有类别斑块，再计算跨类别 SIMI；增加显式对称相似系数，默认同类 1、异类 0 | `test_similarity_relabeling_and_coefficients` |
| 3 | CONTAG 只按实际存在类别归一化，并使用 `p_i*g_ik/sum_k(g_ik)`；二维和六面三维均修正 | `test_contagion_formula_and_absent_classes` |
| 4 | 二维 IJI/AI/COHESION 按列明公式实现；PLADJ/CLUMPY 统一有向邻接计数及背景口径；明确三维扩展或重命名 | `test_standard_aggregation_metrics`、`test_unique_landscape_boundaries` |
| 5 | ENN/PROX/SIMI/CONNECT 使用边缘像元中心距离；PROX/CONNECT 只比较同类；无邻居 PROX=0 | `test_boundary_distance_and_radius`、`test_spatial_index_matches_brute_force` |
| 6 | 聚类写 `label_maps.json`；兼容旧目录时排除聚类汇总 NPZ/CSV；排除当前输出路径和已有结果表，允许重跑 | `test_batch_skips_cluster_summaries_and_its_output`、聚类端到端测试 |
| 7 | 默认保留原图，拒绝已有输出，临时写入+全字段回读校验+原子不覆盖发布；删除需明确指定，并检查来源变化 | `test_merge_preserves_sources_and_rejects_incremental_overwrite`、`test_atomic_save_failure_does_not_delete_or_publish`、`test_new_patient_revision_replaces_only_manifest_entry` |
| 8 | 增加患者 CSV 分组；校验固定模型、物理尺度与单位；存储切片区域/来源；指标和过滤使用切片边界，按底层数据汇总 | `test_manifest_grouping_and_patient_batch`、`test_patient_results_ignore_canvas_layout`、`test_patient_composition_pools_area`、`test_model_and_scale_validation_retains_originals` |
| 9 | 单样本簇 silhouette 为 0，零距离样本保留为 0；补全退化质量分数情况 | `test_singleton_and_zero_distance_quality` |
| 10 | reservoir 填满后继续处理同一文件尾部，且记录与样本对齐的原始行索引 | `test_reservoir_visits_tail_and_tracks_rows` |
| 11 | 读取和固定模型推理允许单行，训练抽样在全局检查最小样本数 | `test_single_row_per_file_is_a_valid_global_sample`、单 tile 推理端到端测试 |
| 12 | 推理计算实际抽样的 silhouette/CH/DB，并记录样本数；CH/DB 使用样本簇的实际中心，inertia 保留全量含义 | `test_training_inference_batch_and_sample_alignment`；三个独立样例与 scikit-learn 对照 |
| 13 | `feature-clustering` extra 声明 h5py，补全安装说明 | wheel 构建、安装到隔离目标目录、依赖 METADATA 检查通过；H5 训练/推理端到端测试 |
| 14 | 去掉斑块对内的列表查找；边缘点/包围球空间索引；按斑块包围盒统计；二维/三维按文件流式批处理 | 随机二维/三维斑块与暴力边缘距离对照；[合成性能记录](review-validation.json) |
| 15 | spacing 要求有限正值；负/非整数阈值、负/NaN 半径、非法 max_classes、错误连通性和系数在 API 边界拒绝 | `test_parameter_validation` |

其他报告建议也已处理：拆出 `feature_io.py`、`grid.py`、`cluster_quality.py` 和 `landscape_common`，移除未使用的旧输出实现；训练/评价的 sampled latent、标签、来源、行号明确对齐；三维无依据的分形维数字段移除；各类边界之和与景观唯一边界分别输出；NIfTI 间距约定在 README 明确说明。加入 26 个回归测试和 Python 3.10/3.12 的 GitHub Actions 配置。

## 已执行验证

- 本地 Python 3.10：26 个测试全部通过，包含 CPU 自编码器训练、固定模型保存/加载、H5 输入、单 tile 推理、批处理重跑、NIfTI 标签和合并失败保护。
- 安装包：在临时副本构建 wheel，检查全部新增包模块及 h5py extra 元数据，在隔离目标目录安装并验证导入及指标计算。没有模拟完全无缓存、无现有依赖的网络安装。
- 入口：全部 Python 文件语法检查、9 个 CLI 入口帮助输出检查通过。
- 聚类质量分数：三个包含 singleton 的独立数据样例与 scikit-learn 数值一致。
- 合成性能：三个重复的中位耗时，8×8/16 斑块约 0.014 s；16×16/64 斑块约 0.102 s；24×24/144 斑块约 0.426 s；200×200/10,000 斑块、半径 3 约 6.209 s。整段检查进程峰值 RSS 约 153.6 MiB。结果仅代表本机合成输入，不外推真实 WSI 吞吐。

复现测试：

```bash
pip install -e '.[feature-clustering]'
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -v
```

性能脚本：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/benchmark_metrics.py --output /tmp/tle2-performance.json
```

## 迁移与验证范围

- 旧标签图若把完整空行/列压掉，必须从原始坐标与真实步长重建。旧患者拼接图若缺少原图/切片范围，不能自动恢复真实拓扑。
- 受影响的历史指标需重算；三维输出字段变更、SIMI 系数及边界计数改变，见指标定义。
- 有限半径的空间索引减小候选数；无限半径精确 PROX/SIMI 仍可能需要 O(P²) 时间。
- 未执行 GPU、全新网络环境安装或整批真实 WSI 分析；未增加原始 SVS 的切块/特征提取流程。患者身份、样本类型/时间点及真实 MPP/步长需由研究输入清单核实。
