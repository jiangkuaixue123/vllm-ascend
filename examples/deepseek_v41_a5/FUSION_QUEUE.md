# 统一融合分析与执行队列

执行起点为 `experiments/flatten-20261109/editable/`，固定性能场景仍为
`v41-c8-i100-o256-v1`。排序先依据真实 trace 中可回收的算子开销、调用频次及实现边界；
它是实现前的收益优先级，实际收益必须由逐项 A/B 更新，不能将共享的 Cast/Mul 时间重复计入多个候选。

已检查目标模型的 prefill/decode、三层 DSpark、HC、Attention/Indexer/Compressor、
FP8/FP4 QDQ、MoE 的 routed/shared expert、Engram、输出 head 和视觉边界。
原始证据、分析覆盖及当前展平基线见 [FLATTENING.md](FLATTENING.md)。

| 顺序 | 可融合区域 | 证据与收益依据 | Triton 实现边界与正确性约束 | 初始状态 |
| ---: | --- | --- | --- | --- |
| 1 | HC 门控与完整 Sinkhorn | 4×4 FP32 的明确 shape 组在 2 秒窗口中约 140 ms；每 HC 有 117 个 sum/add/div 表达式，40 个主模型 block 和 3 个 draft stage 重复执行 | 单 kernel 输出新 pre/post/comb，保留 eps、归一化轴及 20 次迭代；前序 pre 的消费时序不变 | 待实现 |
| 2 | FP8/FP4 QDQ 的全部标量/位运算 | Attention、KV、Indexer、输出投影及 DSpark 共用；大量 Cast、Clip、BitwiseAnd、FloorDiv/Mod 分散执行，家族独占耗时待微基准核对 | group-32 E8M0，compressed-KV 的 group-16 E4M3；精确舍入、符号零、饱和/NaN、scale 下界及 BF16 输出不能改变 | 待实现 |
| 3 | 缓存字段偏移 + slot 索引 + gather/mask | 两个全缓存平面 Slice 组约 49.62 ms，另有 Gather/Select；优先消除大平面切片物化的候选 | 直接从物理页/字段 offset 读需要的 slot；保留 bank、页共享、negative slot、prefix cache、压缩源层和拒绝尾部语义 | 待实现 |
| 4 | HC post 残差混合 | 广播到 `[T,4,4,5120]` 后归约，重复 86 个子层；避免临时大张量和多次 Cast/Mul/Reduce | 单 kernel 按输出 stream 计算，保持 comb 轴方向、FP32 运算和 BF16 舍入 | 待实现 |
| 5 | HC pre + RMSNorm，以及最终 HC pre/norm | 相邻混合、归约、Cast 与归一化；适用于主模型和 draft | 必须保留 pre 混合结果的 BF16 舍入，再转 FP32 计算 RMSNorm；不能用当前新 gate 替换前序 gate | 待实现 |
| 6 | MoE clipped SwiGLU + route factor + 量化 | routed/shared expert 在每层执行；激活、Cast 与 DynamicMxQuant 有可合并空间 | 先融合激活与因子；再验证 MXFP8 scale/rounding。gate/up clamp 不对称，权重必须在 W2 量化前相乘 | 待实现 |
| 7 | Attention/DSpark RMSNorm + RoPE + QDQ + cache write | 每层 q/kv 投影之后的连续向量操作；复用第 2 项后可继续合并，减少中间读写 | 保留 norm 的 BF16 边界、相邻配对 RoPE、inverse、压缩 position 以及 cache write mask；GEMM 与通信仍为边界 | 待实现 |
| 8 | Indexer dot 后的 ReLU、权重、head sum、mask，以及 candidate block 处理 | 索引阶段有 Select/Reduce/Gather，受 source-layer 共享与 64K 容量影响 | 首先保留原生 GEMM 和 top-k；融合 epilogue 与 mask。top-k cutoff 的低列号 tie-break 不得改变 | 待实现 |
| 9 | Compressor softmax-weighted pooling + norm | 只有 source layers 执行，ratio=2 的双 token 聚合及完整组 mask 可合并 | FP32 softmax/加权归约、BF16 输出、group-start RoPE 和 incomplete-group 写缓存条件 | 待实现 |
| 10 | Engram 双 RMS + weighted dot + gate + residual | 仅层 1/14，频次低于 HC；可消除大量 FP32 中间量 | 先保留 host mmap、TP reduce-scatter 和投影边界；只处理 unrotated gate，保持 q/k 权重、gate 激活和 token mask | 待实现 |
| 11 | SFA sink epilogue + mask，随后研究 inverse RoPE/QDQ | 现有 SFA 已融合主体；后续 log/sigmoid/mul/cast/mask 仍分离 | 保留 LSE、sink 零 V 语义、padding slot，不能以普通 softmax 替换原生 SFA | 待实现 |
| 12 | MoE router softplus/sqrt/bias，以及 top-k 后归一化 | 小形状但重复执行；独占耗时比 HC 小，融合前后评分和 route 权重须分开 | 保留原生 top-k 作为初期边界；选择用带 bias 分数，权重用未加 bias 的分数，VL 模态 bias 单独处理 | 待实现 |
| 13 | Draft auxiliary mean/concat、confidence concat/sigmoid、输出 norm | 分散的小操作，需在主热点完成后测量边际收益 | LM head/Markov 的 FP32 GEMM 和跨 rank logits 通信保持原接口；不能改变采样和拒绝逻辑 | 待实现 |
| — | 视觉塔 norm/activation/patch 后处理 | 当前固定文本场景没有执行，无法用此场景给出正吞吐收益 | 单独做图像正确性；不擅自替换固定吞吐场景以证明收益 | 本轮文本排序外 |
| — | 已有 GroupedMatmul、QuantBatchMatmul、SFA 主体 | 已有成熟原生实现；本次先优化其相邻向量链 | 不把重新实现 GEMM/Attention 本身当作免费的融合收益 | 保留原生边界 |
| — | TP/EP 通信、host Engram mmap/hash、调度/请求队列 | 分布式依赖或 Host 任务，不能由普通单设备 Triton kernel 自动覆盖 | 需要独立的通信/Host 优化证据；不得跨依赖关系强行合并 | 保留依赖边界 |

每个候选依次经历：源码片段参考 → NPU 数值与边界测试 → eager/ACLGraph →
微基准 → 独立版本接入 → 固定 8 并发重复计时。可交错进行下一项分析与上一项编译/验证，
但正式服务压测期间不运行其他 NPU 测试。单项无收益或数值不通过时保留证据，默认部署不启用该项。

按实测收益更新执行结果，不将上表的先验排序当作已经实现的提升。
