# 统一融合分析与执行队列

执行起点为 `experiments/flatten-20261109/editable/`，固定性能场景仍为
`v41-c8-i100-o256-v1`。排序先依据真实 trace 中可回收的算子开销、调用频次及实现边界；
它是实现前的收益优先级，实际收益必须由逐项 A/B 更新，不能将共享的 Cast/Mul 时间重复计入多个候选。

已检查目标模型的 prefill/decode、三层 DSpark、HC、Attention/Indexer/Compressor、
FP8/FP4 QDQ、MoE 的 routed/shared expert、Engram、输出 head 和视觉边界。
原始证据、分析覆盖及当前展平基线见 [FLATTENING.md](FLATTENING.md)。
全部实测轮次、数值边界及默认组合见 [FUSION_RESULTS.md](FUSION_RESULTS.md)。

| 顺序 | 可融合区域 | 证据与收益依据 | Triton 实现边界与正确性约束 | 实施状态 |
| ---: | --- | --- | --- | --- |
| 1 | HC 门控与完整 Sinkhorn | 4×4 FP32 的明确 shape 组在 2 秒窗口中约 140 ms；每 HC 有 117 个 sum/add/div 表达式，40 个主模型 block 和 3 个 draft stage 重复执行 | 单 kernel 输出新 pre/post/comb，保留 eps、归一化轴及 20 次迭代；前序 pre 的消费时序不变 | 已实现；HC 服务三轮均值 70.03 token/s |
| 2 | FP8/FP4 QDQ 的全部标量/位运算 | Attention、KV、Indexer、输出投影及 DSpark 共用；大量 Cast、Clip、BitwiseAnd、FloorDiv/Mod 分散执行，家族独占耗时待微基准核对 | group-32 E8M0，compressed-KV 的 group-16 E4M3；精确舍入、符号零、饱和/NaN、scale 下界及 BF16 输出不能改变 | 已实现；累积均值 81.69 token/s；33 项 QDQ 逐值一致测试 |
| 3 | 缓存字段偏移 + slot 索引 + gather/mask | 两个全缓存平面 Slice 组约 49.62 ms，另有 Gather/Select；优先消除大平面切片物化的候选 | 直接从物理页/字段 offset 读需要的 slot；保留 bank、页共享、negative slot、prefix cache、压缩源层和拒绝尾部语义 | 已实现选择性读写；大 raw/index 读取保留原生；累积均值 84.10 token/s |
| 4 | HC post 残差混合 | 广播到 `[T,4,4,5120]` 后归约，重复 86 个子层；避免临时大张量和多次 Cast/Mul/Reduce | 单 kernel 按输出 stream 计算，保持 comb 轴方向、FP32 运算和 BF16 舍入 | 已实现并通过服务正确性；增量吞吐存在轮次波动，保留全部结果 |
| 5 | HC pre + RMSNorm，以及最终 HC pre/norm | 相邻混合、归约、Cast 与归一化；适用于主模型和 draft | 必须保留 pre 混合结果的 BF16 舍入，再转 FP32 计算 RMSNorm；不能用当前新 gate 替换前序 gate | 已实现并保留 BF16 边界；与 post 共 11 项 NPU 测试 |
| 6 | MoE clipped SwiGLU + route factor + 量化 | routed/shared expert 在每层执行；激活、Cast 与 DynamicMxQuant 有可合并空间 | 先融合激活与因子；再验证 MXFP8 scale/rounding。gate/up clamp 不对称，权重必须在 W2 量化前相乘 | 已实现激活/路由因子；MXFP8 量化保持原生；24 项 NPU 测试 |
| 7 | Attention/DSpark RMSNorm + RoPE + QDQ + cache write | 每层 q/kv 投影之后的连续向量操作；复用第 2 项后可继续合并，减少中间读写 | 保留 norm 的 BF16 边界、相邻配对 RoPE、inverse、压缩 position 以及 cache write mask；GEMM 与通信仍为边界 | 已实现 KV norm/RoPE/FP8 QDQ；19 项逐值一致/异常值测试 |
| 8 | Indexer dot 后的 ReLU、权重、head sum、mask，以及 candidate block 处理 | 索引阶段有 Select/Reduce/Gather，受 source-layer 共享与 64K 容量影响 | 首先保留原生 GEMM 和 top-k；融合 epilogue 与 mask。top-k cutoff 的低列号 tie-break 不得改变 | 已实现 decode GEMM epilogue；保留原生 top-k；10 项 NPU 测试 |
| 9 | Compressor softmax-weighted pooling + norm | 只有 source layers 执行，ratio=2 的双 token 聚合及完整组 mask 可合并 | FP32 softmax/加权归约、BF16 输出、group-start RoPE 和 incomplete-group 写缓存条件 | 已实现 ratio=2 pooling/norm；12 项 NPU 测试 |
| 10 | Engram 双 RMS + weighted dot + gate + residual | 仅层 1/14，频次低于 HC；可消除大量 FP32 中间量 | 先保留 host mmap、TP reduce-scatter 和投影边界；只处理 unrotated gate，保持 q/k 权重、gate 激活和 token mask | 已实现两个相邻 Triton kernel；10 项 NPU 测试 |
| 11 | SFA sink epilogue + mask，随后研究 inverse RoPE/QDQ | 现有 SFA 已融合主体；后续 log/sigmoid/mul/cast/mask 仍分离 | 保留 LSE、sink 零 V 语义、padding slot，不能以普通 softmax 替换原生 SFA | 已实现 SFA sink/mask epilogue；4 项 NPU 测试 |
| 12 | MoE router softplus/sqrt/bias，以及 top-k 后归一化 | 小形状但重复执行；独占耗时比 HC 小，融合前后评分和 route 权重须分开 | 保留原生 top-k 作为初期边界；选择用带 bias 分数，权重用未加 bias 的分数，VL 模态 bias 单独处理 | 已实现 top-k 后 gather/normalize/scale；评分和选择保持原生；30 项 NPU 测试 |
| 13 | Draft auxiliary mean/concat、confidence concat/sigmoid、输出 norm | 分散的小操作，需在主热点完成后测量边际收益 | LM head/Markov 的 FP32 GEMM 和跨 rank logits 通信保持原接口；不能改变采样和拒绝逻辑 | 已实现 confidence concat/FP32 projection，sigmoid 保留原生；10 项 NPU 测试 |
| — | 视觉塔 norm/activation/patch 后处理 | 当前固定文本场景没有执行，无法用此场景给出正吞吐收益 | 单独做图像正确性；不擅自替换固定吞吐场景以证明收益 | 本轮文本排序外 |
| — | 已有 GroupedMatmul、QuantBatchMatmul、SFA 主体 | 已有成熟原生实现；本次先优化其相邻向量链 | 不把重新实现 GEMM/Attention 本身当作免费的融合收益 | 保留原生边界 |
| — | TP/EP 通信、host Engram mmap/hash、调度/请求队列 | 分布式依赖或 Host 任务，不能由普通单设备 Triton kernel 自动覆盖 | 需要独立的通信/Host 优化证据；不得跨依赖关系强行合并 | 保留依赖边界 |

每个候选依次经历：源码片段参考 → NPU 数值与边界测试 → eager/ACLGraph →
微基准 → 独立版本接入 → 固定 8 并发重复计时。可交错进行下一项分析与上一项编译/验证，
但正式服务压测期间不运行其他 NPU 测试。单项无收益或数值不通过时保留证据，默认部署不启用该项。

按实测收益更新执行结果，不将上表的先验排序当作已经实现的提升。

## 数值、编译与收益边界

完整算子 NPU 回归为 186 项通过，包含空行、实际 decode/prefill 行数、ACLGraph、
QDQ 舍入边界、缓存位模式、无效 slot 和重复写入。CPU 模型回归 303 项通过；
可编辑回调的实例隔离、根对象回调和 SHA 校验另通过 13 项测试。

QDQ 与 KV norm/RoPE/QDQ 在测试中逐值一致；缓存比较包含未写区域的原始位模式。
HC、Compressor、Engram、SFA 和 MoE 的 BF16 结果保留原有舍入边界，允许测试中
明确给出的归约误差；Router 和 confidence 使用 FP32 容差，不能描述为全模型位级等价。

A5 Triton 编译器在部分“归约后直接连接门控”结构上中止。Engram 因此拆为
归一化点积与门控残差两个 kernel；confidence 融合 concat/投影并保留原生 sigmoid。
失败日志保留，未将失败候选接入服务。新 kernel 的 token 行数改为运行时参数并关闭
该参数的数值特化；修改后重新通过全部 186 项 NPU 测试。

后半部分按 Attention 组合（KV、Indexer、Compressor）和输出处理组合
（Engram、SFA、Router、confidence）完成服务验证。前者首三轮均值 97.39 token/s，
后者 94.15 token/s，因此默认保留前者。前者重启确认三轮后，六轮总均值为 97.84 token/s。逐算子微基准和组合服务吞吐
分别记录，不能将组合收益宣称为某一个 kernel 的独立端到端收益。

微基准初期每图调用一次，约 14 微秒的图重放开销会掩盖小算子差异。后半部分使用
每图 20 次调用、80 次重放，结果除以总调用次数；固定服务基准及其预热方式没有改变。
原始记录位于 `experiments/fusion-20261109/`，正式服务记录位于 `logs/fusion-*.json`。
