# V4.1 展平与融合实验入口

本次以 `d4ac431` 的 A5 FP8/FP4 适配为基础，借鉴
[Agentic-Compiler](https://github.com/jojoinfra/Agentic-Compiler/tree/c237f86844598388d2d20d5782344bd538bf617e)
的源码内联与等价性验证方法。展平是后续优化的起点；仅展平阶段没有新增融合算子。
后续 Triton 实施状态见 [FUSION_QUEUE.md](FUSION_QUEUE.md)，实测与默认组合见
[FUSION_RESULTS.md](FUSION_RESULTS.md)。

## 可执行源码

部署目录为 `/home/jojo/vllm-ascend-d4ac431`，展平源码位于
`experiments/flatten-20261109/editable/`：

| 文件 | 内容 | 行数 |
| --- | --- | --- |
| `target_forward.py` | 40 层目标模型 prefill，以及 eager/图像分支 | 16590 |
| `target__forward_decode.py` | 40 层目标模型 decode | 8292 |
| `draft_forward.py` | 3 层 DSpark 前向 | 618 |
| `manifest.json` | 静态配置、检查点配置校验值和源文件 SHA256 | — |

`deepseek_v41_flatten=true` 和 `deepseek_v41_flatten_source_dir` 使服务直接编译执行这些文件。
文件是接收现有模型实例的回调，参数与缓存仍由原模型持有；它们不是独立模型检查点。
主干、block、HC、RMSNorm 和 Sinkhorn 已展开。
attention/KV、Engram、量化 MoE、通信接口仍是明确的调用边界，尚未展开为完整的纯算子图。

每次后续优化复制整个目录到新的实验目录，修改代码并更新 manifest 中相应文件的 SHA256，
保留原目录不动。修改静态配置需重新生成；修改文件后不更新清单会拒绝加载。
新增的融合实现放在适当的算子模块中，由展平前向显式调用。
检查点、并行方式和固定性能场景保持不变。

## 正确性与计时

初次展平通过 321 项 CPU 回归、4 项真实 NPU 等价性检查（含 ACLGraph），
并通过 DSpark 的拒绝尾部/缓存密集参考计算对比、真实服务 10/10 API 检查。
可编辑源码加载另外通过 12 项测试，包括静态配置不符和源文件被改动时拒绝加载。
导出为可加载目录时，三个前向文件的 SHA256 与最初测量的版本完全相同。

固定场景是 `v41-c8-i100-o256-v1`：物理 4–7 卡，TP4/EP4，8 个固定输入，
每请求输出 256 token，零缓存命中。每轮有既定的 64-token 预热。
保留首次固定基线 51.9727 token/s，不重设基线。

最初 A/B 各三轮，无 profiler 正式计时：

| 版本 | 三轮吞吐（token/s） | 平均 |
| --- | --- | --- |
| 展平前 | 56.012 / 59.319 / 59.914 | 58.415 |
| 初次展平 | 55.265 / 57.997 / 55.813 | 56.358 |

平均差异为 -3.52%，没有观察到可靠提速。初次展平服务配置了 profiler 接口，但正式计时未开启采集；
最终的可编辑源码服务去掉了 profiler 配置，另行运行相同基准，结果保存在
`logs/flatten-editable-{1,2,3}.json`。这一步用于建立可优化的执行入口，不以单纯展开 Python 带来提速为目标。

最终可编辑源码版本三轮为 **53.620 / 56.013 / 55.147 token/s**，平均 **54.927 token/s**，
相对本轮展平前均值为 **-5.97%**，相对原固定基线为 **+5.68%**。
两种比较都保留，不能选择有利的参照宣称优化成功。这个版本作为后续融合优化的不可变参照保留。
完整信息见 `experiments/flatten-20261109/final-results.json`。
展平阶段的启动脚本保存在实验目录，后续默认入口以部署根目录 `start.sh` 和
`deployment-state.json` 的实际记录为准。融合实验保留原来的 4 张卡、FP8/FP4、DSpark5 和 ACLGraph。

## 设备证据与覆盖

采集复用了参考仓库的 `profile-capture`、`profile-analyze` 流程。
完整 8 并发请求的原始 trace 保存在实验目录 `capture/native/`。
由于全量硬件计数器转换耗时较长，该轮离线转换中止，原始文件和转换状态完整保留。
另一次相同固定工作负载的短 decode 窗口已完成四个 rank 的解析，位于 `capture-short/native/`。

参考工具 `profile_tool.py` 成功发现 4 个 rank，格式检查随后明确拒绝并发请求归属。
保留真实重叠请求时间，使用 `analyze-native.py` 直接查询原生数据库，分析整批请求中的
2.000068 秒 decode 窗口，不把共享 kernel 分摊到某一个请求。
查询结果为 `native-short-summary.json`；每项保留源数据库路径、TASK 行号、shape/dtype 和关联 API。

rank 0 的详细计算 kernel 累计约 775.97 ms，各 rank 约 775.97–783.88 ms。
以下数据为 rank 0，不能跨 rank 相加作为请求延迟：

| 算子类型 | 窗口内次数 | 累计时间（ms） |
| --- | ---: | ---: |
| ReduceSum | 23836 | 71.55 |
| RealDiv | 23729 | 69.07 |
| Mul | 19401 | 67.00 |
| Add | 33712 | 57.00 |
| Slice | 2399 | 54.42 |
| Cast | 19758 | 51.25 |
| GroupedMatmul | 564 | 39.38 |

其中 HC 的 `192 → 48` FP32 归约、`192 / 48` FP32 除法和 `48 + scalar` 加法，
各观测到 20631 次，合计约 139.89 ms，占该 rank 详细计算 kernel 累计时间约 18.0%。
它们与目标模型本窗口每 rank 12 行、4×4 HC 矩阵的源码计算相符；这是源码和 shape 的关联，
没有逐个 Python 调用栈证明所有同 shape 事件都来自唯一代码行。

A5 原生事件包含 `EVENT WAIT`、`MODEL_WAIT_COMPLETE`、CCU 和 UB 窗口。
分析将等待与计算分开；CCU/UB 窗口可能包含等待，不能用其持续时间计算网络带宽或硬件利用率。
最大的约 18–21 ms 间隙邻接 `aclmdlRIExecuteAsync`，但缺少 CPU 调度和完整异步依赖证据，
不能直接归因为 GIL、线程调度或通信瓶颈。前端 profiler 也报告了线程注册警告，Host 覆盖是部分的。
短窗口不覆盖完整 prefill、全部输出尾部或所有上下文长度；算子累计时间不是端到端可节省时间。

## 融合机会

下表中的行号指上述展平目录中的首层位置；每层有对应的重复区域。

| 优先级 | 入口与候选 | 依据及必须保持的语义 |
| --- | --- | --- |
| 1 | `target__forward_decode.py:40`，pre/post 门控与 Sinkhorn | 每个 HC 有 39 组 `sum + eps + divide`，共 117 个此类算子表达式。4×4 FP32 中间矩阵适合在一个 kernel 内迭代，减少中间读写与调度。保持 softmax、eps 的加入位置、归一化方向和 20 次迭代语义。draft 对应 `draft_forward.py:36`。 |
| 2 | `target__forward_decode.py:90`，HC pre 混合与 RMSNorm | 融合混合、归约、归一化及权重缩放，减少 Cast/Mul/Reduce。必须保留 HC pre 输出先舍入 BF16、RMSNorm 再转 FP32 的数值边界。 |
| 2 | `target__forward_decode.py:105`，HC post 残差混合 | 避免物化 `[T,4,4,5120]` 广播中间量。保持 comb 的输入/输出轴方向和最终 BF16 舍入。跨层融合须额外保留 Engram 注入和 DSpark 辅助状态采样位置。 |
| 3 | MoE 调用边界：`fused_experts.py:_apply_mlp` | 合并 clipped SwiGLU、路由权重相乘、BF16 转换与动态 MXFP8 量化。gate/up 截断不对称，路由权重必须在 W2 激活量化之前应用；保持 group-32 scale、舍入和空路由/padding 语义。此处内部尚未展平，应先单独展开和验证。 |
| 4 | Attention 调用边界：`native_attention.py` | SFA 输出后的 `log(total) + maximum → sigmoid(lse-sink) → scale/cast/mask` 可研究合入 epilogue。保持 sink 为零 V 的 softmax 分母贡献、负 slot mask 和 FP32 中间精度；需展开该边界后单独验证。 |

另一个值得调查的非融合问题是 `Slice`：两个热点输入包含整个 `103296×1024` 缓存平面，
FP32/BF16 两组累计约 49.62 ms。`attention.py:field` 的字段切片及其后续索引是否导致大平面物化，
需要进一步关联实际调用；候选是把字段偏移与所需 slot 的 gather 合并，不能仅根据输入 shape
断言已经发生了多少字节的复制。必须验证 prefix cache、页共享、拒绝尾部及所有 cache bank。

仓库已有 V4 的 `npu_hc_pre_v2`/`npu_hc_post` 源码，但不是 V4.1 当前路径中的直接替换件。
V4.1 子层消费前一个子层返回的 pre gate，同时生成供下一个子层使用的新 pre gate；
V4 的 fused pre 接口返回按当前 gate 混合后的 y，且不返回这个新 pre 向量。
因此首先融合独立的 V4.1 门控/Sinkhorn，或设计显式接收前序 pre、返回新 pre 的接口，
不能直接替换并跳过时序等价验证。

## 下一次实验的验证顺序

1. 保留当前 `editable/`；为单一候选建立新目录，记录代码/算子版本和 manifest。
2. 以原展平片段作参考，检查 FP32/BF16 数值、空输入、active mask、padding 和代表性 token 数。
   初始目标是逐位等价；需要容差时单独说明误差、舍入变化及其对模型输出的验证。
3. 通过真实 NPU eager 与 ACLGraph 检查；MoE/缓存改动还要覆盖通信和缓存语义。
4. 运行真实 API 正确性检查，再执行固定基准并保存全部有效结果。保持原固定基线，同时与当前展平版本对照。
5. 另采 profiler 验证 kernel 数量/中间读写变化；以无 profiler 的重复测量判断实际收益。
