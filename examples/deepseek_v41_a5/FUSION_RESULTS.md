# V4.1 A5 Triton 融合实测

统一机会分析及实现前排序见 [FUSION_QUEUE.md](FUSION_QUEUE.md)。本轮在可编辑展平源码上
实施 13 类候选中的可验证向量链，逐项做真实 NPU 数值、ACLGraph、微基准与服务验证。
GEMM、原生 SFA 主体、MXFP8 动态量化、通信和 host Engram 读取仍保留原接口。
这些边界及尚未合并的子链已列入队列，不能把本轮描述为全模型已经完全融合。

部署根目录为 `/home/jojo/vllm-ascend-d4ac431`；以下实验与日志路径均相对此目录。
实现提交为 `10bfed22d3399958f094bccad8f07a3330c5f4af`，是用户指定
`d4ac4319acdaf480689c39b6320a998712f80523` 的后代。上游 vLLM 仍为
`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`，权重保持原 FP8/FP4。

## 选定部署

默认保留 Attention 组合：HC gate/Sinkhorn、FP8/FP4 QDQ、选择性缓存读写、HC pre/post、
MoE 激活/route factor、KV norm/RoPE/QDQ、Indexer epilogue 和 Compressor pool/norm。
源码 bundle 为 `experiments/fusion-20261109/pool-v1/`，镜像为
`jojo/vllm-ascend:d4ac431-fusion-runtime-10bfed22`。

最初三轮均值 **97.3865 token/s**；恢复该版本后的确认三轮均值 **98.3025 token/s**。
全部六轮平均 **97.8445 token/s**，范围 **91.6257–102.7374 token/s**。
相对展平起点 **+78.14%**，相对原固定基线 **+88.26%**。
选择依据是最初三轮组合均值，随后保留全部确认轮次；没有仅汇报更快的确认结果。

新增 Engram/SFA sink/Router/confidence 的完整组合均值为 94.1541，低于 Attention 组合；
这些 kernel 的代码、测试和候选 bundle 保留，默认不启用。完整组合移除 HC pre/post 与
MoE 激活后均值为 92.7728，比完整组合低约 1.47%；差异相对轮次波动较小，不能据此量化
这两类融合各自的独立收益。后续若优化这些边界，应继续做同场景移除对照。

## 固定测试及全部轮次

场景 `v41-c8-i100-o256-v1` 未更改：4 张 A5（物理 4–7）、TP4/EP4、8 个固定输入，
每请求输出 256 token；每轮一次既定 64-token 预热，正式计时无 profiler、无其他 NPU 测试。
有效结果必须满足 8/8 成功、2048 输出 token、0 缓存命中，输入 token IDs 校验通过，
服务端与客户端计数一致。总吞吐为 2048 除以整批端到端用时，包含首 token 等待。
不剔除第一轮，也不选择最快一轮作为均值。

<!-- BEGIN RESULTS TABLE -->
| 阶段 | 三轮吞吐（token/s） | 均值 | 相对原固定基线 |
| --- | --- | ---: | ---: |
| 可编辑展平参照 | 53.620 / 56.013 / 55.147 | 54.927 | +5.68% |
| HC 门控/Sinkhorn | 63.211 / 73.303 / 73.586 | 70.034 | +34.75% |
| + FP8/FP4 QDQ | 71.681 / 87.306 / 86.084 | 81.690 | +57.18% |
| + 选择性缓存读写 | 70.078 / 90.463 / 91.770 | 84.104 | +61.82% |
| + HC pre/post（最初三轮） | 63.104 / 89.487 / 93.810 | 82.134 | +58.03% |
| + MoE 激活（静态行数） | 59.481 / 97.403 / 78.951 | 78.612 | +51.26% |
| 同一组合，运行时行数 | 95.487 / 92.873 / 93.817 | 94.059 | +80.98% |
| + KV/Indexer/Compressor | 91.626 / 99.524 / 101.010 | 97.386 | +87.38% |
| + Engram/SFA/Router/confidence | 85.568 / 96.187 / 100.708 | 94.154 | +81.16% |
| 完整组合移除 HC pre/post 与 MoE 激活 | 89.505 / 99.100 / 89.714 | 92.773 | +78.50% |
| Attention 组合重启确认 | 94.197 / 97.973 / 102.737 | 98.302 | +89.14% |
<!-- END RESULTS TABLE -->

原固定基线为 **51.9727 token/s**；历史 **46.1295 token/s** 使用随机输入，仅作历史参考。
展平前另一次三轮均值为 58.4150 token/s，展平优化起点为 54.9266 token/s。
基线文件和原展平目录均保留不变。各轮完整请求、耗时、TTFT、draft 接受率和服务配置
保存在 `logs/fusion-<stage>-throughput-<n>.json`。

HC pre/post 的补测三轮为 93.7344 / 92.0556 / 89.6486，均值 **91.8129 token/s**；
它们与最初三轮分开记录，六轮总均值 86.9733，不用补测替代最初较慢的结果。
早期 MoE 组合三轮波动也完整保留。

运行时行数版本只改变 Triton 的特化方式：token 行数改为运行时参数，关闭该参数的数值特化，
数学实现不变。其三轮为 95.4866 / 92.8728 / 93.8174，早期明显的首轮下降在这组中未出现。
源码和现象支持额外 shape 编译是一个可能原因，但没有完整编译时间归因，不能认定为唯一原因。

服务参数一直保持 DSpark5、FULL_DECODE_ONLY ACLGraph（24/48）、max length 65536、
max batched tokens 512、max seqs 16、显存比例 0.93。自动探测的 KV 容量有差异：
HC 为 103296 token，QDQ 为 98048，后续多数版本为 103424；本场景的输入与输出总量远低于容量。
这些实际差异仍记录在 `experiments/fusion-20261109/stage-runtime-metadata.json`，没有隐藏。
各阶段顺序运行，未随机化实验顺序；小幅增量和三轮波动不足以证明统计显著性。

## 算子收益与启用原则

最强的早期服务证据来自 HC 门控/Sinkhorn 和 QDQ：累计吞吐从展平的 54.93 提升至
70.03，再至 81.69 token/s。后续向量链按累计组合测量；不能把组合提升分配给某一个 kernel，
也不能将微基准加速比直接当作服务加速比。

| 算子链及代表形状 | 原实现 → Triton（微秒） | 说明 |
| --- | ---: | --- |
| HC gate/Sinkhorn，12 行 | 204.57 → 13.62 | 原 20 次迭代完整保留 |
| FP8 E8M0 QDQ，256×4096 | 128.92 → 37.61 | 测试中逐值一致 |
| FP4 E8M0 QDQ，256×4096 | 122.37 → 36.61 | 保留分组、舍入和符号语义 |
| Cache FP32 双 slot 读取，48×2×512 | 约 203 → 17 | 避免先物化整个字段平面 |
| Cache raw 写入，48×512 | 156.60 → 16.05 | 无效 slot、重复 slot 均有测试 |
| HC pre + norm，12×4×5120 | 36.96 → 15.03 | 保留混合后的 BF16 舍入 |
| HC post，12×4×5120 | 21.60 → 14.52 | 避免广播中间大张量 |
| MoE 激活，768 行 | 44.20 → 13.97 | 激活/路由因子融合，量化保持原生 |
| KV norm/RoPE/QDQ，48 行 | 41.11 → 15.87 | 参考实现已经使用融合 QDQ |
| Indexer epilogue，8192 keys | 28.12 → 15.64 | 保留 GEMM 和 top-k |
| Compressor pool/norm，12×2×512 | 20.70 → 1.73 | 以下五行采用批量图微基准 |
| Engram gate/residual，12×4×5120 | 65.93 → 9.44 | 两个 Triton kernel |
| SFA sink/mask，12×1×64 | 22.33 → 6.08 | 原生 SFA 主体不变 |
| Router top-k 后权重，12 行 | 7.04 → 2.30 | 原生评分、选择及 tie-break 不变 |
| Draft confidence，8×5376 | 11.03 → 4.15 | concat/投影融合，sigmoid 保留原生 |

初期微基准每图调用一次，约 14 微秒的图重放开销会掩盖小算子差异；最后五行采用
每图 20 次调用、80 次重放，并除以总调用次数。两种方法不能直接混为相同计时精度。
原始数据及脚本位于 `experiments/fusion-20261109/*micro*.json` 和相邻 Python 文件。

缓存读取存在明确反例：12×8192×128 的 index 读取约 127.5 → 467.4 微秒，
12×640 的 raw 读取约 143 → 156 微秒。因此实际 bundle 只启用小行数读取、压缩源 FP32
字段读取及融合写入，大 raw/index 读取保留原生。失败的 tile 尝试和回退依据完整保留。

Engram 和 confidence 的部分单 kernel 版本触发 A5 Triton 编译器中止；通过拆分实现解决。
未把编译失败的版本接入服务，也没有为追求一个“大算子”而改变模型语义。

## 正确性与覆盖边界

- 全部新增 Triton 算子 **186 项真实 NPU 测试通过**，包括 eager、ACLGraph、空行、
  实际 decode/prefill 行数、异常值、QDQ 舍入边界、缓存位模式及无效/重复 slot。
- CPU 模型回归 **303 项通过**；实例回调、根对象回调与 SHA 校验另外 **13 项通过**。
  测试选择有重叠，不能简单相加作为独立测试总数。
- 各服务候选均运行 API 文本/流式 10 项、GSM8K 前 8 道题、一个图像算术用例。
  GSM8K 仅为小样本集成回归，不是完整基准准确率评测。
- 最终部署单独通过前缀缓存复用检查：相同长输入两次均回答正确，缓存命中从 0 增至
  1024 token。该正确性检查在全部吞吐测试之后执行，不计入性能数据。
- QDQ 与 KV norm/RoPE/QDQ 的测试逐值一致；缓存测试比较完整 bank，包含未写区域。
  HC、Compressor、Engram、SFA、MoE 的测试保留 BF16 边界并允许明确的归约容差；
  Router/confidence 使用 FP32 容差。因此没有宣称全模型逐位等价。
- 原服务的重复请求本来就可能因批处理/归约而产生不同长文本，不能单靠全文字节相等判定正确性。
  本轮没有扩大到全部上下文长度、完整模型准确率数据集或长期稳定性测试。

日志：`logs/fusion-runtime-rows-npu-tests.log`、`logs/fusion-cpu-regression.log`、
`logs/fusion-root-callback-tests.log`、`logs/fusion-final-prefix-cache.json`
和各阶段 `api/gsm8k/vision` JSON。

## 可复现入口与下一轮

实现位于 `vllm_ascend/ops/triton/`：

| 文件 | 内容 |
| --- | --- |
| `deepseek_v41.py` | HC gate/Sinkhorn，FP8/FP4 QDQ |
| `deepseek_v41_cache.py` | 分页缓存直接寻址、读取、masked scatter |
| `deepseek_v41_hc.py` | HC pre + RMSNorm、post 残差混合 |
| `deepseek_v41_moe.py` | routed/shared clipped SwiGLU 和 route factor |
| `deepseek_v41_kv.py` | KV norm + RoPE + FP8 QDQ |
| `deepseek_v41_indexer.py` | Indexer GEMM epilogue |
| `deepseek_v41_epilogues.py` | Compressor pool/norm、SFA sink/mask |
| `deepseek_v41_engram.py` | Engram weighted dot 与 gate/residual |
| `deepseek_v41_router.py` | Router 权重、draft confidence |

测试位于 `tests/ut/models/deepseek_v41/test_triton*.py`。
实例回调加载与完整性检查位于 `vllm_ascend/models/deepseek_v41/flatten.py`。

每个阶段保留独立 bundle、manifest SHA、源码提交、镜像和服务结果。回调只绑定当前模型实例，
不修改类或全局函数；服务加载时校验文件哈希与模型静态配置。原始展平参照仍在
`experiments/flatten-20261109/editable/`。

后续按以下顺序继续：

1. 复制选定 bundle 到新的候选目录，展开相邻链、修改 Triton 实现，更新 manifest。
2. 先跑对应的数值和真实 NPU/ACLGraph 测试，再做代表形状微基准；服务压测期间不运行它们。
3. 为代码提交构建独立镜像，准备对应 `serve-<stage>.sh` / `start-<stage>.sh`。
4. 使用 `run-candidate.sh <全新名称> <start脚本的路径>` 停止并保留上一候选、启动新候选；
   健康后自动完成 API、GSM8K、图像回归和三轮固定吞吐。入口拒绝重用名称、并行验证
   和操作其他部署的容器；验证失败会保留失败容器并重启上一版本。该入口通过 shell 语法
   和 ShellCheck，自动回退分支未进行真实故障注入。使用全新名称，禁止覆盖既有日志。
5. 在 `summarize-results.py` 登记新增阶段与 bundle 映射并汇总全部轮次；需要时做移除对照，
   仅保留有证据支持的默认组合。

这套顺序允许下一候选的源码分析/编译与上一候选的验证交错；同一组 NPU 的正式计时串行进行。
当前默认镜像、bundle、容器 ID 和结果由根目录 `deployment-state.json` 记录，
默认启动入口为 `start.sh`，原始 `serve.sh` 保留用于回退。
