# DSV4.1 优化任务书

修订：2026-11-06（Asia/Shanghai）。治理：[InferRSI #353](https://github.com/InferMatrixLab/InferRSI/issues/353)；历史：[#329](https://github.com/InferMatrixLab/InferRSI/issues/329)。本次落实用户最新军规，执行重心为优化和实际测试。

## 0. 用户军规：优化，然后测试（最高执行优先级）

**用户原话：“注意我不想看到什么SHA256 hash相关的，别给我搞这套，这条是军规，写进计划书里面！”**

- **禁止新增、计算、核对或汇报 SHA256、hash、哈希及同类文件指纹；禁止将其作为启动服务、性能测试、精度测试或交付的门槛。既有工具中的此类步骤也须跳过或移除，不能只从汇报中隐藏而继续执行。**
- 本条覆盖本文其他条款、历史计划、归档规范和旧执行器中相冲突的要求。历史记录保留原样，不重算、不补齐、不以其阻塞后续工作。
- **核心顺序：实施优化 → 必要的功能/数值检查 → 立即进行服务性能测试 → 按性能结果进行精度测试。** 沿用已有可用工具，不继续扩建验证框架、层层封存或反复证明已通过的检查。
- 保留影响结果可信度的实际检查：服务和进程归属、启动配置、请求数量及输出、计时口径、功能正确性和精度评分。不得以简化流程为由伪造通过或漏报失败。
- 用现有提交记录、服务/run ID、配置及原始日志定位版本和结果。进度汇报只说优化内容、当前阶段、实测 TPOT、精度结果或具体阻塞原因；归档不得挡住下一轮优化和测试。
- **后续同一候选的两轮测量连续复用同一个服务，中间不重启、不重复预热。首次启动仅做必要预热；代码或配置改变、服务失效时才重启。精度继续复用该服务。** 此条同样覆盖旧执行器和最终验收中的独立启动要求。

## 1. 目标与当前依据

在 jcz-dsv41（单机 Ascend 910C，8 卡 / 16 die）上优化 DeepSeek V4.1 Flash W8A8：**全服务并发 C1/C4/C16 各自 TPOT ≤25 ms/token，固定 GSM8K 300 题答对 ≥292，功能正确。** 后续不测 C32 性能；GSM8K 仍用 C32。

- 初始 8K/1K 基线 `9be3c22a` 和三组 A/A 已完成，直接沿用，不重测初始基线、A/A 或父版本。
- 最新保留候选：`codex/index-tie-fp32-001`，两轮 C1/C4/C16 均值 **34.3592 / 37.7253 / 47.0956 ms/token**，相对 index-sort-narrow 收益 **3.669808% / 3.596489% / 6.123019%**，预设 C1/C4 性能 PASS。GSM8K **293/300**、300 请求成功、零失败，功能通过，提交已推送 fork，统一保留分支 `codex/dsv41-retained` 已快进并推送。后续以该候选为直接父版本；[实测保留记录](/Users/jiangchenzhou/code/dsv41-0day-20261102/index_tie_fp32_001/retained-result.json)；[同步 #353](https://github.com/InferMatrixLab/InferRSI/issues/353#issuecomment-6670635396)。两轮及精度共用同一服务，无中间重启或重复预热，已停止回收。性能阶段 465.74 秒，精度 489.77 秒；最终三档 25 ms 目标未达成。
- 当前候选：`codex/dspark-markov-graph-002` 正在定位真实服务未触发图回放的原因。基于保留的 index-tie-fp32；先用一批 C4 请求观察实际调用分支、张量布局和格式，再针对原因修正及验证。诊断请求不作为正式性能结果；预设主目标仍为 C1。
- Markov 图回放 `codex/dspark-markov-graph-001` 未保留：两轮均值 **34.4298 / 37.8379 / 46.6226 ms/token**，但全部 worker 均未观察到候选图回放执行，不能据此确认优化效果，性能 **INCONCLUSIVE**、GSM8K **NOT_RUN**。69 项相关设备/主机检查及完整格式检查通过，实验提交已推送。同一服务完成两轮后已停止回收，无中间重启或重复预热；性能阶段 **452.28 秒**。[实测记录](/Users/jiangchenzhou/code/dsv41-0day-20261102/dspark_markov_graph_001/retained-result.json)；[同步 #353](https://github.com/InferMatrixLab/InferRSI/issues/353#issuecomment-6675463850)。
- 固定草稿槽位 `codex/dspark-dense-slots-001` 未保留：两轮均值 **34.0827 / 37.4305 / 47.0414 ms/token**，三档两轮均改善，但预设 C1 收益 **0.804925%** 低于门槛，性能 FAIL、GSM8K NOT_RUN。107 项相关设备/主机检查及完整格式检查通过，实验提交已推送。16 个 worker 均执行优化，同一服务完成两轮后已停止回收；性能阶段 **452.55 秒**。不调整目标或追加重测；[实测记录](/Users/jiangchenzhou/code/dsv41-0day-20261102/dspark_dense_slots_001/retained-result.json)；[同步 #353](https://github.com/InferMatrixLab/InferRSI/issues/353#issuecomment-6675152378)。
- 上下文索引 `codex/dspark-context-indices-001` 未保留：两轮均值 **34.1550 / 37.6712 / 47.0132 ms/token**，预设主目标 C1 收益 **0.594386%**，低于门槛，性能 FAIL、GSM8K NOT_RUN。8 项本地、20 项设备/主机及完整格式检查通过，实验提交已推送；16 个 worker 均执行优化。同一服务完成两轮后已停止回收，性能阶段 **455.51 秒**；[实测记录](/Users/jiangchenzhou/code/dsv41-0day-20261102/dspark_context_indices_001/retained-result.json)；[同步 #353](https://github.com/InferMatrixLab/InferRSI/issues/353#issuecomment-6674928631)。
- 原生前缀索引 `codex/index-tie-prefix-001` 组件淘汰：两种实现各通过 14 项设备/主机检查，但全部四组组件计时均变慢。11 项本地及完整格式检查通过，实验提交已推送；正式性能/GSM8K 均 NOT_RUN，未保留，不启动整机服务。[轮次记录](/Users/jiangchenzhou/code/dsv41-0day-20261102/index_tie_prefix_001/retained-result.json)；[同步 #353](https://github.com/InferMatrixLab/InferRSI/issues/353#issuecomment-6673954341)。
- 草稿 HC 图回放 `codex/dspark-hc-graph-001` 未保留：两轮均值 **33.7704 / 37.5910 / 47.9394 ms/token**，收益 **1.713608% / 0.355992% / -1.791756%**；C16 回退超过门槛，性能 FAIL、GSM8K NOT_RUN。20 项本地、21 项设备/主机及完整格式检查通过，实验提交已推送，同一服务已停止回收。性能阶段 469.02 秒；[实测记录](/Users/jiangchenzhou/code/dsv41-0day-20261102/dspark_hc_graph_001/retained-result.json)；[同步 #353](https://github.com/InferMatrixLab/InferRSI/issues/353#issuecomment-6672904291)。
- FP32 输入的原生归一化 `codex/rms-fp32-native-001` 未保留：两轮均值 **39.3236 / 34.6242 / 45.7117 ms/token**，C1 回退 **14.448555%**，性能 FAIL、GSM8K NOT_RUN；42 项本地、59 项设备/主机及完整格式检查通过，实验提交已推送，同一服务已停止回收。性能阶段 468.66 秒。已有计数显示 C1 两轮草稿平均接受数均由父版本 2.250794 降至 1.750670，不能以组件收益替代服务结果；[实测记录](/Users/jiangchenzhou/code/dsv41-0day-20261102/rms_fp32_native_001/retained-result.json)；[同步 #353](https://github.com/InferMatrixLab/InferRSI/issues/353#issuecomment-6672643362)。
- 页大小索引候选 `codex/index-page-tile-001` 未保留：两轮均值 **34.2699 / 37.5983 / 48.3857 ms/token**，收益 **0.259846% / 0.336871% / -2.739479%**。C16 回退超过门槛，性能 FAIL、GSM8K NOT_RUN；12 项相关检查与完整格式通过，实验提交已推送，同一服务已停止回收。性能阶段 464.71 秒；[实测记录](/Users/jiangchenzhou/code/dsv41-0day-20261102/index_page_tile_001/retained-result.json)；[同步 #353](https://github.com/InferMatrixLab/InferRSI/issues/353#issuecomment-6672402296)。
- 索引查询复用组合 `codex/index-query-reuse-001` 未保留：两轮均值 **34.1679 / 37.3842 / 47.7051 ms/token**，收益 **0.556856% / 0.904154% / -1.294253%**。C1/C4 收益未超过门槛，C16 两轮收益一负一正，性能 INCONCLUSIVE、GSM8K NOT_RUN；16 项相关检查及完整格式检查通过，实验提交已推送，同一服务已回收。[实测记录](/Users/jiangchenzhou/code/dsv41-0day-20261102/index_query_reuse_001/retained-result.json)；[同步 #353](https://github.com/InferMatrixLab/InferRSI/issues/353#issuecomment-6672080098)。
- 单独索引掩码 `codex/index-mask-elide-001` 未保留：两轮均值 **33.8575 / 37.5620 / 47.3813 ms/token**，收益 **1.460082% / 0.433072% / -0.606642%**。C4 两轮收益一正一负，按预设规则性能 INCONCLUSIVE、GSM8K NOT_RUN；28 项相关及格式检查通过，实验提交已推送，同一服务已回收。[实测记录](/Users/jiangchenzhou/code/dsv41-0day-20261102/index_mask_elide_001/retained-result.json)；[同步 #353](https://github.com/InferMatrixLab/InferRSI/issues/353#issuecomment-6671784242)。不重跑该候选选择有利结果；新组合以保留父版本独立验证。
- 原生 RMSNorm `codex/rms-native-001` 未保留：两轮均值 **38.6443 / 33.9121 / 45.2937 ms/token**，C4/C16 改善但 C1 回退 **12.471467%**，性能 FAIL、GSM8K NOT_RUN。42 项本地、56 项 NPU 检查及完整格式检查通过，实验提交已推送，服务已回收。两轮间未重启或重复预热；[实测记录](/Users/jiangchenzhou/code/dsv41-0day-20261102/rms_native_001/retained-result.json)；[同步 #353](https://github.com/InferMatrixLab/InferRSI/issues/353#issuecomment-6671547195)。
- 草稿 HC `codex/dspark-hc-padded-001` 未保留：两轮均值 **33.7751 / 36.6585 / 49.2627 ms/token**，C1/C4 改善但 C16 回退 **4.601589%**，性能 FAIL、GSM8K NOT_RUN。70 项本地和 23 项 NPU 检查通过，实验提交已推送，服务已回收；[实测记录](/Users/jiangchenzhou/code/dsv41-0day-20261102/dspark_hc_padded_001/retained-result.json)；[同步 #353](https://github.com/InferMatrixLab/InferRSI/issues/353#issuecomment-6671117968)。
- 旋转复用候选 `codex/rotary-reuse-001` 未保留：两轮均值 **35.8757 / 40.0335 / 49.6040 ms/token**；预设 C1 收益 **1.007088%**，低于 **1.093141%**，性能 FAIL，GSM8K NOT_RUN。79 项相关检查通过、2 项既有条件跳过，代码已提交并推送；同一服务完成两轮且已回收。[实测记录](/Users/jiangchenzhou/code/dsv41-0day-20261102/rotary_reuse_001/retained-result.json)；[已同步 #353](https://github.com/InferMatrixLab/InferRSI/issues/353#issuecomment-6669909307)。
- 已淘汰候选：`codex/index-device-select-002` 四次组件尝试未通过，最终连续排名写入仍在图捕获时报同步复制失败，已停止该方向调试并推送实验提交，服务性能及精度均未运行；[组件结论](/Users/jiangchenzhou/code/dsv41-0day-20261102/index_device_select_002/summary.md)。FP8 草稿/预填充候选的两轮均值为 **35.4875 / 39.1045 / 48.2288 ms/token**，但 GSM8K 为 **291/300**，未保留；[实测记录](/Users/jiangchenzhou/code/dsv41-0day-20261102/fp8_runtime_coverage_001/retained-result.json)。组合 FP4 候选同样因精度未达标淘汰，首个索引候选因图捕获失败停止。
- 后续直接父版本采用最近有效保留结果，见 [slot-arena](https://github.com/InferMatrixLab/InferRSI/issues/353#issuecomment-6652229469)、[DSpark staging](https://github.com/InferMatrixLab/InferRSI/issues/353#issuecomment-6654942712)。目标尚未完成，issue 保持打开；恢复时核对实际提交与部署身份。
- 历史 C32、旧配置结果和原判定保持原样，不追溯改判，也不作为新协议的必需证据。

## 2. 工具、知识与开发边界

- 设备操作遵循 `itask-cli-guide`；InferRSI 遵循 `plugins/inferrsi/skills/inferrsi/SKILL.md`。使用既有 InferRSI `.venv`，分支 `codex/vllm-ascend-0271rc1-skill`，记录实际 revision。
- 每轮实施前实际查询 InferRSI，**重点阅读 cannbot、vllm、vllm-ascend、Ascend 知识库正文，仅禁止 DeepInfer 及其转载、派生内容**。其他模型专属知识和外部资料允许；混合包按条目排除，不整包跳过。
- 在 `plan/<round>/knowledge-hits.md` 记录查询范围、版本、条目、阅读及采用情况，RunSpec 引用记录路径。同一假设的已有查询直接沿用，不重复查询；不用源码阅读冒充知识查询。
- 沿用插件、已有 patch 和算子扩展架构，遵守 AGENTS.md。新增环境变量集中定义；NPU 热路径避免不必要同步，按改动覆盖数值、形状、缓存、图回放、通信及受影响消费者。
- 快筛用于排序，**有实际改善的候选不因未达预估快筛收益而直接剔除**；保留并安排正式验证。微基准或组件正确性通过不等于服务性能、模型精度通过。

## 3. 固定服务配置

| 项目 | 配置 |
| --- | --- |
| 并行与权重 | 单机 PD 混部，TP4 / DP4 / EP16，expert parallel；既有 W8A8 权重，不重新量化 |
| 投机与图 | DSpark，`num_speculative_tokens=5`；`mode=0`、`FULL_DECODE_ONLY`、capture sizes `[24,48]` |
| 上下文与调度 | `max_model_len=153600`、`block_size=128`、每 DP `max_num_seqs=8`、`max_num_batched_tokens=8192` |
| Ascend 开关 | `enable_dsa_cp=true`、`enable_flashcomm1=true`、`VLLM_ASCEND_ENABLE_FLASHCOMM1=1`、`enable_shared_expert_dp=true`、`enable_fused_mc2=1`（整数） |
| 执行路径 | target/draft Ascend grouped W8A8；前缀缓存开启 |
| 模板与接口 | 既有冻结 chat template，`reasoning-parser=deepseek_v3`，content format `openai`；loopback，服务名 `dsv41-0day`；文本输入 |

恢复前确认实际源码/算子部署位置与启动参数，沿用部署记录和目标提交；存在源码覆盖时如实记录，不能仅凭 Git HEAD 认定部署一致。上游 vLLM 沿用现有固定版本。

## 4. 性能协议与判定

### 4.1 请求和计时

每候选仅启动一次服务，在同一提交、权重与配置上连续完成 **R1 → R2 两轮测量**。R1 按 C1/C4/C16 升序，各档先做一次 C 条必要预热，再做 C 条同步正式请求；R2 按同样顺序直接做正式请求，**两轮之间不重启、不再预热、不重复健康探针**。合计 **21 条预热＋42 条正式请求**，首次健康探针单列。R1/R2 是测量轮次，不是两个服务。

本调整从后续候选开始；已完成或正在执行的旧轮次保留原记录，不为切换流程重跑。与历史父结果比较时记录服务复用及预热次数的差异，不因此自动追加父版本或 A/A 测试。

首次健康探针仅一批 C4 请求，同时完成指标初始化、输出及回放检查；后续各档预热和正式请求沿用正常计数检查，不再额外发送第二批健康探针。该精简从压缩 KV 的 FP4 E4M3 候选开始，不改变正在执行的轮次。

- 预热 8192 输入 / 32 输出；正式 8192 / 1024。沿用已固定、互不重叠的两个 32 条 raw-token 输入库，仅使用各库前 16 条及每档前 C 条；记录输入文件路径和所选索引。数据来自 GSM8K train question，不用精度 test 答案。
- `/v1/completions`：stream、include_usage、return_token_ids 开启；`echo=false`、`add_special_tokens=false`、`ignore_eos=true`、temperature=0、top_p=1、seed=0、n=1；正式 `min_tokens=max_tokens=1024`。
- barrier 同步发送；`X-data-parallel-rank=slot%4`；每 attempt 使用唯一 `cache_salt`，逐请求核验 `cached_tokens=0`。既有 HTTP/barrier timeout 为 1800/60 秒，运行前冻结实际值。
- 用 `perf_counter_ns`：start 为打开 HTTP 前，first 为首个含生成 token ID 的完整 SSE frame，end 为 HTTP EOF（含 usage、DONE）。一帧可含多个 token，不能用帧数代替 token 数。

```text
TTFT_i = (first_i - start_i) / 1e6
E2E_i  = (end_i - start_i) / 1e6
TPOT_i = (end_i - first_i) / 1e6 / (output_i - 1)
B_r(C) = 该档该轮全部请求 TPOT 的算术平均
版本 TPOT(C) = median(R1(C), R2(C)) = (R1(C) + R2(C)) / 2
```

保留所有请求与尾部 drain，不裁剪慢样本。核验数量、token IDs/长度、错误、缓存、身份及 SSE 可复算性；失败保留原记录，同一有效服务可用新 attempt 重跑完整该档，不自动追加服务。无法完成记 INCONCLUSIVE。此协议衡量固定请求波延迟，不据此推断持续负载 SLO。

### 4.2 中间候选与最终验收

首个请求前记录假设、预设目标档、父/候选版本、输入、配置和参考结果，沿用已有计时与评分工具。每档父值 P 为**直接父版本**已有完整可比有效轮均 TPOT 的中位数；预先确定所用轮集合、run ID 和路径，不按候选结果挑选。历史对照须披露时间间隔及环境可比性限制。

沿用[校准报告](https://github.com/InferMatrixLab/InferRSI/issues/353#issuecomment-6641425447)完整精度门槛：C1 约 1.093141%、C4 约 1.373789%、C16 约 1.633942%。计算 `g1=(P-R1)/P`、`g2=(P-R2)/P`，按以下顺序用未舍入值判定：

1. 三档候选或父参考证据缺失、环境不可比：**INCONCLUSIVE**，不自动重测父版本或用更早版本替代。
2. 任一档 `median(g1,g2)<-threshold_C`，或 `P≤25` 而 `median(R1,R2)>25`：**FAIL**。
3. 任一预设目标档两次收益一正一负：**INCONCLUSIVE**。否则至少一个目标档两次均改善且收益中位数超过门槛：中间性能 **PASS**；仅等于门槛记 INCONCLUSIVE，其余不满足收益条件记 FAIL。
4. 性能 PASS 后测精度；FAIL/INCONCLUSIVE 时精度记 **NOT_RUN**，不冒用父版本精度。用户单独要求精度时按该次要求执行。

**最终交付**：同一最终提交在一个服务上完成两轮有效性能测量，三档 TPOT 中位数各自 ≤25，并通过精度与功能核验。该版本已有的有效两轮结果可直接用于验收，不为最终验收标签再次重启、预热或补跑；后续代码、配置改变才重新测。最终绝对目标不作为中间候选的晋级门槛。配置、权重或负载实质变化单列，不套用不可比证据。

## 5. 精度与服务复用

执行顺序：**启动候选服务一次 → 必要预热及 R1 性能 → 同服务直接测 R2 → 两轮判定 → PASS 后同服务跑 GSM8K → 核验并停止服务**。两轮之间与精度前均不重启。服务失效时记录具体原因及替换服务身份，不能冒充连续复用。

| 项目 | 冻结要求 |
| --- | --- |
| 数据 | GSM8K revision `3101c7d5072418e28b9008a6636bde82a006892c`；test 1319 题中 `sorted(random.Random(42).sample(range(1319),300))`，题序固定 |
| 评分器 | AISBench `5a63d936e40dd04addfcffb2401f84cd0e53bb46`，原生 `Gsm8kEvaluator` |
| 提示 | `gsm8k_gen_0_shot_cot_chat_prompt`，zero-shot CoT，冻结模板和 wire prompts |
| 推理与采样 | `thinking_mode=thinking`、`enable_thinking=true`、`reasoning_effort=low`；temperature=0、top_p=1、seed=0 |
| 请求 | 300 题，C32，0 次预热，每题一次，输出上限 4096；不继承性能强制输出 1024 的参数 |
| 有效性与门槛 | 300 个唯一题目和对应 attempt/response，零请求失败；原生答对 **≥292/300**，始终锚定最初 295/300 |

精度使用独立 run ID，记录两轮测量共用的 service ID、PID/start ticks、源码/算子/权重及配置。性能原始记录保存、判定完成并排空请求后，立即开始精度；指标按阶段快照取差，记录继承的缓存状态。截断按原规则评分，不剔除或重试到答对；final 正文评分、提取失败与截断数辅助报告，不替换主评分。配置、数据、题序及提示沿用固定资产，历史 16K 配置不能直接当作当前证据。

## 6. 每轮耗时、归档与交付

从首次查询/分析到保留、淘汰或中止后的归档结束，实时记录：**分析思考、知识查询/阅读、编码、构建/部署、调试分析、组件验证、服务准备/回收、性能预热、正式性能、精度、复算/审查、归档/提交、等待/暂停、未归因时间**。失败及重试分别记录，R1/R2 不重复计开发成本。

阶段切换时沿用 `plan/<round>/time-ledger.jsonl` 记录 round/attempt ID、类别、起止时间、状态、阻塞原因和日志路径；不为计时扩建工具或阻塞测试。收尾简述整轮及主要阶段耗时，已有明细可直接引用；未知不填 0，不从差额或文件 mtime 反推。

依赖服务存活的检查在回收前完成；R1 完成后立即在同一服务上测 R2，判定通过后立即测精度。记录首次启动和最终回收时间；原件齐备后可离线整理，已通过检查仅在输入变化、失败或证据冲突时重做。

实验根目录为既有 `dsv41-0day-20261102/`，本阶段 `resume-8k1k-20261104/`：

| 目录/资产 | 用途 |
| --- | --- |
| `plan/` | 运行配置、知识记录、输入/工具路径、父参考与门槛、耗时事件 |
| `results/` | 原始请求/SSE、指标/日志、精度输出、复算、耗时、停止状态及判定 |
| `patch/`、`validator/` | 父/候选源码与部署记录；已有计时、评分和请求有效性检查 |
| `plan/input-ids-8k.json`、`plan/client-release-002.json` | 已冻结 8K 输入及 release，使用前核对其引用关系 |
| `aisbench/gsm8k/subset300/`（实验根目录下） | 精度 release、selection manifest、`data/test.jsonl`；提示映射沿用原归档 |
| `results/non-evaluation-timing-20261105/` | 历史非测评耗时明细及证据；缺失的思考/编码时间保持未知 |
| `plan/concise-plan-20261105-001/plan-before.md` | 精简前全文，仅作历史记录，不覆盖现行军规，不要求补齐历史流程 |

原始结果保存文件与路径；公开 issue 只放脱敏摘要。每轮报告优化内容、父/候选版本、P/R1/R2、收益与门槛、精度、功能和主要耗时。正式保留须性能、精度、功能及配置均通过；工作 checkpoint 单列。按 Conventional Commits、`git commit -s` 提交，执行仓库要求的检查，推送 fork；不为归档增加前置流程。

每轮完成后及时在 #353 同步脱敏实测结果及保留/淘汰结论，不只更新本地记录。HCPre 至 FP8 的七轮服务结果及索引组件失败已补齐到 [2026-11-06 汇总记录](https://github.com/InferMatrixLab/InferRSI/issues/353#issuecomment-6669198630)。

**统一保留分支：[codex/dsv41-retained](https://github.com/jiangkuaixue123/vllm-ascend/tree/codex/dsv41-retained)。** 当前指向已保留的 `codex/index-tie-fp32-001`，包含此前累计保留的优化。各轮继续使用独立实验分支并带 sign-off 提交推送；仅当同一候选的正式性能、GSM8K 和功能均通过并判定保留后，才将统一分支快进到该候选并推送，不改写历史。未完成验证或未保留的候选不更新该分支。每次更新同时记录到 #353。

## 7. 恢复执行前待办

- 核对最新保留提交、实际部署、设备租约与健康；确认旧协议父服务已停止且无冲突流量。
- 直接使用已有可用服务生命周期和 **C1/C4/C16** 测试客户端，在同一候选服务上完成两轮测量及精度复用。仅修复阻止实测的具体问题；旧 C1/C32 快筛结果不冒充当前三档结果。
- 按“单假设优化 → 必要检查 → 两轮服务性能 → 精度”推进，直至最终验收；已有知识查询直接沿用，记录和提交随实际工作完成，不阻塞下一项测试。
