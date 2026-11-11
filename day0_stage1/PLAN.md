# DeepSeek V4.1 A5 仓内交付计划

阶段一已用仓内 vllm-ascend 发行包完成八卡复验。阶段二的独立 GSM8K 复测已经逐题核验，投机与目标单独解码的分歧已记录。阶段三已有一轮全量 GPQA-Diamond 官方结果，正在与 A5 0.30 镜像逐题对照，并验证前缀一致性及单请求长上下文。各阶段的模型及权重加载器均随同一仓安装，使用普通服务入口。每个独立改动提交并推送到开发分支，实际执行情况写入 [PROGRESS.md](PROGRESS.md)。

| 顺序 | 交付内容 | 检查方式 |
| --- | --- | --- |
| 1 | 使用基线 DSV4 的模型注册入口登记 V4.1，通过 Ascend 权重加载器入口注册 V4.1 loader。模型、加载器、Engram 实现和缺省聊天模板随 vllm-ascend 发行。 | 安装发行包，在新进程中核对模型类、加载器、模板及 wheel 文件清单。检查 DSV4 注册和其他模型的默认加载路径。 |
| 2 | 从标准 `vllm serve <模型目录>` 启动。调用方提供普通硬件拓扑与 eager 配置，使用原始权重和 checkpoint tokenizer。 | 在指定镜像的新容器里运行八卡 DP8 / EP8 / TP1 / PP1，保存实际标准 CLI 和各 rank 的设备、专家、权重与 Engram 放置。 |
| 3 | 在这份仓内发行包上复验阶段一。 | 核对 HCCL、独立 Engram 输出、模型列表、模板 IDs、四个 HTTP 请求、prefill 末位 logits 与连续 decode。按原先的数值检查规则记录第一处分歧。 |
| 4 | 恢复阶段二，接入原生量化、融合路由与专家执行、decode 图回放、DSpark 提案验证及拒绝回滚。 | 在同一组合服务上复跑功能验证和已冻结的 AISBench GSM8K 验收，保存逐题结果与实际并发。 |
| 5 | 执行阶段三，接入前缀缓存，验证绑核、单请求长上下文和 GPQA-Diamond。 | 记录命中与隔离、逐 rank 和线程 affinity、长文本使用量与检索输出、全集逐题评分。 |

实现以冻结的官方 HF 模型语义、指定镜像以及基线 DSV4 注册接口为依据。其他分支已有的 DSV4.1 实现不在来源范围内。精确的环境、对照版本、权重哈希、样本和结果存放在私有验收记录中。[重放说明](patch/REPLAY.md)

## 阶段二的服务与评测

复验从同一仓的 wheel 和 `vllm serve` 启动。完整候选使用 eager prefill、`FULL_DECODE_ONLY`、原始权重及 `--speculative-config '{"method":"dspark","num_speculative_tokens":5}'`。冻结的验收记录提供上下文、页数、批宽、dtype、镜像和并行配置。对同一候选还会用关闭投机的确定性请求作对照。

先检查各 rank 的目标和草稿权重、原生 experts runner、实际通信方式与图捕获。服务就绪后重放阶段一四问，并同时发起独立请求。记录有实际请求时的 decode replay、已接收与被拒绝的草稿 token、拒绝后的请求状态以及共享专家的放置和通信。共享专家按本地输出检查，路由专家按 EP8 的实际 dispatch/combine 检查。[并发功能检查](../day0_stage2/validator/check_concurrent_chat.py)

独立 AISBench 客户端先核验官方评分器和输入顺序，支持用 `--dry-run` 检查最终任务配置。正式运行用 `--num-warmups 0` 关闭 AISBench 从正式题取样的预热，记录每题唯一请求、HTTP 状态、模型输出和在途数，并与 AISBench 的逐题评分对照。题量、并发数、分数门槛以及单次尝试预算均由冻结验收文件提供。[评测启动器](../day0_stage2/validator/aisbench_gsm8k.py) · [HTTP 观测器](../day0_stage2/validator/observe_gsm8k.py)

### TP1 的共享专家与通信

2026-11-10 按任务方要求，阶段二暂缓适配 `shared_expert_dp_enabled` 和 `sequence_parallel_enabled`。两项仅记录实际值，均可为 `false`，不作为阶段二通过条件。原生共享专家的本地计算与数值、路由专家的 EP8 分发和合并仍须验证。

在已复验的 DP8 / EP8 / TP1 拓扑中，CLI 传入了 `enable_shared_expert_dp` 和 `enable_flashcomm1`。八个 rank 的主干和草稿账本均记录两个 effective 字段为 `false`。继续使用同一拓扑验证 DSpark 接受与拒绝、无投机对照、真实 decode 图重放、独立并发请求和冻结的 GSM8K 任务。[框架设置](../vllm_ascend/ascend_config.py) · [逐层账本](../vllm_ascend/models/deepseek_v41/fused_moe.py)

### 普通服务入口

在指定 A5 镜像中安装仓库构建的 wheel，设置 `MODEL_DIR` 为只读原始权重目录、`SERVED_MODEL` 为服务别名。下例取自已完成的八卡组合服务。HCCL 在该环境使用 AIV。手动给定页数用于重放同一部署配置。

```sh
export HCCL_OP_EXPANSION_MODE=AIV
vllm serve "$MODEL_DIR" --served-model-name "$SERVED_MODEL" \
  --api-server-count 1 --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 --data-parallel-size 8 --data-parallel-size-local 8 \
  --enable-expert-parallel --tensor-parallel-size 1 --pipeline-parallel-size 1 \
  --block-size 128 --num-gpu-blocks-override 145 --kv-cache-memory-bytes 16777216 \
  --max-model-len 4096 --max-num-seqs 4 --max-num-batched-tokens 4096 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[6,12,18,24]}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}' \
  --additional-config '{"enable_shared_expert_dp":true,"enable_flashcomm1":true}' \
  --default-chat-template-kwargs '{"thinking_mode":"chat","reasoning_effort":"high"}' \
  --no-enable-prefix-caching --enable-chunked-prefill --no-async-scheduling --generation-config vllm
```

示例中的两个并行开关保留已测命令行原值，在 TP1 下的生效状态见上节。阶段一 eager 配置和安装重放步骤见[阶段一说明](patch/REPLAY.md)。

## 阶段三的服务与验收

继续用指定 A5 镜像、原始 checkpoint、仓内 wheel 和普通 `vllm serve`。DP8/EP8、TP1、PP1、PCP1，使用阶段二的 native experts、eager prefill、DSpark 和真实 decode 图回放。在同一份候选上记录各项生效配置、输入及结果。

| 项目 | 实测与判定 |
| --- | --- |
| 前缀缓存 | 开关分别冷启动，固定同一 DP rank，测冷请求、同前缀不同后缀的命中、改变一个前缀块的 miss，以及请求交错和页复用。核对实际 cached tokens、输出 token、Engram 历史、压缩索引状态和 DSpark 接受或回滚。 |
| CPU 绑核 | 分别传入 `enable_cpu_binding=false/true`。记录镜像默认值、NPU 对应 NUMA 节点，以及每个 worker 主线程、ACL 和释放线程的实际 CPU affinity。 |
| 单请求 1M | 最终组合设置 `--max-model-len 1048576`。逐档发出已知首、中、尾检索提示，边界按实际 1,048,560 个输入及 16 个输出核对，同时记录每 rank 的 KV 与 HBM 峰值，并发送超限请求。容量归属实际承载该请求的 rank。 |
| GPQA-Diamond | 用 AISBench 单次遍历 198 道唯一题。指定 A5 上的 0.30 参考镜像运行自身随镜代码；本分支在原指定镜像独立运行。两边对齐原始权重、八卡拓扑、服务参数、题单、选项、模板、解码与评分器。双方均需完整计分，候选答对数至少等于参考答对数；按题目 ID 报告判分差异。 |

A5 正式对照优先使用 `quay.io/ascend/vllm-ascend:v0.30.0-a5`，记录实际拉取摘要、镜像软件栈和主机驱动。指南的单机示例启用 V2、DP8/TP1/EP8、原始 MXFP、共享 CPU Engram 和 DSpark 5；按双方冻结的服务预算与输入运行 GPQA。RC1 结果另列。[官方 0.30.0 950DT 部署指南](https://docs.vllm.ai/projects/ascend/zh-cn/v0.30.0/tutorials/models/DeepSeek-V4.1-Flash.html)

参考镜像的量化方法按原始 checkpoint 的 E4M3、UE8M0 和 FP4 experts 核对。记录镜像实际选择的量化实现，使用原权重只读挂载及原镜像提供的 CLI 参数。

两侧服务默认同时传入 `thinking_mode=chat` 和与其等义的 `enable_thinking=false`，保留冻结的 `reasoning_effort=high`。这覆盖 HF 模板和参考镜像原生 tokenizer 接受的模式字段。正式请求正文保持冻结。

GPQA 两侧启动后先逐题调用服务的 `/tokenize`，核对全部输入 ID 与同一份 HF 模板结果。两轮官方生成结束后，复算逐题评分并按原题序列出双方均对、仅参考答对、仅候选答对、双方均错和不可提取数。保留每侧实际启动配置和只读权重身份。[配对验证](../day0_stage3/validator/gpqa_pair.md)

本阶段保持 DSA context parallel 关闭，`shared_expert_dp_enabled` 和 `sequence_parallel_enabled` 继续只记录实际状态。阶段二已有的数值差异保留在对应候选的结果中。逐项原始证据归档后更新[进展](PROGRESS.md)。

[阶段三启动脚本](../day0_stage3/serve.sh) 通过普通 CLI 选择缓存与绑核配置。设置 `MODEL_DIR`、`SERVED_MODEL`、`PORT`、`MAX_MODEL_LEN`、`PREFIX_CACHING=true|false` 和 `CPU_BINDING=true|false`。服务向 API 返回 cached tokens，KV 页数使用标准 vLLM 内存规划器。

长上下文运行可用 `GPU_MEMORY_UTILIZATION` 设置 vLLM 标准单卡显存比例。每轮保存请求的比例、各 rank 实际可用内存与自动规划的 token 容量，再实际发送该档输入。
当预填充激活的实际峰值限制单卡容量时，用 `MAX_NUM_BATCHED_TOKENS` 调整单步调度预算，默认 4,096。单请求 1M 输入与最终用量的判据保持原值。
最终组合的各档长请求使用独立的 vLLM `cache_salt`，核对定长请求的实际 cached tokens 为零，再用相同 salt 做自然停词复测。归档中记录 salt 摘要。
每档在冷请求之后先按相同 `min_tokens`、相同输出预算和相同 salt 再发一次热请求，比较完整输出；随后另做自然停止复核。
前缀一致性需要固定同一 wheel、绑核、rank 和输入，对缓存开关独立重启服务。发生生成分歧时比对首次核验的目标层输入、输出和未经采样处理的 logits，并保留关闭诊断的评测回执。
如需复现首两层注意力差异，可在已有 `diagnose-first-verify` 的诊断结果目录创建 `diagnose-attention-window-rank-0`，文件内容为所需回执数（1 至 32）。八卡服务从普通入口重新启动后，指定 rank 保存首个核验的逻辑位置、物理槽、实际缓存字节和投影。正式长请求与 GPQA 使用关闭诊断的配置。
同 rank 的交错请求使用两个独立 salt。先预填充左侧，再同时发送左侧命中、右侧冷请求；下一对均应命中。顺序组使用相同输入和生成参数，核对各自完整输出与实际 HTTP 重叠。

有窗口诊断时，在 CPU 用捕获的原始 BF16 KV 重算每组 UE8M0 指数和每个 E4M3FN 码，按活跃行逐字节核对。两次服务独立保留回执，并检查归档摘要。[窗口量化核对](../day0_stage3/validator/window_quantization.py)

若相同输入在压缩注意力层发生首差，可在指定 rank 的结果目录创建 `diagnose-compressed-rank-0`，其中登记实际双源层、固定短提示长度和回执数，例如 `{"layer":8,"prompt_tokens":265,"limit":8}`。与 `diagnose-first-verify` 一起使用，同次回放归档原始窗口、压缩索引及物理页、压缩字节、Q 和输出投影。正式评测关闭这些诊断。[压缩层诊断](../vllm_ascend/models/deepseek_v41/compressed_verify_trace.py)

在独立诊断服务上固定左侧提示、salt、命中数、rank 和采样参数，右侧依次使用相同提示与一个 token 不同的提示。通过 `diagnose-overlap-head-rank-0` JSON 指定提示长度、采样位置和偏移、关注的 token、top-k 及回执上限，在实际单请求与双请求核验中比较采样前 logits、物理页和输出。[目标 head 取证](../vllm_ascend/models/deepseek_v41/overlap_logits.py)

配对时绑定原镜像的实际标识。若使用 OCI 导入，逐层验证官方 registry 清单、Linux arm64 子镜像、config ID 和 Docker 实际 RootFS。运行中保存八个 worker 的实际绑核与 DSpark 指标，再核对两侧的配置及 198 道题的逐项结果。

压缩层取证使用同一次回放的 RoPE 后 BF16 源值，分别重算压缩 KV 的 E2M1/E4M3 与索引键的 E2M1/UE8M0，并核对实际写入字节及物理页读回值。

按任务方确认，GPQA 放行只采用官方 v0.30.0 A5 正式版的全量正确题数。对双方使用同一份冻结输入及 scorer；需要修正模型时，使用评测题单之外的固定输入比对首个数值差异，并在修复后的仓内 wheel 上重新运行全量评测。

官方 950DT 配置使用 `HCCL_OP_EXPANSION_MODE=CCU_SCHED` 和 `HCCL_BUFFSIZE=1600`。启动脚本接受显式通信模式，缺省沿用先前的 AIV。记录每次运行的通信设置，正式版切换通信配置后，候选也用相同设置重新测量。[官方单机指南](https://docs.vllm.ai/projects/ascend/zh-cn/v0.30.0/tutorials/models/DeepSeek-V4.1-Flash.html)

按任务方要求，为正式版另冻结客户端并发 8 和 16。两档沿用全部 198 题、显示顺序、提示、生成预算与官方评分，先运行 8，必要时运行 16；用于数值对比的候选在成功的同一并发独立重跑。 每次记录实际并发峰值、唯一请求数和完成状态。

正式发行包的单卡索引控制同时记录元数据和计算的同步点。在指南给出的容器权限与网络配置下分别试验发布二进制和同一发行包编译的算子；端到端正确率仍由原始模型服务和全部 198 个官方评分确定。

按任务方更新的精度目标，候选修正后在原 198 题、非思考模式、贪心解码和 8,192 输出上限下独立重跑。每个修复先在评测之外的固定输入验证数值因果。另按作者的最大推理力度及采样参数运行独立条件，逐轮报告配置、截断和答对数。 [固定版本模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/README.md)

候选优先使用作者模型卡的 Instruct 条件复测 GPQA-Diamond 全部 198 题。开启思考，推理力度 100，temperature 1.0，top_p 0.95，每题独立采样一次。输出预算采用作者通用建议的 262,144 token。归档实际 HTTP 参数、逐题完整输入 ID、输出长度、结束原因和官方评分。修复滑窗寻址后，先用独立短提示检查调度页表与真实读取，再在关闭诊断的服务运行评测。[固定版本模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/README.md)

作者条件的 GPQA 完成、归档后，在同一仓内 wheel 上按冻结的常规聊天模板启动独立服务。依次运行 262,128 和 1,048,560 个输入 token，各预留 16 个输出 token。每档包含冷请求、热前缀重放和自然停词检索，记录实际用量、返回的三个标记及八卡资源采样。
