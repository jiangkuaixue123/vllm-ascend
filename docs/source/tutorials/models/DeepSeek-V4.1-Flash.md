# DeepSeek-V4.1-Flash

## 简介

本教程部署 DeepSeek-V4.1-Flash 的 40 层语言模型和 32 层图像 encoder。实测使用 Atlas A3 单机 16 个 NPU die，每个 die 配备 64 GiB HBM，权重为 Ascend W8A8 动态量化检查点。模型保留 384 个路由专家，每个 token 选择 6 个，并在第 1、14 层使用 Engram。官方参考版本为 `314113f8a7dadb388215396ba5fbee2c610a04d6`。[模型与配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/314113f8a7dadb388215396ba5fbee2c610a04d6)

## 支持情况

| 功能 | 本次验证结果 |
| --- | --- |
| 文本生成 | 真实 40 层权重通过文本生成、中文、多轮和流式请求，图模式的单请求输出与 eager 对齐 |
| 专家并行 | TP8、DP2、EP16 运行通过，每个 EP rank 持有 24 个完整路由专家 |
| Engram | 主机 mmap 按 TP 分片读取命中行，在设备端反量化，在当前 DP 的 TP 组内归约。decode 的命中行在 replay 前更新 |
| ACLGraph | `FULL_DECODE_ONLY` 已在真实 40 层模型上完成 capture 和 replay，每个 DP 使用 1、2、4、8 行捕获尺寸覆盖 1–8 条 decode 请求。16 个真实 GPQA 请求时，两组都已验证 8 个活跃行重放。prefill 和混合批次使用 eager |
| FlashComm1 | TP 内分片 token；路由专家通过 EP16 收集输入并将输出归约回 token 所在的 rank |
| DSA-CP | 在各 TP rank 的本地 token 上计算完整查询头，使用 all-to-all 转换为输出投影的分头布局。该开关会启用序列并行 |
| 共享专家 DP | 各 TP rank 加载完整共享专家并处理不同 token。三开关联合在 TP8 / DP2 / EP16 下完成 21 条合成请求的逐 token 对照 |
| DSpark | 读取检查点中的三层草稿、独立 embedding 和输出头。草稿在 eager 下提议 5 个 token，目标模型按 6 个位置校验 |
| 图片输入 | ViT、aligner 和图像分隔符从 W8A8 检查点的 BF16 张量加载。可通过聊天 API 的 `image_url` 内容块提供图片 |
| 前缀缓存 | 同图与文本的 prompt 可复用目标和草稿的已完成 KV 块。Engram 按页恢复历史，图像行会打断相邻的文本 n-gram |
| A3 注意力算子 | SWA 和 DSA 使用 `npu_sparse_attn_sharedkv`。V4.1 的 1×/2× 压缩候选与窗口 token 一起进入显式 slot 模式，使用一次联合 softmax。DSpark 的草稿块保留块内双向注意力 |

## 环境准备

=== "Atlas A3 Docker"

    按 A3 安装说明将 16 个 die 与驱动挂入容器，并将权重目录以只读方式挂载。vLLM 使用原版 v0.27.1，全部适配代码随 vllm-ascend 安装。在容器内检查可见设备和固定基线。[Docker 安装指南](../../getting_started/installation.md#installation-prebuilt-image)

    ```bash
    npu-smi info
    git -C /vllm-workspace/vllm rev-parse HEAD
    git -C /vllm-workspace/vllm-ascend rev-parse HEAD
    ```

=== "Atlas A2 Docker"

    A2 环境尚未验证。本教程的设备数、权重占用与 KV 容量均来自 A3 单机。

| 组件 | 验证版本 |
| --- | --- |
| vLLM | v0.27.1，`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`，原版 |
| vLLM Ascend | v0.27.1rc1，`3b31886237ed65c435e0965b10e8002ff5769766`，包括 V4.1 配置、模型、算子、图输入准备和 ModelSlim 描述加载 |
| Python / Transformers | 3.12.13 / 5.14.1 |
| PyTorch / torch-npu | 2.10.0+cpu / 2.10.0.post4 |

安装包含本次适配的 vllm-ascend。插件通过自身的 `ascend_model` 入口注册模型和 Transformers 配置。检查插件模块的实际路径。

```bash
cd /workspace
git -C /vllm-workspace/vllm status --short
python -c 'from vllm.plugins import load_general_plugins; load_general_plugins(); from vllm_ascend.models.deepseek_v41 import model; print(model.__file__)'
```

准备包含 `quant_model_weights.safetensors.index.json`、`quant_model_description.json` 和 `optional/quarot.safetensors` 的 W8A8 检查点。实测模型有 142 个分片，合计约 727.68 GiB。两张 Engram 表保留 INT8 权重与每组 32 列的 E8M0 scale，共约 188.83 GiB。按索引选择 `engram-extra` 中的 q/k/wkv，可避免加载数值编号分片中未被索引引用的副本。

实测检查点缺少聊天模板。将固定版本的 `chat_template.jinja` 下载到单独目录并校验 SHA256，然后将 `MODEL_PATH` 和 `CHAT_TEMPLATE` 指向容器中的实际文件。[官方聊天模板](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/chat_template.jinja)

```bash
MODEL_PATH=/models/DeepSeek-V4.1-Flash-W8A8
CHAT_TEMPLATE=/models/DeepSeek-V4.1-Flash-reference/chat_template.jinja
test -r "$MODEL_PATH/quant_model_weights.safetensors.index.json"
printf '%s  %s\n' d959d804d5101b79a49b6ff1bf3c54cd5affa9a5a78c503d925f3090040b31c3 "$CHAT_TEMPLATE" | sha256sum -c -
```

## 单机部署

在 `/workspace` 启动服务。固定 TP8 和 DP2 后，`--enable-expert-parallel` 形成 EP16。以下命令用于 Diamond 评测，32,768 上下文包含提示和输出。两个 DP 各允许 8 条序列；均匀 decode 批次补齐到 1、2、4 或 8 行，空行的 KV 保持原值。

```bash
cd /workspace
vllm serve "$MODEL_PATH" \
  --served-model-name dsv41-0day --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 8 --data-parallel-size 2 --enable-expert-parallel \
  --max-model-len 32768 --max-num-batched-tokens 128 --max-num-seqs 8 \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}' \
  --no-enable-prefix-caching \
  --chat-template "$CHAT_TEMPLATE" --reasoning-parser deepseek_v3
```

测 eager 时将 `--compilation-config` 及其值替换为 `--enforce-eager`，其余参数保持一致。

### 启用并行优化

在上面的命令中添加以下参数可启用三个开关。联合验证时，使用 `--max-model-len 512`、`--max-num-seqs 8` 和下面的整图配置替换原图配置。每个 DP 的实际 1–8 条均匀 decode 请求使用同一个按 TP8 对齐的图。

```bash
--additional-config '{"enable_dsa_cp":true,"enable_flashcomm1":true,"enable_shared_expert_dp":true}' \
--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[8]}'
```

在同一节点的 512 上下文上，使用固定的 32 个输出 token 和贪心解码。三个开关联合开启时，两 DP 分别接收 1/0、3/1、8/8 个请求，合计 21 条输出与此前无开关部署逐 token 一致。8/8 批次的实测吞吐分别为 26.99 和 21.75 输出 token/s。

### DSpark 与 fused MoE

该检查点训练了三层 DSpark 草稿。目标模型第 37、38、39 层把 Engram 注入后的 HC 输入提供给草稿。在上面的 TP8 / DP2 / EP16 服务命令中保留三项并行优化，添加投机参数并替换图配置。

```bash
--additional-config '{"enable_dsa_cp":true,"enable_flashcomm1":true,"enable_shared_expert_dp":true}' \
--speculative-config '{"method":"dspark","num_speculative_tokens":5}' \
--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[24,48]}'
```

草稿在 eager 下计算，目标模型对锚点和 5 个提议位置做因果校验。每个 DP 最多 8 个请求，24 与 48 个 token 的整图尺寸同时满足 TP8 和每请求 6 个校验位置的对齐要求。prefill 和混合批次走 eager。

文本目标与草稿的路由专家均由 Ascend fused MoE runner 执行。目标每个 token 选择 6/384 个专家，草稿选择 3/128 个。分组矩阵乘在 SwiGLU 之后乘入 FP32 路由权重，再把激活量化给 W2。开启共享专家 DP 时，每个 TP rank 持有该层完整的共享专家。

## 功能验证

在同一容器的另一个终端查询模型列表并提交聊天请求。`thinking_mode=chat` 指定直接回答，便于复现短请求。

```bash
curl --fail --noproxy '*' http://127.0.0.1:8000/v1/models
curl --fail --noproxy '*' http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv41-0day","messages":[{"role":"user","content":"What is 2 + 2? Answer with only the number."}],"chat_template_kwargs":{"thinking_mode":"chat"},"temperature":0,"seed":0,"max_tokens":32}'
```

在纯插件服务中，文本补全 `The capital of France is` 返回 `Paris.`。图模式的中文、多轮和流式请求均返回 HTTP 200，多轮算术回答 `14`，流式回答正常结束。模型在 decode 期间实际重放了 ACLGraph。

## 图像输入与前缀缓存

以下配置用于短请求验证。3 个开关同时打开，DSpark 每次提出 5 个 token。原生注意力在 A3 上使用插件随附的 CANN 自定义算子。目标 decode 使用 24、48 两个图尺寸，prefill 和草稿使用 eager。

```bash
cd /workspace
vllm serve "$MODEL_PATH" \
  --served-model-name dsv41-0day --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 8 --data-parallel-size 2 --enable-expert-parallel \
  --max-model-len 512 --max-num-batched-tokens 128 --max-num-seqs 8 \
  --enable-prefix-caching --enable-prompt-tokens-details --limit-mm-per-prompt '{"image":1}' \
  --chat-template "$CHAT_TEMPLATE" --chat-template-content-format openai --reasoning-parser deepseek_v3 \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[24,48]}' \
  --additional-config '{"enable_dsa_cp":true,"enable_flashcomm1":true,"enable_shared_expert_dp":true}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}'
```

通过 `/v1/chat/completions` 发送带 `image_url` 和 `text` 的用户消息。processor 根据图片几何尺寸展开占位符，encoder 输出包含每行的换行向量和起止向量。检查前缀命中时，将重复请求发往相同的 DP，比较响应中的 `usage.prompt_tokens_details.cached_tokens` 和贪心输出 token。选择足够长的 prompt，使其除 DSpark 校验预留外还包含可复用的完整 KV 块。

## 准确率评测

| 数据集 | 样本数 | 协议 | 准确率 |
| --- | ---: | --- | --- |
| GPQA Diamond | 198 | 零样本，seed 0，temperature 0，top-p 1，最大输出 16,384 token，并发 16 | 本轮已停止，完整准确率为空 |

使用数据 revision `56686c06f5e19865c153de0fdb11be3890014df7`，Diamond CSV 的 SHA256 为 `41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305`。模板参数为 `enable_thinking=true`、`reasoning_effort=high`。服务端用基线 `deepseek_v3` parser 分离推理和最终回答，评测器只读取最终回答。运行 198 题之前，用非数据集选择题验证了这一边界。固定选项乱序后对 198 题逐题计分，HTTP 错误、截断、无法解析均计错，V005 停测时共发出 33 题，其中 17 题有终态、16 题正确。V006 停测时共发出 52 题，其中 36 题有终态、34 题正确。两轮结果分别保留。

## 性能与容量

真实权重加载约占 35.12 GiB/rank。32,768 上下文配置下，每个 DP 的日志报告可用 KV 约 19.94 GiB/rank、65,280 个 token，满长请求并发估计 1.99。该配置已完成短请求与不同长度并发验证，整段 32,768 token 输入的实测仍待补充。

配置文件的理论位置上限为 1,048,576。当前缓存约为 320 KiB/token/rank，单条 128K 请求约需 40 GiB/rank，超出现有 KV 空间。128K 加全局 16 条满长序列也超出此配置的容量。此容量估计来自参考缓存布局。

单请求对照使用同一版纯插件、原版 vLLM、TP8 / DP2 / EP16。上下文设为 256，关闭前缀缓存，使用相同的 20 个输入 token，固定生成 32 个 token。每组预热一次，再串行测量三次；每次输出的 token ID 均一致。TPOT 按第一个和最后一个生成 token 的客户端到达时间差除以 31 计算，TTFT 包含 prefill。

| 单行对照 | TPOT 中位数（ms/token） | TTFT 中位数（ms） |
| --- | ---: | ---: |
| Eager | 501.30 | 1024.37 |
| FULL_DECODE_ONLY，捕获尺寸 1 | 289.30 | 1155.18 |

该对照的 TPOT 降低 42.29%，TTFT 增加 130.80 ms。多行图和 32,768 上下文属于不同的服务配置。

在上文的 32,768 上下文、多尺寸图部署中，使用同一输入和 32 个输出 token，3 次单请求 TPOT 分别为 340.80、340.99、340.65 ms/token。中位数为 **340.80 ms/token**，TTFT 中位数为 **1246.51 ms**。两组 DP 的 16 并发实测另行验证，已在真实数据集请求之后观察到每组 8 行的 FULL 重放。

### Fused MoE 与 DSpark 单请求性能

在同一份 fused MoE 插件和 W8A8 检查点上固定 TP8 / DP2 / EP16、512 上下文、128 个 batched tokens、每 DP 最多 8 条序列。每组向 DP0 单独发送相同的 20-token 提示，贪心生成 32 个 token。预热一次后串行测五次，三组的输入和生成 token ID 均一致。TPOT 为最后一个与第一个输出 token 的客户端到达时间差除以 31，投机返回的一帧可含多个 token。

| 配置 | 单请求图尺寸 | TPOT 中位数（ms/token） | TTFT 中位数（ms） |
| --- | ---: | ---: | ---: |
| 三开关关闭，DSpark 关闭 | 1 | 209.41 | 856.25 |
| 三开关开启，DSpark 关闭 | 8 | 344.89 | 1154.63 |
| 三开关开启，DSpark 开启 | 24 | 59.16 | 964.44 |

相对全关组，三开关加 DSpark 的 TPOT 降低 71.75%。相对只开三项并行优化的组，加入 DSpark 后降低 82.85%。在这条提示的五次正式请求中，草稿共提议 150 个 token，目标接受 150 个。目标端实际重放了 24 行图，其中有 6 个活跃校验位置。

### A3 原生 SWA 和 DSA 的单请求性能

在上文图像与前缀缓存的部署配置下，对照组和原生算子组均使用 Ascend fused MoE、三个并行开关及 DSpark5。固定相同的 20 个输入 token、32 个贪心输出 token，预热一次后串行测五次。两组每次生成的 token ID 完全一致。

| 注意力实现 | TPOT 中位数（ms/token） | TPOT 范围（ms/token） | TTFT 中位数（ms） |
| --- | ---: | ---: | ---: |
| 插件原注意力 | 58.51 | 58.28–61.80 | 968.27 |
| A3 原生 SWA 和 DSA | 43.38 | 43.11–43.94 | 812.37 |

原生算子使该负载的 TPOT 降低 15.13 ms/token（25.86%）。两个 DP 都捕获了 24、48 行 FULL decode 图。单请求阶段记录到 24 行图的重放，6 行活跃，5 次正式请求的 150 个提议均被接受。图像与长文本另行验证了前缀缓存，同一 DP 上重复请求命中 128 个 token，换图或换 DP 首次请求命中 0 个 token。
