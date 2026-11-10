# 本机 A5 FP8/FP4 部署脚本

首轮 Triton 融合的版本化回调、构建/启动入口和候选验证流水见
[fusion/README.md](fusion/README.md)。该目录对应历史六轮平均 97.84 token/s 的组合。

这些脚本是本次部署目录中入口的版本化副本，部署布局为：

```text
/home/jojo/vllm-ascend-d4ac431/
  source/                 # 本适配仓库
  vllm/                   # 原版 v0.27.1
  reference/chat_template.jinja
  reference/gsm8k-test.jsonl
  reference/vision-request.json
  logs/
  start.sh
  serve.sh
  serve-eager.sh
  build-runtime.sh
  verify-service.py
```

不要直接在本 examples 子目录运行 `start.sh`；先将脚本复制到已有部署目录，
这样 `/deployment` 挂载才包含 source、reference 和启动脚本。
权重固定只读挂载本机 `/mnt/weight/DeepSeek-V4.1-Flash-1017`，设备固定为物理 4–7。
`build-runtime.sh` 在带相同挂载的隔离 builder 中运行，依赖本机 CANN 9.2 基础镜像环境。
运行 builder 前先停止服务，运行服务前必须停止 builder。

聊天模板从固定版本下载：

```bash
curl -fL --retry 3 \
  https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/314113f8a7dadb388215396ba5fbee2c610a04d6/chat_template.jinja \
  -o reference/chat_template.jinja
printf '%s  %s\n' d959d804d5101b79a49b6ff1bf3c54cd5affa9a5a78c503d925f3090040b31c3 \
  reference/chat_template.jinja | sha256sum -c -
```

完整参数、验证结果与限制见
[A5 教程](../../docs/source/tutorials/models/DeepSeek-V4.1-Flash-A5-FP8.md)。

## 后续固定性能验证

用户指定的固定场景为 `v41-c8-i100-o256-v1`。每次优化后均使用同一入口：

```bash
cd /home/jojo/vllm-ascend-d4ac431
python3 benchmark-throughput.py
```

仓库内等价入口为 `python3 examples/deepseek_v41_a5/benchmark-fixed.py`。
`--describe` 只展示场景；`--output <新文件路径>` 指定报告位置，已有文件会被拒绝覆盖。
默认报告保存到部署根目录 `logs/fixed-throughput/`，文件名含时间。

| 项目 | 固定值 |
| --- | --- |
| 设备、并行、权重 | 4 张 A5，物理 4–7，TP4/EP4，同一 FP8/FP4 检查点 |
| 请求 | 同时发送 8 条固定的图书馆主题中文提示词，一轮共 8 条 |
| 输入 token 数 | 97、100、98、95、96、96、98、101；同时检查每条 token IDs 的 SHA256 |
| 输出 | 每请求 256 token，总计 2048 token；`min_tokens=max_tokens=256`、`ignore_eos=true` |
| 采样和协议 | temperature=0、seed=0、thinking_mode=chat、流式返回并统计 usage |
| 预热 | 每轮先执行 1 条固定提示词，输出 64 token，不计入吞吐 |
| 缓存 | 每请求独立 cache_salt；输入内容不变，cached_tokens 必须为 0 |
| 吞吐公式 | 2048 / 整批端到端耗时，包含预填充和生成，保留客户端调度开销 |
| 附加指标 | 平均 TTFT、平均请求耗时、解码速率、DSpark 接受率、服务端计数 |

`fixed-throughput/scenario.json` 保存完整输入和参数，`scenario.sha256` 固定其版本。
tokenizer/模板改变输入、短输出、缓存命中、请求失败、并行业务导致服务端计数不符，均使结果无效。
服务需空闲后再测。脚本只发送验证请求，不修改服务参数或清空全局缓存。

上次 46.1295 token/s、平均 42.7814 秒、平均 TTFT 6.5519 秒保存在
`fixed-throughput/historical-reference.json`。旧脚本使用随机任务编号且没有保存编号，无法逐字重放。
现已固定新的任务编号，并匹配上次每条请求的 token 数；`baseline.json` 保存固定场景首次有效结果，
以后以它进行精确输入的对比，不把随机输入测试当作同输入基线。
脚本自动记录实际镜像和服务参数，输出相对基线的吞吐变化百分比，但不会覆盖基线或自动判定优化成功。
一次结果存在波动；需要判断较小差异时，重复相同命令并保留所有结果。
固定版首次有效结果为 51.9727 token/s，复跑为 56.7803 token/s；两次服务参数相同，
这约 9.25% 的变化体现了短测波动，不能作为代码优化收益。基线仍保留首次结果，未替换为更高值。

## 推理主干展平实验

参考 `jojoinfra/Agentic-Compiler` 的 `c237f86844598388d2d20d5782344bd538bf617e`，
采用从源码内联、展开静态循环、验证数值等价的方法。该仓库的 DeepSeek 样例是小配置训练模型，
本实现直接处理当前 d4ac431 适配后的推理代码，不替换权重、量化格式或服务框架。

`vllm_ascend/models/deepseek_v41/flatten.py` 从当前方法的 AST 生成前向：

- 展开 40 个目标模型 block 和 3 个 DSpark stage，内联 block 的 HC 与 RMSNorm 计算。
- 展开每个 HC 的 20 次 Sinkhorn 归一化，保持张量操作顺序、FP32 中间结果和 BF16 转换。
- 保留 attention/KV cache、Engram、量化 MoE、TP/EP 通信及服务元数据接口。
- 保留空批次参与专家通信、图像路由、decode active mask、DSpark 辅助状态的语义。
- 在内联结束后删除局部临时引用，避免将所有层的临时张量延长到整个前向结束。

这是推理主干的源码展平，尚不是所有 attention/MoE 内部算子的纯 Torch 展平，
也没有新增融合算子、自动微分或训练支持。ACLGraph 的捕获和重放保持原样。
只对当前模型实例绑定生成的方法，原来的模块、参数所有权和 checkpoint 加载接口保持不变。

在原有 `--additional-config` 字典中添加 `"deepseek_v41_flatten":true` 可启用；默认关闭。
在已有运行环境中导出实际检查点对应的显式代码，不需要加载权重：

```bash
python examples/deepseek_v41_a5/export-flat-forward.py \
  --config /models/deepseek-v41/config.json \
  --output /tmp/v41-flat-review
```

输出包含目标模型 prefill/decode、draft 前向及 SHA256 清单。
这些文件是以原模块 globals 和原模型实例执行的回调源码，不是独立模型检查点。
将 `"deepseek_v41_flatten_source_dir":"/tmp/v41-flat-review"` 一并加入 `--additional-config`，
即可直接执行该目录中的代码，作为后续融合实验的入口。路径必须在容器内可访问。
加载时检查静态配置和每个源文件的 SHA256；后续优化应复制出新目录，修改代码、更新实验清单，
保留原始展平版本用于 A/B 对照。不要重新生成并覆盖已经手动优化过的版本。
每次改变模型静态配置都要重新生成；每次改变代码都要重新验证。

本次实验的完整生成代码、原始 A/B 结果、采集脚本和 trace 位于部署根目录
`experiments/flatten-20261109/`。正式计时使用固定场景，profiling 另行运行；
不得将 profiler 开启期间的吞吐与无采集结果直接比较。

详细性能证据、可直接执行的源码入口、融合候选及验证顺序见 [展平与融合实验](FLATTENING.md)。
