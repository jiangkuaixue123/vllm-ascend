# 本机 A5 FP8/FP4 部署脚本

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
