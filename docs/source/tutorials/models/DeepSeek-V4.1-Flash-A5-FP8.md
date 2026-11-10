# DeepSeek-V4.1-Flash：A5 原始 FP8/FP4 部署

## 简介

本次在 `d4ac4319acdaf480689c39b6320a998712f80523` 上增加原始 FP8/FP4 检查点适配，
保留该提交的 V4.1 模型、Engram、DSpark、KV 管理和调度实现。vLLM 固定为原版 v0.27.1
（`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`），未替换为另一套 V4.1 模型。

验证权重为本机 `/mnt/weight/DeepSeek-V4.1-Flash-1017`：40 层、384 个路由专家、
每 token 选择 6 个专家；两层 Engram；三层 DSpark 草稿；32 层视觉编码器。
稠密矩阵使用 FP8 E4M3、32×32 分块 E8M0 scale，专家使用 packed FP4、每 32 列一个 scale。
模型参考为 [DeepSeek-V4.1-Flash 固定版本](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/314113f8a7dadb388215396ba5fbee2c610a04d6)。

## 支持情况

| 功能 | 2026-11-09 实测状态 |
| --- | --- |
| FP8/FP4 真实权重 | supported；目标每 rank 1640 个参数、草稿每 rank 105 个参数完整加载 |
| TP4 / EP4 | supported；目标每 rank 96 个专家，草稿每 rank 32 个专家 |
| DSA-CP / 共享专家 DP | supported；主模型和草稿真实推理通过 |
| ACLGraph | supported；目标 FULL_DECODE_ONLY，24/48 token 捕获并实际 replay；草稿 eager |
| DSpark5 | supported；监测窗口中接受 582 / 1050 个草稿 token，非固定性能指标 |
| 图片 | supported；算式图片两次返回 HTTP 200 和正确答案 5 |
| 前缀缓存 | supported；1117-token 文本第二次命中 896 token，结果一致 |
| 中文 / 流式 / 并发 | supported；10 项常规请求、16 路并发算术请求均通过 |
| FlashComm1 独立开关 | not-applicable；该基线配置已废弃此开关，使用 DSA-CP 自动启用的 SP |
| A2 / A3 原始 FP4 | unsupported in this deployment；本次原始 FP4 运算路径依赖 A5，未在 A2/A3 验证 |

## 环境准备

=== "Atlas A5 Docker"

    本机使用物理 NPU 4–7，每卡 96 GiB，容器内编号 0–3。主机提供约 2.2 TiB RAM，
    足够容纳 mmap 的 Engram 页和权重文件缓存。权重共约 475 GiB，48 个分片；
    Engram 表约 188 GiB 保留在主机，按命中行读取。

    已构建镜像 `jojo/vllm-ascend:d4ac431-a5-fp8-fp4`。挂载驱动、URMA、设备和只读权重的
    完整参数见 `examples/deepseek_v41_a5/start.sh`。同一组 NPU 不得同时分配给 builder
    和运行容器；builder 即使只运行 `sleep` 也必须停止。

=== "Atlas A3 Docker"

    本次没有验证 A3 上的原始 FP8/FP4 路径。原提交的 W8A8 部署见
    [A3 教程](DeepSeek-V4.1-Flash.md)，不要将本教程的 A5 算子配置套用到 A3。

=== "Atlas A2 Docker"

    本次没有验证 A2 上的原始 FP8/FP4 路径。

| 组件 | 本次版本 |
| --- | --- |
| 驱动 / CANN | 25.1.rc3.2 / 9.2 |
| Python / Transformers | 3.12.13 / 5.14.1；未升级 Transformers |
| PyTorch / torch-npu | 2.10.0+cpu / 2.10.0.post4 |
| triton-ascend | 3.2.2 |
| 安装目录 | `/opt/d4ac431/vllm`、`/opt/d4ac431/vllm-ascend`、`/opt/d4ac431/.venv` |

复用部署目录 `/home/jojo/vllm-ascend-d4ac431`，其中 `source/` 是本适配仓库，
`vllm/` 是固定版本的上游仓库。`build-runtime.sh` 在隔离 builder 中安装两者，
为此模型编译 A5 `sparse_flash_attention` 及其 C++ 注册扩展。
运行时通过 editable 安装加载代码，不依赖 `PYTHONPATH` 覆盖。

聊天模板来自上述官方固定版本，保存为 `reference/chat_template.jinja`，SHA256：
`d959d804d5101b79a49b6ff1bf3c54cd5affa9a5a78c503d925f3090040b31c3`。

## 部署

当前主机已有容器时使用 `docker start jojo-v41-d4ac431`；首次创建使用部署目录的 `start.sh`。
脚本以服务为容器主进程，配置健康检查、日志轮转和 `unless-stopped` 重启策略，
仅向主机 `127.0.0.1:8000` 发布端口。完整环境变量与命令见
`examples/deepseek_v41_a5/serve.sh`，容器内可直接执行：

```bash
cd /workspace
bash /deployment/serve.sh
```

实际参数：TP4、EP4、`--quantization deepseek_v4_fp8`、BF16 激活、64K 上下文、
512-token prefill 批次、最多 16 条序列、HBM 利用率 0.93，DSpark5、解码图 24/48。
1–8 条均匀 decode 请求可以使用这些图尺寸，更大或混合批次使用 eager。
模型配置理论上限为 1,048,576 token；本配置 KV 容量为 103,296 token，
所以 128K + bs16 基线不可容纳，仅单条 128K 就超过本次 KV 容量。
64K 是配置上限，本次未做 64K 满长度精度测试，也不能同时容纳 16 条 64K 请求。

可选回退：停掉当前服务后，在同一容器环境运行 `bash /deployment/serve-eager.sh`。
此配置使用 32K、最多 8 条序列、无 DSpark、`--enforce-eager`，文本真实权重检查 10/10 通过。
未遇到 Dynamo 的 interpolate/contiguous 问题，无需 `TORCHDYNAMO_DISABLE=1`。

## 功能验证

```bash
curl --noproxy '*' -fsS http://127.0.0.1:8000/health
curl --noproxy '*' -fsS http://127.0.0.1:8000/v1/models
curl --noproxy '*' -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1","messages":[{"role":"user","content":"计算2+3，只输出结果。"}],"temperature":0,"max_tokens":16,"chat_template_kwargs":{"thinking_mode":"chat"}}'
```

预期 `choices[0].message.content` 为 `5`。使用 `--chat-template-content-format openai`
支持文本及 `image_url` 内容块。该模板的自动 string 检测会使图片请求报 HTTP 400。
本机 HTTP 代理不适用于 localhost，请保留 `--noproxy '*'`。

完整功能验证命令和日志位置见 [运行手册](DeepSeek-V4.1-Flash-A5-runbook.md)。

## 精度检查

| 检查 | 设置 | 结果 |
| --- | --- | --- |
| 基础算术 | 贪心、8 题 | 8/8，eager 和 DSpark + graph 一致 |
| 并发算术 | 同时提交 16 请求 | 16/16 |
| GSM8K 前 8 个 test 样本 | zero-shot、温度 0、最多 512 输出 token、4 路请求 | 8/8，numeric_exact_match=1.0 |
| 图片算式 | 一张 640×480 图片 | 2/2 次回答正确 |

GSM8K 检查是小样本集成回归，不是完整 GSM8K 或官方精度分数。数据来自
[OpenAI GSM8K test](https://github.com/openai/grade-school-math/blob/master/grade_school_math/data/test.jsonl)，
配置记录文件 SHA256，评测程序拒绝其他快照。配置为
`tests/e2e/models/configs/DeepSeek-V4.1-Flash-A5-FP8.yaml`，使用专用 API runner
`tests/e2e/models/eval_v41_api.py`，不交给通用 `test_lm_eval_correctness.py`。

## 性能

目标加草稿权重约 77.3862 GiB/卡，KV 预算 5.94 GiB/卡。
在混合验证负载中，16 路短算术请求约 1.15–9.54 秒完成；8 个 GSM8K 请求各约 6.46–22.04 秒。
1117-token 重复文本请求从 5.875 秒降到 1.331 秒，第二次命中 896 个缓存 token。
这些是本机功能测试观测值，包含并发、冷启动和调度影响，不构成稳态吞吐基准或加速比结论。

详细根因、回退记录和证据见 [适配报告](DeepSeek-V4.1-Flash-A5-report.md)。
