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
