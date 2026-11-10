# A5 FP8/FP4 运行手册

## 服务和路径

- 主机部署根目录：`/home/jojo/vllm-ascend-d4ac431`。
- 容器：`jojo-v41-d4ac431`；镜像：`jojo/vllm-ascend:d4ac431-a5-fp8-fp4`。
- API：`http://127.0.0.1:8000/v1`；模型名：`deepseek-v4.1`。
- 权重只读挂载：`/mnt/weight/DeepSeek-V4.1-Flash-1017` → `/models/deepseek-v41`。
- 服务占用物理 NPU 4–7；容器内使用 `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`。

## 启停

```bash
cd /home/jojo/vllm-ascend-d4ac431
docker start jojo-v41-d4ac431
docker logs --tail 60 -f jojo-v41-d4ac431
```

容器不存在时，运行 `bash start.sh` 创建。容器已有时脚本拒绝覆盖，避免误删状态。
停止使用 `docker stop --time 60 jojo-v41-d4ac431`，重新启动使用 `docker start`。
已配置健康检查和 `unless-stopped` 重启策略；人工停止后不会自动重启。
不要同时启动 `jojo-v41-d4ac431-build`，它持有同一组设备。

新容器需要数分钟加载权重和捕获图。除 `/health`、`/v1/models` 外，必须验证真实生成：

```bash
curl --noproxy '*' -fsS http://127.0.0.1:8000/health
python3 verify-service.py --quick --output logs/api-manual-quick.json
python3 verify-service.py --output logs/api-manual-full.json
```

容器中直接启动的等价入口：

```bash
cd /workspace
bash /deployment/serve.sh
```

同一容器不要启动第二份服务。需要 eager 隔离时先停止当前服务，在调试容器中执行
`bash /deployment/serve-eager.sh`；创建新 eager 容器时可使用 `bash start.sh --eager`。
两个入口均保持端口 8000，默认入口为 DSpark5 + 解码图。

## 图片与小样本评测

部署目录已有图片请求示例，可检查输出是否为 `5`：

```bash
curl --noproxy '*' -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' --data-binary @reference/vision-request.json
docker exec jojo-v41-d4ac431 /opt/d4ac431/.venv/bin/python \
  /deployment/source/tests/e2e/models/eval_v41_api.py \
  --config /deployment/source/tests/e2e/models/configs/DeepSeek-V4.1-Flash-A5-FP8.yaml \
  --dataset /deployment/reference/gsm8k-test.jsonl --output /tmp/gsm8k-manual.json
docker cp jojo-v41-d4ac431:/tmp/gsm8k-manual.json logs/gsm8k-manual.json
```

评测在容器内使用已安装的 PyYAML 和 regex。数据下载地址及 SHA256 见教程与 YAML；结果只代表固定前 8 题。

## 证据与限制

所有下列日志均位于部署根目录 `logs/`：

| 文件 | 内容 |
| --- | --- |
| `build-runtime-v41.log` | 编译结果 |
| `test-v41-final-regression2.log` | CPU 回归 |
| `test-native-a5.log` | 8 个真实 NPU 数值及图测试，6 个 A3 专用元数据测试跳过 |
| `serve-eager-attempt3.log`、`api-eager-verification.json` | eager 真权重与 10/10 请求 |
| `serve-graph2.log` | 真权重、DSpark、解码图捕获与 replay |
| `api-graph-verification.json`、`api-concurrency16-graph.json` | 10/10 常规与 16/16 并发 |
| `api-vision-graph.json`、`api-vision-prefix-graph.json` | 两次图片答案 |
| `api-prefix-graph.json` | 896 个前缀 token 命中 |
| `gsm8k-first8-graph.json` | 全部 8 个问题、输出、答案、耗时 |
| `metrics-graph.txt` | DSpark 接受量、前缀缓存计数 |
| `serve-final.log`、`api-final-verification.json` | 固化镜像后的启动和生成复验 |

64K 是单请求配置上限，KV 总容量有限；未进行满长准确性测试。
API 仅发布到本机回环地址，没有配置鉴权，不应直接改为公网监听。
