# 首轮 Triton 融合的可复现入口

本目录对应 `FUSION_RESULTS.md` 中六轮平均 **97.8445 token/s** 的首轮优化。
分支基于 `dsv41-a5-fp8-fp4`，保留该分支从 `d4ac431` 开始的八个适配提交，
随后逐项迁入原来的固定基准、展平、融合和测试提交。

## 源码与配置

- `selected/`：当时选定的 `pool-v1`，含目标模型、DSpark、Attention、缓存及 MoE 回调。
  HC/Sinkhorn、QDQ、选择性缓存读写、HC pre/post、MoE 激活、KV/Indexer/Compressor 融合启用。
- `optional/`：完整组合额外需要的 Engram、SFA sink、Router、confidence 回调及清单。
  其余文件复用 `selected/`。该组合历史三轮均值为 94.1541 token/s，默认不启用。
- `provenance.json`：原始提交、各文件原 SHA256、规范化 AST 指纹。
  回调仅增加注释和格式化；AST 对比仅统一 `del a, b` 与 `del (a, b)` 的等价写法。
  实际算子代码保持原始历史提交内容。
- `historical-results.json`：全部阶段的各轮吞吐、正确性摘要及原始报告指纹。
  这些是 2026-11-09 的历史测量，不是本次迁移后重新测得的性能。

回调通过 `flatten.py` 绑定到模型实例，在原方法的全局命名空间执行。
文件中的外部符号和未删除的 SSA 临时变量是保留测量代码的需要。
`manifest.json` 固定静态配置并校验全部源码 SHA256；记录的历史绝对路径仅用于追溯。
当前静态配置是 40 层目标模型、3 层 draft、HC 宽度 4、20 次 Sinkhorn 迭代。

生成一个新的可编辑候选目录（已有目录会被拒绝覆盖）：

```bash
python3 examples/deepseek_v41_a5/fusion/prepare-bundle.py \
  --profile selected --output /tmp/v41-fusion-candidate \
  --config /mnt/weight/DeepSeek-V4.1-Flash-1017/config.json
```

需要完整的尾部融合组合时，使用 `--profile optional`；脚本会自动补齐共享回调。
修改候选代码后需要更新其 manifest，并重新运行对应数值、ACLGraph 和固定场景验证。
GEMM、SFA 主体、动态 MXFP8 量化、通信和 host Engram 读取仍使用原有实现。

## 镜像与部署布局

沿用父目录 README 的部署布局和 CANN 9.2 环境，`source/` 应 checkout 到本分支。
从干净、已提交的源码构建镜像，无需重新编译未改动的原生 SFA：

```bash
bash examples/deepseek_v41_a5/fusion/build-image.sh
```

构建依赖本机已有 `jojo/vllm-ascend:d4ac431-a5-fp8-fp4` 基础镜像。
脚本通过 Git bundle 把当前提交及历史放入镜像，记录实际源码 commit；不会启动服务。

将本目录复制到部署根目录下 `experiments/fusion-20261109/`，而不是直接从仓库 examples
目录启动。该目录须不存在；已有实验应保留，使用另一个部署副本。
部署根目录还需包含 `source/`、`reference/chat_template.jinja`、
`reference/gsm8k-test.jsonl`、`reference/vision-request.json` 和父目录的 `serve-eager.sh`。
后三个验证资源沿用原始部署，图像用例的预期答案为 `5`，GSM8K 只验证前 8 题。

设备空闲时，启动入口是：

```bash
bash experiments/fusion-20261109/start-attention.sh
```

该入口固定物理 4–7 卡、TP4/EP4、DSpark5 和 24/48 decode ACLGraph，使用 `selected/`。
`--eager` 沿用原部署的 eager 回退入口；其配置不启用这些展平回调。
新建分支和打包过程不改变当前运行服务，不代表已经切换线上默认版本。

## 验证和候选流水

CPU 测试包括 `test_flatten.py`、`test_fusion_bundle.py`；真实 NPU 测试为
`tests/ut/models/deepseek_v41/test_triton*.py`。历史 186 项 NPU 测试通过。
本次只迁移相同算子源码并核对回调 AST；未重新运行 NPU 或端到端吞吐测试。

服务健康后运行：

```bash
bash experiments/fusion-20261109/validate-when-ready.sh new-stage-name
```

它会执行 API、GSM8K 小样本、图像回归和三轮固定场景测试。
固定场景仍为 `v41-c8-i100-o256-v1`，8 并发、每请求 256 输出 token、零缓存命中。
需要复现选定版本的六轮测量时，以另一个全新阶段名再执行一次，保留全部结果。

后续准备独立镜像、候选目录和新的 `start-*.sh`，完成算子数值与 ACLGraph 检查后运行：

```bash
bash experiments/fusion-20261109/run-candidate.sh new-stage-name \
  experiments/fusion-20261109/start-candidate.sh
```

流水入口检查容器归属、串行锁和阶段名；保存旧容器，验证失败时尝试恢复旧容器。
自动回退分支未进行真实故障注入。它会停止属于本部署的当前服务，只有计划切换候选时才运行。
正式计时期间不要并行运行其他 NPU 测试。
