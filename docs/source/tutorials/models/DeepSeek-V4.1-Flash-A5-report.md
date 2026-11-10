# d4ac431 原始 FP8/FP4 适配报告

## 范围与根因

用户要求保留 `d4ac431` 并在当前 A5 主机使用已有 FP8/FP4 权重。
因此所有运行代码以该提交为基线，vLLM 保持原版 v0.27.1。
模型结构和部署参数见 [教程](DeepSeek-V4.1-Flash-A5-FP8.md)。

原提交针对 A3 ModelSlim W8A8：它假定量化索引、INT8 Engram、逐通道专家 scale、
独立 DSpark embedding/head，以及 A3 shared-KV 算子。
本地检查点采用标准 safetensors 索引、FP8 Engram、packed FP4 专家及 E8M0 分组 scale，
DSpark 复用主模型 embedding/head；A5 需要对应的 sparse flash attention 算子。
仅更换启动参数无法修复这些格式与算子契约差异。

## 代码变化

1. `engram.py` 兼容标准索引和 E4M3 表，保留主机 mmap、分片命中行读取和 E8M0 反量化。
2. `fused_weights.py` 映射 `.scale`，按原始 FP4 尺寸读取专家，保留 packed bits，交由现有 EP/TP loader 分片。
3. `model.py` 支持 FP8 Engram 投影、无旋转的原始权重及 TP 局部 BF16 投影反量化。
4. FP8 linear 支持 32×32 与 128×128 scale 展开；V4.1 使用匹配原始检查点的 E8M0 向上取整。
5. `fused_experts.py` 调用 A5 MXFP4 分组矩阵乘；FP32 路由系数仍在 down projection 量化前乘入。
6. A5 native attention 保留原提交的 latent/KV/slot 语义，将 sink 归一化与 SFA LSE 合并。
   CANN 补丁使负数 padding 不参与 softmax、全空行归零，并补齐 A5 编译所需 barrier 接口。
   只加载所需 C++ 注册，不启用 A2/A3 专用全局 custom-op 路径。
7. DSpark 原始 FP8 加载器将缺失的独立 embedding/head 映射到共享源张量，保持草稿 FP32 输出头。

所有新增运行逻辑均在 vllm-ascend 中；没有修改上游 vLLM，也没有转换或覆盖原始模型分片。
单元测试覆盖 FP8 小数值、TP 分片、缺 scale 报错、FP4 packed bits、scale 布局、草稿共享权重、
真实 NPU GEMM、稀疏索引重复/空行及图 replay。
仓库 `bash format.sh ci` 的全部检查通过，记录为 `logs/format-ci-final2.log`。
ARM 主机上的检查使用官方 ARM64 ShellCheck/Gitleaks 二进制，未跳过对应检查。

## 验证与回退记录

| 阶段 | 结果及证据（部署目录 `logs/`） |
| --- | --- |
| 合成张量与 CPU | 310 passed、80 deselected；加载器、模型数学、缓存、DSpark 和量化回归；`test-v41-final-regression2.log` |
| 真实 NPU 算子 | 8 passed；6 个 A3 元数据专用用例跳过；`test-native-a5.log` |
| 真权重 eager | 目标完整加载，10/10 文本/流式请求；`serve-eager-attempt3.log`、`api-eager-verification.json` |
| 真权重 DSpark + graph | 目标 1640、草稿 105 参数/rank；capture 和真实 replay；`serve-graph2.log` |
| API 功能 | 10/10 常规、16/16 并发、两次图片答案正确、缓存命中；对应 `api-*.json` |
| 小样本精度 | 固定 GSM8K 前 8 题全对；`gsm8k-first8-graph.json`，不推断完整精度 |

没有运行完整模型的 dummy 启动。合成张量测试证明布局、算子和数值契约；
只有真权重启动与请求证明了完整分片加载、端到端生成、视觉路径及 DSpark/graph 集成。
配置理论上下文为 1M，本次运行设置 64K、KV 总容量 103,296 token；
128K + bs16 超出当前四卡与草稿模型配置的缓存预算，未宣称通过。
不占用其他工作负载所在的设备来扩大容量。

遇到的问题及最终处理：

- builder 与运行容器共用设备时，NPU 初始化失败。停止 builder 后恢复；不是 Triton 版本不兼容。
- 原始 A5 SFA 对负数 padding 的处理不能直接满足模型语义；通过自定义 CANN 算子修正并做数值与 replay 对照。
- 首次 DSpark 真权重启动缺少 `mtp.0.embed.weight`；查验索引后增加共享权重映射，重新完整加载通过。
- eager 文本服务就绪后，图片请求因模板 string 检测返回 HTTP 400。改为 OpenAI 内容块后，两次图片请求均通过。
- 先使用 eager 隔离模型和量化问题，再启用 DSpark 与图；最终默认启用后者。
  没有把端口监听或模型列表当作完整成功，最终均检查了真实生成。

完整特性状态见教程表格：EP、ACLGraph、MTP/DSpark、图片均通过。
FlashComm1 独立开关在本提交的配置层已被废弃，DSA-CP 自动启用的 SP 路径已实际执行。
没有对应的来源 GitHub issue，本次只交付本地代码、镜像、服务、报告和签署提交。
