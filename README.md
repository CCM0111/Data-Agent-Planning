# Data-Agent-Planning

最新 DataAgent 主流程，包含 suitable30 的 30 个数据集、31 个预测任务。

## 安装与运行

建议 Python 3.11，在项目根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.data-agent.txt
cp .env.example .env
python -u run.py
```

在 .env 填写自己的 OPENAI_API_KEY。API 地址及模型由 config/main.yaml 配置。
指定单个任务可使用 --task，输出目录可使用 --output。

## 默认主流程

初始生成五个方法，在 API 所选 action 对应的 coreset 上评价，按 coreset 分数选择前三个进行全量训练。

后续每轮生成两个候选，在本轮 API 所选 action 对应的 coreset 上评价，仅将 coreset 排名第一的候选进行全量训练。按照既有深度策略，超过第 3 层时只生成一个候选并直接进行全量评估。

每轮采样包含受保护的 15% 引导样本，区域 action 作用于其余 85%。baseline 使用本轮重新构建的 coreset。在线执行只训练所选 action。

默认最多 12 次成功全量评估，包括初始化三次，后续最多九次。失败次数及 API、时间限制可能使实际完成数减少。搜索以验证指标反馈，记录 test_access=false。

任务按清单顺序执行，共享从空库开始的 skill 经验。完整轮次写入经验，后续任务检索使用。发生异常时保留报错及断点并继续下一个任务。重新执行命令会跳过完成任务并尝试恢复其余任务。

## 数据

datasets/suitable30/tasks.json 是任务清单，NOMAD 包含两个预测目标。各任务保留原训练、验证和测试划分。data_manifest.json 提供文件大小与 SHA-256，来源及说明保存在 task.json 和清单中。

数据通过 Git LFS 管理，克隆后运行 git lfs pull。代码遵循 LICENSE，数据使用条件应依据原始来源核对。

## 其他入口

scripts/validate_suitable30.py 是单独的研究验证工具，会全量评估所有候选及多个 action。默认 run.py 调用 data_agent.mcts_skill_workflow.run_skill_task。

environment_versions.json 记录打包环境相关版本。当前交付验证覆盖入口导入、配置加载和调度约束，完整主流程的真实 API 训练效果需通过运行得到。
