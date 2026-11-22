# Data-Agent-Planning

本目录是 2026-11-22 的 DataAgent 流程代码快照，包含 suitable30 的冻结数据划分。30 个数据集对应 31 个预测任务，NOMAD 包含两个目标。来源说明保存在任务清单及各任务的 task.json 中。

## 安装

建议 Python 3.11。从本目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.data-agent.txt
cp .env.example .env
```

在 .env 中填写自己的 OPENAI_API_KEY。API 模型和地址由 config/workflow_skill_mcts.yaml 配置。requirements.data-agent.txt 提供基础依赖，environment_versions.json 记录打包环境的相关版本。

## 运行本次验证流程

```bash
python -u scripts/validate_suitable30.py \
  --cold-sequential --skip-failed \
  --output results/suitable30_validation \
  --env-file .env
```

顺序执行清单里的任务，从空 skill 经验库开始，完整轮次的经验写入共享数据库供后续检索。失败任务保留报错和已完成记录，继续下一个任务。再次执行同一命令可恢复未完成任务并跳过已记录失败的任务。

初始生成五个方法，全部进行全量训练，扩展全量排名前三的方法。后续每轮生成两个候选，全部进行全量训练，选择全量最佳候选继续扩展，最多到第 2 层。每个完整任务包含 7 轮比较和 17 次全量评估。

每轮记录原始 85% 加保护的 15% coreset，以及选定区域上可用的 baseline、cover、gentle_cover、focus 对照。若 agent 选择 baseline，程序预先确定一个区域用于 action 对照。训练评分使用验证集，验证程序记录 test_access=false。

方法轨迹、action 改动、排名、全量预测和奖励保存在输出目录。可以生成汇总：

```bash
python scripts/summarize_suitable30_validation.py results/suitable30_validation
```

常规流程入口为 run_data_agent_workflow.py 和 scripts/run_mcts_skill_campaign.py，可用 --help 查看参数。本次全候选、全 action 验证协议由 scripts/validate_suitable30.py 实现。

## 数据与复现

datasets/suitable30/tasks.json 是可迁移的任务清单，各任务目录包含训练、验证、测试及已有协议文件。请在项目根目录运行命令。data_manifest.json 给出数据文件大小与 SHA-256。

数据文件使用 Git LFS，克隆后需要运行 git lfs pull。代码使用仓库 LICENSE，数据使用条件请依据 task.json 中的原始来源核对。

本包包括运行需要的 data_agent 模块、入口脚本、配置和少量 InternAgent 模型适配依赖。最新的参数兼容、修复反馈和断点恢复代码已包含。检查覆盖配置加载、入口导入和数据完整性，完整数据集的重新训练需实际执行上述命令。

## 当前入口说明

本快照的上述运行命令启动全候选、全 action 验证程序。核心主流程代码已包含最新修改，但默认配置仍含校准和额外评估调度。严格的“初始 coreset 前三全量、后续 coreset 第一全量、只训练所选 action”默认入口仍待统一。
