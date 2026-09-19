# V3 最终整合与人工确认

## 三方来源

- 框架：保留 v2 已同步的上游版本与 MT 接入，不退回 original 的旧 evaluator。
- neural metrics：复用 BSC original。MetricX models.py 撤回 v5 改动；COMET/Kiwi/XCOMET/BLEURT/MetricX metric.py 均与 original 完全一致。没有第二套复制实现。
- 新增部分：双环境运行、已有译文读取、分数保存和环境记录。
- 配置：保留 v2 的 GPFS checkpoint、指标开关、prompt 和任务配置。
- 传统指标：保留 v2 的语言 tokenizer/亚洲 TER 行为，不冒然回退；因此并不保证所有传统分数与 original 相同。

## 人工确认：三个边界

1. `lm_eval/api/mt_task.py` 与 `neural_scoring/score.py`：生成阶段跳过 neural、三列文本对齐、指标入参和逐句输出。
   辅助 `neural_scoring/legacy_metrics/__init__.py` 只把包搜索路径指向原指标目录。
2. `scripts/run_mt_v3.sh` 与 `scripts/launch_flores_v3.sbatch`：GPFS 路径、环境、模型、参数。已有生成文件会复用，评分重新执行；修改生成配置时使用新输出文件名。
3. 环境和指标配置：`requirements-generation-v5.txt`、`requirements-neural-v4.txt`、`pyproject.toml`、`lm_eval/extra_metrics/mt_metrics_config.yaml`。在 GPFS 固定译文上验证逐句分数，尤其最大绝对差值。依赖文件尚非经过 GPU 验证的完整 lockfile。

不用重审纯上游文件或原版 neural 算法。两个旧任务文件 holisticbias/perturbations 仅把顶层导入换成延迟调用，属于环境隔离接入，不改数据/任务逻辑。
GPFS 的新增 tasks/data 不在审核范围。特殊 holistic/toxicity 聚合不支持本双环境 wrapper。

## 已完成自动检查

- 六个 neural 核心文件与 BSC original 逐字节比较通过。
- 四项评分输入/输出/失败保护测试通过。
- 实际 v2 JSON 的 1012 条 MT 三列解析通过。
- Python/Bash 语法检查；独立指标加载路径检查。

未完成：GPFS GPU 生成、真实 neural inference、依赖安装完整验证。不能宣称实际分数已验收。
上次 v2 清单中的其他模型后端及 workflow 问题仍存在，但不是普通 HF/FLORES 双环境运行的逐文件审核要求。使用那些功能时再处理。
