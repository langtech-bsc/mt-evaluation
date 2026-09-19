# mt-evaluation-v3：生成与 neural metrics 分离

本地生成，基于 fork 的 `a1cc303e0a3864a6365876df686945e762cd7d08`，未向 GitHub 发布，也未合并新的上游提交。
保留现有 tasks、prompts 和传统指标逻辑。GPFS 的任务和 data 脚本仍按你现有部署补齐。

## 一次提交完成全流程

编辑 `scripts/launch_flores_v3.sbatch` 的项目路径、两个 Python 路径及模型路径。
该文件已沿用你提供的 Slurm 配置、语言对与生成参数。提交前创建 `slurm_logs_xixian`。
执行 `sbatch scripts/launch_flores_v3.sbatch`。

每个方向输出两份文件：
- `results_....json`：v5 生成的文本和传统指标。
- `results_....neural.json`：v4 的 neural metrics、逐句分数、相同文本、输入文件 SHA256 和依赖版本。

第二份文件只包含 neural 结果，避免沿用过期的 group 分数或 stderr。
生成进程退出后再启动评分进程，因此生成模型的显存会释放。
已存在的生成结果会复用；评分会重新执行。修改模型、prompt 或生成参数后必须使用新的输出路径。
评分失败不会产生新的完整评分文件，可再次运行恢复；旧评分文件如存在，其 hash 可核对输入。

## 环境

生成环境：在项目根目录执行
```bash
/path/to/venv-v5/bin/python -m pip install -e . -r requirements-generation-v5.txt
```
评分环境：独立 Python 3.10/3.11 venv，执行
```bash
/path/to/venv-neural-v4/bin/python -m pip install -r requirements-neural-v4.txt
/path/to/venv-neural-v4/bin/python -m pip check
```
评分环境不要安装整个项目；`python -m neural_scoring.score` 在项目根目录即可运行。
不要运行旧 v5 patches。若复用旧环境，必须确保 bleurt/comet 安装文件没有被 patch；推荐新建环境。
CUDA/PyTorch 按集群配置安装。requirements 是初始约束，并非经过 GPU 验证的完整 lockfile。
Transformers 固定为你 v1 的 4.53.3；其他依赖应优先沿用你已验证的 v1 版本。
首次成功后对两个环境分别 `pip freeze` 保存锁定版本。

指标开关、checkpoint、batch size 继续使用 `lm_eval/extra_metrics/mt_metrics_config.yaml`。
也可设置绝对路径 `MT_METRICS_CONFIG` 使用另一份配置。默认包括 COMET、Kiwi、BLEURT、MetricX 和 QE；XCOMET 可打开。
本次仅将这些 neural metrics 拆开；没有改动 BLONDE 等其他指标算法。

## 单独评分
```bash
/path/to/venv-neural-v4/bin/python -m neural_scoring.score \
  --input results/example.json --output results/example.neural.json
```
不会加载生成模型，不导入 lm_eval，不重新生成。

## 与上游同步

继续按需要合并 EleutherAI/lm-evaluation-harness 的发布版本，以获取模型支持和 evaluator 修复。
不必实时追 main。保留 fork 的 Git 历史后应用本目录的变更，便于继续合并上游；本目录是源代码快照，不含 .git。
`neural_scoring/` 通过 JSON 与生成侧连接，独立固定依赖，不随上游 Transformers 升级。
每次同步重点检查 `MTask` 对 request、process_results、aggregation 的接入和结果 JSON 格式。
现有 upstream sync workflow 继承自 v2，本次未审核或改变其自动推送策略。

## 改动及验证范围

- MTask 使用延迟导入；wrapper 设置 `MT_DEFER_NEURAL_METRICS=1`，生成阶段禁用 neural 计算。
- 主安装移除 COMET、BLEURT、detoxify 依赖；评分独立安装。
- MetricX models.py 恢复 BSC 原版，保留许可证与来源；不复制维护第二套指标实现。
- 所有 neural metric 实现直接复用原目录，与 BSC original 逐字节一致；BLEURT 保留原版 GPU 要求。
- 修复 samples 未传入 doc_iterator，并在选择 samples 时禁用 request cache 防止错配。
- 检查逐句文本长度、评分长度和有限值；评分输出原子写入。

本地验证：Python/Bash 语法、真实 v2 JSON 输入解析、模拟指标的数据对齐与失败处理。
未在 GPFS 执行模型生成或真实 neural inference；使用前应以少量样本完成两环境 smoke test。

## 最终整合范围（2026-09-15）

本版基于 v2 上游同步框架，复用 original BSC 指标实现；不是把整个框架退回旧版。
`V3_METRIC_PROVENANCE.json` 记录六个核心实现与 BSC original 的逐字节一致性。
评分加载使用 `neural_scoring/legacy_metrics/__init__.py` 的包路径，直接访问原指标目录，避免导入 lm_eval 主运行时。
两个旧任务 holisticbias/perturbations 仅调整顶层导入为延迟导入，防止注册任务时加载 neural 依赖。
本 wrapper 支持标准 MT 三列结果，不支持 holistic/toxicity 自定义聚合；此类任务需要单独适配。
保留 v2 的 BLEU 语言 tokenizer、亚洲语言 TER、prompt、任务映射和 JSON 路径行为；这些不是 neural v5 patches，不能通过回退 neural 代码自动恢复旧版数值。
原有 MT 后端与自动同步 workflow 的已知问题不在此次双环境整合中修复；不代表整个仓库已经验收。
