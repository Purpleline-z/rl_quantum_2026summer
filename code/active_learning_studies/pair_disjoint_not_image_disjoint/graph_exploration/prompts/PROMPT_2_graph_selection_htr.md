# Prompt 2(8 小时):调好学习器之后,图还能帮选择吗?——冷启动与 HTR 导向的图选择

你是接手 RHEED 主动学习项目的 Claude。先读 `code/active_learning_studies/pair_disjoint_not_image_disjoint/graph_exploration/prompts/COMMON_RULES.md` 并严格遵守。工作分支:`claude/graph-selection-htr-<日期>`(从 `claude/integration` 切出)。种子:开发 8000–8019;确认 1(调好学习器下的主实验)= 8100–8134;确认 2(冷启动,10 条随机初始判断)= 8200–8234;确认 3(HTR 导向规则,若时间够)= 8300–8334。

## 为什么做这个
`new_directions` 发现:学习器用 `aw8` 后,`vopt_u` 等选择规则不再显著优于 Random(冷启动下 A 划分边缘),而 HTR——实验室最关心的类型——在任何学习器下都没有确认的增益。我们已经知道图传播得到的类型后验携带一点超出类型本身的信息,也知道"类型相关的权重"(而不是预测 decisive)才是 `typed_decisive_coverage` 的有效成分。两个没回答的问题:(1)图规则在调好的学习器下还有没有用,特别是冷启动(选择最可能有用的时候);(2)能否用图专门瞄准 HTR 与 √13(RT13)之间的边界(图上的"桥"图像、类型后验最模糊的判断)来改进 HTR。

## 目标
(A) 在 `aw8_all` 学习器下重测冻结的图规则:`typed_decisive_coverage`、`gvopt_lap3`、基线 `vopt_u` 和 Random,常规预算与冷启动各一次(`nd_select.py` 已有"选择器 × 学习器"的框架,把图规则接进去)。(B) 设计并确认一个 HTR 导向的图选择规则,使 HTR 的留出判断准确率/AUC 在 `aw8_all` 学习器下优于 Random,同时其他类型的准确率下降不超过 0.01;若做不到,写出机制分析(这些规则把预算花在了哪些类型/图像上,为什么没用)。

## 协议扩展
无(学习器 = `aw8_all`,协议不变)。新增的评估:按类型的留出指标(每个种子每类型只有约 13–16 个 decisive HTR 测试判断,必须跨种子汇总,按预先登记的方式检验)。选择规则只能使用:候选判断、已揭示判断、当前模型、候选/已标注/参考图像的特征(图只建在这些图像 + 参考图上,和 `graph_typed.py` 一致)。

## 开发方案(≤8 个选择规则变体,开发种子上筛选)
1. 把 `typed_decisive_coverage`(及其 `_typeonly`、`_shuffled` 对照)、`gvopt_lap3`(及 `_shuffled`)接进 `nd_select.py`,在 `aw8_all` 下与 Random/`vopt_u` 配对比较。
2. 机制分析:这些规则选出的集合的类型构成、图像覆盖、每个类型的预算占比,与 Random 比较;检验"类型相关的预算分配"这一假设(如按类型配额的覆盖规则)是否能复现收益。
3. HTR 导向的图规则:(a) 用参考图的标签传播得到 HTR 与 RT13 的类型后验,选后验最模糊的判断(类型 k ∈ {HTR, RT13});(b) "桥"图像:kNN 图上同时靠近 HTR 与 RT13 参考图的图像(例如两类标签传播分数都高);(c) (a)(b)与覆盖/`vopt_u` 的组合;每个都配"打乱图"对照。
4. 冷启动版本:全部规则在 10 条随机初始判断下再测。

## 文献(开发前读,写进 `notes_graph_selection_literature.md`,只写搜索结果确认的内容)
图上的主动学习与 Σ-optimality(已有笔记)、边界/桥节点与标签传播的类别模糊度、目标导向/类别优先的主动学习(class-priority、cost-sensitive)、冷启动主动学习(TypiClust、ProbCover 等已有笔记)、"弃权式标注者"(不适用/平局)与主动学习、`new_directions/literature/lit_selection.md`(别重复)。

## 判定与交付
- 预注册:冻结 ≤4 个候选(含对照)写入 `PREREGISTRATION_S1.md` 并提交,再跑确认。成功标准:(A) 某图规则在 `aw8_all` 下相对 Random 的 AUC 与准确率增益在冷启动或常规预算的两个划分里 Holm 后显著(家族 = 候选数 × 划分 × 指标);(B) HTR 规则的 HTR 准确率与 AUC 增益 Holm 后显著且其他类型准确率损失 ≤ 0.01。任何图相关的说法必须优于对应的打乱图对照。否则写"未确认"并给出机制解释。
- 结果文档:`graph_exploration/RESULTS_graph_selection.md`(生成的表格、类型构成的机制分析、所有变体清单、限制)、`SUMMARY_ZH_graph_selection.md`、`HANDOFF_graph_selection.md`、`GPU_TASKS_graph_selection.md`(例如:端到端微调下的重测)。
