# Prompt 1(8 小时):图增强的学习器——用全部图像的图改进 reward head 本身

你是接手 RHEED 主动学习项目的 Claude。先读 `code/active_learning_studies/pair_disjoint_not_image_disjoint/graph_exploration/prompts/COMMON_RULES.md` 并严格遵守。工作分支:`claude/graph-learner-<日期>`(从 `claude/integration` 切出)。种子:开发 7000–7019;确认 1 = 7100–7134;确认 2(重复,若时间够)= 7200–7234;冷启动确认(只有 10 条随机初始判断,若时间够)= 7300–7334。

## 为什么做这个
选择规则的收益很小且依赖 schedule,而一个便宜的学习器改动(参考图锚点权重 0.25→8)的收益是选择规则的 1–2 倍,并吃掉了选择规则的优势。图至今只被用在"选哪个标注"上,没有用在"模型怎么学"上。图真正独特的东西是**没有标注的图像**:仓库里 1124 张轨迹图 + 154 张理想图,而 168 个标注组只覆盖 284 张。冻结特征的 head 看不到这些图像的结构;图(半监督/流形正则化/GCN)正好能把它们用上。这也是导师建议的"再接 GCN"一步,这里作为学习器改进来做。

## 目标
找到一个图增强的学习器,在**预先登记的新种子**上显著优于调好的基线 `aw8_all`(或在不能用 `aw8_all` 的终点上优于 `aw8`),终点不变(留出配对组的 AUC、准确率、log-loss;两个划分),标注集用 `nd_core.py` 的"相同随机标注集"配对比较;若做不到,清楚写出负结果和原因。

## 协议扩展(必须写进预注册)
- 学习器可以使用"无标注图像的特征"建图:节点 = 全部轨迹图像 + 全部 154 张理想图,**排除**该种子下验证/测试组的图像和外部测试理想图(身份安全,按内容 hash)。边 = 余弦 kNN(k 先固定 10,开发种子上最多扫 {5,10,20})。
- 其余(特征、head 结构、损失、schedule 的 lr/steps 规则、终点)与 `nd_learners.py` 里 `aw8_all` 一致。

## 开发方案(≤8 个变体,开发种子上筛选,全部与 `aw8_all` 配对)
1. 流形正则化(LapRLS 风格):在 head 的类型分数 s_k(x) 上加 λ·Σ_ij A_ij (s_k(i) − s_k(j))²,对无标注节点也生效;λ 扫一个小网格。
2. 类型后验先验:对每个无标注图像用标签传播(参考图 one-hot → kNN 图)得到类型后验 q(x),加一项把"类型 k 的分数排序"拉向 q 的排序(软锚点;与已被否定的伪标签自训练不同:这里目标来自参考图的图传播,不来自模型自己的预测)。
3. SGC/GCN 编码器:两层图卷积(先 SGC 一步无参数平滑,再带参数的 GCN)作用在冻结特征上,per-type BT head,锚点损失不变。
4. 1+2 组合、1+3 组合。
5. 对照:每个获胜候选都要有"打乱图"(节点身份置换,度序列不变)对照和"无图但同样加了无标注图像做 dropout/噪声正则"的对照,判断是图结构还是额外正则在起作用。
每个变体同时记录整体指标和按类型指标(特别是 HTR);冷启动(10 条随机初始判断)也要在开发种子上看一遍。

## 文献(开发前先读,写进 `notes_graph_learner_literature.md`,只写搜索结果确认的内容)
Manifold regularization(Belkin、Niyogi、Sindhwani)、Laplacian 正则化最小二乘、label spreading(Zhou 等)、GCN(Kipf & Welling)、SGC(Wu 等)、低标注率下的图半监督学习(Poisson learning)、带无标注数据的偏好学习/排序、以及 `new_directions/literature/` 里已有的内容(别重复)。

## 判定与交付
- 预注册:冻结 ≤4 个候选(含各自的打乱图对照)和家族划分,写入 `PREREGISTRATION_G1.md` 并提交,再跑确认。成功标准:某候选在两个划分里的 AUC 和准确率相对 `aw8_all` 的配对差值都为正,且 Holm(家族 = 候选数 × 2 划分 × 2 指标)后至少 3 个检验 p<0.05,并且优于它自己的打乱图对照(未校正 p<0.05 在至少一个划分)。否则只报告"未确认"。
- 结果文档:`graph_exploration/RESULTS_graph_learner.md`(含生成的表格、机制分析、所有变体清单、限制)、`SUMMARY_ZH_graph_learner.md`、`HANDOFF_graph_learner.md`、`GPU_TASKS_graph_learner.md`(例如:端到端微调里加图正则化、更强的编码器的特征)。
