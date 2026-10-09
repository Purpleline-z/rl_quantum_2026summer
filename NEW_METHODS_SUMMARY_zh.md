# 昨晚新方法探索：一页总结（中文）

分支 `claude/new-methods`（已合并进 `claude/frozen-encoder-strategies`，报告新增 §5.15）。详细英文版：`code/active_learning_studies/pair_disjoint_not_image_disjoint/NEW_METHODS_RESULTS.md`；过程日志：`NEW_METHODS_LOG.md`。

## 一句话结论
在 judgment 单位、重新调好的 head 训练参数下，**“最小化候选池整体预测方差”的最优设计类规则（I-optimal 设计 `vopt_u`，以及之前的 Fisher D-optimal、BALD×P(decisive)）显著优于随机选择**；这个结论在多组预先登记（pre-registered）的新种子上复现。但这是一个**方法族**的效应，不是新规则独有；效应不大（主协议下 accuracy +1~2 个点，AUC +0.006~0.013）；而且**只在已有标注很少时存在**。

## 具体数字（均为相对 Random 的平均增益，4 个 split×条件格子合并；Holm 校正）
| 实验（新种子，预先登记） | 结果 |
|---|---|
| 主协议，700–734 | `vopt_u` accuracy +0.0095（Holm 0.0004），AUC +0.0058（Holm 0.13，不显著），log-loss 不显著 |
| 主协议复现，800–834（4 个方法×{AUC, acc}，Holm 8） | **8 个检验全部显著**：`vopt_u` AUC +0.0127、acc +0.0167、log-loss +0.035 |
| 冷启动（初始 10 条随机 judgment），900–934（Holm 12） | **12 个检验全显著**：`vopt_u_inf1` log-loss +0.061、AUC +0.020、acc +0.024 |
| 跨“世界”复现（图像不相交的两半数据互为训练/测试），1200–1234 | `vopt_u` AUC +0.0067、acc +0.0133、log-loss +0.017 全显著；但两个方向差异大（H1→H2 弱，H2→H1 强） |
| HTR 定向（只问 HTR），1000–1034 | HTR accuracy +0.071、AUC +0.078（相对 Random；AUC 在 33 个种子上有定义）；比“随机问 HTR”还高 +0.021/+0.011；但代价：c6x2 −0.030、1x1 −0.015 |
| **HTR 优先权重 ×4（`vopt_hw4`）**，1100–1134 | HTR accuracy +0.067、AUC +0.058、log-loss +0.183，**同时全类型 accuracy +0.016、AUC +0.014，其它类型没掉**（推荐的形式） |

## 重要的限制（请务必读）
1. 所有“新种子”其实是**同一批 168 个 pair 组的新随机划分**，不是新数据；p 值偏乐观。跨世界实验（两半图像不相交）缓解了一部分，但两半来自同一批实验。
2. **与已有标注数量强相关**：冷启动增益最大（acc +0.019）；30 条 judgment 起步 +0.017；20 组 +0.007；40 组 +0.001；60 组（约 175 条）−0.003。也就是说，实验室现在已有几百条标注，在**同一分布**上再选的收益大概率很小；真正的检验要用**新数据（新生长批次）**。
3. 效应依赖 head 的训练参数：用旧参数（lr 0.01）增益消失。
4. 第一次预先登记（700–734）比开发时弱（AUC 不显著），说明开发数字有“赢家诅咒”；可信的是预先登记的结果，不是开发结果。
5. 不确定性采样（uncertainty）、coverage 类（core-set 等）、NTK 版设计、类型配额、P(decisive) 因子都**没有带来额外收益**；“知道一条 judgment 是否 decisive”也没价值（ties / not_apply 一样有用）。
6. 事后诸葛亮（hindsight）oracle 能比随机多 +0.04~0.05 accuracy、log-loss 好 0.16~0.20，但**单条 judgment 的真实效用无法由任何可观测特征预测**（|ρ|≤0.06），所以没法学出一个逐条打分函数。

## 给你的建议（我的看法）
- 对教授可以说：**“在标注很少的阶段，主动选择（最优设计类）有稳定、可复现的小幅收益；uncertainty 类没有；收益随已有标注增多迅速消失。因此值得在新数据上做前瞻性实验，而不是在旧数据上继续调方法。”** 这比“没有策略比随机好”更准确，也诚实。
- 如果实验室目标是 HTR，推荐 `vopt_hw4`（HTR 权重是个可调旋钮，4 是开发时定的，没细调）。
- 下一步最有价值：**新生长数据到手后做前瞻性对比**（random vs `vopt_u` / `vopt_hw4`）。

## 需要你明天做的（GPU/联网/需要人）
- 第二个编码器（ImageNet 或更大的自监督模型）：这台机器下载不到权重（只通 PyPI）。需要你在 Colab 跑或放开网络域名。
- 更大的候选池：需要新数据或模拟器图片 + 标注来源确认。
- 用顺序无关的端到端微调替代冻结编码器 + 小 head，再检验这个效应是否仍在（GPU）。
- 与实验室确认 HTR 优先权重取多少。

## 复现
`code/active_learning_studies/pair_disjoint_not_image_disjoint/new_methods_run.sh`；所有确认性表格：`results/new_methods/CONFIRMATORY_TABLES.md`（`new_methods_final_tables.py` 生成）；预先登记文件 `NEW_METHODS_PREREGISTRATION*.md`（1–6，每个文件末尾记录了提交哈希）。
