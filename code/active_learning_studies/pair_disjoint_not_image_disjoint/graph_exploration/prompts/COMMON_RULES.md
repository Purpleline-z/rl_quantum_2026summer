# 共同背景与规则(两个 8 小时 prompt 都先读这个文件)

## 读什么(按顺序)
1. 仓库根目录 `SESSION_CONTEXT_FOR_NEXT_CLAUDE.md`(项目背景、分支历史、坑)。
2. `code/ACADEMIC_REPORT_DRAFT.md` 的 §5.12–5.20(尤其 §5.14 judgment-unit 协议、§5.15 方差缩减设计与 head schedule、§5.16–5.20 图相关结果)。
3. `code/active_learning_studies/pair_disjoint_not_image_disjoint/graph_exploration/` 下的 `HANDOFF_graph_run.md`、`AUTONOMOUS_RUN_LOG.md`、`OVERNIGHT2_PLAN.md`、`notes_overnight2_literature.md`。
4. `code/active_learning_studies/new_directions/` 下的 `NEW_DIRECTIONS_RESULTS.md`、`README.md`、`nd_core.py`(另一个会话做的"学习器侧"研究;它的 harness 用同一批 168 个标注配对组、同一协议,可直接复用)。
工作分支:从 `claude/integration` 切出你自己的新分支(名字见各 prompt),只推自己的分支,**不要推 `main`**,不要删任何分支。

## 已经确立的事实(别重复发现,也别和它们矛盾)
- 比较图(图像为节点、标注配对组为边)近似完美匹配:168 条边、284 个节点、117 个连通分量,排序型 GNN 在它上面没有可恢复的结构;图必须建在图像相似度上(冻结 SimCLR 特征的 kNN)。
- 所有实验用同一批 168 个配对组(换随机种子重新划分,不是新数据),所以 Wilcoxon p 偏乐观;只有预先登记、在没用过的种子上的确认才算证据;在开发种子上挑方法已经两次高估效应(赢家诅咒)。
- head schedule 决定结论:旧 schedule(lr 0.01,100 步)下图规则无效;重新调过的 schedule(`results/pair_endpoint_study/schedule_judgment_unit.json`)下冻结的类型感知图规则 `typed_decisive_coverage` 优于 Random(105 个新种子合并,四个块 Holm<0.01);更温和的 lr 0.0003 下图正则化设计与不用图的 `vopt_u`、以及打乱图对照都优于 Random,图本身没有可测的贡献。图的类型后验比"只用类型"的 decisive 预测多出一点 AUC(+0.009 到 +0.013)。
- `new_directions` 的关键发现:把参考图排序损失的权重 `aw` 从 0.25 提到 8(`aw8`,或把全部 144 张理想图作锚点的 `aw8_all`)比换选择规则收益更大(AUC +0.012 到 +0.024,35 个新种子预注册确认),并且**吃掉了选择规则的优势**:在 `aw8` 学习器下 `vopt_u` 不再显著优于 Random。HTR(实验室最关心的类型)没有确认的增益。以后任何"方法优于基线"的说法,比较对象必须是调好的学习器(`aw8_all`),不是旧默认头。
- 其他没用的方向(别再试):类型耦合、贝叶斯头、时间对比适配器(有害)、tie/not_apply 建模、伪标签自训练、文件名元数据、镜像先验、混合锚点。

## 硬性规则
- 协议不动:judgment-unit 协议(一次查询 = 一个 (pair, type) 判断;预算 10/20/40/60 个判断;划分 A 和 B;留出配对组的偏好预测终点:log-loss、AUC、准确率;Random = 多次抽样平均)。允许改动的东西必须在各 prompt 里明确写出,并在 PREREGISTRATION 里标注"协议扩展"。
- 种子:**已用过**的有 42,79,123,202,303、400–459、500–534、600–629、700–1739、2000–2019、3000–6034;你的种子段见各 prompt。在 `new_directions/nd_core.py` 的 `USED_SEEDS_BY_EARLIER_WORK` 里加入你的段,并保持断言通过。
- 流程:开发种子(≤20 个)只用来设计和筛选;至多冻结 4 个候选写进日志并提交,**之后才**跑确认种子;确认只跑一次;Holm 校正写明家族;与对照(打乱图、只用类型、随机)的配对差值标注"未校正"。失败如实写,列出试过的所有变体。图相关的说法必须有"打乱图"对照。
- 身份安全:验证/测试组的图像、外部测试理想图不能进入图的节点、标注或选择器能看到的数据;参考图(理想图)是所有策略共有的训练锚点,可以用。
- 学习器类对比用相同的随机标注集配对(见 `nd_core.py`);选择规则对比用相同的初始模型与候选池。
- 运行方式:4 个 CPU 核,每个进程 1 个 torch 线程(`OMP_NUM_THREADS=1`);**容器重启会杀掉后台进程**——所有长任务写成幂等的重启脚本(已完成的 cell 跳过),每次回来先 `uptime` 和 `ps` 检查;前台用不超过 9.5 分钟的等待循环(不要用 sleep 链),等待时读文献、写文档;`git add` 只加具体路径;旧测试会改写 `results/active_learning_v1.8_seed42/manifests`,提交前用 `git checkout -- ` 恢复;不要编辑正在运行的 shell 脚本;结果文件每隔几分钟提交推送(仓库的 Stop hook 要求工作区干净)。
- 文献:`WebSearch` 可用,arXiv/CVF/ar5iv 被代理拦截(GitHub raw 可读);只写搜索结果明确说过的内容,标注"(摘要)"。
- 对用户:中文、讲人话、不夸大;发现和假设分开;每次汇报包含"哪些是预先登记的确认,哪些只是开发观察"。
- 交付物:(1) 代码 + 单元测试;(2) 所有表格由脚本从结果文件生成(不手抄);(3) 一份独立的结果文档(见各 prompt),**不要改** `code/ACADEMIC_REPORT_DRAFT.md`(两个会话并行,避免冲突,之后由用户统一整合);(4) 一页中文摘要;(5) 给下一个 Claude 的 handoff;(6) 需要 GPU 的任务清单留给用户。
- 时间表(8 小时):0–0.5h 读材料、搭环境、重建特征缓存、跑测试;0.5–1.5h 文献与设计,写 PREREGISTRATION_0(开发方案);1.5–4h 开发筛选(后台跑,前台读文献);4h 冻结候选并提交预注册;4–7h 确认实验(后台),前台写文档和做机制分析;7–8h 汇总、测试、推送、写 handoff。每完成一个阶段更新日志并推送。
