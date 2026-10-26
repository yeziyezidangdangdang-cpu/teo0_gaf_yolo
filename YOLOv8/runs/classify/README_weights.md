# CWRU 双分支 (GASF + GADF) 训练输出说明

## 为什么会出现 gasf5、且 gasf 和 gadf 数量不对齐？

**之前的行为（已改）：**  
`train_cwru_dual.py` 每次运行会先写 GASF（生成 `cwru_gasf` 或递增为 `cwru_gasf2`、`cwru_gasf3`…），再写 GADF。若在 **GASF 跑完、GADF 还没跑完** 时中断（关机、断电、手动停），就会多出一个 gasf 目录而没有对应的 gadf，所以会出现 **5 个 gasf + 4 个 gadf** 这类不对齐。

**现在的行为（已修复）：**  
脚本里已加上 **exist_ok=True**：每次完整运行都会**覆盖**同一对目录 `cwru_gasf` 和 `cwru_gadf`，不再自动递增。因此：
- **之后**每次完整跑完，目录里**只会有一对**：`cwru_gasf`、`cwru_gadf`，数量对齐。
- 你当前看到的 `cwru_gasf5`、`cwru_gadf4` 等是**之前**多次运行/中断留下的历史目录，可以手动删除；删除后只保留 `cwru_gasf` 和 `cwru_gadf` 即可。

## 评估和推理应使用哪一组权重？

双分支评估/推理必须使用 **同一轮** 的 GASF + GADF 权重（同一“配对”）：

| run | GASF 权重目录 | GADF 权重目录 | 说明 |
|-----|----------------|----------------|------|
| 1（默认） | cwru_gasf  | cwru_gadf  | **固定一对**：训练脚本已开 exist_ok，每次完整运行会覆盖此对，推荐用这组 |
| 2 | cwru_gasf2 | cwru_gadf2 | 历史第 2 轮 |
| 3 | cwru_gasf3 | cwru_gadf3 | 历史第 3 轮 |
| … | … | … | 若某轮只跑完 GASF 就中断，则无对应 gadf，不能单独用于双分支 |

**推荐**：默认 `--run 1`（即 cwru_gasf + cwru_gadf）。下次完整跑完一次 `train_cwru_dual.py` 后，评估/预测不写 `--run` 即会用最新结果。若要用某次历史编号，可加 `--run 2`、`--run 3` 等。

---

## 输出指标说明（是否正确）

| 脚本 | 输出 | 计算方式 | 说明 |
|------|------|----------|------|
| **train_cwru_dual** | 单分支目录 `cwru_gasf*` / `cwru_gadf*` | Ultralytics 自带：每个目录内有 `results.csv`、`weights/best.pt`、混淆矩阵等 | 正确。单分支验证集准确率见 `results.csv` 的 `metrics/accuracy_top1`。 |
| **eval_cwru_dual** | 总体准确率、每类准确率、4×4 混淆矩阵 | 总体 = 对角线之和 / 总样本数；每类 = 该类正确数 / 该类真实总数；混淆矩阵行=真实、列=预测 | 正确。双分支融合为两模型概率平均后 argmax。 |
| **predict_cwru_dual** | 预测类别、置信度、各类概率 | 两模型 softmax 输出做平均，argmax 为类别，对应概率为置信度 | 正确。 |

结论：当前各脚本的指标定义与实现一致，无需改计算方式；只需保证评估/预测使用同一组配对权重（如默认的 run=3）。
