# 证据包（Evidence）

本目录存放 README 中引用的可视化证据，全部由脚本生成、可复现。

| 文件 | 说明 | 生成方式 |
|---|---|---|
| `architecture.png` | 三段式架构图：离线索引 → 在线检索 → 输出与约束 | `python scripts/make_evidence.py` |
| `metrics.png` | 示例语料离线评测结果（Recall@3 / 拒答准确率 / 耗时） | 同上（读取 `eval/results/metrics.json`） |
| `terminal-demo.gif` | 真实运行截图：两次命中生成 + 一次证据不足拒答 | 同上（调用 `app.demo_engine` 实跑） |

## 为什么放图

面试官不会读 500 行 README。**一张架构图 + 一段真实运行输出，比十段文字更能说明"这东西真的跑起来了"。**

## 复现

```bash
python scripts/eval_demo.py     # 先产出 eval/results/metrics.json
python scripts/make_evidence.py # 再生成三张图
```

> 图中所有数字都来自 `eval/results/metrics.json`，不存在手写数字。
