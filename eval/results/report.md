# 离线评测报告（示例语料）

- 生成时间：`2026-11-09T08:40:25+00:00`
- 语料切块：16；评测集：23 题（事实题 19 / 拒答题 4）
- 检索通道：idf 加权词面重叠（无向量库、无大模型）
- 证据阈值：0.35

| 指标 | 数值 |
|---|---|
| Recall@3 | **94.74%** |
| 拒答准确率 | **100.00%** |
| 平均耗时 | 1.59 ms |
| 通过题数 | 22 / 23 |

## 逐题结果

| ID | 类型 | 问题 | 命中 | 最高分 | 结果 |
|---|---|---|---|---|---|
| q01 | factual | 稳盈货币基金A的起购金额是多少？ | 01_fund#0, 04_service#0, 04_service#1 | 0.646 | ✅ |
| q02 | factual | 货币基金赎回资金多久到账？ | 01_fund#0, 04_service#0, 03_wealth#1 | 0.560 | ✅ |
| q03 | factual | 交易日15点之后提交的申购什么时候确认？ | 01_fund#1, 01_fund#0, 01_fund#2 | 0.606 | ✅ |
| q04 | factual | 这只基金的管理费是多少？ | 01_fund#2, 03_wealth#3, 01_fund#0 | 1.000 | ✅ |
| q05 | factual | 货币基金保本吗？ | 03_wealth#0, 01_fund#0, 04_service#0 | 0.606 | ❌ |
| q06 | factual | 产品风险等级分为几级？ | 02_suitability#0, 02_suitability#2, 02_suitability#3 | 0.628 | ✅ |
| q07 | factual | 投资者可以购买高于自身风险等级的产品吗？ | 02_suitability#0, 02_suitability#2, 02_suitability#3 | 1.000 | ✅ |
| q08 | factual | 风险测评结果的有效期是多久？ | 02_suitability#1, 02_suitability#0, 01_fund#3 | 0.677 | ✅ |
| q09 | factual | 冷静期是几个小时？ | 02_suitability#2 | 0.408 | ✅ |
| q10 | factual | 65周岁以上投资者购买高风险产品有什么要求？ | 02_suitability#3, 02_suitability#0, 02_suitability#2 | 0.638 | ✅ |
| q11 | factual | 安享90天的业绩比较基准是多少？ | 03_wealth#0, 03_wealth#3, 03_wealth#1 | 1.000 | ✅ |
| q12 | factual | 定期开放产品多久开放一次？ | 03_wealth#1, 04_service#0, 03_wealth#0 | 0.470 | ✅ |
| q13 | factual | 封闭期内可以赎回吗？ | 03_wealth#1, 02_suitability#2, 01_fund#0 | 1.000 | ✅ |
| q14 | factual | 产品多久披露一次运作报告？ | 03_wealth#2, 03_wealth#1, 01_fund#0 | 0.537 | ✅ |
| q15 | factual | 超额业绩报酬怎么计提？ | 03_wealth#3, 01_fund#2, 01_fund#3 | 1.000 | ✅ |
| q16 | factual | 赎回资金一直没有到账怎么办？ | 01_fund#0, 03_wealth#1, 04_service#0 | 0.216 | ✅ |
| q17 | factual | 七日年化收益率是什么意思？ | 01_fund#0, 04_service#1, 03_wealth#0 | 0.590 | ✅ |
| q18 | factual | 投诉多久会被受理？ | 04_service#2 | 0.579 | ✅ |
| q19 | factual | 密码连续输错会有什么后果？ | 04_service#3 | 0.408 | ✅ |
| q20 | refusal | 信用卡年费是多少？ | - | 0.000 | ✅ |
| q21 | refusal | 股票开户佣金是多少？ | - | 0.000 | ✅ |
| q22 | refusal | 房贷利率现在多少？ | - | 0.000 | ✅ |
| q23 | refusal | 基金分红需要缴多少税？ | 01_fund#0, 01_fund#2, 01_fund#3 | 0.309 | ✅ |

> 说明：本报告基于 `demo/corpus/` 的**示例语料**，不代表生产语料上的效果；生产语料评测见 `docs/baseline/` 下的交付审计。
