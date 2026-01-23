# Drift sanity checks summary

- Seed: 123

- Permutations: 500

- Min n per group: 10


## (1) Placebo / Negative Control

### shape_24D

- Energy: p<0.05: 0.083, p<0.01: 0.000

- MMD^2:  p<0.05: 0.000, p<0.01: 0.000


### level_3D

- Energy: p<0.05: 0.583, p<0.01: 0.583

- MMD^2:  p<0.05: 0.583, p<0.01: 0.583


Expected under null: ~5% and ~1% (roughly). If far larger across many classes, suspect preprocessing artifacts or strong dependence.

## (2) Year-by-year drift

### shape_24D

Top 10 (Energy) smallest p-values:


| class_id          | comparison        |   n_a |   n_b |   stat_energy |   p_energy |
|:------------------|:------------------|------:|------:|--------------:|-----------:|
| weekday__summer   | year_2023_vs_2024 |    66 |    65 |     0.319137  | 0.00199601 |
| weekday__summer   | year_2022_vs_2023 |    66 |    66 |     0.0887042 | 0.0299401  |
| weekday__shoulder | year_2024_vs_2025 |   110 |   110 |     0.0716781 | 0.0299401  |
| weekday__shoulder | year_2023_vs_2024 |   109 |   110 |     0.0767563 | 0.0459082  |
| weekday__summer   | year_2024_vs_2025 |    65 |    65 |     0.0387787 | 0.149701   |
| weekday__shoulder | year_2022_vs_2023 |   109 |   109 |     0.0245624 | 0.163673   |
| weekend__winter   | year_2023_vs_2024 |    35 |    34 |     0.094459  | 0.169661   |
| weekend__shoulder | year_2023_vs_2024 |    42 |    41 |     0.0597168 | 0.183633   |
| weekday__winter   | year_2024_vs_2025 |    87 |    86 |     0.0235593 | 0.199601   |
| weekend__summer   | year_2023_vs_2024 |    26 |    27 |    -0.0236311 | 0.44511    |


Top 10 (MMD^2) smallest p-values:


| class_id          | comparison        |   n_a |   n_b |   stat_mmd2 |     p_mmd2 |
|:------------------|:------------------|------:|------:|------------:|-----------:|
| weekday__summer   | year_2023_vs_2024 |    66 |    65 |  0.04983    | 0.00199601 |
| weekday__shoulder | year_2024_vs_2025 |   110 |   110 |  0.00867495 | 0.0359281  |
| weekday__summer   | year_2022_vs_2023 |    66 |    66 |  0.0104425  | 0.0499002  |
| weekday__shoulder | year_2023_vs_2024 |   109 |   110 |  0.00895704 | 0.0538922  |
| weekday__summer   | year_2024_vs_2025 |    65 |    65 |  0.00670893 | 0.125749   |
| weekend__winter   | year_2023_vs_2024 |    35 |    34 |  0.0103737  | 0.183633   |
| weekend__shoulder | year_2023_vs_2024 |    42 |    41 |  0.00729211 | 0.191617   |
| weekday__shoulder | year_2022_vs_2023 |   109 |   109 |  0.00244545 | 0.199601   |
| weekday__winter   | year_2024_vs_2025 |    87 |    86 |  0.00267846 | 0.209581   |
| weekend__summer   | year_2023_vs_2024 |    26 |    27 | -0.00250149 | 0.42515    |


### level_3D

Top 10 (Energy) smallest p-values:


| class_id          | comparison        |   n_a |   n_b |   stat_energy |   p_energy |
|:------------------|:------------------|------:|------:|--------------:|-----------:|
| weekday__summer   | year_2022_vs_2023 |    66 |    66 |     442.145   | 0.00199601 |
| weekend__summer   | year_2022_vs_2023 |    26 |    26 |     428.345   | 0.00199601 |
| weekday__shoulder | year_2022_vs_2023 |   109 |   109 |     223.656   | 0.00199601 |
| weekday__winter   | year_2022_vs_2023 |    85 |    85 |     174.057   | 0.00199601 |
| weekend__shoulder | year_2022_vs_2023 |    42 |    42 |     173.03    | 0.00199601 |
| weekend__winter   | year_2022_vs_2023 |    35 |    35 |     170.525   | 0.00199601 |
| weekday__winter   | year_2023_vs_2024 |    85 |    87 |      13.6203  | 0.00199601 |
| weekday__summer   | year_2023_vs_2024 |    66 |    65 |      11.4052  | 0.00199601 |
| weekday__shoulder | year_2023_vs_2024 |   109 |   110 |       6.34393 | 0.00199601 |
| weekend__shoulder | year_2023_vs_2024 |    42 |    41 |      10.6115  | 0.00399202 |


Top 10 (MMD^2) smallest p-values:


| class_id          | comparison        |   n_a |   n_b |   stat_mmd2 |     p_mmd2 |
|:------------------|:------------------|------:|------:|------------:|-----------:|
| weekend__summer   | year_2022_vs_2023 |    26 |    26 |   0.825298  | 0.00199601 |
| weekday__summer   | year_2022_vs_2023 |    66 |    66 |   0.77389   | 0.00199601 |
| weekday__shoulder | year_2022_vs_2023 |   109 |   109 |   0.616931  | 0.00199601 |
| weekend__winter   | year_2022_vs_2023 |    35 |    35 |   0.529337  | 0.00199601 |
| weekend__shoulder | year_2022_vs_2023 |    42 |    42 |   0.500338  | 0.00199601 |
| weekday__winter   | year_2022_vs_2023 |    85 |    85 |   0.460913  | 0.00199601 |
| weekday__winter   | year_2023_vs_2024 |    85 |    87 |   0.10661   | 0.00199601 |
| weekday__summer   | year_2023_vs_2024 |    66 |    65 |   0.101696  | 0.00199601 |
| weekday__shoulder | year_2023_vs_2024 |   109 |   110 |   0.0617472 | 0.00199601 |
| weekday__winter   | year_2024_vs_2025 |    87 |    86 |   0.0497405 | 0.00199601 |


## (3) Block permutation vs i.i.d permutation

Flagged cases where iid p<0.01 but blockperm p>0.05 (dependence may explain some significance):


| class_id          | comparison                  |        iid |     month |       week | feature_space   | stat     | block_method   |
|:------------------|:----------------------------|-----------:|----------:|-----------:|:----------------|:---------|:---------------|
| weekday__shoulder | blockperm_year_2024_vs_2025 | 0.00399202 | 0.0858283 | 0.0499002  | level_3D        | p_energy | month          |
| weekday__summer   | blockperm_year_2022_vs_2023 | 0.00199601 | 0.10978   | 0.00199601 | level_3D        | p_energy | month          |
| weekday__summer   | blockperm_year_2023_vs_2024 | 0.00199601 | 0.0598802 | 0.00199601 | level_3D        | p_energy | month          |
| weekday__winter   | blockperm_year_2023_vs_2024 | 0.00199601 | 0.159681  | 0.0319361  | level_3D        | p_energy | month          |
| weekday__winter   | blockperm_year_2024_vs_2025 | 0.00199601 | 0.177645  | 0.0399202  | level_3D        | p_energy | month          |
| weekend__shoulder | blockperm_year_2023_vs_2024 | 0.00399202 | 0.117764  | 0.011976   | level_3D        | p_energy | month          |
| weekend__summer   | blockperm_year_2022_vs_2023 | 0.00199601 | 0.0958084 | 0.00199601 | level_3D        | p_energy | month          |
| weekend__winter   | blockperm_year_2023_vs_2024 | 0.00798403 | 0.0698603 | 0.0359281  | level_3D        | p_energy | month          |
| weekend__winter   | blockperm_year_2024_vs_2025 | 0.00998004 | 0.115768  | 0.0319361  | level_3D        | p_energy | month          |
| weekday__shoulder | blockperm_year_2024_vs_2025 | 0.00598802 | 0.147705  | 0.0578842  | level_3D        | p_mmd2   | week           |
| weekday__shoulder | blockperm_year_2024_vs_2025 | 0.00598802 | 0.147705  | 0.0578842  | level_3D        | p_mmd2   | month          |
| weekday__summer   | blockperm_year_2022_vs_2023 | 0.00199601 | 0.0538922 | 0.00199601 | level_3D        | p_mmd2   | month          |
| weekday__summer   | blockperm_year_2023_vs_2024 | 0.00199601 | 0.0518962 | 0.00199601 | level_3D        | p_mmd2   | month          |
| weekday__winter   | blockperm_year_2023_vs_2024 | 0.00199601 | 0.163673  | 0.0179641  | level_3D        | p_mmd2   | month          |
| weekday__winter   | blockperm_year_2024_vs_2025 | 0.00199601 | 0.227545  | 0.0419162  | level_3D        | p_mmd2   | month          |
| weekend__shoulder | blockperm_year_2023_vs_2024 | 0.00199601 | 0.0938124 | 0.00798403 | level_3D        | p_mmd2   | month          |
| weekend__summer   | blockperm_year_2022_vs_2023 | 0.00199601 | 0.0978044 | 0.00199601 | level_3D        | p_mmd2   | month          |
| weekend__winter   | blockperm_year_2023_vs_2024 | 0.00399202 | 0.0538922 | 0.0259481  | level_3D        | p_mmd2   | month          |


## Interpretation guide

- **Placebo red flag**: if placebo tests show a very high fraction of p<0.01 across many classes, suspect pipeline artifacts (time alignment, DST handling, differing preprocessing by year) or strong dependence.
- **Year-by-year**: if drift jumps sharply at one transition (e.g., 2022→2023 strong, others weak), that suggests a regime shift rather than gradual drift.
- **Block permutation**: if iid permutation shows p<<0.01 but weekly/monthly block permutation becomes non-significant, some of the iid significance may be driven by temporal dependence.
- **Next steps**: increase permutations (e.g., 2000), add block sizes (14-day), verify timestamp alignment & DST, and run a placebo that splits within the same year.
