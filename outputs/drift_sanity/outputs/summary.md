# Drift sanity checks summary

## Placebo modes

**Placebo A (2022 vs 2023):** kept for context (early shift inside train).
Not a *true* placebo for **level_3D** because price levels genuinely changed between 2022 and 2023.

**Placebo B (true placebo):** within-year random halves (2024 and 2025).
Under the null, significant fractions should be near ~5% (p<0.05) and ~1% (p<0.01).

**Placebo C (true placebo):** within-year time split Jan–Jun vs Jul–Dec (2024 and 2025).
This is stricter and probes mild temporal dependence; it can naturally be a bit more sensitive.

## True placebo summary (B/C only)

| placebo_type                 | feature_space   | stat     |   n_tests |   frac_p_lt_0p05 |   frac_p_lt_0p01 |
|:-----------------------------|:----------------|:---------|----------:|-----------------:|-----------------:|
| placebo_B_random_halves_2024 | level_3D        | p_energy |         6 |         0        |         0        |
| placebo_B_random_halves_2024 | level_3D        | p_mmd2   |         6 |         0        |         0        |
| placebo_B_random_halves_2024 | shape_24D       | p_energy |         6 |         0        |         0        |
| placebo_B_random_halves_2024 | shape_24D       | p_mmd2   |         6 |         0        |         0        |
| placebo_B_random_halves_2025 | level_3D        | p_energy |         6 |         0        |         0        |
| placebo_B_random_halves_2025 | level_3D        | p_mmd2   |         6 |         0        |         0        |
| placebo_B_random_halves_2025 | shape_24D       | p_energy |         6 |         0        |         0        |
| placebo_B_random_halves_2025 | shape_24D       | p_mmd2   |         6 |         0        |         0        |
| placebo_C_time_halves_2024   | level_3D        | p_energy |         6 |         0.5      |         0.333333 |
| placebo_C_time_halves_2024   | level_3D        | p_mmd2   |         6 |         0.666667 |         0.333333 |
| placebo_C_time_halves_2024   | shape_24D       | p_energy |         6 |         0.166667 |         0        |
| placebo_C_time_halves_2024   | shape_24D       | p_mmd2   |         6 |         0.166667 |         0        |
| placebo_C_time_halves_2025   | level_3D        | p_energy |         5 |         0.8      |         0.4      |
| placebo_C_time_halves_2025   | level_3D        | p_mmd2   |         5 |         0.8      |         0.6      |
| placebo_C_time_halves_2025   | shape_24D       | p_energy |         5 |         0.4      |         0.2      |
| placebo_C_time_halves_2025   | shape_24D       | p_mmd2   |         5 |         0.4      |         0.2      |

## Red-flag heuristics

If Placebo B/C shows very high significant fractions (e.g., >20% at p<0.01), that can indicate:
- time dependence not handled by iid permutation,
- preprocessing mismatch between groups,
- DST alignment / missing-hour artifacts,
- or a test that’s too sensitive for your dependence structure.

## Year-by-year drift (iid permutation)

| class_id          | comparison   | feature_space   |   p_energy |     p_mmd2 | p_energy_sig_0.01   | p_mmd2_sig_0.01   |
|:------------------|:-------------|:----------------|-----------:|-----------:|:--------------------|:------------------|
| weekday__shoulder | 2022_vs_2023 | level_3D        | 0.00199601 | 0.00199601 | True                | True              |
| weekday__summer   | 2022_vs_2023 | level_3D        | 0.00199601 | 0.00199601 | True                | True              |
| weekday__winter   | 2022_vs_2023 | level_3D        | 0.00199601 | 0.00199601 | True                | True              |
| weekend__shoulder | 2022_vs_2023 | level_3D        | 0.00199601 | 0.00199601 | True                | True              |
| weekend__summer   | 2022_vs_2023 | level_3D        | 0.00199601 | 0.00199601 | True                | True              |
| weekend__winter   | 2022_vs_2023 | level_3D        | 0.00199601 | 0.00199601 | True                | True              |
| weekday__shoulder | 2023_vs_2024 | level_3D        | 0.00199601 | 0.00199601 | True                | True              |
| weekday__summer   | 2023_vs_2024 | level_3D        | 0.00199601 | 0.00199601 | True                | True              |
| weekday__winter   | 2023_vs_2024 | level_3D        | 0.00199601 | 0.00199601 | True                | True              |
| weekend__shoulder | 2023_vs_2024 | level_3D        | 0.00199601 | 0.00399202 | True                | True              |
| weekend__summer   | 2023_vs_2024 | level_3D        | 0.0199601  | 0.0299401  | False               | False             |
| weekend__winter   | 2023_vs_2024 | level_3D        | 0.00598802 | 0.00399202 | True                | True              |
| weekday__shoulder | 2024_vs_2025 | level_3D        | 0.00798403 | 0.00399202 | True                | True              |
| weekday__summer   | 2024_vs_2025 | level_3D        | 0.516966   | 0.550898   | False               | False             |
| weekday__winter   | 2024_vs_2025 | level_3D        | 0.00199601 | 0.00199601 | True                | True              |
| weekend__shoulder | 2024_vs_2025 | level_3D        | 0.48503    | 0.329341   | False               | False             |
| weekend__summer   | 2024_vs_2025 | level_3D        | 0.0419162  | 0.0678643  | False               | False             |
| weekend__winter   | 2024_vs_2025 | level_3D        | 0.0199601  | 0.0339321  | False               | False             |
| weekday__shoulder | 2022_vs_2023 | shape_24D       | 0.157685   | 0.177645   | False               | False             |
| weekday__summer   | 2022_vs_2023 | shape_24D       | 0.0578842  | 0.0778443  | False               | False             |
| weekday__winter   | 2022_vs_2023 | shape_24D       | 0.650699   | 0.712575   | False               | False             |
| weekend__shoulder | 2022_vs_2023 | shape_24D       | 0.874251   | 0.916168   | False               | False             |
| weekend__summer   | 2022_vs_2023 | shape_24D       | 0.708583   | 0.720559   | False               | False             |
| weekend__winter   | 2022_vs_2023 | shape_24D       | 0.634731   | 0.592814   | False               | False             |
| weekday__shoulder | 2023_vs_2024 | shape_24D       | 0.0299401  | 0.0319361  | False               | False             |
| weekday__summer   | 2023_vs_2024 | shape_24D       | 0.00199601 | 0.00199601 | True                | True              |
| weekday__winter   | 2023_vs_2024 | shape_24D       | 0.467066   | 0.572854   | False               | False             |
| weekend__shoulder | 2023_vs_2024 | shape_24D       | 0.159681   | 0.167665   | False               | False             |
| weekend__summer   | 2023_vs_2024 | shape_24D       | 0.429142   | 0.417166   | False               | False             |
| weekend__winter   | 2023_vs_2024 | shape_24D       | 0.141717   | 0.153693   | False               | False             |
| weekday__shoulder | 2024_vs_2025 | shape_24D       | 0.0399202  | 0.0439122  | False               | False             |
| weekday__summer   | 2024_vs_2025 | shape_24D       | 0.167665   | 0.143713   | False               | False             |
| weekday__winter   | 2024_vs_2025 | shape_24D       | 0.241517   | 0.251497   | False               | False             |
| weekend__shoulder | 2024_vs_2025 | shape_24D       | 0.816367   | 0.832335   | False               | False             |
| weekend__summer   | 2024_vs_2025 | shape_24D       | 0.652695   | 0.668663   | False               | False             |
| weekend__winter   | 2024_vs_2025 | shape_24D       | 0.704591   | 0.748503   | False               | False             |

## Block permutation sensitivity

We flag cases where **iid** reports strong significance (p<0.01) but **biweek** or **month** loses significance (p>0.05).
| class_id          | comparison   | feature_space   | stat     |      p_iid |   p_biweek |   p_month | flag_reason                                       |
|:------------------|:-------------|:----------------|:---------|-----------:|-----------:|----------:|:--------------------------------------------------|
| weekday__shoulder | 2023_vs_2024 | level_3D        | p_energy | 0.00199601 | 0.011976   | 0.0538922 | iid<0.01 but month>0.05                           |
| weekday__shoulder | 2024_vs_2025 | level_3D        | p_energy | 0.00998004 | 0.0938124  | 0.11976   | iid<0.01 but biweek>0.05; iid<0.01 but month>0.05 |
| weekday__summer   | 2022_vs_2023 | level_3D        | p_energy | 0.00199601 | 0.00199601 | 0.103792  | iid<0.01 but month>0.05                           |
| weekday__summer   | 2023_vs_2024 | level_3D        | p_energy | 0.00199601 | 0.00399202 | 0.0918164 | iid<0.01 but month>0.05                           |
| weekday__summer   | 2023_vs_2024 | shape_24D       | p_energy | 0.00199601 | 0.0379242  | 0.0518962 | iid<0.01 but month>0.05                           |
| weekday__winter   | 2023_vs_2024 | level_3D        | p_energy | 0.00199601 | 0.0698603  | 0.171657  | iid<0.01 but biweek>0.05; iid<0.01 but month>0.05 |
| weekday__winter   | 2024_vs_2025 | level_3D        | p_energy | 0.00199601 | 0.0558882  | 0.225549  | iid<0.01 but biweek>0.05; iid<0.01 but month>0.05 |
| weekend__shoulder | 2023_vs_2024 | level_3D        | p_energy | 0.00199601 | 0.0479042  | 0.125749  | iid<0.01 but month>0.05                           |
| weekend__summer   | 2022_vs_2023 | level_3D        | p_energy | 0.00199601 | 0.00199601 | 0.0958084 | iid<0.01 but month>0.05                           |
| weekend__winter   | 2023_vs_2024 | level_3D        | p_energy | 0.00998004 | 0.0638723  | 0.0738523 | iid<0.01 but biweek>0.05; iid<0.01 but month>0.05 |
| weekend__winter   | 2024_vs_2025 | level_3D        | p_energy | 0.00399202 | 0.0678643  | 0.10978   | iid<0.01 but biweek>0.05; iid<0.01 but month>0.05 |
| weekday__shoulder | 2024_vs_2025 | level_3D        | p_mmd2   | 0.00399202 | 0.131737   | 0.165669  | iid<0.01 but biweek>0.05; iid<0.01 but month>0.05 |
| weekday__summer   | 2022_vs_2023 | level_3D        | p_mmd2   | 0.00199601 | 0.00199601 | 0.0598802 | iid<0.01 but month>0.05                           |
| weekday__summer   | 2023_vs_2024 | level_3D        | p_mmd2   | 0.00199601 | 0.00399202 | 0.0918164 | iid<0.01 but month>0.05                           |
| weekday__summer   | 2023_vs_2024 | shape_24D       | p_mmd2   | 0.00199601 | 0.0359281  | 0.0518962 | iid<0.01 but month>0.05                           |
| weekday__winter   | 2023_vs_2024 | level_3D        | p_mmd2   | 0.00199601 | 0.0578842  | 0.169661  | iid<0.01 but biweek>0.05; iid<0.01 but month>0.05 |
| weekday__winter   | 2024_vs_2025 | level_3D        | p_mmd2   | 0.00199601 | 0.0698603  | 0.261477  | iid<0.01 but biweek>0.05; iid<0.01 but month>0.05 |
| weekend__shoulder | 2023_vs_2024 | level_3D        | p_mmd2   | 0.00199601 | 0.0319361  | 0.117764  | iid<0.01 but month>0.05                           |
| weekend__summer   | 2022_vs_2023 | level_3D        | p_mmd2   | 0.00199601 | 0.00199601 | 0.0958084 | iid<0.01 but month>0.05                           |
| weekend__winter   | 2023_vs_2024 | level_3D        | p_mmd2   | 0.00598802 | 0.0459082  | 0.0758483 | iid<0.01 but month>0.05                           |

## Interpretation guide

### How to read Placebo A vs true placebo B/C

- **Placebo A (2022 vs 2023)** is useful context for *within-train* changes, but it is **not a null** for level drift.
- **Placebo B/C** are the actual checks of false positive rate under (approximately) stationary conditions.

### What placebo red flags mean

If true placebos (B/C) show too many significant results, likely causes include:
- pipeline artifacts (different preprocessing between subsets),
- time alignment/DST errors causing systematic shape differences,
- dependence (days not i.i.d.) making iid permutation too optimistic,
- or hidden conditioning leakage.

### What to do next if you see red flags

- Increase permutations to 2000 for more stable p-values (and a smaller p-value floor).
- Prefer **block permutations** (biweek/month) for inference if dependence is strong.
- Add a negative-control drift test within a single year using multiple random splits.
- Verify DST handling: missing hours, duplicated hours, day boundary alignment.
- Check if 2025 downsampling or timestamp parsing differs from earlier years.
