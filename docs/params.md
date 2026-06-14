我将分两步说明：1) 列出脚本里通用参数及它们在命令行中的映射；2) 指出两个脚本常见的局部覆盖与差异。目标文件：run_experiments.py 和 test_experiments.py。

**参数 — 全局与命令行映射**  
- **`benchmark`**: 要跑的工作负载（例如 tpcc、ycsba/ycsbf 等）。对应 CLI `-b`.  
- **`database`**: 数据库/数据集名字（`epic` 或 gacco）。对应 CLI `-d`.  
- **`num_warehouses`**: 仓库/分区数量（TPCC 的规模参数）。对应 CLI `-w`.  
- **`skew_factor`**: 键访问偏斜程度（0.0 均匀 → 越大越集中）。对应 CLI `-a`.  
- **`fullread`**: 是否启用全表/全字段读取（字符串 `"true"`/`"false"`）。对应 CLI `-r`.  
- **`cpu_exec_num_threads`**: CPU 执行线程数（用于 CPU 执行时线程并发）。对应 CLI `-c`.  
- **`num_epochs`**: 运行的 epoch/批次数（高层循环次数）。对应 CLI `-e`.  
- **`num_txns`**: 每次实验/一个 epoch 中提交的事务数（决定工作量大小）。对应 CLI `-s`.  
- **`split_fields`**: 是否按字段拆分存储/访问（字符串）。对应 CLI `-f`.  
- **`commutative_ops`**: 是否启用可交换操作优化（字符串）。对应 CLI `-m`.  
- **`num_records`**: 表中记录数（数据规模）。对应 CLI `-n`.  
- **`exec_device`**: 执行设备，通常 `"gpu"` 或 `"cpu"`。对应 CLI `-x`.  
- **`num_repeat`**: 每个参数组合重复运行次数（用于取平均/稳定性）。不是 CLI 参数，控制脚本循环。  
- **`epic_driver_path` / `epic_micro_driver_path`**: 可执行二进制路径。  
- **`output_path` / output/err templates**: 输出文件夹与输出/错误文件命名模板（脚本会把 stdout/（test 脚本中）stderr 写入文件）。  
- **`cmd_template` / `micro_cmd_template`**: 拼接命令的格式字符串；`micro_cmd_template` 在末尾多一个 `-p`，对应 `abort_rate`（microbench 的中止率）。

**脚本内部常见局部覆盖与范围设置**  
- 各个实验函数（如 `epic_tpcc_experiment`, `gacco_ycsb_experiment` 等）会局部设置或覆盖一些参数：例如 `database`、`benchmark`、`exec_device`、`num_txns`、`skew_factor`、`split_fields`、`abort_rate` 等。  
- 这些函数通过循环不同取值（例如 `for skew_factor in [...]`、`for num_warehouses in [...]`）构成参数网格，交替调用 `cmd_template.format(...)` 执行实际命令。

**run_experiments.py vs test_experiments.py 的参数差异要点**  
- **范围大小**: run_experiments.py 定义并遍历大量取值（多个 benchmarks、多个 `skew_factor`、长的 `num_txns` 列表、多个 `num_warehouses` 等）。  
- **重复次数**: run_experiments.py 默认 `num_repeat = 3`，test_experiments.py 默认 `num_repeat = 1`。  
- **stderr 处理**: test_experiments.py 额外捕获并写入 `.err` 文件；run_experiments.py 只保存 `stdout`。  
- **microbench**: run_experiments.py 循环多个 `abort_rate` 值（0..50）；test_experiments.py 通常只用单个测试值（例如 50）。  
- **快速/验收目的**: test_experiments.py 仅保留少量/边界或典型值（便于快速 smoke test）；run_experiments.py 用于全面长跑基准测试。

---

# 逐函数参数网格对照 (run_experiments.py vs test_experiments.py)

## 全局默认值差异

| 参数 | CLI | run | test | 说明 |
|------|-----|-----|------|------|
| `benchmark` | `-b` | `"tpcc"` | `"tpcc"` | |
| `database` | `-d` | `"epic"` | `"epic"` | |
| `num_warehouses` | `-w` | `1` | `1` | |
| `skew_factor` | `-a` | `0.0` | `0.0` | |
| `fullread` | `-r` | `"true"` | `"true"` | |
| `cpu_exec_num_threads` | `-c` | `32` | `32` | |
| `num_epochs` | `-e` | `5` | `5` | |
| `num_txns` | `-s` | `100000` | `100000` | |
| `split_fields` | `-f` | `"true"` | `"true"` | |
| `commutative_ops` | `-m` | `"false"` | `"false"` | |
| `num_records` | `-n` | `10000000` | `10000000` | |
| `exec_device` | `-x` | `"gpu"` | `"gpu"` | |
| **`num_repeat`** | — | **3** | **1** | ⚠️ 唯一不同的默认值 |

**stderr 处理**: test 比 run 多了 `import sys`，额外定义了 `err_file_template` / `micro_err_file_template`，会把 stderr 写入独立 `.err` 文件；run 只保存 stdout。

---

## epic_ycsb_experiment (epic GPU YCSB)

| 参数 | run | test |
|------|-----|------|
| `split_fields` | `["true", "false"]` | `["true", "false"]` |
| `benchmark` | `["ycsba","ycsbb","ycsbc","ycsbf"]` | `["ycsbf"]` |
| `skew_factor` | `[0.0,0.2,0.4,0.6,0.8,0.9,0.95,0.99]` | `[0.99]` |
| 组合数×repeat | 64 | 2 |

## epic_tpcc_experiment (epic GPU TPCC)

| 参数 | run | test |
|------|-----|------|
| `num_warehouses` | `[1,2,4,8,16,32,64]` | `[64]` |
| 组合数×repeat | 7 | 1 |

## epic_tpcc_full_experiment (epic GPU TPCC-Full)

| 参数 | run | test |
|------|-----|------|
| `num_warehouses` | `[1,2,4,8,16,32,64]` | `[64]` |
| 组合数×repeat | 7 | 1 |

## gacco_ycsb_experiment (gacco GPU YCSB)

| 参数 | run | test |
|------|-----|------|
| `num_txns`(局部覆盖) | `32768` | `32768` |
| `benchmark` | `["ycsba","ycsbb","ycsbc","ycsbf"]` | `["ycsbf"]` |
| `skew_factor` | `[0.0,0.2,0.4,0.6,0.8,0.9,0.95,0.99]` | `[0.99]` |
| 组合数×repeat | 32 | 1 |

## gacco_tpcc_experiment (gacco GPU TPCC)

| 参数 | run | test |
|------|-----|------|
| `num_txns`(局部覆盖) | `32768` | `32768` |
| `benchmark` | `["tpccn","tpccp"]` | `["tpccn","tpccp"]` |
| `commutative_ops` | `["true","false"]` | `["true","false"]` |
| `num_warehouses` | `[1,2,4,8,16,32,64]` | `[64]` |
| 组合数×repeat | 28 | 4 |

## epic_cpu_tpcc_experiment (epic CPU TPCC)

| 参数 | run | test |
|------|-----|------|
| `num_warehouses` | `[1,2,4,8,16,32,64]` | `[64]` |
| 组合数×repeat | 7 | 1 |

## epic_cpu_tpcc_full_experiment (epic CPU TPCC-Full)

| 参数 | run | test |
|------|-----|------|
| `num_warehouses` | `[1,2,4,8,16,32,64]` | `[64]` |
| 组合数×repeat | 7 | 1 |

## epic_cpu_ycsb_experiment (epic CPU YCSB)

| 参数 | run | test |
|------|-----|------|
| `benchmark` | `["ycsba","ycsbb","ycsbc","ycsbf"]` | `["ycsbf"]` |
| `skew_factor` | `[0.0,0.2,0.4,0.6,0.8,0.9,0.95,0.99]` | `[0.99]` |
| 组合数×repeat | 32 | 1 |

## epic_ycsb_epoch_size_experiment (YCSB epoch 大小扫参)

| 参数 | run | test |
|------|-----|------|
| `skew_factor` | `[0.99, 0.0]` | `[0.99, 0.0]` |
| `num_txns` (skew=0.99) | 24个值 `[500..70000]` | `[70000]` (仅最大值) |
| `num_txns` (skew=0.0) | 32个值 `[500..200000]` | `[200000]` (仅最大值) |
| 组合数×repeat | 56 | 2 |

## epic_tpcc_epoch_size_experiment (TPCC epoch 大小扫参)

| 参数 | run | test |
|------|-----|------|
| `num_warehouses` | `[1, 64]` | `[1, 64]` |
| `num_txns` (w=1) | 20个值 `[500..40000]` | `[40000]` (仅最大值) |
| `num_txns` (w=64) | 32个值 `[500..200000]` | `[200000]` (仅最大值) |
| 组合数×repeat | 52 | 2 |

## epic_microbenchmark

| 参数 | run | test |
|------|-----|------|
| `abort_rate` (`-p`) | `[0,5,10,15,20,25,30,35,40,45,50]` | `[50]` (仅最大值) |
| 组合数×repeat | 11 | 1 |

## gacco_tpcc_epoch_size_experiment

| 参数 | run | test |
|------|-----|------|
| `benchmark` | `["tpccn","tpccp"]` | `["tpccn","tpccp"]` |
| `num_warehouses` | `[1, 64]` | `[1, 64]` |
| `num_txns` (w=1) | 5个值 `[500..4000]` | `[4000]` (仅最大值) |
| `num_txns` (w=64) | 17个值 `[500..25000]` | `[25000]` (仅最大值) |
| 组合数×repeat | 44 | 4 |

## gacco_ycsb_epoch_size_experiment

| 参数 | run | test |
|------|-----|------|
| `skew_factor` | `[0.99, 0.0]` | `[0.99, 0.0]` |
| `num_txns` (skew=0.99) | 7个值 `[500..6000]` | `[6000]` (仅最大值) |
| `num_txns` (skew=0.0) | 21个值 `[500..45000]` | `[45000]` (仅最大值) |
| 组合数×repeat | 28 | 2 |

---

## 总结

`test_experiments.py` 是 `run_experiments.py` 的**最小边界子集**版本：
- `num_repeat`: test=1, run=3
- test 额外保存 stderr 到 `.err` 文件，方便定位失败原因
- 每个实验函数只保留极值/边界条件（skew=0.99, warehouse=64, max txns, max abort_rate），run 扫全量
- 两者执行的函数列表和顺序完全一致（13个函数）
