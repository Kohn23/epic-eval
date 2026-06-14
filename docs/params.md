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

如果你想，我可以：
- 为每个函数列出 run vs test 中具体被遍历的数值对照表；或  
- 抽取并生成一个表格（CSV/markdown）列出每个 CLI 标志、取值集合与在哪些函数出现。你想要哪个？