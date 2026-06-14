# GPU Occupancy & SM 资源利用率分析指南

本文档对应 `profile_experiments.py` 的 Nsight Compute profiling 策略，说明选取了哪些 metrics、如何解读它们，以及如何判断是否存在“SM block 数达到上限但每个 block 仅剩 1 个 warp 活跃”的尾部效应。

---

## 一、Profiling 策略

`profile_experiments.py` 对每个 GPU 实验运行 **两轮** ncu：

| 轮次 | `--set` | 关注点 | 输出文件后缀 |
|------|---------|--------|-------------|
| 第 1 轮 | `occupancy` | block 上限、理论/实际占用率 | `__occupancy.ncu-rep` / `.csv` |
| 第 2 轮 | `sm` | scheduler 统计、warp 状态、stall 原因 | `__sm.ncu-rep` / `.csv` |

两轮收集的 CSV 合并起来即可覆盖“SM 资源是否浪费”的全部关键指标。

> 也可以直接 `--set full` 一轮搞定，但数据量较大；两轮分拆更适合快速定位问题。

---

### 1.1 核函数筛选（避免全量 profiling 爆炸）

EPIC 一次运行会启动 **上百个** GPU kernel，但绝大多数是 CUB/Thrust 的初始化/填充/复制类
小 kernel（单次 < 10 µs），profiling 所有 kernel 会严重拖慢采集速度且产出大量无价值的噪声数据。

因此 `profile_experiments.py` 通过 `--kernel-name` 正则只保留以下 **时间主导** 的 kernel：

| 分类 | 匹配正则 | 包含的 kernel | 耗时级别 |
|------|---------|-------------|---------|
| **执行器** | `gpuNoSplitPiecewiseExec\|gpuPiecewiseExec` | YCSB / TPCC 主执行循环 | ~61 ms (绝对主导) |
| **执行规划器** | `scatterRW\|calcOperation` | scatter RW 位置、计算操作类型 | ~140 µs / ~25 µs |
| **索引** | `index(Ycsb\|Tpcc)\|prepare(Ycsb\|Tpcc)Index` | cuCollections 哈希表查找/插入准备 | ~160 µs |
| **提交器** | `submit(Ycsb\|Tpcc)\|prepareSubmit(Ycsb\|Tpcc)` | 事务结果提交与准备 | ~90 µs |
| **哈希表** | `cuco::detail::(insert\|find\|initialize)` | cuCollections 底层哈希插入/查找/初始化 | ~475 µs / ~296 µs |
| **CUB 重量级** | `DeviceRadixSortOnesweep\|DeviceScanByKeyKernel\|DeviceSelectSweep` | 基数排序 / 按键扫描 / 选择压缩 | 17~47 µs |

**被跳过的 kernel 类型**：`DeviceScanInit`, `DeviceCompactInit`, `DeviceReduceSingleTile`,
`DeviceReduceKernel`, `DeviceRadixSortHistogram`, `DeviceRadixSortExclusiveSum`,
所有 `thrust::cuda_cub::` 前缀的 kernel（`__uninitialized_fill`, `__tabulate`,
`__uninitialized_copy`, `for_each_f` 等）。

> 筛除后 profiling 时间从 ~20 分钟降到 ~2 分钟。

---

## 二、选取的 Metrics 及含义

### 2.1 Occupancy 组 (`--set occupancy`)

这些指标回答：**SM 上最多能塞多少 block/warp？实际塞了多少？**

| Metric | 含义 | 如何解读 |
|--------|------|---------|
| `Block Limit SM block` | 硬件层面 SM 可容纳的最大 block 数（如 Volta+ 为 32） | 若此值 ≤ 其他限制指标，说明已撞硬件上限 |
| `Block Limit Registers` | 受寄存器用量限制的最大 block 数 | 若此值最小 → 瓶颈是寄存器溢出 |
| `Block Limit Shared Mem` | 受 shared memory 用量限制的最大 block 数 | 若此值最小 → 瓶颈是 shared memory |
| `Block Limit Warps` | 受 block 内 warp 数限制的最大 block 数 | 若此值最小 → block 本身 warp 太少 |
| `Theoretical Active Warps per SM` | SM 理论上最多可同时活跃的 warp 数 | 由硬件 + 资源配置决定 |
| `Achieved Active Warps Per SM` | 内核执行期间 SM 上 **平均** 活跃 warp 数 | < Theoretical 说明有空闲 |
| `Achieved Occupancy [%]` | `Achieved / Theoretical × 100%` | < 50% 需重点关注 |

**判断尾部效应的第一步**：
- 若 `Block Limit SM block` ≤ 其他 `Block Limit *`，说明 **SM 的 block 数确实已达硬件上限**，无法再增加 block。
- 此时若 `Achieved Active Warps Per SM` 又明显低于 `Theoretical Active Warps per SM`，就说明虽然块数满了，但很多 warp 并未真正工作。

```
示例（理想情况）:
  Block Limit SM block           32
  Block Limit Registers          4     ← 寄存器限制了 block 数，而非硬件上限
  Theoretical Active Warps      128
  Achieved Active Warps         120   ← 接近理论值 → 利用率高

示例（尾部效应嫌疑）:
  Block Limit SM block           32   ← 硬件上限先到
  Block Limit Registers         64   ← 寄存器很充足
  Theoretical Active Warps      128
  Achieved Active Warps          50   ← 大幅低于理论 → 可能存在尾部效应
```

---

### 2.2 SM / Scheduler 组 (`--set sm`)

这些指标回答：**那些 resident 的 warp 到底在干什么？是活跃、stall、还是已经结束了？**

| Metric | 含义 | 如何解读 |
|--------|------|---------|
| `Active Warps Per SM` | 内核执行期间每个 SM 上平均 **resident** warp 数 | 多 vs 少 → 反映 warps 是否成功驻留 |
| `Eligible Warps Per SM` | 平均 **准备好发出指令** 的 warp 数 | **远小于 Active Warps** → 大量 warp 处于 stall/等待 |
| `No Eligible Issue Rate [%]` | **没有任何 warp 可发射指令** 的周期占比 | **高 (>40%)** → SM 空转，可能存在尾部效应 |
| `Waves Per SM` | 每个 SM 上分配的 thread-block **波次**数 | **不是整数** → 存在尾部效应（tail effect），即最后一波 block 没有填满 SM |
| `Warp Cycles Per Issued Instruction` | 每个 warp 发出一条指令的平均等待周期 | 越高 → 延迟越大 |
| `Warp Cycles Per Active Cycle` | 在 warp 活跃期间的每条指令平均等待周期 | 排除长时间 stall 后的实际延迟 |

**scheduler_statistics 典型输出示例**：

```
Active Warps Per SM              31.78   ← resident 的 warp 不少
Eligible Warps Per SM             0.76   ← 但能干活的不多！
No Eligible Issue Rate [%]       50.97   ← 一半时间调度器空等
```

> 这种模式（Active 高, Eligible 低）强烈暗示 **大量 warp 已经结束工作**，只剩下极少数 warp 在运行——这正是尾部效应的典型特征。

---

### 2.3 Warp State Statistics（可选深度分析）

在 Nsight Compute GUI 的 **Warp State** 面板中，warp 会被分类为多种状态：

| 状态 | 含义 | 高占比时的推断 |
|------|------|---------------|
| **Active** | 正在执行指令 | 正常 |
| **Stall - Long Scoreboard** | 等待全局内存 / 纹理加载 | 内存瓶颈 |
| **Stall - Short Scoreboard** | 等待共享内存 / 寄存器依赖 | 共享内存瓶颈 |
| **Stall - Wait** | 等待同步屏障 `__syncthreads()` | block 内 warp 步调不均 |
| **Stall - Selected** | 被调度器选中但未发射 | 指令发射瓶颈 |
| **Stall - Not Selected** | 有资格发射但未被选中 | 调度竞争 |
| **Stall - Other** | 其他原因（含已结束） | ⚠️ 若占比极高 → 可能是 warp 已完成工作 |

**关键判断**：如果 **Wait / Other 占比极高，且 Active 占比极低**，而内存指标（Long Scoreboard）并不高，则尾部效应的可能性非常大。

---

### 2.4 快速验证指标

| Metric | 正常范围 | 尾部效应嫌疑值 |
|--------|---------|---------------|
| `Achieved Occupancy` | > 60% | < 40% |
| `Eligible Warps Per SM` / `Active Warps Per SM` | > 0.6 | < 0.2 |
| `No Eligible Issue Rate` | < 20% | > 40% |
| `Waves Per SM` | 接近整数 | **不是整数** |
| `Long Scoreboard %` | 有但不能解释全部 stall | **很低但 Eligible Warps 也很低** |

---

## 三、分析流程：如何判断尾部效应

### Step 1：确认 Block 是否达到硬件上限

在 `occupancy` 报告中查看 `Block Limit *` 系列：
- 若 `Block Limit SM block` 是当前的最小值（或与其他限制指标相等），说明 block 数确实已达硬件上限。

### Step 2：检查 Occupancy 是否偏低

- `Achieved Occupancy < 50%` → 继续往下分析
- `Achieved Occupancy > 70%` → 利用率 OK，尾部效应可能性低

### Step 3：检查 Scheduler 统计

- `Eligible Warps Per SM` 远小于 `Active Warps Per SM` → 大量 warp 未参与调度
- `No Eligible Issue Rate > 40%` → SM 大量空转

### Step 4：检查 Waves Per SM

- 若 `Waves Per SM` 不是整数（如 2.3），说明最后一波 block 未填满 SM → 存在尾部效应。

### Step 5：交叉验证 Warp State

- 在 GUI 中打开 **Warp State Statistics** 面板
- 若 `Stall - Other` 或 `Stall - Wait` 占比异常高，且 `Stall - Long Scoreboard`（内存等待）占比不高 → 可确认 warp 已完成而非因内存而 stall

### Step 6（可选）：控制变量实验

如果怀疑尾部效应严重，可以编写一个对比实验：
- 构造一个 **workload 极不均匀** 的内核：大部分线程很快结束，只有少量线程执行长循环
- 对比该内核与 **workload 均匀** 的内核在同样 metrics 下的表现
- 若均匀 workload 下 `Eligible Warps` 正常，而不均匀下明显偏低 → 确认尾部效应

---

## 四、命令行分析（无需 GUI）

以下 ncu 命令直接从 CSV 提取关键值：

```bash
# 从 occupancy CSV 提取 block limit 和 occupancy 指标
grep -E "Block Limit|Occupancy|Active Warps" profile_output/*__occupancy.csv

# 从 sm CSV 提取 scheduler 和 warp 状态指标
grep -E "eligible|Active Warps|Waves Per SM|No Eligible" profile_output/*__sm.csv
```

或用 Python 快速分析：

```python
import pandas as pd

occ = pd.read_csv("profile_output/profile__epic_tpcc__...__occupancy.csv")
sm  = pd.read_csv("profile_output/profile__epic_tpcc__...__sm.csv")

# 关键指标
print("=== Occupancy ===")
print(occ[["metric_name", "metric_value"]].to_string())

print("\n=== Scheduler ===")
print(sm[["metric_name", "metric_value"]].to_string())
```

---

## 五、关于“平均值”的注意事项

Nsight Compute 的 `Achieved Active Warps Per SM` 和 `Eligible Warps Per SM` 是**全内核执行期间的平均值**。如果尾部效应只占总执行时间的很小一部分，这些平均值可能变化不明显。

此时需要结合：
- **Timeline 视图**（GUI）：拖动时间轴，观察尾部阶段是否出现 warp 活跃数骤降
- **Waves Per SM**：非整数即表明尾部效应确实存在
- **Warp State 时序图**：查看 Stall - Other 在时间轴上的分布

---

## 六、指标速查表

| 关心的现象 | 去哪个 set 找 | 关键指标名 |
|-----------|--------------|-----------|
| Block 数是否到硬件上限 | `occupancy` | `Block Limit SM block` |
| 寄存器/共享内存是否限制 block 数 | `occupancy` | `Block Limit Registers` / `Shared Mem` |
| 理论 vs 实际占用率 | `occupancy` | `Theoretical / Achieved Active Warps Per SM` |
| Warp 是 resident 还是 eligible | `sm` | `Active / Eligible Warps Per SM` |
| 调度器空转比例 | `sm` | `No Eligible Issue Rate` |
| 是否存在尾部效应 | `sm` | `Waves Per SM`（非整数 = 是） |
| Warp 在等什么（内存/同步/其他） | `sm` → Warp State 面板 | 各 Stall 原因的百分比 |
| 内存是否瓶颈 | `sm` | `Stall - Long Scoreboard %` |
