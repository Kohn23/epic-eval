

## Epic 完整工作流程 — 按代码执行顺序阅读

```
main.cpp 入口
  │
  ├─ ① 构造期: epic::tpcc::TpccDb(config)        [benchmarks/tpcc.cpp:41-238]
  │    初始化所有数据结构、分配 GPU 内存、创建 Planner/Index/Submitter/Executor
  │    ↓ 此阶段涉及的阅读顺序 ↓
  │
  ├─ ② benchmark->loadInitialData()              [benchmarks/tpcc.cpp:327]
  │    加载 TPCC 初始数据到索引
  │
  ├─ ③ benchmark->generateTxns()                 [benchmarks/tpcc.cpp:298]
  │    生成所有 epoch 的事务输入
  │
  └─ ④ benchmark->runBenchmark()                 [benchmarks/tpcc.cpp:338]
       对每个 epoch 循环执行 ⑥~⑫ 共 7 个子阶段
```

---

### 阶段 ①：构造期 — 数据结构初始化

**阅读入口**：tpcc.cpp 第 41 行 `TpccDb::TpccDb(TpccConfig config)`

**阅读顺序**：

| 步骤 | 文件 | 行/关键内容 | 说明 |
|------|------|------------|------|
| 1.1 | tpcc_table.h | `WarehouseKey`/`WarehouseValue` 等所有表结构 | 理解 schema 定义方式（bitfield 键、cache-line 对齐） |
| 1.2 | storage.h | `Record<ValueType>`（双版本：version1/value1 + version2/value2） | **核心数据结构**：MVCC 的基础 |
| 1.3 | tpcc_storage.h | `TpccRecords` / `TpccVersions` | 所有表的指针集合 |
| 1.4 | tpcc_txn.h | `TxnInput` → `TxnParams` → `TxnExecPlan` 三层结构 | 事务数据的三阶段形态 |
| 1.5 | txn.h | `BaseTxn`、`TxnArray<T>`、`BaseTxnSize<T>` | 事务数组模板基础设施 |
| 1.6 | `benchmarks/tpcc.cpp:41-120` | 创建 `TxnArray`、`TxnBridge`、`Index` | 理解数据在各阶段间的流动管道 |
| 1.7 | `benchmarks/tpcc.cpp:120-170` | 创建 9 个 `GpuTableExecutionPlanner` | 每个表一个 planner |
| 1.8 | gpu_execution_planner.h | 模板类声明、`op_t` 编码格式 | 理解 `op_t` 的位布局 |
| 1.9 | `gpu_execution_planner.cu:40-80` | `Initialize()` — GPU 内存分配 | planner 初始化 |
| 1.10 | `benchmarks/tpcc.cpp:170-200` | 创建 `TpccGpuSubmitter`，绑定所有表的 `SubmitDest` | 提交器连接 planner |
| 1.11 | `benchmarks/tpcc.cpp:200-240` | **分配 records & versions GPU 内存** | 核心存储分配 |

---

### 阶段 ②：加载初始数据

| 步骤 | 文件 | 说明 |
|------|------|------|
| 2.1 | `benchmarks/tpcc.cpp:327` | `index->loadInitialData()` → `gpu_aux_index.loadInitialData()` |

---

### 阶段 ③：生成事务

| 步骤 | 文件 | 说明 |
|------|------|------|
| 3.1 | `benchmarks/tpcc.cpp:298-325` | 按 `txn_mix` 比例随机生成 `NewOrder`/`Payment` 等 5 种事务 |
| 3.2 | tpcc_txn_gen.cpp | 具体的事务字段随机填充逻辑 |

---

### 阶段 ④：每个 Epoch 的 7 步循环

这是核心运行时流程。入口在 `benchmarks/tpcc.cpp:338` 的 `for (epoch_id ...)` 循环。

```
每个 epoch 内部:
  ├─ 步骤A: GPU Aux Index (辅助索引范围查询)
  ├─ 步骤B: 传输 ①  CPU→GPU
  ├─ 步骤C: GPU Index (主索引: TxnInput → TxnParams)
  ├─ 步骤D: 传输 ②  GPU内部
  ├─ 步骤E: Submit (提交操作到各表 planner)
  ├─ 步骤F: Initialize (GPU编译执行计划: 排序/扫描/散射)
  ├─ 步骤G: 传输 ③  GPU内部
  └─ 步骤H: Execute (GPU执行事务)
```

#### 步骤 B：传输 ① — CPU → GPU

| 文件 | 行 | 说明 |
|------|-----|------|
| `benchmarks/tpcc.cpp:365` | `input_index_bridge.StartTransfer()` | `TxnBridge` 将 CPU 上的 `TxnInput` 拷贝到 GPU |
| `txn_bridge.h:30` | `Link()` / `StartTransfer()` / `FinishTransfer()` | 理解 CPU↔GPU 数据传输机制 |

#### 步骤 C：GPU Index — 主索引查找

| 文件 | 行 | 说明 |
|------|-----|------|
| `benchmarks/tpcc.cpp:415` | `index->indexTxns(index_input, index_output, ...)` | **TxnInput → TxnParams 转换** |
| tpcc_gpu_index.cu | 全文件 | GPU kernel：将逻辑 key 转为物理 record_id |
| tpcc_gpu_index.h | 全文件 | 索引类声明 |

#### 步骤 E：Submit — 向各表提交操作

| 文件 | 行 | 说明 |
|------|-----|------|
| `benchmarks/tpcc.cpp:490` | `submitter->submit(initialization_input)` | 遍历所有事务，将每个读写操作分发到对应表的 planner |
| tpcc_gpu_submitter.cu | 全文件 | **GPU kernel**：`prepareSubmitTpccTxn` 计算每个表的操作数，然后 scatter 操作到各表的 `d_submitted_ops` |

#### 步骤 F：Initialize — GPU 编译执行计划 ⭐ **核心创新**

| 文件 | 行 | 说明 |
|------|-----|------|
| `benchmarks/tpcc.cpp:498` | `warehouse_planner->InitializeExecutionPlan()` ... | 对 9 个表依次执行 |
| `gpu_execution_planner.cu:196` | `InitializeExecutionPlan()` | **主函数入口** |
| `gpu_execution_planner.cu:220` | `cub::DeviceRadixSort::SortKeys` | ① 按 record_id 排序所有操作 |
| `gpu_execution_planner.cu:245` | `cub::DeviceScan::ExclusiveSumByKey` | ② 计算每个记录之前有几个写操作 (`d_write_ops_before`) |
| `gpu_execution_planner.cu:262` | `cub::DeviceScan::ExclusiveSumByKey`（反向） | ③ 计算每个记录之后有几个写操作 (`d_write_ops_after`) |
| `gpu_execution_planner.cu:267` | `calcOperationType` kernel | ④ 根据前后写操作数判断操作类型（RECORD_A_READ / RECORD_B_WRITE / VERSION_READ / VERSION_WRITE） |
| `gpu_execution_planner.cu:275` | `cub::DeviceScan::ExclusiveSum`（version write） | ⑤ 计算 version write 的索引 |
| `gpu_execution_planner.cu:145` | `scatterRWLocation` kernel | ⑥ 将位置信息写入各事务的 `TxnExecPlan` |

#### 步骤 H：Execute — GPU 执行事务 ⭐ **核心创新**

| 文件 | 行 | 说明 |
|------|-----|------|
| `benchmarks/tpcc.cpp:625` | `executor->execute(epoch_id)` | 启动 GPU kernel |
| `benchmarks/tpcc_gpu_executor.cu:390` | `GpuExecutor::execute()` | 启动 `gpuExecKernel` |
| `benchmarks/tpcc_gpu_executor.cu:30-110` | `gpuExecTpccTxn(NewOrder...)` | **最核心的执行逻辑** |
| `gpu_storage.cuh:27` | `gpuReadFromTableCoop()` | Warp 协作读取：leader lane 选择 A/B 副本，广播 |
| gpu_storage.cuh | `gpuWriteToTableCoop()` | Warp 协作写入 |
| `benchmarks/tpcc_gpu_executor.cu:30-40` | `offsetof(StockValue, s_quantity)` | 编译期字段偏移，lane 分工 |

---

### 推荐阅读路线（由浅入深）

```
第一遍（理解数据结构）:
  txn.h → storage.h → benchmarks/tpcc_table.h → benchmarks/tpcc_storage.h → benchmarks/tpcc_txn.h

第二遍（理解初始化流程）:
  main.cpp → benchmarks/tpcc.cpp (构造函数) → gpu_execution_planner.h → gpu_execution_planner.cu (Initialize)

第三遍（理解运行流程）:
  benchmarks/tpcc.cpp (runBenchmark) → benchmarks/tpcc_gpu_index.cu → benchmarks/tpcc_gpu_submitter.cu

第四遍（理解核心创新）:
  gpu_execution_planner.cu (InitializeExecutionPlan + scatterRWLocation)
  → gpu_storage.cuh (gpuReadFromTableCoop / gpuWriteToTableCoop)
  → benchmarks/tpcc_gpu_executor.cu (gpuExecTpccTxn)
```

### 对照阅读（理解差异）

阅读 Epic 版的同时对照 gacco 下的同名文件，体会差异：
- tpcc_gpu_executor.cu — 纯锁 + 串行读写整条记录
- gpu_execution_planner.cu — 无执行计划编译，仅排序后直接加锁