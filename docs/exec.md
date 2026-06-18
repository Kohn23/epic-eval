# tpcc_gpu_executor.cu 内部详解

> 源文件: `benchmarks/tpcc_gpu_executor.cu`, `benchmarks/tpcc_gpu_executor.h`
> 辅助文件: `gpu_storage.cuh` (存储读写), `util_warp_memory.cuh` (Warp 内存拷贝)

---

## 目录

1. [文件结构总览](#1-文件结构总览)
2. [GpuExecutor 类](#2-gpuexecutor-类)
3. [gpuExecKernel — 主执行 Kernel](#3-gpuexeckernel--主执行-kernel)
4. [事务调度机制](#4-事务调度机制)
5. [五种事务的执行逻辑](#5-五种事务的执行逻辑)
6. [存储层: 版本化读写](#6-存储层-版本化读写)
7. [配置参数与启动参数](#7-配置参数与启动参数)
8. [完整数据流](#8-完整数据流)

---

## 1. 文件结构总览

```
tpcc_gpu_executor.cu
├── namespace epic::tpcc
│   ├── namespace { (anonymous)
│   │   ├── 常量定义: block_size=128, num_warps=4, txn_counter
│   │   ├── gpuExecTpccTxn() ×5 重载 (5种事务类型的 device 函数)
│   │   ├── CachableTxnParams / CachableTxnExecPlan (union 用于 shared memory 缓存)
│   │   ├── gpuExecKernel()           ← ★ 主 kernel
│   │   └── } // anonymous namespace
│   │
│   └── GpuExecutor::execute()         ← 入口函数 (CPU 端调用)
│       ├── cudaMemcpyToSymbol 清零 txn_counter
│       ├── gpuExecKernel<<<25000, 128>>> 启动
│       └── cudaDeviceSynchronize() 等待完成
│
├── 依赖
│   ├── gpu_storage.cuh       → gpuReadFromTableCoop / gpuWriteToTableCoop
│   ├── gpu_txn.cuh           → GpuPackedTxnArray, getTxn()
│   ├── util_warp_memory.cuh  → warpMemcpy()
│   ├── util_arch.h           → kDeviceWarpSize (32)
│   └── tpcc_gpu_txn.cuh      → FixedSizeTxn, TpccTxnType
```

---

## 2. GpuExecutor 类

**头文件**: `benchmarks/tpcc_gpu_executor.h`

```cpp
template <typename TxnParamArrayType, typename TxnExecPlanArrayType>
class GpuExecutor : public Executor<TxnParamArrayType, TxnExecPlanArrayType> {
    // 继承自 Executor 基类的成员:
    //   TpccRecords  records;   // 9 张表的 Record 数组指针
    //   TpccVersions versions;  // 9 张表的 Version 数组指针
    //   TpccConfig   config;    // 配置 (num_txns, num_warehouses, epochs...)
    //   TxnParamArrayType  txn;   // 事务参数数组 (GpuPackedTxnArray)
    //   TxnExecPlanArrayType plan; // 执行计划数组 (GpuPackedTxnArray)

    void execute(uint32_t epoch) override;
};
```

**`execute()` 方法** (`tpcc_gpu_executor.cu:390-439`):

```cpp
void GpuExecutor::execute(uint32_t epoch) {
    // Step 1: 清零全局 txn_counter (device symbol)
    cudaMemcpyToSymbol(txn_counter, &zero, sizeof(uint32_t));

    // Step 2: 计算 block 数
    // 公式: num_blocks = ceil(num_txns * warp_size / block_size)
    //   = ceil(100,000 * 32 / 128) = 25,000
    uint32_t num_blocks = (config.num_txns * kDeviceWarpSize + block_size - 1) / block_size;

    // Step 3: Launch kernel (异步)
    gpuExecKernel<<<25000, 128>>>(
        records, versions,
        TpccGpuTxnArrayT(txn),   // 事务参数
        TpccGpuTxnArrayT(plan),  // 执行计划
        config.num_txns, epoch);

    // Step 4: 同步等待 GPU 完成
    cudaDeviceSynchronize();  // ← 62ms 主要在这里
}
```

---

## 3. gpuExecKernel — 主执行 Kernel

**签名**: `__global__ void gpuExecKernel(records, versions, txn, plan, num_txns, epoch)`

**启动配置**: `<<<25000, 128>>>` — 25,000 blocks × 128 threads = 3,200,000 threads

### 3.1 Block 内组织结构

```
每个 Block: 128 threads = 4 Warps × 32 lanes

Block (128 threads)
├── Warp 0: lane 0..31  ← 合作处理事务 A
├── Warp 1: lane 0..31  ← 合作处理事务 B
├── Warp 2: lane 0..31  ← 合作处理事务 C
└── Warp 3: lane 0..31  ← 合作处理事务 D

每个 Warp 内部:
├── lane 0 (leader): 负责获取 txn_id、决定读/写地址
├── lane 1..31:       各负责 Record/Version 中一个 32-bit 字段的读/写
```

### 3.2 Shared Memory 布局

每个 block 使用 ~4KB shared memory:

```
__shared__ uint8_t cached_txn_param[4][sizeof(CachableTxnParams)];  // Warp×参数缓存
__shared__ uint8_t cached_exec_plan[4][sizeof(CachableTxnExecPlan)]; // Warp×执行计划缓存
__shared__ uint32_t warp_counter;  // 每个 block 内部 warp 间的事务分配计数器
```

`CachableTxnParams` union 覆盖 NewOrder / Payment / OrderStatus 三种最大事务类型。

Delivery 和 StockLevel 由于参数过大（含变长数组），直接从 global memory 访问，不经过 shared memory 缓存。

### 3.3 Kernel 主循环伪代码

```
gpuExecKernel:
    warp_id = threadIdx.x / 32
    lane_id = threadIdx.x % 32

    // Step 1: block 内第一个线程从全局 txn_counter 抢一批事务
    if thread 0:
        warp_counter = atomicAdd(&txn_counter, 4)  // 一次抢 4 个 warp 的事务

    __syncthreads()  // 保证 warp_counter 对 block 内所有线程可见

    // Step 2: 每个 warp 从 warp_counter 中抢自己的 txn_id
    if lane 0:
        warp_txn_id = atomicAdd(&warp_counter, 1)
    warp_txn_id = __shfl(warp_txn_id, lane 0)  // broadcast

    if warp_txn_id >= num_txns: return

    // Step 3: 从 global memory 加载 txn_param 和 exec_plan 到 shared memory
    load_to_shared_memory()

    // Step 4: 根据事务类型分发执行
    switch txn_type:
        case NEW_ORDER:    → gpuExecTpccTxn(NewOrder...)
        case PAYMENT:      → gpuExecTpccTxn(Payment...)
        case ORDER_STATUS: → gpuExecTpccTxn(OrderStatus...)
        case DELIVERY:     → gpuExecTpccTxn(Delivery...)
        case STOCK_LEVEL:  → gpuExecTpccTxn(StockLevel...)
```

---

## 4. 事务调度机制

### 4.1 两级原子计数器

```
全局: txn_counter (device symbol, 初始值 0)
        ↓ atomicAdd(&txn_counter, 4)  ← 每个 block 的 thread 0 执行一次
        ↓ 返回: block_i 的起始 warp_id 范围

Block 内: warp_counter (shared memory)
        ↓ atomicAdd(&warp_counter, 1)  ← 每个 warp 的 lane 0 执行
        ↓ 返回: 该 warp 要处理的事务 id
```

### 4.2 调度示意 (25,000 blocks × 4 warps/block = 100,000 warps)

```
时间 →
Block 0:  txn_counter 0 →4  → warp0:txn0  warp1:txn1  warp2:txn2  warp3:txn3
Block 1:  txn_counter 4 →8  → warp0:txn4  warp1:txn5  warp2:txn6  warp3:txn7
Block 2:  txn_counter 8 →12 → warp0:txn8  warp1:txn9  ...
...
Block N:  ... → 100,000 个事务全部分配完毕
```

- 每个 block 抢占 4 个 warp 的事务槽位
- 每个 warp 独立处理 1 个事务
- 25,000 blocks × 4 warps = 正好 100,000 个事务

### 4.3 kv 的 warp id 计算

```cpp
uint32_t num_blocks = (num_txns * kDeviceWarpSize + block_size - 1) / block_size;
//                 = (100000 * 32 + 127) / 128
//                 = 3,200,000 / 128 = 25,000
```

关键: 这个公式确保总共启动的 warp 数 ≥ num_txns。如果 num_txns 不能被 num_warps 整除，多余的 warp 会在 `warp_txn_id >= num_txns` 时提前 return。

---

## 5. 五种事务的执行逻辑

### 5.1 NewOrder (45% of tpccfull mix)

**操作序列** (每个 item 最多 5 个，因此有循环):

```
第 1 阶段: 固定操作 (不受 num_items 影响)
┌──────────────────────────────────────────────────────┐
│ 1. Read  Warehouse    (warehouse_record,  read_loc)  │ ← 验证
│ 2. Read  District     (district_record,  read_loc)   │ ← 读 d_next_o_id
│    └─ lane[d_next_o_id_offset]: result = next_order_id│
│ 3. Write District     (district_record, write_loc)   │ ← 更新 d_next_o_id
│ 4. Read  Customer     (customer_record, read_loc)    │ ← 读 c_discount 等
│ 5. Write Order        (order_record,    write_loc)   │ ← 插入新订单
│ 6. Write NewOrder     (new_order_record, write_loc)  │ ← 插入 new_order
└──────────────────────────────────────────────────────┘

第 2 阶段: 对每个 item (最多 15 个) 循环
┌──────────────────────────────────────────────────────┐
│ for i in 0..num_items:                               │
│   7. Read  Item       (item_record, item_loc)        │
│   8. Read  Stock      (stock_record, stock_read_loc) │
│      └─ lane[s_quantity_offset]:                     │
│           result = result > qty+10 ? result-qty      │
│                                    : result+91-qty   │
│   9. Write Stock      (stock_record, stock_write_loc)│
│  10. Write OrderLine  (orderline_record, ol_loc)     │
│      └─ lane[ol_i_id]:      result = item_id         │
│      └─ lane[ol_amount]:    result = order_quantities│
│      └─ lane[ol_supply_w]:  result = warehouse_id    │
│      └─ lane[ol_quantity]:  result = order_quantities│
└──────────────────────────────────────────────────────┘
```

**总操作数**: 6 + 4 × num_items (典型 num_items≈15 → 66 次存储操作)

**关键字段偏移量**:
| 字段 | 偏移(32-bit words) | 赋值逻辑 |
|------|-------------------|---------|
| `s_quantity` | offsetof / 4 | 递减或回卷: `result > qty+10 ? result-qty : result+91-qty` |
| `d_next_o_id` | offsetof / 4 | 直接设为 `params->next_order_id` |
| `ol_i_id` | offsetof / 4 | = `item_id` |
| `ol_amount` | offsetof / 4 | = `order_quantities` |
| `ol_quantity` | offsetof / 4 | = `order_quantities` |
| `ol_supply_w_id` | offsetof / 4 | = `warehouse_id` |

### 5.2 Payment (43%)

**全读写，无只读操作**:

```
1. Read  Warehouse  →   lane[w_ytd_offset]:   result += payment_amount
2. Write Warehouse
3. Read  District   →   lane[d_ytd_offset]:   result += payment_amount
4. Write District
5. Read  Customer   →   lane[c_balance_offset]:     result -= payment_amount
                    →   lane[c_ytd_payment_offset]: result += payment_amount
                    →   lane[c_payment_cnt_offset]: result += 1
6. Write Customer
```

**总操作数**: 6 (3 组读写)

| 字段 | 修改 |
|------|------|
| `w_ytd` | += payment_amount |
| `d_ytd` | += payment_amount |
| `c_balance` | -= payment_amount |
| `c_ytd_payment` | += payment_amount |
| `c_payment_cnt` | += 1 |

### 5.3 OrderStatus (4%)

**全只读**:

```
1. Read Customer                    ← 读客户信息
2. Read Order                       ← 读订单信息
3. for i in 0..num_items:
     Read OrderLine                 ← 读每个订单行的信息
```

无写操作。所有 lane 读同一个字段，Warp 内合作读取。

### 5.4 Delivery (4%)

**最复杂的事务，10 个 district 各一个独立子事务**:

```
for i in 0..9 (10 districts):
    1. Read  NewOrder       ← 获取要 delivery 的 order_id
    2. Read  Order          ← 读订单
       └─ lane[o_carrier_id]: result = carrier_id
    3. Write Order          ← 更新 carrier_id
    4. for j in 0..num_items[i]:
          Read  OrderLine   ← 读订单行
          └─ lane[ol_amount]: amount += result
          └─ lane[ol_delivery_d]: result = delivery_d
          Write OrderLine   ← 直接写 loc_record_b
    5. __shfl_sync 广播 amount 到所有 lane
    6. Read  Customer
       └─ lane[c_balance]: result += amount
       └─ lane[c_delivery_cnt]: result += 1
    7. Write Customer
```

**特殊之处**:
- Delivery 的参数太大不使用 shared memory 缓存，直接从 global memory 访问
- 使用 `__shfl_sync` 跨 lane 广播 `amount` 累加值
- OrderLine 的 Write 固定使用 `loc_record_b`（不用 Version）

### 5.5 StockLevel (4%)

**只读 + 本地计数**:

```
num_low_stock = 0
for i in 0..num_items (最多 380: 20 订单 × 19 items):
    Read Stock
    if lane[s_quantity_offset] and result < threshold:
        num_low_stock++
params->num_low_stock = num_low_stock  ← 输出结果
```

---

## 6. 存储层: 版本化读写

所有表操作通过 `gpu_storage.cuh` 中的两个函数实现:

### 6.1 gpuReadFromTableCoop

```cpp
// 32 线程 Warp 合作读取一个 Record 或 Version
template<typename ValueType>
__device__ void gpuReadFromTableCoop(
    Record<ValueType> *record,
    Version<ValueType> *version,
    uint32_t record_id,   // 要读的记录 id
    uint32_t read_loc,    // 读写位置 (loc_record_a / loc_record_b / version_index)
    uint32_t epoch,       // 当前 epoch 号
    uint32_t &result,     // 输出: lane_id 对应字段的值
    uint32_t lane_id)
```

**内部逻辑** (leader lane 决定读哪个 value，广播指针后 32 线程并行读):

```
leader lane:
  if read_loc == loc_record_a:
      读 version1, version2
      选 epochs 前的最新版本 → 读 record_a
  else if read_loc == loc_record_b:
      读 version1, version2
      选本 epoch 要写的版本
      如果该版本还没写入: while(version != epoch) {} spin 等待
  else:
      是 version 读，索引到 version[read_loc].value
      如果还没写入: while(version[read_loc].version != epoch) {} spin 等待

所有 lane:
  __shfl 广播 value 指针
  并行读取: lane_i 负责第 i 个 32-bit 字段
```

**Record A vs Record B 判断逻辑**:

```
Record 有两个 value slot (value1, value2) 和两个 version 号 (version1, version2)

Record A (epoch 开始时的状态):
  取 version1 和 version2 中 不是 epoch 且最大 的那个 slot

Record B (本 epoch 第一个写入的状态):
  取 version1 和 version2 中 等于 epoch 的那个 slot
  如果还没有等于 epoch 的 → spin wait
```

### 6.2 gpuWriteToTableCoop

```cpp
template<typename ValueType>
__device__ void gpuWriteToTableCoop(
    Record<ValueType> *record,
    Version<ValueType> *version,
    uint32_t record_id,
    uint32_t write_loc,
    uint32_t epoch,
    uint32_t data,        // lane_id 对应字段要写入的值
    uint32_t lane_id)
```

**内部逻辑**:

```
leader lane:
  if write_loc == loc_record_b:
      找到 version 较小的 slot (即不是 record_a 的那个)
      设置 value_to_write 和 version_to_update 指针
  else:
      是 version 写，设置 value_to_write = &version[write_loc]

所有 lane:
  __shfl 广播 value_to_write 指针
  并行写入: lane_i 写入第 i 个字段

__threadfence()  ← 确保写入对所有线程可见
__syncwarp()     ← Warp 内同步

leader lane:
  atomicExch(version_to_update, epoch)  ← 发布版本号，表示写入完成
```

### 6.3 版本链示例

```
epoch=5 时，Stock[s_id=100] 的 Record 被 3 个事务同时写:

时间 →   t1: Write Record B    t2: Read Record B     t3: Write Version 0
         stock_loc=loc_rec_b   stock_loc=loc_rec_b    stock_loc=0 (version write)

Record[100]:
  value1: {qty=50}  ver1=0   ← record_a (epoch 前的最新值)
  value2: {qty=45}  ver2=5   ← record_b (epoch 5 第一个写)

Version[0]:
  value:  {qty=40}  ver=5    ← t3 的 version write

t1: write value1, set version1=5  (选 version 较小的 slot)
t2: read version1=5 → 发现是 epoch 5 → 读 value1
t3: write Version[0], set version[0]=5
```

---

## 7. 配置参数与启动参数

### Kernel 启动参数

| 参数 | 值 | 计算 |
|------|---|------|
| `block_size` | 128 | 编译期常量 |
| `num_warps` | 4 | = 128 / 32 |
| `num_blocks` | 25,000 | = ceil(100,000 × 32 / 128) |
| 总 threads | 3,200,000 | = 25,000 × 128 |
| 总 warps | 100,000 | = 25,000 × 4 |
| **事务数** | **100,000** | = num_txns |
| Shared memory/block | ~4 KB | 4×(CachableTxnParams + CachableTxnExecPlan) + 4B |

### Shared Memory 缓存策略

| 事务类型 | 使用 shared memory? | 原因 |
|----------|:---:|------|
| NewOrder | ✅ | 参数拟合 CachableTxnParams union |
| Payment | ✅ | 参数拟合 CachableTxnParams union |
| OrderStatus | ✅ | 参数拟合 CachableTxnParams union |
| Delivery | ❌ | 含变长数组 `num_items[10]` + `orderline_ids[10][n]` |
| StockLevel | ❌ | 含变长数组 `stock_ids[n]` (n 可达 380) |

```
static_assert(sizeof(CachableTxnExecPlan) + sizeof(CachableTxnParams) < 1000);
// 确保缓存能放入 ~1KB，留足 shared memory 空间给 4 个 warp
```

---

## 8. 完整数据流

```
Phase 5 产出 (ExecutionPlan)
    │
    │  每个事务的 ExecutionPlan 包含:
    │  ├─ warehouse_loc, district_loc, customer_loc, ...
    │  ├─ 每个 item 的 item_loc, stock_read_loc, stock_write_loc, orderline_loc
    │  └─ 每个字段的 loc 是以下之一:
    │       loc_record_a, loc_record_b, 或 version_index (0,1,2,...)
    │
    ▼
gpuExecKernel<<<25000, 128>>>
    │
    ├── warp_id = threadIdx.x / 32
    ├── lane_id = threadIdx.x % 32
    │
    ├── [调度] 两级 atomicAdd 分配 txn_id
    │
    ├── [加载] warpMemcpy: 32 线程并行拷贝 txn_param + exec_plan → shared memory
    │
    ├── [执行] 按 txn_type 分发:
    │   ┌─────────────────────────────────────────────────────────┐
    │   │ gpuExecTpccTxn(txn_type):                              │
    │   │   反复调用:                                             │
    │   │     gpuReadFromTableCoop(record, version, id, loc, ...)│
    │   │     gpuWriteToTableCoop(record, version, id, loc, ...) │
    │   │                                                        │
    │   │ 每次调用:                                              │
    │   │   lane 0: 决定读/写哪个 value slot (Record A / B / Ver)│
    │   │   lane 0..31: 并行读/写 Record 的 32-bit 字段          │
    │   │   写: __threadfence + atomicExch 发布版本号              │
    │   └─────────────────────────────────────────────────────────┘
    │
    ▼
cudaDeviceSynchronize() ← 等待所有 25,000 blocks 完成
```

**存储模型总结**:

```
每个表的存储结构:
┌──────────────────────────────────────────────────────────┐
│ Record[]  数组 (每行有 value1, value2, version1, version2) │
│ Version[] 数组 (多版本临时存储, 仅当同 record 有多个写时使用) │
│                                                          │
│ 版本选择规则 (由 gpuReadFromTableCoop leader lane 决策):    │
│                                                          │
│ loc_record_a: 读 epoch 前的最新版本                        │
│   → 选 version1/version2 中最大且 ≠ epoch 的那个 slot       │
│                                                          │
│ loc_record_b: 读/写本 epoch 第一个版本                      │
│   → 选 version1/version2 中较小的 slot (将被本 epoch 覆盖)  │
│   → 如果目标 slot 还没写入: 自旋等待 (while ver≠epoch)       │
│                                                          │
│ version_index (≥0): 读/写临时版本                          │
│   → 直接索引 Version[version_index]                        │
│   → 如果还没写入: 自旋等待 (while ver≠epoch)                 │
└──────────────────────────────────────────────────────────┘
```
