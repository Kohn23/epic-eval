//
// Created on 2026-06-14.
//
// Transaction scheduler for TPC-C that reorders transactions so that
// GPU blocks contain similarly-sized transactions operating on the
// same warehouse, preventing long txns from straggling short ones
// and improving version-chain locality.
//

#ifndef TPCC_TXN_SCHEDULER_H
#define TPCC_TXN_SCHEDULER_H

#include <cstdint>
#include <cstring>
#include <vector>

#include "tpcc_config.h"
#include "tpcc_txn.h"
#include "txn.h"
#include "util_log.h"

namespace epic::tpcc {

/** Number of op-count sub-buckets within each warehouse bucket. */
constexpr uint32_t kOpsBuckets = 5;

/**
 * Extracts the primary warehouse_id from any TPC-C transaction input.
 * All TxnInput types place the warehouse_id field at offset 0 of `data`.
 */
inline uint32_t getTpccTxnWarehouseId(BaseTxn *txn)
{
    return reinterpret_cast<uint32_t *>(txn->data)[0];
}

/**
 * Computes the exact number of table operations (reads + writes) that a
 * transaction will perform during execution.  Derived from the per-op counts
 * in TpccGpuSubmitter::prepareSubmitTpccTxn (tpcc_gpu_submitter.cu):
 *
 *   PAYMENT:      wh(2)+dist(2)+cust(2)                      = 6
 *   ORDER_STATUS: cust(1)+order(1)+orderline(n)              = 2 + n
 *   STOCK_LEVEL:  stock(n)                                    = n
 *   NEW_ORDER:    6 + item(n)+orderline(n)+stock(2n)         = 6 + 4n
 *   DELIVERY:     50 + orderline(2*sum)                      = 50 + 2*sum(n)
 */
inline uint32_t getTpccTxnOpCount(BaseTxn *txn)
{
    switch (static_cast<TpccTxnType>(txn->txn_type))
    {
    case TpccTxnType::PAYMENT:
        return 6;
    case TpccTxnType::ORDER_STATUS: {
        auto *p = reinterpret_cast<OrderStatusTxnInput *>(txn->data);
        return 2 + p->num_items;
    }
    case TpccTxnType::STOCK_LEVEL: {
        auto *p = reinterpret_cast<StockLevelTxnInput *>(txn->data);
        return p->num_items;
    }
    case TpccTxnType::NEW_ORDER: {
        auto *p = reinterpret_cast<NewOrderTxnInput<FixedSizeTxn> *>(txn->data);
        return 6 + 4 * p->num_items;
    }
    case TpccTxnType::DELIVERY: {
        auto *p = reinterpret_cast<DeliveryTxnInput *>(txn->data);
        uint32_t total_items = 0;
        for (uint32_t i = 0; i < 10; ++i)
            total_items += p->num_items[i];
        return 50 + 2 * total_items;
    }
    default:
        return 6;
    }
}

/**
 * Map an operation count to a sub-bucket index within a warehouse group.
 *
 * Sub-bucket boundaries (covering payment=6 to delivery=350+):
 *   0:  ops <= 6       (PAYMENT)
 *   1:  ops 7-17       (ORDER_STATUS, small STOCK_LEVEL)
 *   2:  ops 18-35      (small NEW_ORDER, large STOCK_LEVEL)
 *   3:  ops 36-70      (large NEW_ORDER, small DELIVERY)
 *   4:  ops > 70       (DELIVERY)
 */
inline uint32_t getTpccTxnOpsBucket(BaseTxn *txn)
{
    uint32_t ops = getTpccTxnOpCount(txn);
    if (ops <= 6)   return 0;
    if (ops <= 17)  return 1;
    if (ops <= 35)  return 2;
    if (ops <= 70)  return 3;
    return 4;
}

/**
 * Composite sort key: warehouse_id (primary) × kOpsBuckets + ops_bucket (secondary).
 * This groups transactions by warehouse first, then by op-count within
 * each warehouse, giving both version-chain locality and homogeneous blocks.
 */
inline uint32_t getTpccTxnCost(BaseTxn *txn, uint32_t num_warehouses)
{
    uint32_t wh = getTpccTxnWarehouseId(txn);
    // warehouse_id is 1-based, map to 0-based
    uint32_t wh_idx = (wh > 0 && wh <= num_warehouses) ? wh - 1 : 0;
    uint32_t ops_bucket = getTpccTxnOpsBucket(txn);
    return wh_idx * kOpsBuckets + ops_bucket;
}

/**
 * Reorders transactions in a PackedTxnArray by (warehouse_id, op_count) so that:
 * 1. Same-warehouse txns are contiguous → version-chain locality
 * 2. Within each warehouse, similar-length txns are together → homogeneous blocks
 *
 * Algorithm: 3-pass bucket sort
 *   Pass 1 — Count: tally txns per (warehouse × ops_bucket)
 *   Pass 2 — Prefix sum: compute output start offset per bucket
 *   Pass 3 — Scatter: copy each txn to its bucket region in output buffer
 *
 * @param txn_array  The packed transaction array to reorder in-place (CPU-side)
 * @param config     TPC-C configuration (for num_warehouses and logging)
 */
inline void scheduleTpccTxns(PackedTxnArray<TpccTxn> &txn_array, const TpccConfig &config)
{
    auto &logger = Logger::GetInstance();

    if (txn_array.num_txns == 0)
    {
        logger.Info("Scheduler: no txns to schedule, skipping");
        return;
    }

    uint32_t num_buckets = config.num_warehouses * kOpsBuckets;

    /* --- Pass 1: Count txns per bucket --- */
    std::vector<uint32_t> bucket_counts(num_buckets, 0);

    for (uint32_t i = 0; i < txn_array.num_txns; ++i)
    {
        BaseTxn *txn = txn_array.getTxn(i);
        uint32_t bucket = getTpccTxnCost(txn, config.num_warehouses);
        ++bucket_counts[bucket];
    }

    /* --- Pass 2: Compute bucket start offsets (prefix sum of counts) --- */
    std::vector<uint32_t> bucket_starts(num_buckets, 0);
    uint32_t offset = 0;
    for (uint32_t b = 0; b < num_buckets; ++b)
    {
        bucket_starts[b] = offset;
        offset += bucket_counts[b];
    }

    /* bucket_cursors tracks write position within each bucket during scatter */
    std::vector<uint32_t> bucket_cursors = bucket_starts;

    /* --- Pass 3: Build new index and scatter txns --- */
    /* Allocate output buffers */
    size_t txn_capacity = txn_array.capacity;
    uint8_t *old_txns = txn_array.txns;
    uint32_t *old_index = txn_array.index;

    uint8_t *new_txns = static_cast<uint8_t *>(Malloc(txn_capacity));
    uint32_t *new_index = static_cast<uint32_t *>(Malloc((txn_array.num_txns + 1) * sizeof(uint32_t)));

    uint32_t new_byte_offset = 0;

    for (uint32_t i = 0; i < txn_array.num_txns; ++i)
    {
        BaseTxn *txn = txn_array.getTxn(i);
        uint32_t bucket = getTpccTxnCost(txn, config.num_warehouses);

        /* txn byte size: from current index layout */
        uint32_t txn_begin = old_index[i];
        uint32_t txn_end = (i + 1 < txn_array.num_txns) ? old_index[i + 1] : txn_array.size;
        uint32_t txn_size = txn_end - txn_begin;

        /* Place txn in output at new position */
        uint32_t dst_txn_id = bucket_cursors[bucket];
        ++bucket_cursors[bucket];

        /* Copy txn data to new position */
        uint32_t dst_byte_offset = new_byte_offset;
        std::memcpy(&new_txns[dst_byte_offset], &old_txns[txn_begin], txn_size);
        new_index[dst_txn_id] = dst_byte_offset;
        new_byte_offset += txn_size;
    }
    /* Set the final size entry */
    new_index[txn_array.num_txns] = new_byte_offset;

    /* Swap in the new buffers */
    txn_array.txns = new_txns;
    txn_array.index = new_index;
    txn_array.size = new_byte_offset;

    /* Free old buffers */
    Free(old_txns);
    Free(old_index);

    /* Count how many buckets are non-empty for logging */
    uint32_t nonempty = 0;
    for (uint32_t b = 0; b < num_buckets; ++b)
        if (bucket_counts[b] > 0) ++nonempty;

    logger.Info("Scheduler: reordered {} txns into {}/{} non-empty buckets "
                "(warehouses={} × ops_buckets={}), packed size {} bytes",
        txn_array.num_txns, nonempty, num_buckets,
        config.num_warehouses, kOpsBuckets, new_byte_offset);
}

} // namespace epic::tpcc

#endif // TPCC_TXN_SCHEDULER_H
