//
// Created on 2026-06-14.
//
// Transaction scheduler for TPC-C that reorders transactions by operation cost
// so that GPU blocks contain similarly-sized transactions, preventing long txns
// from blocking short ones within the same warp/block.
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
        return 6; /* unknown types fallback to medium */
    }
}

/**
 * Map an operation count to a cost bucket index.
 *
 * Bucket boundaries (covering payment=6 to delivery=350+):
 *   bucket 0:  ops <= 6       (PAYMENT)
 *   bucket 1:  ops 7-17       (ORDER_STATUS, small STOCK_LEVEL)
 *   bucket 2:  ops 18-35      (small NEW_ORDER, large STOCK_LEVEL)
 *   bucket 3:  ops 36-70      (large NEW_ORDER, small DELIVERY)
 *   bucket 4:  ops > 70       (DELIVERY)
 */
inline uint32_t getTpccTxnCost(BaseTxn *txn)
{
    uint32_t ops = getTpccTxnOpCount(txn);
    if (ops <= 6)   return 0;
    if (ops <= 17)  return 1;
    if (ops <= 35)  return 2;
    if (ops <= 70)  return 3;
    return 4;
}

/** Number of distinct cost buckets. Must be >= max(getTpccTxnCost) + 1. */
constexpr uint32_t kNumCostBuckets = 5;

/**
 * Reorders transactions in a PackedTxnArray by operation cost so that
 * similarly-sized transactions are contiguous. This causes GPU blocks
 * (which consume txns sequentially) to be homogeneously loaded,
 * preventing long txns from straggling short ones.
 *
 * Algorithm: 3-pass bucket sort
 *   Pass 1 — Count: tally txns per cost bucket
 *   Pass 2 — Prefix sum: compute output start offset per bucket
 *   Pass 3 — Scatter: copy each txn to its bucket region in output buffer
 *
 * @param txn_array  The packed transaction array to reorder in-place (CPU-side)
 * @param config     TPC-C configuration (for logging)
 */
inline void scheduleTpccTxns(PackedTxnArray<TpccTxn> &txn_array, const TpccConfig &config)
{
    auto &logger = Logger::GetInstance();

    if (txn_array.num_txns == 0)
    {
        logger.Info("Scheduler: no txns to schedule, skipping");
        return;
    }

    /* --- Pass 1: Count txns per bucket --- */
    std::vector<uint32_t> bucket_counts(kNumCostBuckets, 0);

    for (uint32_t i = 0; i < txn_array.num_txns; ++i)
    {
        BaseTxn *txn = txn_array.getTxn(i);
        uint32_t cost = getTpccTxnCost(txn);
        ++bucket_counts[cost];
    }

    /* --- Pass 2: Compute bucket start offsets (prefix sum of counts) --- */
    std::vector<uint32_t> bucket_starts(kNumCostBuckets, 0);
    uint32_t offset = 0;
    for (uint32_t b = 0; b < kNumCostBuckets; ++b)
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
        uint32_t cost = getTpccTxnCost(txn);

        /* txn byte size: from current index layout */
        uint32_t txn_begin = old_index[i];
        uint32_t txn_end = (i + 1 < txn_array.num_txns) ? old_index[i + 1] : txn_array.size;
        uint32_t txn_size = txn_end - txn_begin;

        /* Place txn in output at new position */
        uint32_t dst_txn_id = bucket_cursors[cost];
        ++bucket_cursors[cost];

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

    logger.Info("Scheduler: reordered {} txns into {} cost buckets, packed size {} bytes",
        txn_array.num_txns, kNumCostBuckets, new_byte_offset);
}

} // namespace epic::tpcc

#endif // TPCC_TXN_SCHEDULER_H
