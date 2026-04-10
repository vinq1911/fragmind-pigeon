/*
 * fragpigeon.h — C definitions for fragmind-pigeon shared-memory layouts.
 *
 * These structs match the Go binary layout byte-for-byte. No cgo needed:
 * just mmap the same shm files the Go pigeon creates and cast the pointers.
 *
 * All multi-byte fields are little-endian.
 */

#ifndef FRAGPIGEON_H
#define FRAGPIGEON_H

#include <stdint.h>
#include <stdatomic.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Ring Buffer (SPSC shared-memory)
 * ================================================================
 *
 * Layout: [RingCtrl: 64B] [Slot 0] [Slot 1] ...
 * Each slot: [Header: 64B] [Payload: hdr.len bytes]
 *
 * Producer: atomic store ProdIdx after writing slot
 * Consumer: atomic store ConsIdx after reading slot
 */

typedef struct {
    _Atomic uint64_t cap_slots;    /* number of slots (power of 2) */
    _Atomic uint64_t prod_idx;     /* producer write index */
    _Atomic uint64_t cons_idx;     /* consumer read index */
    uint32_t         slot_size;    /* bytes per slot */
    uint32_t         _pad0;
    uint64_t         prod_evtfd;   /* eventfd for producer (0xFFFFFFFFFFFFFFFF = none) */
    uint64_t         cons_evtfd;   /* eventfd for consumer (0xFFFFFFFFFFFFFFFF = none) */
    uint64_t         _reserved[4];
} fp_ring_ctrl_t;

_Static_assert(sizeof(fp_ring_ctrl_t) == 64, "ring ctrl must be 64 bytes");

/* ================================================================
 * Message Header (64 bytes, little-endian)
 * ================================================================ */

#define FP_HDR_SIZE 64

/* Message kinds */
#define FP_KIND_PROCESS  0x0001
#define FP_KIND_LEARN    0x0002
#define FP_KIND_SHARE    0x0003
#define FP_KIND_PING     0x0004

/* Message flags */
#define FP_FLAG_END_OF_STREAM 0x0001
#define FP_FLAG_URGENT        0x0002
#define FP_FLAG_REPLY         0x0004
#define FP_FLAG_DROP_OK       0x0008
#define FP_FLAG_LOA_PTR       0x0010

typedef struct {
    uint32_t len;            /* payload length in bytes */
    uint16_t kind;           /* FP_KIND_* */
    uint16_t flags;          /* FP_FLAG_* */
    uint64_t ts_ns;          /* timestamp (nanoseconds since epoch) */
    uint64_t concept_id;     /* concept identifier for routing */
    uint16_t concept_bits;   /* prefix bits for COI matching */
    uint16_t schema_id;      /* payload schema (FP_SCHEMA_*) */
    uint32_t src_id;         /* source fragment ID */
    uint32_t msg_id;         /* message sequence number */
    uint16_t hop;            /* hop count */
    uint16_t ver;            /* protocol version */
    uint64_t trace_id;       /* distributed trace ID */
    uint32_t checksum32;     /* CRC32 of payload */
    uint32_t _reserved;
} fp_header_t;

_Static_assert(sizeof(fp_header_t) == 64, "header must be 64 bytes");

/* ================================================================
 * Schema IDs
 * ================================================================ */

#define FP_SCHEMA_RAW           0
#define FP_SCHEMA_WEIGHT_SHARD  1
#define FP_SCHEMA_KV_CACHE      2
#define FP_SCHEMA_ACTIVATION    3
#define FP_SCHEMA_TOKEN_BATCH   4
#define FP_SCHEMA_GRADIENT      5
#define FP_SCHEMA_CONTROL       0xFFFF

/* Data types */
#define FP_DTYPE_F32    0
#define FP_DTYPE_F16    1
#define FP_DTYPE_BF16   2
#define FP_DTYPE_FP8E4  3
#define FP_DTYPE_FP8E5  4
#define FP_DTYPE_I8     5
#define FP_DTYPE_I32    6
#define FP_DTYPE_U32    7

/* ================================================================
 * LOA (Large Object Attach) Pool
 * ================================================================
 *
 * Layout: [LOAHeader: 64B] [SlotMeta[0]: 32B] ... [padding] [Data region]
 *
 * Data region is page-aligned (4096B). Each slot is slot_size bytes,
 * contiguous in the data region. Adjacent slots are adjacent in memory.
 */

#define FP_LOA_MAGIC 0x4C4F41504F4F4C31ULL  /* "LOAPOOL1" */

typedef struct {
    uint64_t magic;          /* FP_LOA_MAGIC */
    uint32_t version;
    uint32_t num_slots;
    uint32_t slot_size;      /* bytes per slot */
    uint16_t pool_id;
    uint8_t  _pad0[2];
    uint32_t data_base;      /* byte offset where slot data begins */
    uint8_t  _reserved[32];
} fp_loa_header_t;

_Static_assert(sizeof(fp_loa_header_t) == 64, "LOA header must be 64 bytes");

/* Slot states */
#define FP_SLOT_FREE       0
#define FP_SLOT_ALLOCATING 1
#define FP_SLOT_READY      2

typedef struct {
    _Atomic uint32_t state;     /* FP_SLOT_* */
    _Atomic int32_t  ref_cnt;   /* reference count */
    uint32_t         owner;     /* owning fragment ID */
    uint32_t         size;      /* actual payload size */
    uint8_t          _reserved[16];
} fp_slot_meta_t;

_Static_assert(sizeof(fp_slot_meta_t) == 32, "slot meta must be 32 bytes");

/* LOA reference (sent over ring when FP_FLAG_LOA_PTR is set) */
typedef struct {
    uint16_t pool_id;
    uint16_t slot_id;
    uint32_t offset;
    uint32_t length;
} fp_loa_ref_t;

_Static_assert(sizeof(fp_loa_ref_t) == 12, "LOA ref must be 12 bytes");

/* Multi-slot LOA reference */
typedef struct {
    uint16_t pool_id;
    uint16_t start_slot;
    uint16_t num_slots;
    uint16_t _pad;
    uint32_t offset;
    uint32_t length;
} fp_loa_multi_ref_t;

_Static_assert(sizeof(fp_loa_multi_ref_t) == 16, "LOA multi ref must be 16 bytes");

/* ================================================================
 * COI (Concept of Interest)
 * ================================================================
 *
 * Shared-memory COI table layout:
 * [COIHeader: 64B] [Entry[0]: 16B] [Entry[1]: 16B] ...
 *
 * Uses seqlock: reader spins while header.seq is odd.
 */

typedef struct {
    _Atomic uint64_t seq;        /* seqlock: odd = writing */
    _Atomic uint32_t version;
    _Atomic uint32_t count;      /* number of entries */
    _Atomic uint64_t updated_ns; /* last update timestamp */
    uint8_t          _reserved[48];
} fp_coi_header_t;

_Static_assert(sizeof(fp_coi_header_t) == 64, "COI header must be 64 bytes");

typedef struct {
    uint64_t concept_id;
    uint16_t bits;
    uint16_t schema_id;
    uint16_t flags;
    uint16_t _pad;
} fp_coi_entry_t;

_Static_assert(sizeof(fp_coi_entry_t) == 16, "COI entry must be 16 bytes");

/* ================================================================
 * Schema Metadata Structs
 * ================================================================ */

typedef struct {
    uint32_t model_id;
    uint16_t layer_start;
    uint16_t layer_end;
    uint8_t  dtype;
    uint8_t  _pad[3];
    uint32_t num_elements;
    uint16_t shape[4];
    uint32_t checksum;
} fp_weight_shard_meta_t;

_Static_assert(sizeof(fp_weight_shard_meta_t) == 32, "weight shard meta must be 32 bytes");

typedef struct {
    uint32_t model_id;
    uint16_t layer;
    uint16_t head_start;
    uint16_t head_end;
    uint32_t seq_start;
    uint32_t seq_len;
    uint16_t head_dim;
    uint8_t  dtype;
    uint8_t  _pad;
    uint32_t num_elements;
    uint32_t checksum;
} fp_kv_cache_meta_t;

_Static_assert(sizeof(fp_kv_cache_meta_t) == 32, "kv cache meta must be 32 bytes");

typedef struct {
    uint32_t model_id;
    uint16_t layer;
    uint8_t  dtype;
    uint8_t  _pad;
    uint32_t batch_idx;
    uint32_t seq_len;
    uint32_t hidden_dim;
    uint32_t checksum;
} fp_activation_meta_t;

_Static_assert(sizeof(fp_activation_meta_t) == 24, "activation meta must be 24 bytes");

typedef struct {
    uint32_t model_id;
    uint32_t batch_idx;
    uint32_t num_tokens;
    uint16_t max_seq_len;
    uint16_t _pad;
} fp_token_batch_meta_t;

_Static_assert(sizeof(fp_token_batch_meta_t) == 16, "token batch meta must be 16 bytes");

/* ================================================================
 * Inline helpers
 * ================================================================ */

static inline void *fp_ring_slot(fp_ring_ctrl_t *ctrl, void *base, uint64_t idx) {
    uint64_t cap = atomic_load_explicit(&ctrl->cap_slots, memory_order_relaxed);
    uint64_t off = (idx & (cap - 1)) * ctrl->slot_size;
    return (char *)base + off;
}

static inline int fp_ring_try_write(fp_ring_ctrl_t *ctrl, void *slots_base,
                                     const fp_header_t *hdr, const void *payload) {
    uint64_t prod = atomic_load_explicit(&ctrl->prod_idx, memory_order_acquire);
    uint64_t cons = atomic_load_explicit(&ctrl->cons_idx, memory_order_acquire);
    uint64_t cap  = atomic_load_explicit(&ctrl->cap_slots, memory_order_relaxed);
    if (prod - cons >= cap) return 0; /* full */
    if (hdr->len + FP_HDR_SIZE > ctrl->slot_size) return 0; /* too big */

    void *slot = fp_ring_slot(ctrl, slots_base, prod);
    __builtin_memcpy(slot, hdr, FP_HDR_SIZE);
    __builtin_memcpy((char *)slot + FP_HDR_SIZE, payload, hdr->len);
    atomic_store_explicit(&ctrl->prod_idx, prod + 1, memory_order_release);
    return 1;
}

static inline int fp_ring_try_read(fp_ring_ctrl_t *ctrl, void *slots_base,
                                    fp_header_t *hdr_out, void **payload_out) {
    uint64_t prod = atomic_load_explicit(&ctrl->prod_idx, memory_order_acquire);
    uint64_t cons = atomic_load_explicit(&ctrl->cons_idx, memory_order_acquire);
    if (prod == cons) return 0; /* empty */

    void *slot = fp_ring_slot(ctrl, slots_base, cons);
    __builtin_memcpy(hdr_out, slot, FP_HDR_SIZE);
    *payload_out = (char *)slot + FP_HDR_SIZE;
    atomic_store_explicit(&ctrl->cons_idx, cons + 1, memory_order_release);
    return 1;
}

static inline void *fp_loa_slot_data(fp_loa_header_t *hdr, void *pool_base, uint16_t slot_id) {
    return (char *)pool_base + hdr->data_base + (uint64_t)slot_id * hdr->slot_size;
}

static inline fp_slot_meta_t *fp_loa_slot_meta(void *pool_base, uint16_t slot_id) {
    return (fp_slot_meta_t *)((char *)pool_base + 64 + (uint64_t)slot_id * 32);
}

#ifdef __cplusplus
}
#endif

#endif /* FRAGPIGEON_H */
