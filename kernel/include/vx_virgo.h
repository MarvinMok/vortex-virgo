// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include "tensor_cfg.h"

#define FP32 0
#define FP16 1

namespace vortex {

namespace virgo {
    
typedef struct {
    uint32_t src_addr_A;
    uint32_t src_addr_B;
    uint32_t dst_addr;
    uint32_t data_type;
    uint32_t num_rows_A; // [rows_A x cols_A] x [cols_A x cols_B] = [rows_A x cols_B]
    uint32_t num_cols_A; 
    uint32_t num_cols_B;
} virgo_compute_t;

typedef struct {
    uint32_t src_addr;
    uint32_t dst_addr;
    uint32_t data_type_size;
    uint32_t num_rows;
    uint32_t num_cols;
    uint32_t src_stride;
    uint32_t dst_stride;
} dma_load_t;


uint32_t dma_counters[32] = {0};
uint32_t dma_tags[32] = {0};

uint32_t compute_counters[32] = {0};
uint32_t compute_tags[32] = {0};

static __attribute__((always_inline)) uint32_t dma_fence(uint32_t num_ops) {
    
    if (num_ops == 0) {
        return 0;
    }
    
    //only one thread needs to read
    vx_tmc_one();
    
    uint32_t local_core_id = vx_core_id(); 
    uint32_t global_warp_id = local_core_id * vx_num_warps() + vx_warp_id();

     if (num_ops > dma_counters[global_warp_id]) {
        num_ops = dma_counters[global_warp_id];
    }
    
    volatile uint32_t* MMIO_READ_ADDR = (volatile uint32_t*)(0x0000F900);
    uint32_t hw_counter = *MMIO_READ_ADDR;
    
    vx_printf("dma fence core: %d, warp: %d, hw: %d, counter: %d, num_ops: %d\n", local_core_id, vx_warp_id(), hw_counter, dma_counters[global_warp_id], num_ops);
    
    while (dma_counters[global_warp_id] != 0 && hw_counter > dma_counters[global_warp_id] - num_ops) {
        hw_counter = *MMIO_READ_ADDR;
    }
    
    if (dma_counters[global_warp_id] > 0) {
        dma_counters[global_warp_id]-= num_ops;
    }
    //set all threads active
    vx_tmc(-1);
    
    return dma_counters[global_warp_id];
}

static __attribute__((always_inline)) uint32_t compute_fence(uint32_t num_ops) {
    
    if (num_ops == 0) {
        return 0;
    }

    //only one thread needs to read
    vx_tmc_one();
    uint32_t local_core_id = vx_core_id(); 
    uint32_t global_warp_id = local_core_id * vx_num_warps() + vx_warp_id();

    if (num_ops > compute_counters[global_warp_id]) {
        num_ops = compute_counters[global_warp_id];
    }

    volatile uint32_t* MMIO_READ_ADDR = (volatile uint32_t*)(0x0000EC00);
    uint32_t hw_counter = *MMIO_READ_ADDR;

    vx_printf("compute fence core: %d, warp: %d, hw: %d, counter: %d, num_ops: %d\n", local_core_id, vx_warp_id(), hw_counter, compute_counters[global_warp_id], num_ops);

    while (compute_counters[global_warp_id] != 0 && hw_counter > compute_counters[global_warp_id] - num_ops) {
        hw_counter = *MMIO_READ_ADDR;
    }

    
    if (compute_counters[global_warp_id] > 0) {
        compute_counters[global_warp_id] -= num_ops;
    }
    //set all threads active
    vx_tmc(-1);

    return compute_counters[global_warp_id];
}


template <typename T>
static __attribute__((always_inline)) uint32_t dma_load(T* src_addr, T* dst_addr, uint32_t num_rows, uint32_t num_cols, uint32_t src_stride, uint32_t dst_stride) {
   
    //only one core does the dma load
    vx_tmc_one();

    uint32_t core_id = vx_core_id();
    uint32_t wid = vx_warp_id();
    uint32_t num_cores = vx_num_cores(); // Total cores in the cluster/processor context
    uint32_t global_warp_id = core_id * vx_num_warps() + wid;

    dma_load_t dma_load = {
        .src_addr = reinterpret_cast<uint32_t>(src_addr),
        .dst_addr = reinterpret_cast<uint32_t>(dst_addr),
        .data_type_size = sizeof(T),
        .num_rows = num_rows,
        .num_cols = num_cols,
        .src_stride = src_stride,
        .dst_stride = dst_stride
    };
    
    // Calculate offset based on global warp ID to avoid collision in MMIO space if hardware supports it
    // Assuming MMIO space is partitioned per warp as implied by previous code
    uint32_t offset = global_warp_id * (sizeof(dma_load_t) + 4);
    
    // vx_printf("dma_load: core=%d warp=%d offset=%d tag=%d\n", core_id, wid, offset, dma_tags[global_warp_id]);
    
    volatile uint32_t* MMIO_WRITE_ADDR = (volatile uint32_t*)(0x0000F800 + offset);
    MMIO_WRITE_ADDR[0] = dma_load.src_addr;
    MMIO_WRITE_ADDR[1] = dma_load.dst_addr;
    MMIO_WRITE_ADDR[2] = dma_load.data_type_size;
    MMIO_WRITE_ADDR[3] = dma_load.num_rows;
    MMIO_WRITE_ADDR[4] = dma_load.num_cols;
    MMIO_WRITE_ADDR[5] = dma_load.src_stride;
    MMIO_WRITE_ADDR[6] = dma_load.dst_stride;
    
    volatile uint32_t* MMIO_COMMIT_ADDR = MMIO_WRITE_ADDR + 7;
    *(MMIO_COMMIT_ADDR) = dma_tags[global_warp_id];

    //set all threads active
    dma_tags[global_warp_id] = (dma_tags[global_warp_id] + 1) % 4;
    dma_counters[global_warp_id]++;
    
    vx_tmc(-1);

    return 0;
}

template <typename T>
static __attribute__((always_inline)) uint32_t compute(T* src_addr_A, T* src_addr_B, T* dst_addr, uint32_t num_rows_A, uint32_t num_cols_A, uint32_t num_cols_B) {
   
    //only one thread per warp
    vx_tmc_one();

    uint32_t core_id = vx_core_id();
    uint32_t wid = vx_warp_id();
    uint32_t num_cores = vx_num_cores(); // Total cores in the cluster/processor context
    uint32_t global_warp_id = core_id * vx_num_warps() + wid;

    virgo_compute_t virgo_compute;
    if (std::is_same_v<T, float>) {
        virgo_compute.src_addr_A = reinterpret_cast<uint32_t>(src_addr_A);
        virgo_compute.src_addr_B = reinterpret_cast<uint32_t>(src_addr_B);
        virgo_compute.dst_addr = reinterpret_cast<uint32_t>(dst_addr);
        virgo_compute.data_type = 0;
        virgo_compute.num_rows_A = num_rows_A;
        virgo_compute.num_cols_A = num_cols_A;
        virgo_compute.num_cols_B = num_cols_B;
    } else if (std::is_same_v<T, vortex::tensor::fp16>) {
        virgo_compute.src_addr_A = reinterpret_cast<uint32_t>(src_addr_A);
        virgo_compute.src_addr_B = reinterpret_cast<uint32_t>(src_addr_B);
        virgo_compute.dst_addr = reinterpret_cast<uint32_t>(dst_addr);
        virgo_compute.data_type = 1; // other data types, only FP16 rn
        virgo_compute.num_rows_A = num_rows_A;
        virgo_compute.num_cols_A = num_cols_A;
        virgo_compute.num_cols_B = num_cols_B;
    }
    
    // start of MMIO_VIRGO addressing E800 is start of MMIO_VIRGO addressing
    volatile uint32_t* MMIO_VIRGO_WRITE_ADDR = (volatile uint32_t*) (0x0000E800 + global_warp_id * (sizeof(virgo_compute_t) + 4));
    MMIO_VIRGO_WRITE_ADDR[0] = virgo_compute.src_addr_A;
    MMIO_VIRGO_WRITE_ADDR[1] = virgo_compute.src_addr_B;
    MMIO_VIRGO_WRITE_ADDR[2] = virgo_compute.dst_addr;
    MMIO_VIRGO_WRITE_ADDR[3] = virgo_compute.data_type;
    MMIO_VIRGO_WRITE_ADDR[4] = virgo_compute.num_rows_A;
    MMIO_VIRGO_WRITE_ADDR[5] = virgo_compute.num_cols_A;
    MMIO_VIRGO_WRITE_ADDR[6] = virgo_compute.num_cols_B;

    volatile uint32_t* MMIO_VIRGO_COMMIT_ADDR = MMIO_VIRGO_WRITE_ADDR + 7;
    *(MMIO_VIRGO_COMMIT_ADDR) = compute_tags[global_warp_id]; // write tag to COMMIT address

    compute_tags[global_warp_id] = (compute_tags[global_warp_id] + 1) % 4;

    //set all threads active
    vx_tmc(-1);

    return 0;
}

} // namespace virgo

} // namespace vortex


