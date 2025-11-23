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

namespace vortex {

namespace virgo {
    
typedef struct {
    uint32_t src_addr;
    uint32_t dst_addr;
    uint32_t data_type_size;
    uint32_t num_rows;
    uint32_t num_cols;
    uint32_t row_stride;
    uint32_t core_id;
    uint32_t wid;
} dma_load_t;

uint32_t counter = 0;

static __attribute__((always_inline)) uint32_t fence() {

    //only one thread needs to read
    vx_tmc_one();
    
    uint32_t local_core_id = vx_core_id(); // % cores_per_cluster;
    uint32_t* MMIO_READ_ADDR = (uint32_t*)(0x0000F900);
    uint32_t hw_counter = *MMIO_READ_ADDR;
    vx_printf("fence core: %d, fence: %d\n", local_core_id, hw_counter);
    while (counter != 0 && counter >= hw_counter) {
        hw_counter = *MMIO_READ_ADDR;
    }
    
    //set all threads active
    vx_tmc(-1);
    counter = counter == 0 ? 0 : counter - 1;
    //uint32_t cores_per_cluster = vx_num_cores() / vx_num_clusters();

    return counter;
}

template <typename T>
static __attribute__((always_inline)) uint32_t dma_load(T* src_addr, T* dst_addr, uint32_t num_rows, uint32_t num_cols, uint32_t row_stride) {
   
    //only one core does the dma load
    vx_tmc_one();

    uint32_t core_id = vx_core_id();
    uint32_t wid = vx_warp_id();

    dma_load_t dma_load = {
        .src_addr = reinterpret_cast<uint32_t>(src_addr),
        .dst_addr = reinterpret_cast<uint32_t>(dst_addr),
        .data_type_size = sizeof(T),
        .num_rows = num_rows,
        .num_cols = num_cols,
        .row_stride = row_stride,
        .core_id = core_id, //maybe unused
        .wid = wid
    };
    
    dma_load_t* MMIO_WRITE_ADDR = (dma_load_t* )0x0000F800;
    uint8_t* MMIO_COMMIT_ADDR = (uint8_t*)0x0000F840;
    *(MMIO_WRITE_ADDR) = dma_load;
    *(MMIO_COMMIT_ADDR) = vx_core_id() * vx_num_warps() + vx_warp_id(); //write unique values to prevent memory coalescence

    //set all threads active
    vx_tmc(-1);
    counter++;

    return 0;
}

} // namespace virgo

} // namespace vortex


