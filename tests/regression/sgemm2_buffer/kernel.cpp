#include <vx_spawn.h>
#include <vx_virgo.h>
#include <vx_print.h>
#include "common.h"

void kernel_body(kernel_arg_t *arg) {
	// Setup buffer arguments
  vx_printf("kernel start\n");
  auto A_ptr = reinterpret_cast<TYPE*>(arg->A_addr);
  auto B_ptr = reinterpret_cast<TYPE*>(arg->B_addr);
  auto C_ptr = reinterpret_cast<TYPE*>(arg->C_addr);

  // Allocate local memory for the tile of matrix A & B & C
	auto local_ptr = __local_mem(5 * blockDim.x * blockDim.y * sizeof(TYPE));
  auto local_A_0 = (TYPE*)local_ptr;
  auto local_B_0 = (TYPE*)local_ptr + blockDim.x * blockDim.y;
  auto local_A_1 = (TYPE*)local_ptr + 2 * blockDim.x * blockDim.y;
  auto local_B_1 = (TYPE*)local_ptr + 3 * blockDim.x * blockDim.y;
  auto local_C = (TYPE*)local_ptr + 4 * blockDim.x * blockDim.y;
  uint32_t accum_C = 0x0;
  auto size = arg->size;
  const auto tile_size = arg->tile_size;
  auto num_cores = vx_num_cores();

  // Determine global row and column indices
  auto g_row = blockIdx.x * blockDim.x;
  auto g_col = blockIdx.y * blockDim.y;
  //vx_printf("kernel gg %d %d %d\n", g_row, g_col, tile_size);
  // Determine local row and column indices
  auto l_row = threadIdx.x;
  auto l_col = threadIdx.y;
  auto C_ptr_local = &C_ptr[g_row * size + blockIdx.y * blockDim.y];
  vx_barrier(1<<31, vx_num_cores());
  //load A and B
  vortex::virgo::dma_load<TYPE>(
      &A_ptr[g_row * size],
      local_A_0,
      tile_size,
      tile_size,
    size,
      tile_size
    );

    vortex::virgo::dma_load<TYPE>(
      &B_ptr[blockIdx.y * blockDim.y],
      local_B_0,
      tile_size,
      tile_size,
      size,
      tile_size
    );
  vx_barrier(1<<31, num_cores);
  //vx_printf("kernel gemm start %d\n", tile_size);

  TYPE sum(0);
  // Loop over tiles
  for (uint32_t k = 0; k < size; k += tile_size) {
    //vx_printf("loop start fence %d\n", k);
    vortex::virgo::dma_fence(2);
    //vx_printf("num cores %d\n", vx_num_cores());
    vx_barrier(1<<31, vx_num_cores());
    //p is for producer (dma load), and c  is for consumer (matrix multiply!)
    auto local_A_p = (k / tile_size) % 2 ? local_A_0 : local_A_1;
    auto local_B_p = (k / tile_size) % 2 ? local_B_0 : local_B_1;
    auto local_A_c = (k / tile_size) % 2 ? local_A_1 : local_A_0;
    auto local_B_c = (k / tile_size) % 2 ? local_B_1 : local_B_0;

    //compute this tile
    if (k + 4 < size) {
      //vx_printf("dma load core %d warp %d\n", vx_core_id(), vx_warp_id());
      // Load tile of matrix A & B to local memory
      vortex::virgo::dma_load<TYPE>(
        &A_ptr[g_row * size + k + 4],
        local_A_p,
        4,
        4, 
        size,
        4
      );

      vortex::virgo::dma_load<TYPE>(
        &B_ptr[(k + 4) * size + g_col],
        local_B_p,
        4,
        4,
        size,
        4
      );
    }
    for (uint32_t j = 0; j < 4; ++j) {
      TYPE a = local_A_c[threadIdx.x * 4 + j];
      TYPE b = local_B_c[j * 4 + threadIdx.y];
      sum += (a * b);
      // vx_printf("computed partial sum %f %f %f, addr= %p %p, other= %d %d %p %p\n", sum, local_A_c[threadIdx.x * 4 + j], local_B_c[j * 4 + threadIdx.y]
      //   , &local_A_c[threadIdx.x * 4 + j], &local_B_c[j * 4 + threadIdx.y],
      //   threadIdx.x * 4 + j, j * 4 + threadIdx.y, &local_A_c[0], &local_B_c[0]);
    }
  }
  vx_barrier(1<<31, vx_num_cores());
  //x_printf("kernel store C\n");
  C_ptr[(g_row + threadIdx.x) * size + g_col + threadIdx.y] = sum;
  // vx_printf("kernel fin %f, %p, %d, %d, %d, %d, %d, %d, %d\n", 
  //   sum, &C_ptr[(g_row + threadIdx.x) * size + (blockIdx.y * blockDim.y) + threadIdx.y], 
  //   (g_row + threadIdx.x) * size + (blockIdx.y * blockDim.y) + threadIdx.y,
  //   g_row, (blockIdx.y * blockDim.y), threadIdx.x, threadIdx.y, tile_size, size);
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(2, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
