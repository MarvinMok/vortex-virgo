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
  uint32_t accum_C = 12345;
  auto size = arg->size;
  auto tile_size = arg->tile_size;

  // Determine global row and column indices
  auto g_row = blockIdx.x * blockDim.x;
  auto g_col = blockIdx.y * blockDim.y;

  // Determine local row and column indices
  auto l_row = threadIdx.x;
  auto l_col = threadIdx.y;

  auto C_ptr_local = &C_ptr[g_row * size + g_col];
  vortex::virgo::dma_load<TYPE>(
    C_ptr_local,
    local_C,
    tile_size,
    tile_size,
    size, 
    tile_size       
  );
  vx_barrier(1<<31, vx_num_cores());
  vortex::virgo::dma_fence(1);
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
      &B_ptr[g_col],
      local_B_0,
      tile_size,
      tile_size,
      size,
      tile_size
    );
  vx_printf("kernel gemm start %d\n", tile_size);

  
  // Loop over tiles
  for (uint32_t k = 0; k < size; k += tile_size) {
    vx_printf("loop start fence %d\n", k);
    vortex::virgo::dma_fence(2);
    vx_printf("num cores %d\n", vx_num_cores());
    vx_barrier(1<<31, vx_num_cores());
    //p is for producer (dma load), and c  is for consumer (matrix multiply!)
    //hard coded since threadblock size is 4 by 4, so tile_size is 4
    auto local_A_p = (k >> 2) & 1 ? local_A_0 : local_A_1;
    auto local_B_p = (k >> 2) & 1 ? local_B_0 : local_B_1;
    auto local_A_c = (k >> 2) & 1 ? local_A_1 : local_A_0;
    auto local_B_c = (k >> 2) & 1 ? local_B_1 : local_B_0;
    bool accum = (k != 0);
    bool store = (k + tile_size >= size);

    // pack accumulator address, accum, and store flags together
    // store flags tells us whether to write back to memory or not
    // accum flag tells us whether to add to value in accumulator memory, or to simply just store
    // basically, store = 1 on last compute call for an output tile, and accum = 1 at the start of a new compuete call for an output tile
    // idk why but for some reason doing this inside vx_virgo made it break... i think its a compiler error or smtg
    uint32_t accum_packed = (accum_C & 0x7FFF) | (accum << 31) | (store << 30);
    //compute this tile
    vortex::virgo::compute<float>(local_A_c, local_B_c, local_C, accum_packed, tile_size, tile_size, tile_size);
    if (k + tile_size < size) {
      vx_printf("dma load core %d warp %d\n", vx_core_id(), vx_warp_id());
      // Load tile of matrix A & B to local memory
      vortex::virgo::dma_load<TYPE>(
        &A_ptr[g_row * size + k + tile_size],
        local_A_p,
        tile_size,
        tile_size, 
        size,
        tile_size
      );

      vortex::virgo::dma_load<TYPE>(
        &B_ptr[(k + tile_size) * size + g_col],
        local_B_p,
        tile_size,
        tile_size,
        size,
        tile_size
      );
    }
   
    vx_printf("compute fence %d\n", k);
    vortex::virgo::compute_fence(1);
    vx_printf("here2\n");
    
  }

  vx_printf("kernel store C\n");

  // Store tile of matrix C from local memory
  vortex::virgo::dma_load<TYPE>(
    local_C,
    C_ptr_local,
    tile_size,
    tile_size,
    tile_size, 
    size       
  );
  vortex::virgo::dma_fence(1);
  vx_printf("kernel fin\n");

}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(2, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
