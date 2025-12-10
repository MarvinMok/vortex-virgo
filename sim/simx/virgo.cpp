#include <iostream>
#include <iomanip>
#include <string.h>
#include <assert.h>
#include <util.h>
#include "types.h"
#include "arch.h"
#include "mem.h"
#include "cluster.h"
#include "virgo.h"
#include <bitset>
#include "tensor_cfg.h"

#define FP32 0
#define FP16 1

using namespace vortex;

AccumulatorMem::AccumulatorMem(const SimContext& ctx, const char* name, uint64_t capacity)
    : SimObject(ctx, name)
    , ReadReqIn(this)
    , ReadRspOut(this)
    , WriteReqIn(this)
    , WriteRspOut(this)
    , ram_(capacity)
    , mmu_(0)
    , capacity_(capacity) {
    mmu_.attach(ram_, 0, capacity - 1);
}

AccumulatorMem::~AccumulatorMem() {}

void AccumulatorMem::read(void* data, uint64_t addr, uint32_t size) {
    mmu_.read(data, addr, size, 0);
}

void AccumulatorMem::write(const void* data, uint64_t addr, uint32_t size) {
    mmu_.write(data, addr, size, 0);
}

void AccumulatorMem::tick() {
    // Handle Read Requests
    if (!ReadReqIn.empty()) {
        auto& req = ReadReqIn.front();
        // Assert address adjacency
        uint64_t base_addr = req.addrs[0];
        if (!req.mask.all()) {
            std::cout << "Warning: AccumulatorMem ReadReq mask not all valid!" << std::endl;
        }
        for (size_t i = 1; i < req.addrs.size(); ++i) {
            if (req.mask.test(i)) {
                // Assuming 4-byte words, adjacent means +4
                if (req.addrs[i] != base_addr + i * 4) {
                    std::cout << "Warning: AccumulatorMem ReadReq addresses not adjacent!" << std::endl;
                }
            }
        }
        
        // Send Response
        LsuRsp rsp(req.mask.size());
        rsp.tag = req.tag;
        rsp.cid = req.cid;
        rsp.uuid = req.uuid;
        rsp.mask = req.mask;
        ReadRspOut.push(rsp, 1);
        ReadReqIn.pop();
    }

    // Handle Write Requests
    if (!WriteReqIn.empty()) {
        auto& req = WriteReqIn.front();
        // Assert address adjacency
        uint64_t base_addr = req.addrs[0];
        if (!req.mask.all()) {
            std::cout << "Error: AccumulatorMem WriteReq mask not all valid!" << std::endl;
            std::abort();
        }
        for (size_t i = 1; i < req.addrs.size(); ++i) {
            if (req.mask.test(i)) {
                // Assuming 4-byte words, adjacent means +4
                if (req.addrs[i] != base_addr + i * 4) {
                    std::cout << "Error: AccumulatorMem WriteReq addresses not adjacent!" << std::endl;
                    std::abort();
                }
            }
        }
        
        // Send Response
        LsuRsp rsp(req.mask.size());
        rsp.tag = req.tag;
        rsp.cid = req.cid;
        rsp.uuid = req.uuid;
        rsp.mask = req.mask;
        WriteRspOut.push(rsp, 1);
        WriteReqIn.pop();
    }
}
void AccumulatorMem::reset() {
    for (uint64_t i = 0; i < ram_.size(); i += 4) {
        uint32_t zero = 0;
        ram_.write(&zero, i, 4);
    }
}
uint64_t AccumulatorMem::size() const { return capacity_; }

ScratchPadMem::ScratchPadMem(const SimContext& ctx, const char* name, uint64_t capacity)
    : SimObject(ctx, name)
    , ReadReqIn(this)
    , ReadRspOut(this)
    , WriteReqIn(this)
    , WriteRspOut(this)
    , BankWriteReqOut(SCRATCHPAD_BANKS, this)
    , BankWriteRspIn(SCRATCHPAD_BANKS, this)
    , BankReadReqOut(SCRATCHPAD_BANKS, this)
    , BankReadRspIn(SCRATCHPAD_BANKS, this)
    , ram_(capacity)
    , mmu_(0)
    , capacity_(capacity) {
    mmu_.attach(ram_, 0, capacity - 1);
}

ScratchPadMem::~ScratchPadMem() {}

void ScratchPadMem::read(void* data, uint64_t addr, uint32_t size) {
    mmu_.read(data, addr, size, 0);
}

void ScratchPadMem::write(const void* data, uint64_t addr, uint32_t size) {
    mmu_.write(data, addr, size, 0);
}

void ScratchPadMem::tick() {
    
    //handle writes
    if (!WriteReqIn.empty()) {
        auto& req = WriteReqIn.front();
        // Assert address adjacency
        uint64_t base_addr = req.addrs[0];
        if (!req.mask.all()) { // signifies that the LsuReq contains the maximum number of memory addresses (which we want)
            std::cout << "Error: ScratchPadMem WriteReq mask not all valid!" << std::endl;
            std::abort();
        }
        for (size_t i = 1; i < req.addrs.size(); ++i) {
            if (req.mask.test(i)) {
                if (req.addrs[i] != base_addr + i * 4) {
                    std::cout << "Error: ScratchPadMem WriteReq addresses not adjacent!" << std::endl;
                    std::abort();
                }
            }
        }
        
        // Split to banks
        for (size_t i = 0; i < req.mask.size(); ++i) {
            if (req.mask.test(i)) {
                uint64_t addr = req.addrs[i];
                uint32_t bank_id = (addr / 4) % SCRATCHPAD_BANKS;
                
                MemReq mem_req;
                mem_req.addr = addr;
                mem_req.write = true;
                mem_req.tag = req.tag;
                mem_req.cid = req.cid;
                mem_req.uuid = req.uuid;
                
                BankWriteReqOut.at(bank_id).push(mem_req, 1);
            }
        }
        WriteReqIn.pop();
    
    }
    
    //handle reads
    if (!ReadReqIn.empty()) {
        auto& req = ReadReqIn.front();
        // Assert address adjacency
        uint64_t base_addr = req.addrs[0];
        if (!req.mask.all()) {
            std::cout << "Error: AccumulatorMem ReadReq mask not all valid!" << std::endl;
            std::abort();
        }
        for (size_t i = 1; i < req.addrs.size(); ++i) {
            if (req.mask.test(i)) {
                if (req.addrs[i] != base_addr + i * 4) {
                    std::cout << "Error: ScratchPadMem ReadReq addresses not adjacent!" << std::endl;
                    std::abort();
                }
            }
        }

        for (size_t i = 0; i < req.mask.size(); ++i) {
            if (req.mask.test(i)) {
                uint64_t addr = req.addrs[i];
                uint32_t bank_id = (addr / 4) % SCRATCHPAD_BANKS;
                
                MemReq mem_req;
                mem_req.addr = addr;
                mem_req.write = false;
                mem_req.tag = req.tag;
                mem_req.cid = req.cid;
                mem_req.uuid = req.uuid;
                
                BankReadReqOut.at(bank_id).push(mem_req, 1);
            }
        }
        ReadReqIn.pop();
    }

    //each bank handles one read and one write request at a time;
    for (uint32_t i = 0; i < BankReadReqOut.size(); ++i) {
        if (!BankReadReqOut.at(i).empty()) {
            auto& req = BankReadReqOut.at(i).front();
            MemRsp bank_rsp{req.tag, req.cid, req.uuid};
            BankReadRspIn.at(i).push(bank_rsp, 1);
            BankReadReqOut.at(i).pop();
        }
        if (!BankWriteReqOut.at(i).empty()) {
            auto& req = BankWriteReqOut.at(i).front();
            MemRsp bank_rsp{req.tag, req.cid, req.uuid};
            BankWriteRspIn.at(i).push(bank_rsp, 1);
            BankWriteReqOut.at(i).pop();
        }

    }

    //handle Bank read response
    for (uint32_t i = 0; i < BankReadRspIn.size(); ++i) {
        if (!BankReadRspIn.at(i).empty()) {
            auto& rsp = BankReadRspIn.at(i).front();
            LsuRsp in_rsp(BankReadRspIn.size());
            in_rsp.mask.set(i);
            in_rsp.tag = rsp.tag;
            in_rsp.cid = rsp.cid;
            in_rsp.uuid = rsp.uuid;
            for (uint32_t j = i + 1; j < BankReadRspIn.size(); ++j) {
                if (BankReadRspIn.at(j).empty())
                    continue;
                auto& other_rsp = BankReadRspIn.at(j).front();
                if (rsp.tag == other_rsp.tag) {
                    in_rsp.mask.set(j);
                    BankReadRspIn.at(j).pop();
                }
            }
            BankReadRspIn.at(i).pop();
            ReadRspOut.push(in_rsp, 1);
            break;
        }
    }
    
    //handle Bank write response
    for (uint32_t i = 0; i < BankWriteRspIn.size(); ++i) {
        if (!BankWriteRspIn.at(i).empty()) {
            auto& rsp = BankWriteRspIn.at(i).front();
            LsuRsp in_rsp(BankWriteRspIn.size());
            in_rsp.mask.set(i);
            in_rsp.tag = rsp.tag;
            in_rsp.cid = rsp.cid;
            in_rsp.uuid = rsp.uuid;
            for (uint32_t j = i + 1; j < BankWriteRspIn.size(); ++j) {
                if (BankWriteRspIn.at(j).empty())
                    continue;
                auto& other_rsp = BankWriteRspIn.at(j).front();
                if (rsp.tag == other_rsp.tag) {
                    in_rsp.mask.set(j);
                    BankWriteRspIn.at(j).pop();
                }
            }
            BankWriteRspIn.at(i).pop();
            WriteRspOut.push(in_rsp, 1);
            break;
        }
    }
}

void ScratchPadMem::reset() {
    for (uint64_t i = 0; i < ram_.size(); i += 4) {
        uint32_t zero = 0;
        ram_.write(&zero, i, 4);
    }
}
uint64_t ScratchPadMem::size() const { return capacity_; }

LocalMemReader::LocalMemReader(const SimContext& ctx, const char* name) 
    : SimObject(ctx, name) 
    , ReqIn(this)
    , RspIn(this)
    , CoreArbReqOut(this) // to CoreArbiter
    , CoreArbRspOut(this)
    , ScratchpadReqOut(this) // to scratchpad
    , ScratchpadRspOut(this)
    , VirgoReqIn(this) // reqs from Virgo MM controller
    , VirgoRspIn(this)
    , pending_reqs(16) // 16 for Safety!
{

}

LocalMemReader::~LocalMemReader() {}

/* typedef struct {
uint32_t src_addr_A;
uint32_t src_addr_B;
uint32_t dst_addr;
uint32_t data_type;
uint32_t num_rows_A; // [rows_A x cols_A] x [cols_A x cols_B] = [rows_A x cols_B]
uint32_t num_cols_A; 
uint32_t num_cols_B;
uint32_t accum_addr;
uint32_t tag;
bool store;
bool accum;
} virgo_queue_t; 
*/

/* BitVector<> mask;
std::vector<uint64_t> addrs;
bool     write;
uint32_t tag;
uint32_t cid;
uint64_t uuid; 
LsuReq type */

void LocalMemReader::tick() {
    if (!VirgoReqIn.empty()) {
        auto& virgo_req = VirgoReqIn.front(); // type virgo_queue_t
        LsuReq req(LSU_CHANNELS); // first need to "read" from local memory, generate separate LsuReqs for each row of the A and B matrixes

        uint32_t data_type_size;
        if (virgo_req.data_type == FP32) {
            data_type_size = 4; // FP32
        } else {
            data_type_size = 2; // fp16
        }

        // generate one LSU req per cycle, iterate over each row of the A matrix every cycle till done.
        if (!lmem_state_A_.done) {
            for (uint32_t col_num = 0; col_num < virgo_req.num_cols_A; ++col_num) {
                uint32_t src_index = lmem_state_A_.row * virgo_req.num_cols_A + col_num;
                uint64_t src_addr = virgo_req.src_addr_A + src_index * data_type_size;

                req.mask.set(col_num);
                req.addrs.at(col_num) = src_addr;
            }

            req.write = false;
            bool is_last_req = false;
            if (lmem_state_A_.row == virgo_req.num_rows_A - 1) {
                lmem_state_A_.done = true;
                is_last_req = true;
            }
            uint32_t count = req.mask.count();

            uint32_t tag = pending_reqs.allocate({
                is_last_req, 
                req.addrs, 
                count,
                count,
            });
            req.tag = tag;
            
            // finally, actually send LsuReq to CoreArbiter. One LsuReq per row!
            CoreArbReqOut.push(req); // default delay of 1 cycle
            lmem_state_A_.row++;
        }
        else if (!lmem_state_B_.done) { // lmem_state_A_m.done == true, can move onto the B matrix
            for (uint32_t col_num = 0; col_num < virgo_req.num_cols_B; ++col_num) {
                uint32_t src_index = lmem_state_B_.row * virgo_req.num_cols_B + col_num;
                uint64_t src_addr = virgo_req.src_addr_B + src_index * data_type_size;

                req.mask.set(col_num);
                req.addrs.at(col_num) = src_addr;
            }

            req.write = false;
            bool is_last_req = false;
            if (lmem_state_B_.row == virgo_req.num_cols_A - 1) {
                lmem_state_B_.done = true;
                is_last_req = true;
            }
            uint32_t count = req.mask.count();

            uint32_t tag = pending_reqs.allocate({
                is_last_req, 
                req.addrs, 
                count,
                count,
            });
            req.tag = tag;
            
            // finally, actually send LsuReq to CoreArbiter. One LsuReq per row!
            CoreArbReqOut.push(req); // default delay of 1 cycle
            lmem_state_B_.row++;
        }

        if (lmem_state_A_.done && lmem_state_B_.done) {
            VirgoReqIn.pop(); // finally remove from queue
            lmem_state_A_.reset();
            lmem_state_B_.reset();
        }
    }

    if (!CoreArbRspOut.empty()) { // we've received a response all the way from local mem
        auto& rsp = CoreArbRspOut.front();
        auto& entry = pending_reqs.at(rsp.tag);

        entry.count -= rsp.mask.count();

        if (count == 0) {

            //stuff

            // write to scratchpad
        }

        CoreArbRspOut.pop();
    }

}

void LocalMemReader::reset() {
    // todo - this
}

SystolicArray::SystolicArray(const SimContext& ctx, const char* name) : SimObject(ctx, name) {}
SystolicArray::~SystolicArray() {}
void SystolicArray::tick() {}
void SystolicArray::reset() {}

SysSubArray::SysSubArray(const SimContext& ctx, const char* name) : SimObject(ctx, name) {}
SysSubArray::~SysSubArray() {}
void SysSubArray::tick() {}
void SysSubArray::reset() {}

Virgo_MatMul::Virgo_MatMul(const SimContext& ctx,
                     const Arch &arch,
                     Cluster* cluster)
    : SimObject(ctx, StrFormat("virgo_matmul%d", cluster->id()))
    , ReqIn(this)
    , RspIn(this)
    , DmaReqOut(this)
    , DmaRspIn(this)
    , LMemReqOut(this)
    , LMemRspOut(this)
    , cluster_(cluster)
    , arch_(arch)
    , write_registers(arch.num_cores()*arch.num_warps()*9, 0)
    , read_registers(arch.num_cores()*arch.num_warps(), 0)
    , tag_table(4) // number of max in-flight matmul unit instructions
    , accum_mem_(ctx, StrFormat("accum_mem%d", cluster->id()).c_str(), 1 << LMEM_LOG_SIZE)
    , systolic_array_(ctx, StrFormat("systolic_array%d", cluster->id()).c_str())
    , perf_stats_()
{
    scratchpad_mem_ = ScratchPadMem::Create(StrFormat("scratchpad_mem%d", cluster->id()).c_str(), 1 << LMEM_LOG_SIZE);
    local_mem_reader_ = LocalMemReader::Create(StrFormat("local_mem_reader%d", cluster->id()).c_str());

    // bind localmemreader and scratchpad
    local_mem_reader()->ScratchpadReqOut.bind(&scratchpad_mem()->WriteReqIn);
    scratchpad_mem()->WriteRspOut.bind(&local_mem_reader()->ScratchpadRspOut);

    // bind to local_mem_reader
    LMemReqOut.bind(&local_mem_reader()->VirgoReqIn);
    local_mem_reader()->VirgoRspIn.bind(&LMemRspOut);
}

Virgo_MatMul::~Virgo_MatMul() {}

void Virgo_MatMul::read(const void* data, uint64_t addr, uint32_t /* size */) {
    uint32_t* d = (uint32_t*) data;
    uint32_t super_index = ((static_cast<uint32_t>(addr) - (MMIO_VIRGO_BASE_READ_ADDR)) & 0xFFFF ) >> 2;
    std::cout << "Virgo read from index " << super_index << ", addr  " << std::hex << addr << ", MMIO " << std::hex << MMIO_VIRGO_BASE_READ_ADDR << ", data " << *d << std::endl;
    *d = read_registers.at(super_index);

    perf_stats_.reads++;
}

void Virgo_MatMul::write(const void* data, uint64_t addr, uint32_t /* size */) {
    
    uint32_t* d = (uint32_t*) data;
    uint32_t super_index =((static_cast<uint32_t>(addr) - (MMIO_VIRGO_WRITE_ADDR)) & 0xFFFF ) >> 2;
    uint32_t index = super_index % 9;
    
    uint32_t warp_id = (super_index / 9) % arch_.num_warps();
    uint32_t core_id = (super_index / 9) / arch_.num_warps();
    uint32_t global_warp_id = core_id * arch_.num_warps() + warp_id;
    std::cout << "writing super_index " << super_index << " index " << index << " warp id " << warp_id << " core id " << core_id << std::endl;
    write_registers.at(global_warp_id * 9 + index) = *d;

    perf_stats_.writes++;
    
    if (index == 8) { // 8 index == 9th thing (commit address, tag)
        uint32_t tag = write_registers.at(global_warp_id * 9 + 8);
        tag_table[tag][global_warp_id] = 1; // set to 1
        read_registers.at(global_warp_id)++;
        
        if (tag_table[tag].count() == arch_.num_warps()*arch_.num_cores()) {

            //unpack accum
            uint32_t accum_addr = write_registers.at(global_warp_id * 9 + 7);
            bool accum = (accum_addr >> 31) & 1;
            bool store = (accum_addr >> 30) & 1;
            accum_addr = accum_addr & 0x7FFF;
            virgo_queue_t virgo_compute_entry = {
                .src_addr_A = write_registers.at(global_warp_id * 9 + 0),
                .src_addr_B = write_registers.at(global_warp_id * 9 + 1),
                .dst_addr = write_registers.at(global_warp_id * 9 + 2),
                .data_type = write_registers.at(global_warp_id * 9 + 3),
                .num_rows_A = write_registers.at(global_warp_id * 9 + 4),
                .num_cols_A = write_registers.at(global_warp_id * 9 + 5),
                .num_cols_B = write_registers.at(global_warp_id * 9 + 6),
                .accum_addr = accum_addr,
                .tag = tag,
                .store = store,
                .accum = accum
            };

            tag_table.at(tag).reset();
            virgo_compute_queue_.push(virgo_compute_entry); // add to queue, for timing

            perf_stats_.transfers++;
            MatMul(virgo_compute_entry); // matmul directly
        }
    }   
}

void Virgo_MatMul::MatMul(const virgo_queue_t& virgo_compute_entry) {
    uint64_t src_addr_A = static_cast<uint64_t>(virgo_compute_entry.src_addr_A);
    uint64_t src_addr_B = static_cast<uint64_t>(virgo_compute_entry.src_addr_B);
    uint64_t dst_addr = static_cast<uint64_t>(virgo_compute_entry.dst_addr);
    uint32_t data_type = virgo_compute_entry.data_type;
    uint32_t num_rows_A = virgo_compute_entry.num_rows_A;
    uint32_t num_cols_A = virgo_compute_entry.num_cols_A;
    uint32_t num_cols_B = virgo_compute_entry.num_cols_B;

    auto src_addr_A_type = get_addr_type(src_addr_A);
    auto src_addr_B_type = get_addr_type(src_addr_B);
    auto dst_addr_type = get_addr_type(dst_addr);
    std::cout <<"matmuling" << std::endl;
    std::cout << "src_addr_A " << std::hex << src_addr_A << " src_addr_B " << std::hex << src_addr_B << " dst_addr " << std::hex << dst_addr << std::endl;
    if (data_type == FP32) {
        for (uint32_t i = 0; i < num_rows_A; i++) { // loop over rows of matrix A
            for (uint32_t j = 0; j < num_cols_B; j++) { // Loop over columns of matrix B
                float sum = 0;
                if (virgo_compute_entry.accum) {
                    uint32_t offset = virgo_compute_entry.accum_addr + (i * num_cols_B + j) * 4;
                    if (offset + 4 <= accum_mem_.size()) {
                        accum_mem_.read(&sum, offset, 4);
                    } else {
                        std::cout << "Error: Accumulator memory read out of bounds at offset " << offset << std::endl;
                    }
                }
                // inner loop: Calculate dot product of row i from A and column j from B
                for (uint32_t k = 0; k < num_cols_A; k++) {
                    // array[row_index * width + column_index]
                    float aVal;
                    float bVal;
                    if (src_addr_A_type == AddrType::Shared) {
                        cluster_->local_mem()->read(static_cast<void*>(&aVal), static_cast<uint64_t>(src_addr_A + (i * num_cols_A + k) * 4), 4);
                    } else {
                        std::cout << "Error: Can only read source matrix A from shared memory" << std::endl;
                    }
                    // Write A to scratchpad
                    uint32_t offset_A = virgo_compute_entry.tag * 128 + (i * num_cols_A + k) * 4;
                    if (offset_A + 4 <= scratchpad_mem_->size()) {
                        scratchpad_mem_->write(&aVal, offset_A, 4);
                    } else {
                        std::cout << "Error: Scratchpad memory write A out of bounds at offset " << offset_A << std::endl;
                    }

                    if (src_addr_B_type == AddrType::Shared) {
                        cluster_->local_mem()->read(static_cast<void*>(&bVal), static_cast<uint64_t>(src_addr_B + (k * num_cols_B + j) * 4), 4);
                    } else {
                        std::cout << "Error: Can only read source matrix A from shared memory" << std::endl;
                    }
                    // Write B to scratchpad
                    uint32_t offset_B = virgo_compute_entry.tag * 128 + 64 + (k * num_cols_B + j) * 4;
                    if (offset_B + 4 <= scratchpad_mem_->size()) {
                        scratchpad_mem_->write(&bVal, offset_B, 4);
                    } else {
                        std::cout << "Error: Scratchpad memory write B out of bounds at offset " << offset_B << std::endl;
                    }
                    //  = src_addr_A[i * num_cols_A + k]; 
                    //  = src_addr_B[k * num_cols_B + j];   
                    sum += aVal * bVal;
                    std::cout << "matmul vals: " << aVal << " " << bVal << " " << sum << std::endl;
                }
                // Store result in destination matrix C
                // dest_addr[i * num_cols_B + j] = sum;
                // Store result back to accumulator memory
                uint32_t offset = virgo_compute_entry.accum_addr + (i * num_cols_B + j) * 4;
                if (offset + 4 <= accum_mem_.size()) {
                    accum_mem_.write(&sum, offset, 4);
                } else {
                    std::cout << "Error: Accumulator memory write out of bounds at offset " << offset << std::endl;
                }

                if (virgo_compute_entry.store) {
                    if (dst_addr_type == AddrType::Shared) {
                        cluster_->local_mem()->write(static_cast<void*>(&sum), static_cast<uint64_t>(dst_addr + (i * num_cols_B + j) * 4), 4);
                    } else {
                        mmu_.write(static_cast<void*>(&sum), static_cast<uint64_t>(dst_addr + (i * num_cols_B + j) * 4), 4, 0);
                    }
                }
            }
        }
    } else if (data_type == FP16) {
        // for (int i = 0; i < num_rows_A; i++) { // loop over rows of matrix A
        //     for (int j = 0; j < num_cols_B; j++) { // Loop over columns of matrix B
        //         vortex::tensor::fp16 sum = 0;
        //         // inner loop: Calculate dot product of row i from A and column j from B
        //         for (int k = 0; k < num_cols_A; k++) {
        //             // array[row_index * width + column_index]
        //             vortex::tensor::fp16 aVal = src_addr_A[i * num_cols_A + k]; 
        //             vortex::tensor::fp16 bVal = src_addr_B[k * num_cols_B + j];   
        //             sum += aVal * bVal;
        //         }
        //         // Store result in destination matrix C
        //         C[i * num_cols_B + j] = sum;
        //     }
        // }
    }

    for (uint32_t i = 0; i < arch_.num_cores()*arch_.num_warps(); i++) {
        read_registers.at(i)--;
    }
}

void Virgo_MatMul::attach_ram(RAM* ram) {
#if (XLEN == 64)
    mmu_.attach(*ram, 0, 0x7FFFFFFFFF); //39bit SV39
#else
    mmu_.attach(*ram, 0, 0xFFFFFFFF);
#endif
}

void Virgo_MatMul::reset() {
}

// Virgo MatMul controller
void Virgo_MatMul::tick() {
    if (!ReqIn.empty()) { // ReqIn for MMIO
        auto& req = ReqIn.front();
        std::cout << "Virgo Matmul Tick: Read Req tag=" << req.tag << " mask=" << req.mask << std::endl;
        LsuRsp rsp(LSU_CHANNELS);
        rsp.tag = req.tag;
        rsp.cid = req.cid;
        rsp.uuid = req.uuid;
        rsp.mask = req.mask;
        RspIn.push(rsp, 1);
        ReqIn.pop();
    }

    // typedef struct {
    // uint32_t src_addr_A;
    // uint32_t src_addr_B;
    // uint32_t dst_addr;
    // uint32_t data_type;
    // uint32_t num_rows_A; // [rows_A x cols_A] x [cols_A x cols_B] = [rows_A x cols_B]
    // uint32_t num_cols_A; 
    // uint32_t num_cols_B;
    // uint32_t accum_addr;
    // uint32_t tag;
    // bool store;
    // bool accum;
    // } virgo_queue_t;
    if (!virgo_compute_queue_.empty()) {
        auto virgo_compute_entry = virgo_compute_queue_.front();
        



    }
}

const Virgo_MatMul::PerfStats& Virgo_MatMul::perf_stats() const {
    return perf_stats_;
}