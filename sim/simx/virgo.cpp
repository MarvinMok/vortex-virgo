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
        if (!req.mask.all()) {
            std::cout << "Error: AccumulatorMem WriteReq mask not all valid!" << std::endl;
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

LocalMemReader::LocalMemReader(const SimContext& ctx, const char* name) : SimObject(ctx, name) {}
LocalMemReader::~LocalMemReader() {}
void LocalMemReader::tick() {}
void LocalMemReader::reset() {}

SystolicArray::SystolicArray(const SimContext& ctx, const char* name, uint32_t tile_size, uint32_t array_size) 
    : SimObject(ctx, name)
    , in_a(tile_size * array_size, this)
    , in_b(tile_size * array_size, this)
    , in_c(tile_size * array_size, this)
    , out_c(tile_size * array_size, this)
    , load_en_in(tile_size * array_size, this)
    , flip_reg_in(tile_size * array_size, this)
    , out_a(tile_size * array_size, this)
    , out_weight(tile_size * array_size, this)
    , load_en_out(tile_size * array_size, this)
    , flip_reg_out(tile_size * array_size, this)
    , in_valid(tile_size * array_size, this)
    , out_valid(tile_size * array_size, this)
    , ReqIn(this)
    , RspOut(this)
    , ScratchReadReqOut(this)
    , ScratchReadRspIn(this)
    , AccumWriteReqOut(this)
    , AccumWriteRspIn(this)
    , tile_size_(tile_size)
    , array_size_(array_size)
    , preload_active_(false)
    , compute_active_(false)
    , last_accum_(false)
    , accum_counter_(0)
    , preload_row_counter_(0)
    , compute_cycle_(0)
    , pending_reqs_(tile_size * array_size)
{
    sub_arrays_.resize(array_size * array_size);
    for (uint32_t r = 0; r < array_size; ++r) {
        for (uint32_t c = 0; c < array_size; ++c) {

            // meow meow :D

            auto sub_array = SysSubArray::Create(StrFormat("%s_sub_array_%d_%d", name, r, c).c_str(), tile_size);
            sub_arrays_[r * array_size + c] = sub_array;

            // Bind Ports
            // Left/Right (A, Flip)
            if (c == 0) {
                for (uint32_t k = 0; k < tile_size; ++k) {
                    in_a[r * tile_size + k].bind(&sub_array->in_a[k]);
                    flip_reg_in[r * tile_size + k].bind(&sub_array->flip_reg_in[k]);
                    in_valid[r * tile_size + k].bind(&sub_array->in_valid[k]);
                }
            } else {
                auto left = sub_arrays_[r * array_size + (c - 1)];
                for (uint32_t k = 0; k < tile_size; ++k) {
                    left->out_a[k].bind(&sub_array->in_a[k]);
                    left->flip_reg_out[k].bind(&sub_array->flip_reg_in[k]);
                    left->out_valid[k].bind(&sub_array->in_valid[k]);
                }
            }

            if (c == array_size - 1) {
                for (uint32_t k = 0; k < tile_size; ++k) {
                    sub_array->out_a[k].bind(&out_a[r * tile_size + k]);
                    sub_array->flip_reg_out[k].bind(&flip_reg_out[r * tile_size + k]);
                    sub_array->out_valid[k].bind(&out_valid[r * tile_size + k]);
                }
            }

            // Top/Bottom (C, Weight, LoadEn)
            if (r == 0) {
                for (uint32_t k = 0; k < tile_size; ++k) {
                    in_c[c * tile_size + k].bind(&sub_array->in_c[k]);
                    in_b[c * tile_size + k].bind(&sub_array->in_weight[k]);
                    load_en_in[c * tile_size + k].bind(&sub_array->load_en_in[k]);
                }
            } else {
                auto top = sub_arrays_[(r - 1) * array_size + c];
                for (uint32_t k = 0; k < tile_size; ++k) {
                    top->out_c[k].bind(&sub_array->in_c[k]);
                    top->out_weight[k].bind(&sub_array->in_weight[k]);
                    top->load_en_out[k].bind(&sub_array->load_en_in[k]);
                }
            }

            if (r == array_size - 1) {
                for (uint32_t k = 0; k < tile_size; ++k) {
                    sub_array->out_c[k].bind(&out_c[c * tile_size + k]);
                    sub_array->out_weight[k].bind(&out_weight[c * tile_size + k]);
                    sub_array->load_en_out[k].bind(&load_en_out[c * tile_size + k]);
                }
            }
        }
    }
}
SystolicArray::~SystolicArray() {}
void SystolicArray::tick() {

    // 1. Request Routing
    if (!ReqIn.empty()) {
        auto& req = ReqIn.front();
        preload_queue_.push(req);
        ReqIn.pop();
    }

    // 2. Preload Logic (Matrix B)
    if (!preload_active_ && !preload_queue_.empty()) {
        preload_req_ = preload_queue_.front();
        preload_queue_.pop();
        preload_active_ = true;
        preload_row_counter_ = 0;
    }

    if (preload_active_) {
        // Issue LsuReqs for weights
        if (preload_row_counter_ < preload_req_.num_rows) {
            if (!pending_reqs_.full()) {
                uint32_t row = preload_row_counter_;
                uint32_t cols = preload_req_.num_cols;
                
                LsuReq req(array_size_ * tile_size_);
                req.write = false;
                req.tag = 0; // Will be set by allocate
                
                std::vector<uint32_t> target_indices;
                uint32_t count = 0;

                for (uint32_t c = 0; c < cols && c < array_size_ * tile_size_; ++c) {
                    uint32_t addr = preload_req_.scratchpad_addr_B + (row * cols + c) * 4;
                    req.addrs[c] = addr;
                    req.mask.set(c);
                    target_indices.push_back(c); 
                    count++;
                }

                if (count > 0) {
                    uint32_t tag = pending_reqs_.allocate({
                        preload_req_.tag,
                        true, // is_weight
                        preload_row_counter_,
                        count
                    });
                    req.tag = tag;
                    ScratchReadReqOut.push(req, 1);
                    preload_row_counter_++;
                }
            }
        } 

        //get LsuRsps
        if (!ScratchReadRspIn.empty()) {
            auto& rsp = ScratchReadRspIn.front();
            auto& entry = pending_reqs_.at(rsp.tag);
            
            if (entry.is_weight) {
                entry.count -= rsp.mask.count();
                if (entry.count == 0) {
                    
                    for (uint32_t i = 0; i < array_size_ * tile_size_; ++i) {
                        in_b[i].push(rsp.tag);
                        load_en_in[i].push(true);
                    }

                    if (entry.row_num == preload_req_.num_rows - 1) {
                        compute_queue_.push(preload_req_);
                        preload_active_ = false;
                    }

                    pending_reqs_.release(rsp.tag);
                } 
                ScratchReadRspIn.pop();
            }    
        }
    }

    // 3. Compute Logic (Matrix A)
    if (!compute_active_ && !compute_queue_.empty()) {
        compute_req_ = compute_queue_.front();
        compute_queue_.pop();
        compute_active_ = true;
        compute_cycle_ = 0;
        last_accum_ = false;
        accum_counter_ = 0;
    }

    if (compute_active_) {
        // Issue LsuReqs for A (Anti-Diagonal)
        // i + k == compute_cycle_
        // i: row, k: col
        // i ranges [0, num_rows), k ranges [0, num_cols)
        
        // Only issue if we have pending slots
        if (!pending_reqs_.full()) {
             LsuReq req(array_size_ * tile_size_);
             req.write = false;
             req.tag = 0;

             uint32_t count = 0;
             uint32_t rows = compute_req_.num_rows;
             uint32_t cols = compute_req_.num_cols;

             // Iterate i to find valid (i, k) pairs
             for (uint32_t i = 0; i < rows; ++i) {
                 if (compute_cycle_ >= i) {
                     uint32_t k = compute_cycle_ - i;
                     if (k < cols) {
                         if (count < array_size_ * tile_size_) {
                             uint32_t addr = compute_req_.scratchpad_addr_A + (i * cols + k) * 4;
                             req.addrs[count] = addr;
                             req.mask.set(count);
                             count++;
                         }
                     }
                 }
             }

             if (count > 0) {
                 uint32_t tag = pending_reqs_.allocate({
                     compute_req_.tag,
                     false, // is_weight (is A)
                     compute_cycle_,
                     count
                 });
                 req.tag = tag;
                 ScratchReadReqOut.push(req, 1);
             }
         }

         // Handle Responses
         if (!ScratchReadRspIn.empty()) {
             auto& rsp = ScratchReadRspIn.front();
             auto& entry = pending_reqs_.at(rsp.tag);

             if (!entry.is_weight) {
                 entry.count -= rsp.mask.count();
                 if (entry.count == 0) {
                
                     //skew already calculated
                    for (uint32_t i = 0; i < array_size_ * tile_size_; ++i) {
                        if (rsp.mask.test(i)) {
                            in_a[i].push(rsp.tag);
                            in_valid[i].push(true);
                            flip_reg_in[i].push(rsp.tag & 1);
                        } else {
                            in_valid[i].push(false);
                        }
                    }
                    pending_reqs_.release(rsp.tag);
                 }
                 ScratchReadRspIn.pop();
             }
         }

         //Handle responses from systolic array
         for (uint32_t i = 0; i < array_size_ * tile_size_; ++i) {
            if (!out_c[i].empty()) {
                LsuReq req(array_size_ * tile_size_);
                req.write = true;
                uint32_t count = 0;
                //auto& rsp = out_c[i].front();
                auto& valid = out_valid[i].front();
                if (valid) {
                    req.mask.set(i);
                    //undiagonalize
                    req.addrs[i] = compute_req_.accum_addr + ((array_size_ * tile_size_ - 1 - count) * array_size_ * tile_size_ + (array_size_ * tile_size_ - 1 - accum_counter_)) * 4;
                    count++;
                }
                auto tag = pending_reqs_.allocate({
                    compute_req_.tag,
                    false,
                    compute_cycle_,
                    count
                }); 
                req.tag = tag;
                out_c[i].pop();
                out_valid[i].pop();
                AccumWriteReqOut.push(req, 1);
                accum_counter_++;
            }
         }

         if (!AccumWriteRspIn.empty()) {
            auto& rsp = AccumWriteRspIn.front();
            auto& entry = pending_reqs_.at(rsp.tag);
            entry.count -= rsp.mask.count();
            if (entry.count == 0) {
                pending_reqs_.release(rsp.tag);
            }
            if (rsp.mask.count() == 1) {
                if (last_accum_) {
                    SysArrRsp rsp;
                    rsp.tag = compute_req_.tag;
                    RspOut.push(rsp, 1);
                    compute_active_ = false;
                }
                last_accum_ = true;    
            }
         }  
         compute_cycle_++;
    }
    
    //empty dummy ports
    for (uint32_t i = 0; i < out_a.size(); ++i) {
        if (!out_a[i].empty()) {
            out_a[i].pop();
        }
        if (!out_weight[i].empty()) {
            out_weight[i].pop();
        }
        if (!load_en_out[i].empty()) {
            load_en_out[i].pop();
        }
        if (!flip_reg_out[i].empty()) {
            flip_reg_out[i].pop();
        }
    }
}
void SystolicArray::reset() {
    while (!preload_queue_.empty()) preload_queue_.pop();
    while (!compute_queue_.empty()) compute_queue_.pop();
    preload_active_ = false;
    compute_active_ = false;
    pending_reqs_.clear();
}

SysSubArray::SysSubArray(const SimContext& ctx, const char* name, uint32_t size) 
    : SimObject(ctx, name)
    , in_a(size, this)
    , out_a(size, this)
    , in_c(size, this)
    , out_c(size, this)
    , in_weight(size, this)
    , out_weight(size, this)
    , load_en_in(size, this)
    , load_en_out(size, this)
    , flip_reg_in(size, this)
    , flip_reg_out(size, this)
    , in_valid(size, this)
    , out_valid(size, this)
    , size_(size) 
{}

SysSubArray::~SysSubArray() {}

void SysSubArray::tick() {
    for (uint32_t i = 0; i < size_; ++i) {
        if (!in_a[i].empty()) {
            auto& tag = in_a[i].front();
            out_a[i].push(tag, 1);
            in_a[i].pop();
        }
        if (!in_c[i].empty()) {
            auto& tag = in_c[i].front();
            out_c[i].push(tag, 1);
            in_c[i].pop();
        }
        if (!in_weight[i].empty()) {
            auto& tag = in_weight[i].front();
            out_weight[i].push(tag, 1);
            in_weight[i].pop();
        }
        if (!load_en_in[i].empty()) {
            auto& tag = load_en_in[i].front();
            load_en_out[i].push(tag, 1);
            load_en_in[i].pop();
        }
        if (!flip_reg_in[i].empty()) {
            auto& tag = flip_reg_in[i].front();
            flip_reg_out[i].push(tag, 1);
            flip_reg_in[i].pop();
        }
        if (!in_valid[i].empty()) {
            auto& tag = in_valid[i].front();
            out_valid[i].push(tag, 1);
            in_valid[i].pop();
        }
    }
}
void SysSubArray::reset() {}

Virgo_MatMul::Virgo_MatMul(const SimContext& ctx,
                     const Arch &arch,
                     Cluster* cluster)
    : SimObject(ctx, StrFormat("virgo_matmul%d", cluster->id()))
    , ReqIn(this)
    , RspIn(this)
    , DmaReqOut(this)
    , DmaRspIn(this)
    , ScratchReadReqIn(this)
    , ScratchReadRspOut(this)
    , ScratchWriteReqIn(this)
    , ScratchWriteRspOut(this)
    , cluster_(cluster)
    , arch_(arch)
    , write_registers(arch.num_cores()*arch.num_warps()*9, 0)
    , read_registers(arch.num_cores()*arch.num_warps(), 0)
    , tag_table(4) // number of max in-flight matmul unit instructions
    , accum_mem_(ctx, StrFormat("accum_mem%d", cluster->id()).c_str(), 1 << LMEM_LOG_SIZE)
    , scratchpad_mem_(ctx, StrFormat("scratchpad_mem%d", cluster->id()).c_str(), 1 << LMEM_LOG_SIZE)
    , local_mem_reader_(ctx, StrFormat("local_mem_reader%d", cluster->id()).c_str())
    , systolic_array_(ctx, StrFormat("systolic_array%d", cluster->id()).c_str(), 4, 2)
    , perf_stats_()
{
    // Bind ScratchPad Ports
    ScratchReadReqIn.bind(&scratchpad_mem_.ReadReqIn);
    scratchpad_mem_.ReadRspOut.bind(&ScratchReadRspOut);
    ScratchWriteReqIn.bind(&scratchpad_mem_.WriteReqIn);
    scratchpad_mem_.WriteRspOut.bind(&ScratchWriteRspOut);

    // Bind Accumulator Ports
    systolic_array_.AccumWriteReqOut.bind(&accum_mem_.WriteReqIn);
    accum_mem_.WriteRspOut.bind(&systolic_array_.AccumWriteRspIn);
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
            virgo_queue_t virgo_compute = {
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
            virgo_compute_queue_.push(virgo_compute); // add to queue

            perf_stats_.transfers++;
            MatMul();
        }
    }   
}

void Virgo_MatMul::MatMul() {
    auto virgo_compute = virgo_compute_queue_.front();
    virgo_compute_queue_.pop();

    uint64_t src_addr_A = static_cast<uint64_t>(virgo_compute.src_addr_A);
    uint64_t src_addr_B = static_cast<uint64_t>(virgo_compute.src_addr_B);
    uint64_t dst_addr = static_cast<uint64_t>(virgo_compute.dst_addr);
    uint32_t data_type = virgo_compute.data_type;
    uint32_t num_rows_A = virgo_compute.num_rows_A;
    uint32_t num_cols_A = virgo_compute.num_cols_A;
    uint32_t num_cols_B = virgo_compute.num_cols_B;

    auto src_addr_A_type = get_addr_type(src_addr_A);
    auto src_addr_B_type = get_addr_type(src_addr_B);
    auto dst_addr_type = get_addr_type(dst_addr);
    std::cout <<"matmuling" << std::endl;
    std::cout << "src_addr_A " << std::hex << src_addr_A << " src_addr_B " << std::hex << src_addr_B << " dst_addr " << std::hex << dst_addr << std::endl;
    if (data_type == FP32) {
        for (uint32_t i = 0; i < num_rows_A; i++) { // loop over rows of matrix A
            for (uint32_t j = 0; j < num_cols_B; j++) { // Loop over columns of matrix B
                float sum = 0;
                if (virgo_compute.accum) {
                    uint32_t offset = virgo_compute.accum_addr + (i * num_cols_B + j) * 4;
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
                    uint32_t offset_A = virgo_compute.tag * 128 + (i * num_cols_A + k) * 4;
                    if (offset_A + 4 <= scratchpad_mem_.size()) {
                        scratchpad_mem_.write(&aVal, offset_A, 4);
                    } else {
                        std::cout << "Error: Scratchpad memory write A out of bounds at offset " << offset_A << std::endl;
                    }

                    if (src_addr_B_type == AddrType::Shared) {
                        cluster_->local_mem()->read(static_cast<void*>(&bVal), static_cast<uint64_t>(src_addr_B + (k * num_cols_B + j) * 4), 4);
                    } else {
                        std::cout << "Error: Can only read source matrix A from shared memory" << std::endl;
                    }
                    // Write B to scratchpad
                    uint32_t offset_B = virgo_compute.tag * 128 + 64 + (k * num_cols_B + j) * 4;
                    if (offset_B + 4 <= scratchpad_mem_.size()) {
                        scratchpad_mem_.write(&bVal, offset_B, 4);
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
                uint32_t offset = virgo_compute.accum_addr + (i * num_cols_B + j) * 4;
                if (offset + 4 <= accum_mem_.size()) {
                    accum_mem_.write(&sum, offset, 4);
                } else {
                    std::cout << "Error: Accumulator memory write out of bounds at offset " << offset << std::endl;
                }

                if (virgo_compute.store) {
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

void Virgo_MatMul::tick() {
    // for TIMING simulation, where we'll actually make the matrix multiply a systolic array
    if (!ReqIn.empty()) {
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
}

const Virgo_MatMul::PerfStats& Virgo_MatMul::perf_stats() const {
    return perf_stats_;
}