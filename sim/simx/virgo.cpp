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
        std::cout << "Scratch Read Req In" << std::endl;
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
                
                std::cout << "Pushing Bank ReqReq" << std::endl;
                BankReadReqOut.at(bank_id).push(mem_req, 1);
                std::cout << "Pushed Bank ReqReq" << std::endl;
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
    , lmem_pending_reqs_(16) // 16 for Safety!
    , scratchpad_pending_reqs_(16) // 16 for Safety!
{
}

LocalMemReader::~LocalMemReader() {}

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

            uint32_t tag = lmem_pending_reqs_.allocate({
                is_last_req, 
                true, // is A
                lmem_workload_,
                req.addrs,
                lmem_state_A_.row,
                count,
                count,
            });
            req.tag = tag;
            
            // finally, actually send LsuReq to CoreArbiter. One LsuReq per row!
            //std::cout << "LocalMemReader: Sending LsuReq for Matrix A row=" << lmem_state_A_.row << ", tag=" << req.tag << std::endl;
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

            uint32_t tag = lmem_pending_reqs_.allocate({
                is_last_req, 
                false, // is B
                lmem_workload_,
                req.addrs,
                lmem_state_B_.row,
                count,
                count,
            });
            req.tag = tag;
            
            // finally, actually send LsuReq to CoreArbiter. One LsuReq per row!
            //std::cout << "LocalMemReader: Sending LsuReq for Matrix B row=" << lmem_state_B_.row << ", tag=" << req.tag << std::endl;
            CoreArbReqOut.push(req); // default delay of 1 cycle
            lmem_state_B_.row++;
        }
    }
    if (lmem_state_A_.done && lmem_state_B_.done) {
        VirgoReqIn.pop(); // finally remove from queue
        lmem_state_A_.reset();
        lmem_state_B_.reset();
        lmem_workload_ = !lmem_workload_; // flip workload bit
    }

    // important: scratchpad operations happen IN ORDER.
    if (!CoreArbRspOut.empty()) { // we've received a response all the way from local mem, can write to scratchpad
        auto& rsp = CoreArbRspOut.front();
        auto& entry = lmem_pending_reqs_.at(rsp.tag);

        //std::cout << "LocalMemReader: Received CoreArbRspOut for tag=" << rsp.tag << ", mask.count=" << rsp.mask << std::endl;
        entry.count -= rsp.mask.count(); // response mask contains how many of the addresses have been serviced
        //std::cout << "LocalMemReader: Updated COREARB count for tag=" << rsp.tag << ", entry.count=" << entry.count << std::endl;

        if (entry.count == 0) {
            // write to scratchpad
            LsuReq req(entry.addrs.size());
            req.write = true;
            for (uint32_t i = 0; i < entry.addrs.size(); ++i) {
                req.mask.set(i);
            }

            for (uint32_t col_num = 0; col_num < entry.addrs.size(); ++col_num) {
                uint64_t index = entry.row * entry.addrs.size() + col_num;
                uint64_t data_type_size = entry.addrs.at(1) - entry.addrs.at(0);

                uint64_t base_address;
                if (!entry.workload) { // first workload
                    base_address = (entry.is_A) ? SCRATCHPAD_TILE_A_1 : SCRATCHPAD_TILE_B_1;
                } else {
                    base_address = (entry.is_A) ? SCRATCHPAD_TILE_A_2 : SCRATCHPAD_TILE_B_2;
                }
                uint64_t addr = base_address + index * data_type_size;
                req.addrs.at(col_num) = addr; // add to addrs vector
            }

            uint32_t tag = scratchpad_pending_reqs_.allocate({
                entry.is_last_req,
                entry.is_A,
                entry.workload, // unused???
                req.addrs, // create new addrs
                entry.row,
                static_cast<uint32_t>(entry.addrs.size()),
                static_cast<uint32_t>(entry.addrs.size())
            });
            req.tag = tag;

            // finally, send LsuReq to Scratchpad
            //std::cout << "LocalMemReader: Sending LsuReq to Scratchpad for Matrix " << entry.is_A << ", row=" << entry.row << ", tag=" << req.tag << std::endl;
            ScratchpadReqOut.push(req); // default delay of 1 cycle

            // done with the lmem operations, moving onto scratchpad, so can release the req in lmem_pending_reqs_
            lmem_pending_reqs_.release(rsp.tag); 
        }

        CoreArbRspOut.pop();
    }

    if (!ScratchpadRspOut.empty()) {
        auto& rsp = ScratchpadRspOut.front();
        auto& entry = scratchpad_pending_reqs_.at(rsp.tag);

        entry.count -= rsp.mask.count();
        //std::cout << "LocalMemReader: Received ScratchpadRspOut for tag=" << rsp.tag << ", mask.count=" << rsp.mask << std::endl;
        //std::cout << "LocalMemReader: Updated Scratchpad count for tag=" << rsp.tag << ", entry.count=" << entry.count << std::endl;

        if (entry.count == 0) {
            if (entry.is_last_req) {
                scratchpad_state_.is_last_req_count++;
            }
            if (entry.row == 0) { // save scratchpad addresses
                if (entry.is_A) {
                    scratchpad_state_.scratchpad_address_A_ = entry.addrs.at(0);
                } else {
                    scratchpad_state_.scratchpad_address_B_ = entry.addrs.at(0);
                }
            }

            if (scratchpad_state_.is_last_req_count == 2) {
                // can construct response to virgo controller!
                virgo_rsp_t resp = {
                    scratchpad_state_.scratchpad_address_A_,
                    scratchpad_state_.scratchpad_address_B_
                };

                VirgoRspIn.push(resp);
                scratchpad_state_.reset();
            }

            scratchpad_pending_reqs_.release(rsp.tag);
            ScratchpadRspOut.pop();
        }
    }
}

void LocalMemReader::reset() {
    lmem_state_A_.reset();
    lmem_state_B_.reset();
    lmem_workload_ = 0;
    scratchpad_state_.reset();
    lmem_pending_reqs_.clear();
    scratchpad_pending_reqs_.clear();
}

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
    , preload_queueIn(this)
    , preload_queueOut(this)
    , compute_queueIn(this)
    , compute_queueOut(this)
    , preload_active_(false)
    , compute_active_(false)
    , last_accum_(false)
    , accum_counter_(0)
    , preload_row_counter_(0)
    , compute_counter_(0)
    , pending_scratchpad_reqs_(tile_size * array_size * 2) // a and b rows to enable pipelining
    , pending_array_reqs_(tile_size * array_size)
    , pending_accumulator_reqs_(tile_size * array_size)
{
    preload_queueOut.bind(&preload_queueIn);
    compute_queueOut.bind(&compute_queueIn);
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
        std::cout << "got systolic array req from Virgo controller tag =" << req.tag <<std::endl;
        
        preload_queueOut.push(req);
        ReqIn.pop();
    }

    // 2. Preload Logic (Matrix B)
    if (!preload_active_ && !preload_queueIn.empty()) {
        preload_req_ = preload_queueIn.front();
        preload_queueIn.pop();
        preload_active_ = true;
        preload_row_counter_ = 0;
    }

    if (preload_active_) {
        //std::cout << "preloading" << std::endl;
        // Issue LsuReqs for weights
        if (preload_row_counter_ < preload_req_.num_rows) {
            if (!pending_scratchpad_reqs_.full()) {
                uint32_t row = preload_req_.num_rows - preload_row_counter_ - 1;
                uint32_t cols = preload_req_.num_cols;
                
                LsuReq req(array_size_ * tile_size_);
                req.write = false;
                req.tag = 0; // Will be set by allocate
                
                uint32_t count = 0;

                for (uint32_t c = 0; c < cols && c < array_size_ * tile_size_; ++c) {
                    uint32_t addr = preload_req_.scratchpad_addr_B + (row * cols + c) * 4;
                    req.addrs[c] = addr;
                    req.mask.set(c);
                    count++;
                }

                if (count > 0) {
                    uint32_t tag = pending_scratchpad_reqs_.allocate({
                        preload_req_.tag,
                        true, // is_weight
                        preload_row_counter_,
                        count
                    });
                    req.tag = tag;
                    std::cout << "Scratch req, preload_row_counter_=" << preload_row_counter_ << ", tag=" << tag << std::endl;
                    ScratchReadReqOut.push(req, 1);
                    preload_row_counter_++;
                }
            }
        } 

        //get LsuRsps
    }

    if (!ScratchReadRspIn.empty()) {
        auto& rsp = ScratchReadRspIn.front();
        ScratchReadRspIn.pop();
        std::cout << "scratch Response any, tag=" << rsp.tag << std::endl;
        auto& entry = pending_scratchpad_reqs_.at(rsp.tag);
        
        if (entry.is_weight) {
            //weight preload rsp
            entry.count -= rsp.mask.count();
            if (entry.count == 0) {
                std::cout << "scratchpadRsp" << std::endl;
                
                for (uint32_t i = 0; i < array_size_ * tile_size_; ++i) {
                    in_b[i].push(rsp.tag);
                    load_en_in[i].push(true);
                }

                if (entry.row_num == preload_req_.num_rows - 1) {
                    compute_queueOut.push(preload_req_, tile_size_ * array_size_);
                    preload_active_ = false;
                }

                pending_scratchpad_reqs_.release(rsp.tag);
            } 
        } else {
            //input matrix response
            entry.count -= rsp.mask.count();
            if (entry.count == 0) {
            uint32_t tag = pending_array_reqs_.allocate({
                compute_req_.tag,
                false,
                entry.row_num,
                array_size_ * tile_size_
            });

            for (uint32_t i = 0; i < array_size_ * tile_size_; ++i) {
                in_a[i].push(tag, i + 1);
                in_valid[i].push(true, i + 1);
                in_c[i].push(tag, i + 1);
                flip_reg_in[i].push(tag & 1, i + 1);
            }
            pending_scratchpad_reqs_.release(rsp.tag);
            }
        }
    }
    // 3. Compute Logic (Matrix A)
    if (!compute_active_ && !compute_queueIn.empty()) {
        std::cout << "finished Preload" << std::endl;
        compute_req_ = compute_queueIn.front();
        compute_queueIn.pop();
        compute_active_ = true;
        accum_counter_ = 0;
        compute_counter_ = 0;
    }

    if (compute_active_) {
        // Only issue if we have pending slots
        if (!pending_scratchpad_reqs_.full()) {
            std::cout << "input scratchpad req" << std::endl;
             LsuReq req(array_size_ * tile_size_);
             req.write = false;

             uint32_t count = 0;
             uint32_t cols = compute_req_.num_cols;

             for (uint32_t i = 0; i < array_size_ * tile_size_; ++i) {
                 uint32_t addr = compute_req_.scratchpad_addr_A + (compute_counter_ * cols + i) * 4;
                 req.addrs[i] = addr;
                 req.mask.set(i);
                 count++;
             }

             if (count > 0) {
                 uint32_t tag = pending_scratchpad_reqs_.allocate({
                     compute_req_.tag,
                     false, // is_weight (is A)
                     compute_counter_,
                     count
                 });
                 req.tag = tag;
                 ScratchReadReqOut.push(req, 1);
             }
             compute_counter_++;
         }

         // Handle Responses
         

         //Handle responses from systolic array
         for (uint32_t i = 0; i < array_size_ * tile_size_; ++i) {
            if (!out_c[i].empty()) {
                uint32_t tag = out_c[i].front();
                out_c[i].pop();
                 std::cout << "sysarrayRsp tag=" << tag << std::endl;
                if (tag <= 4 ) {
                    auto& entry = pending_array_reqs_.at(tag);
                    std::cout << "got entry" << std::endl;
                    entry.count--;
                    if (entry.count == 0) {
                        std::cout << "sending Accum" << std::endl;
                        pending_array_reqs_.release(tag);
                        LsuReq req(array_size_ * tile_size_);
                        req.write = true;
                        uint32_t count = 0;     
                        for (uint32_t j = 0; j < array_size_ * tile_size_; ++j) {
                            req.addrs[j] = compute_req_.accum_addr + (entry.row_num * array_size_ * tile_size_ + j) * 4;
                            req.mask.set(j);    
                            count++;
                        }

                        auto accum_tag = pending_accumulator_reqs_.allocate({
                            compute_req_.tag,
                            false,
                            entry.row_num,
                            count
                        });
                        req.tag = accum_tag;
                        AccumWriteReqOut.push(req, 1);
                    }
                } 
            }
         }

         if (!AccumWriteRspIn.empty()) {
            std::cout << "accumRsp" << std::endl;
            auto& rsp = AccumWriteRspIn.front();
            AccumWriteRspIn.pop();
            auto& entry = pending_accumulator_reqs_.at(rsp.tag);
            entry.count -= rsp.mask.count();
            if (entry.count == 0) {
                pending_accumulator_reqs_.release(rsp.tag);
                if (entry.row_num == compute_req_.num_rows - 1) {
                    std::cout << "Send Finished to Control tag=" << compute_req_.tag << std::endl;
                    compute_active_ = false;
                    SysArrRsp rsp;
                    rsp.tag = compute_req_.tag;
                    RspOut.push(rsp);
                }
            }
         }  
         
    }
    
    //empty dummy ports
    for (uint32_t i = 0; i < out_a.size(); ++i) {
        if (!out_a[i].empty()) {
            out_a[i].pop();
        }
        if (!out_valid[i].empty()) {
            out_valid[i].pop();
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
    while (!preload_queueOut.empty()) preload_queueOut.pop();
    while (!compute_queueOut.empty()) compute_queueOut.pop();
    preload_active_ = false;
    compute_active_ = false;
    pending_scratchpad_reqs_.clear();
    pending_array_reqs_.clear();
    pending_accumulator_reqs_.clear();
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
    , LMemReqOut(this)
    , LMemRspOut(this)
    , SysReqOut(this)
    , SysRspOut(this)
    , cluster_(cluster)
    , arch_(arch)
    , write_registers(arch.num_cores()*arch.num_warps()*9, 0)
    , read_registers(arch.num_cores()*arch.num_warps(), 0)
    , tag_table(4) // number of max in-flight matmul unit instructions
    , perf_stats_()
    , pending_sys_reqs_(16)

{
    scratchpad_mem_ = ScratchPadMem::Create(StrFormat("scratchpad_mem%d", cluster->id()).c_str(), 1 << LMEM_LOG_SIZE);
    local_mem_reader_ = LocalMemReader::Create(StrFormat("local_mem_reader%d", cluster->id()).c_str());
    systolic_array_ = SystolicArray::Create(StrFormat("systolic_array%d", cluster->id()).c_str(), 2, 2);
    accum_mem_ = AccumulatorMem::Create(StrFormat("accum_mem%d", cluster->id()).c_str(), 1 << LMEM_LOG_SIZE);

    // bind localmemreader and scratchpad
    local_mem_reader()->ScratchpadReqOut.bind(&scratchpad_mem()->WriteReqIn);
    scratchpad_mem()->WriteRspOut.bind(&local_mem_reader()->ScratchpadRspOut);

    // bind to local_mem_reader
    LMemReqOut.bind(&local_mem_reader()->VirgoReqIn);
    local_mem_reader()->VirgoRspIn.bind(&LMemRspOut);

    inflight_localmem_req = false;

    // bind systolic array to accumulator
    systolic_array_->AccumWriteReqOut.bind(&accum_mem_->WriteReqIn);
    accum_mem_->WriteRspOut.bind(&systolic_array_->AccumWriteRspIn);
    
    // bind systolic array to scratchpad
    systolic_array_->ScratchReadReqOut.bind(&scratchpad_mem_->ReadReqIn);
    scratchpad_mem_->ReadRspOut.bind(&systolic_array_->ScratchReadRspIn);

    // bind virgo MM controller to systolic array
    SysReqOut.bind(&systolic_array_->ReqIn);
    systolic_array_->RspOut.bind(&SysRspOut);
}

Virgo_MatMul::~Virgo_MatMul() {}

void Virgo_MatMul::read(const void* data, uint64_t addr, uint32_t /* size */) {
    uint32_t* d = (uint32_t*) data;
    uint32_t super_index = ((static_cast<uint32_t>(addr) - (MMIO_VIRGO_BASE_READ_ADDR)) & 0xFFFF ) >> 2;
    // std::cout << "Virgo read from index " << super_index << ", addr  " << std::hex << addr << ", MMIO " << std::hex << MMIO_VIRGO_BASE_READ_ADDR << ", data " << *d << std::endl;
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
            virgo_req_t virgo_compute_entry = {
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
            virgo_req_queue_.push(virgo_compute_entry); // add to queue, for timing

            perf_stats_.transfers++;
            MatMul(virgo_compute_entry); // matmul directly
        }
    }   
}

void Virgo_MatMul::MatMul(const virgo_req_t& virgo_compute_entry) {
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
                    if (offset + 4 <= accum_mem_->size()) {
                        accum_mem_->read(&sum, offset, 4);
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
                if (offset + 4 <= accum_mem_->size()) {
                    accum_mem_->write(&sum, offset, 4);
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
}

void Virgo_MatMul::attach_ram(RAM* ram) {
#if (XLEN == 64)
    mmu_.attach(*ram, 0, 0x7FFFFFFFFF); //39bit SV39
#else
    mmu_.attach(*ram, 0, 0xFFFFFFFF);
#endif
}

void Virgo_MatMul::reset() {
    for (int i = 0; i < write_registers.size(); i++) {
        write_registers[i] = 0;
    }
    for (int i = 0; i < read_registers.size(); i++) {
        read_registers[i] = 0;
    }
    for (int i = 0; i < tag_table.size(); i++) {
        tag_table[i].reset();
    }
    
    std::queue<virgo_req_t> empty;
    std::swap(virgo_req_queue_, empty);

    pending_sys_reqs_.clear();
    inflight_localmem_req = false;
}

// Virgo MatMul controller
void Virgo_MatMul::tick() {
    if (!ReqIn.empty()) { // ReqIn for MMIO
        auto& req = ReqIn.front();
        //std::cout << "Virgo MMIO: Read Req tag=" << req.tag << " mask=" << req.mask << std::endl;
        LsuRsp rsp(LSU_CHANNELS);
        rsp.tag = req.tag;
        rsp.cid = req.cid;
        rsp.uuid = req.uuid;
        rsp.mask = req.mask;
        RspIn.push(rsp, 1);
        ReqIn.pop();
    }

    if (!virgo_req_queue_.empty()) {   
        auto virgo_req_entry = virgo_req_queue_.front();
        if (!inflight_localmem_req) {
            std::cout << "matmul start" << std::endl;
            LMemReqOut.push(virgo_req_entry, 6);
            inflight_localmem_req = true;
        }
    }

    if (!LMemRspOut.empty()) {
        // received a response
        std::cout << "received lmem response" << std::endl;
        auto& virgo_rsp = LMemRspOut.front();
        auto& virgo_req = virgo_req_queue_.front();
        virgo_req_queue_.pop(); // can serve next req

        inflight_localmem_req = false;

        uint32_t tag = pending_sys_reqs_.allocate(virgo_req);

        SysArrReq req = {
            virgo_rsp.scratchpad_src_addr_A,
            virgo_rsp.scratchpad_src_addr_B,
            virgo_req.num_rows_A,
            virgo_req.num_cols_A,
            virgo_req.accum_addr,
            virgo_req.data_type,
            tag,
        };
        std::cout << "pushing req to Sys tag=" << req.tag << std::endl;

        SysReqOut.push(req, 1);

        LMemRspOut.pop();
    }

    if (!SysRspOut.empty()) { // received response from systolic array
        std::cout << "received systolic array rsp" << std::endl;
        auto& rsp = SysRspOut.front();
        auto& pending_sys_req = pending_sys_reqs_.at(rsp.tag);
        pending_sys_reqs_.release(rsp.tag);

        if (pending_sys_req.store) {
            std::cout << "move from accum to shared/global mem" << std::endl;
            // need to use DMA to store from accumulator memory into destination address, either shared or global memory
            uint32_t data_type_size = (pending_sys_req.data_type == FP32) ? 4 : 2;

            MatMulDmaReq req = {
                pending_sys_req.accum_addr,
                pending_sys_req.dst_addr,
                pending_sys_req.num_rows_A,
                pending_sys_req.num_cols_B,
                data_type_size,
                rsp.tag
            };
            DmaReqOut.push(req);
        } else {
            // finished request
            std::cout << "decrementing read" << std::endl;
            for (uint32_t i = 0; i < arch_.num_cores()*arch_.num_warps(); i++) {
                read_registers.at(i)--;
            }
        }

        SysRspOut.pop();
    }

    if (!DmaRspIn.empty()) { // received response from DMA
        // auto& rsp = DmaRspIn.front();
        std::cout << "received dma response" << std::endl;

        // received response! happy!
        // indicate that we've finished request.
        std::cout << "decrementing read DMA" << std::endl;
        for (uint32_t i = 0; i < arch_.num_cores()*arch_.num_warps(); i++) {
            read_registers.at(i)--;
        }
        DmaRspIn.pop();
    }


}

const Virgo_MatMul::PerfStats& Virgo_MatMul::perf_stats() const {
    return perf_stats_;
}