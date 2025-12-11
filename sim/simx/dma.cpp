#include "dma.h"
#include "cluster.h"
#include <iostream>

namespace vortex {

// GlobalMemAdapter Implementation

GlobalMemAdapter::GlobalMemAdapter(const SimContext& ctx, const char* name, uint32_t num_inputs)
    : SimObject<GlobalMemAdapter>(ctx, name)
    , DmaReqIn(num_inputs, this)
    , DmaRspOut(num_inputs, this)
    , DmaReqOut(this)
    , DmaRspIn(this)
    , pending_reqs(LSUQ_IN_SIZE)
{
    char sname[100];
    snprintf(sname, 100, "%s-lsu_adapter", name);
    lsu_adapter_ = LsuMemAdapter::Create(sname, LSU_CHANNELS, 1);

    snprintf(sname, 100, "%s-arbiter", name);
    arbiter_ = DmaArbiter::Create(sname, ArbiterType::RoundRobin, num_inputs, 1);
    
    std::cout << "GlobalMemAdapter: num_inputs=" << num_inputs 
              << ", DmaReqIn.size()=" << DmaReqIn.size() 
              << ", arbiter_->ReqIn.size()=" << arbiter_->ReqIn.size() << std::endl;

    for (uint32_t i = 0; i < num_inputs; ++i) {
        DmaReqIn.at(i).bind(&arbiter_->ReqIn.at(i));
        arbiter_->RspIn.at(i).bind(&DmaRspOut.at(i));
    }
}

GlobalMemAdapter::~GlobalMemAdapter() {}

void GlobalMemAdapter::reset() {
    pending_reqs.clear();
}

void GlobalMemAdapter::tick() {
    // 1. Handle Incoming Requests (from Arbiter)
    if (!arbiter_->ReqOut.at(0).empty()) {
        auto& req = arbiter_->ReqOut.at(0).front();
        
        // Allocate tag for tracking
        if (!pending_reqs.full()) {
            //std::cout << "GlobalMemAdapter: Req from Arbiter tag=" << req.tag << " write=" << req.lsuReq.write << " count=" << req.lsuReq.mask.count() << std::endl;
            uint32_t count = req.lsuReq.mask.count();
            uint32_t tag = pending_reqs.allocate({
                req.last_req, 
                req.src_id, 
                0, // input_port not needed with DmaArbiter
                req.lsuReq.write, 
                req.dst_addrs, 
                count,
                count,
                req.tag // Store arbiter's tag
            });
            
            // Send LsuReq to L2
            LsuReq lsu_req = req.lsuReq;
            lsu_req.tag = tag;
            lsu_req.write = false; //hack?
            lsu_adapter_->ReqIn.push(lsu_req, 1);
            
            arbiter_->ReqOut.at(0).pop();
        }
    }

    // 2. Handle Outgoing Responses (from LsuMemAdapter)
    if (!lsu_adapter_->RspIn.empty()) {
        auto& rsp = lsu_adapter_->RspIn.front();
        
        auto& entry = pending_reqs.at(rsp.tag);
        
        entry.count -= rsp.mask.count();
        
        if (entry.count == 0) {
            //std::cout << "GlobalMemAdapter: Full response for tag=" << rsp.tag << " orig_tag=" << entry.tag << " write=" << entry.write << std::endl;
            // Full response received
            if (!entry.write) { // Was a READ, now send WRITE to LocalMemAdapter
                DmaReq req(LSU_CHANNELS);
                req.tag = entry.tag; 
                req.lsuReq.uuid = rsp.uuid;
                req.lsuReq.write = true;
                req.lsuReq.addrs = entry.dst_addrs;
                req.last_req = entry.last_req;
                req.src_id = entry.src_id;
                for (uint32_t i = 0; i < entry.orig_count; ++i) {
                    req.lsuReq.mask.set(i);
                }
                //std::cout << "GlobalMemAdapter: Pushing req.addrs.count=" << req.lsuReq.mask.count() << std::endl;
                // for (uint32_t i = 0; i < LSU_CHANNELS; ++i) {
                //     std::cout << "GlobalMemAdapter: Pushing req.addrs.at(" << i << ")=" << std::hex << req.lsuReq.addrs.at(i) << std::endl;
                // }
                DmaReqOut.push(req, 1);
            } else { // Was a WRITE, send response to Arbiter
                DmaRsp dma_rsp(LSU_CHANNELS);
                dma_rsp.lsuRsp = rsp;
                dma_rsp.last_req = entry.last_req;
                dma_rsp.src_id = entry.src_id;
                dma_rsp.tag = entry.tag; // Restore arbiter tag

                arbiter_->RspOut.at(0).push(dma_rsp, 1); 
            }
            
            pending_reqs.release(rsp.tag);
        }
        lsu_adapter_->RspIn.pop();
    }

    // 3. Handle Incoming Responses from other Adapter (Write Completion)
    if (!DmaRspIn.empty()) {
        auto& rsp = DmaRspIn.front();
        //std::cout << "GlobalMemAdapter: Rsp from other adapter tag=" << rsp.tag << std::endl;
        arbiter_->RspOut.at(0).push(rsp, 1);
        DmaRspIn.pop();
    }
}

// LocalMemAdapter Implementation

LocalMemAdapter::LocalMemAdapter(const SimContext& ctx, const char* name, uint32_t num_inputs)
    : SimObject<LocalMemAdapter>(ctx, name)
    , DmaReqIn(num_inputs, this)
    , DmaRspOut(num_inputs, this)
    , DmaReqOut(this)
    , DmaRspIn(this)
    , LsuReqOut(this)
    , LsuRspIn(this)
    , pending_reqs(LSUQ_IN_SIZE)
{
    char sname[100];
    snprintf(sname, 100, "%s-arbiter", name);
    arbiter_ = DmaArbiter::Create(sname, ArbiterType::RoundRobin, num_inputs, 1);
    
    // std::cout << "LocalMemAdapter: num_inputs=" << num_inputs 
    //           << ", DmaReqIn.size()=" << DmaReqIn.size() 
    //           << ", arbiter_->ReqIn.size()=" << arbiter_->ReqIn.size() << std::endl;
    
    for (uint32_t i = 0; i < num_inputs; ++i) {
        DmaReqIn.at(i).bind(&arbiter_->ReqIn.at(i));
        arbiter_->RspIn.at(i).bind(&DmaRspOut.at(i));
    }
}

LocalMemAdapter::~LocalMemAdapter() {}

void LocalMemAdapter::reset() {
    pending_reqs.clear();
}

void LocalMemAdapter::tick() {
    // 1. Handle Incoming Requests (from Arbiter)
    if (arbiter_->ReqOut.at(0).empty()) {
        static int log_count = 0;
        if (log_count++ < 10) std::cout << "LocalMemAdapter: arbiter ReqOut empty" << std::endl;
    } else {
        auto& req = arbiter_->ReqOut.at(0).front();
        
        if (!pending_reqs.full()) {
            //std::cout << "LocalMemAdapter: Req from Arbiter tag=" << req.tag << " write=" << req.lsuReq.write << std::endl;
            uint32_t count = req.lsuReq.mask.count();
            uint32_t tag = pending_reqs.allocate({
                req.last_req, 
                req.src_id, 
                0, 
                req.lsuReq.write, 
                req.dst_addrs, 
                count,
                count,
                req.tag
            });
            
            LsuReq lsu_req = req.lsuReq;
            lsu_req.tag = tag;
            for (uint32_t i = 0; i < LSU_CHANNELS; ++i) {
                if (lsu_req.mask.test(i)) {
                    //std::cout << "addr " << std::hex << lsu_req.addrs.at(i) << ", i=" << std::dec << i << std::endl;
                    if (get_addr_type(lsu_req.addrs.at(i)) != AddrType::Shared) {
                        std::cout << "Error: LocalMemAdapter address " << std::hex << lsu_req.addrs.at(i) << " is not Shared type! Type=" << get_addr_type(lsu_req.addrs.at(i)) << std::endl;
                        std::abort();
                    }
                }
            }
            //std::cout << "LocalMemAdapter: Pushing LsuReq tag=" << tag << " addr=" << std::hex << lsu_req.addrs[0] << std::dec << " write=" << lsu_req.write << std::endl;
            LsuReqOut.push(lsu_req, 1);
            
            arbiter_->ReqOut.at(0).pop();
        }
    }

    // 2. Handle Outgoing Responses (from CoreArbiter)
    if (!LsuRspIn.empty()) {
        auto& rsp = LsuRspIn.front();
        //std::cout << "LocalMemAdapter: Received LsuRsp tag=" << rsp.tag << std::endl;
        
        auto& entry = pending_reqs.at(rsp.tag);
        
        entry.count -= rsp.mask.count();
        
        if (entry.count == 0) {
            //std::cout << "LocalMemAdapter: Full response for tag=" << rsp.tag << " orig_tag=" << entry.tag << " write=" << entry.write << std::endl;
            if (!entry.write) { // Was a READ, now send WRITE to GlobalMemAdapter
                DmaReq req(LSU_CHANNELS);
                req.tag = entry.tag; // Preserve arbiter tag
                req.lsuReq.uuid = rsp.uuid;
                req.lsuReq.write = true;
                req.lsuReq.addrs = entry.dst_addrs;
                req.last_req = entry.last_req;
                req.src_id = entry.src_id;
                for (uint32_t i = 0; i < entry.orig_count; ++i) {
                    req.lsuReq.mask.set(i);
                }
                DmaReqOut.push(req, 1);
            } else { // Was a WRITE, send response to Arbiter
                DmaRsp dma_rsp(LSU_CHANNELS);
                dma_rsp.lsuRsp = rsp;
                dma_rsp.last_req = entry.last_req;
                dma_rsp.src_id = entry.src_id;
                dma_rsp.tag = entry.tag;

                arbiter_->RspOut.at(0).push(dma_rsp, 1);
            }
            
            pending_reqs.release(rsp.tag);
        }
        LsuRspIn.pop();
    }

    // 3. Handle Incoming Responses from other Adapter
    if (!DmaRspIn.empty()) {
        auto& rsp = DmaRspIn.front();
        //std::cout << "LocalMemAdapter: Rsp from other adapter tag=" << rsp.tag << std::endl;
        arbiter_->RspOut.at(0).push(rsp, 1);
        DmaRspIn.pop();
    }
}

// AccumMemAdapter Implementation

AccumMemAdapter::AccumMemAdapter(const SimContext& ctx, const char* name, uint32_t num_inputs)
    : SimObject<AccumMemAdapter>(ctx, name)
    , DmaReqIn(num_inputs, this)
    , DmaRspOut(num_inputs, this)
    , DmaReqOut(2, this) // 0: Global, 1: Local
    , DmaRspIn(2, this)
    , LsuReqOut(this)
    , LsuRspIn(this)
    , pending_reqs(LSUQ_IN_SIZE)
{
    char sname[100];
    snprintf(sname, 100, "%s-arbiter", name);
    arbiter_ = DmaArbiter::Create(sname, ArbiterType::RoundRobin, num_inputs, 1);

    for (uint32_t i = 0; i < num_inputs; ++i) {
        DmaReqIn.at(i).bind(&arbiter_->ReqIn.at(i));
        arbiter_->RspIn.at(i).bind(&DmaRspOut.at(i));
    }
}

AccumMemAdapter::~AccumMemAdapter() {}

void AccumMemAdapter::reset() {
    pending_reqs.clear();
}

void AccumMemAdapter::tick() {
    // 1. Handle Incoming Requests (from Arbiter/DMA)
    if (!arbiter_->ReqOut.at(0).empty()) {
        auto& req = arbiter_->ReqOut.at(0).front();
        
        if (!pending_reqs.full()) {
             uint32_t count = req.lsuReq.mask.count();
             uint32_t tag = pending_reqs.allocate({
                req.last_req, 
                req.src_id, 
                0, 
                req.lsuReq.write, 
                req.dst_addrs, 
                count,
                count,
                req.tag
            });

            LsuReq lsu_req = req.lsuReq;
            lsu_req.tag = tag;
            // AccumMemAdapter only supports READ from Accumulator
            lsu_req.write = false; 
            
            LsuReqOut.push(lsu_req, 1);
            arbiter_->ReqOut.at(0).pop();
        }
    }

    // 2. Handle Outgoing Responses (from MatMul)
    if (!LsuRspIn.empty()) {
        auto& rsp = LsuRspIn.front();
        auto& entry = pending_reqs.at(rsp.tag);
        entry.count -= rsp.mask.count();

        if (entry.count == 0) {
            // Full response received from Accumulator (Read Done)
            // Now send WRITE to Destination (Global or Local)
            
            DmaReq req(LSU_CHANNELS);
            req.tag = entry.tag; 
            req.lsuReq.uuid = rsp.uuid;
            req.lsuReq.write = true;
            req.lsuReq.addrs = entry.dst_addrs;
            req.last_req = entry.last_req;
            req.src_id = entry.src_id;
            for (uint32_t i = 0; i < entry.orig_count; ++i) {
                req.lsuReq.mask.set(i);
            }

            // Determine target
            // Check first address to determine type
            // Assuming all addresses in a request go to the same memory space
            uint64_t dst_addr = 0;
            for(uint32_t i=0; i<LSU_CHANNELS; ++i) {
                if(req.lsuReq.mask.test(i)) {
                    dst_addr = req.lsuReq.addrs.at(i);
                    break;
                }
            } 
            
            AddrType dst_type = get_addr_type(dst_addr);
            
            if (dst_type == AddrType::Shared) {
                // To LocalMemAdapter (Index 1)
                DmaReqOut.at(1).push(req, 1);
            } else {
                // To GlobalMemAdapter (Index 0)
                DmaReqOut.at(0).push(req, 1);
            }

            pending_reqs.release(rsp.tag);
        }
        LsuRspIn.pop();
    }

    // 3. Handle Incoming Responses from Destination Adapters
    for (uint32_t i = 0; i < DmaRspIn.size(); ++i) {
        if (!DmaRspIn.at(i).empty()) {
            auto& rsp = DmaRspIn.at(i).front();
            arbiter_->RspOut.at(0).push(rsp, 1);
            DmaRspIn.at(i).pop();
        }
    }
}

// Virgo_DMA Implementation

Virgo_DMA::Virgo_DMA(const SimContext& ctx,
                     const Arch &arch,
                     Cluster* cluster)
    : SimObject(ctx, StrFormat("virgo_dma%d", cluster->id()))
    , ReadReqIn(this)
    , ReadRspIn(this)
    , WriteReqIn(this)
    , WriteRspIn(this)
    , MatMulReqIn(this)
    , MatMulRspOut(this)
    , DmaLoadReqIn(this)
    , DmaLoadReqOut(this)
    , cluster_(cluster)
    , arch_(arch)
    , write_registers(arch.num_cores()*arch.num_warps()*8, 0)  
    , read_registers(arch.num_cores()*arch.num_warps(), 0) 
    , tag_table(4)
    , timing_mmio_queue(arch.num_cores()*arch.num_warps(), 0)
    , perf_stats_() 
{
    global_mem_adapter_ = GlobalMemAdapter::Create("global_mem_adapter", 3);
    local_mem_adapter_ = LocalMemAdapter::Create("local_mem_adapter", 3);
    accum_mem_adapter_ = AccumMemAdapter::Create("accum_mem_adapter", 1);

    std::cout << "Virgo_DMA: Connecting adapters..." << std::endl;
    std::cout << "global_mem_adapter_->DmaReqOut binding to local_mem_adapter_->DmaReqIn.at(1)" << std::endl;
    std::cout << "local_mem_adapter_->DmaReqIn size: " << local_mem_adapter_->DmaReqIn.size() << std::endl;

    // Connect Adapters to each other
    // Global -> Local
    global_mem_adapter_->DmaReqOut.bind(&local_mem_adapter_->DmaReqIn.at(1));
    local_mem_adapter_->DmaRspOut.at(1).bind(&global_mem_adapter_->DmaRspIn);

    // Local -> Global
    local_mem_adapter_->DmaReqOut.bind(&global_mem_adapter_->DmaReqIn.at(1));
    global_mem_adapter_->DmaRspOut.at(1).bind(&local_mem_adapter_->DmaRspIn);

    // Accum -> Global
    accum_mem_adapter_->DmaReqOut.at(0).bind(&global_mem_adapter_->DmaReqIn.at(2));
    global_mem_adapter_->DmaRspOut.at(2).bind(&accum_mem_adapter_->DmaRspIn.at(0));

    // Accum -> Local
    accum_mem_adapter_->DmaReqOut.at(1).bind(&local_mem_adapter_->DmaReqIn.at(2));
    local_mem_adapter_->DmaRspOut.at(2).bind(&accum_mem_adapter_->DmaRspIn.at(1));

    // Accum -> MatMul (via Virgo_DMA ports)
    // Removed bindings to AccumReqOut and AccumRspIn as they are removed

    DmaLoadReqOut.bind(&DmaLoadReqIn);
}

Virgo_DMA::~Virgo_DMA() {}

void Virgo_DMA::read(void* data,  uint64_t addr, uint32_t /* size */) {
    uint32_t* d = (uint32_t*)data;
    uint32_t super_index = ((static_cast<uint32_t>(addr) - (MMIO_BASE_READ_ADDR)) & 0xFFFF ) >> 2;
    //std::cout << "DMA read from index " << super_index << ", addr  " << std::hex << addr << ", MMIO " << std::hex << MMIO_BASE_READ_ADDR << ", data " << *d << std::endl;
    if (super_index >= read_registers.size()) {
        std::cout << "Error: read super_index " << super_index << " is out of bounds (max " << read_registers.size() << ")" << std::endl;
        std::abort();
    }
    *d = read_registers.at(super_index);
    perf_stats_.reads++;
}

const Virgo_DMA::PerfStats& Virgo_DMA::perf_stats() const {
    return perf_stats_;
}

void Virgo_DMA::write(const void* data, uint64_t addr, uint32_t /*size*/) {
    uint32_t* d = (uint32_t*)data;
    uint32_t super_index = ((static_cast<uint32_t>(addr) - (MMIO_WRITE_ADDR)) & 0xFFFF ) >> 2;
    uint32_t index = super_index % 8;
    uint32_t warp_id = (super_index / 8) % arch_.num_warps();
    uint32_t core_id = (super_index / 8) / arch_.num_warps();
    uint32_t global_warp_id = core_id * arch_.num_warps() + warp_id;
    std::cout << "DMA write to index " << index << " warp " << warp_id << " core " << core_id << ", addr  " << std::hex << addr << ", MMIO " << std::hex << MMIO_WRITE_ADDR << ", data " << *d << std::endl;
    write_registers.at(global_warp_id * 8 + index) = *d;
    perf_stats_.writes++;
    if (index == 7) {
        read_registers.at(global_warp_id)++;
        tag_table.at(*d).set(global_warp_id);
        timing_mmio_queue.at(global_warp_id)++;
        if (tag_table.at(*d).count() == arch_.num_warps() * arch_.num_cores()) {
            std::cout << "Begin DMA transfer" << std::endl;
            dma_load_t dma_load = {
                .src_addr = write_registers.at(global_warp_id * 8 + 0),
                .dst_addr = write_registers.at(global_warp_id * 8 + 1),
                .data_type_size = write_registers.at(global_warp_id * 8 + 2),
                .num_rows = write_registers.at(global_warp_id * 8 + 3),
                .num_cols = write_registers.at(global_warp_id * 8 + 4),
                .src_stride = write_registers.at(global_warp_id * 8 + 5),
                .dst_stride = write_registers.at(global_warp_id * 8 + 6),
                .is_accum = false,
                .tag = 0,
            };
            tag_table.at(*d).reset();
            
            // Push to queue
            dma_load_queue_.push(dma_load);
            
            dma_transfer(dma_load);
            perf_stats_.transfers++;
        }
    }
}

void Virgo_DMA::dma_transfer(const dma_load_t& dma_load){
    std::cout << "DMA transfer from " << dma_load.src_addr << " to " << dma_load.dst_addr << std::endl;
    std::cout << "Data type size: " << dma_load.data_type_size << std::endl;
    std::cout << "Num rows: " << dma_load.num_rows << std::endl;
    std::cout << "Num cols: " << dma_load.num_cols << std::endl;
    std::cout << "Src stride: " << dma_load.src_stride << std::endl;
    std::cout << "Dst stride: " << dma_load.dst_stride << std::endl;
    auto src_addr_type = get_addr_type(dma_load.src_addr);
    auto dst_addr_type = get_addr_type(dma_load.dst_addr);

    for (uint32_t row = 0; row < dma_load.num_rows; row++) {
        for (uint32_t col = 0; col < dma_load.num_cols; col++) {
            auto src_index = row * dma_load.src_stride + col;
            auto dst_index = row * dma_load.dst_stride + col;
            auto src_addr = dma_load.src_addr + src_index * dma_load.data_type_size;
            auto dst_addr = dma_load.dst_addr + dst_index * dma_load.data_type_size;
            data_transfer(src_addr, dst_addr, dma_load.data_type_size, src_addr_type, dst_addr_type);
        }
    }
}

void Virgo_DMA::data_transfer(uint64_t src_addr, uint64_t dst_addr, uint32_t data_type_size, AddrType src_addr_type, AddrType dst_addr_type) {
    uint64_t data = 0;

    auto true_src_type = get_addr_type(src_addr);
    auto true_dst_type = get_addr_type(dst_addr);
    if (true_src_type != src_addr_type || true_dst_type != dst_addr_type) {
        std::cout << "Error: src_addr_type and dst_addr_type do not match for DMA transfer" << std::endl;
        std::abort();
    }

    if (src_addr_type == AddrType::Shared) {
        cluster_->local_mem()->read(static_cast<void*>(&data), static_cast<uint64_t>(src_addr), data_type_size);
    } else {
        mmu_.read(static_cast<void*>(&data), static_cast<uint64_t>(src_addr), data_type_size, 0);
    }

    std::cout << "Data transfer from " << src_addr << " to " << dst_addr << " with data " << data << std::endl;

    if (dst_addr_type == AddrType::Shared) {
        cluster_->local_mem()->write(static_cast<void*>(&data), static_cast<uint64_t>(dst_addr), data_type_size);
    } else {
        mmu_.write(static_cast<void*>(&data), static_cast<uint64_t>(dst_addr), data_type_size, 0);
    }

}

void Virgo_DMA::attach_ram(RAM* ram) {
#if (XLEN == 64)
    mmu_.attach(*ram, 0, 0x7FFFFFFFFF); //39bit SV39
#else
    mmu_.attach(*ram, 0, 0xFFFFFFFF);
#endif
}

void Virgo_DMA::reset() {
    for (uint32_t i = 0; i < read_registers.size(); i++) {
        read_registers.at(i) = 0;
    }
    for (uint32_t i = 0; i < write_registers.size(); i++) {
        write_registers.at(i) = 0;
    }
    for (uint32_t i = 0; i < tag_table.size(); i++) {
        tag_table.at(i).reset();
    }
    for (uint32_t i = 0; i < timing_mmio_queue.size(); i++) {
        timing_mmio_queue.at(i) = 0;
    }
    dma_state_.reset();
    
    global_mem_adapter_->reset();
    local_mem_adapter_->reset();
}

void Virgo_DMA::tick() {
    // MMIO Read Handling
    if (!ReadReqIn.empty()) {
        auto& req = ReadReqIn.front();
        //std::cout << "DMA Tick: Read Req tag=" << req.tag << " mask=" << req.mask << std::endl;
        LsuRsp rsp(LSU_CHANNELS);
        rsp.tag = req.tag;
        rsp.cid = req.cid;
        rsp.uuid = req.uuid;
        rsp.mask = req.mask;
        ReadRspIn.push(rsp, 1);
        ReadReqIn.pop();
    }

    // MMIO Write Handling
    if (!WriteReqIn.empty()) {
        auto& req = WriteReqIn.front();
        //std::cout << "DMA Tick: Write Req tag=" << req.tag << " mask=" << req.mask << std::endl;
        
        for (uint32_t i = 0; i < req.mask.size(); ++i) {
            if (req.mask.test(i)) {
                uint64_t addr = req.addrs.at(i);
                //std::cout << "DMA Tick: Write Req addr=" << std::hex << addr << std::endl;
                uint32_t super_index = ((static_cast<uint32_t>(addr) - (MMIO_WRITE_ADDR)) & 0xFFFF ) >> 2;
                uint32_t index = super_index % 8;

                if (index == 7) {

                    //check if all are non-zero
                    bool valid = true;
                    for (uint32_t i = 0; i < timing_mmio_queue.size(); i++) {
                        if (timing_mmio_queue.at(i) == 0) {
                            valid = false;
                        }
                    }

                    if (valid) {
                        for (uint32_t i = 0; i < timing_mmio_queue.size(); i++) {
                            timing_mmio_queue.at(i)--;
                        }
                        if (!dma_load_queue_.empty()) {
                            auto dma_load = dma_load_queue_.front();
                            dma_load_queue_.pop();
                            DmaLoadReqOut.push(dma_load, 1);
                        } else {
                            std::cout << "Error: No DMA load requests in queue" << std::endl;
                            std::abort();
                        }
                    }
                }
            }
        }

        LsuRsp rsp(LSU_CHANNELS);
        rsp.tag = req.tag;
        rsp.cid = req.cid;
        rsp.uuid = req.uuid;
        rsp.mask = req.mask;
        WriteRspIn.push(rsp, 1);
        WriteReqIn.pop();
    }
    
    // MatMul Req Handling
    if (!MatMulReqIn.empty()) {
        auto& req = MatMulReqIn.front();
        dma_load_t dma_load = {
            .src_addr = req.src_addr,
            .dst_addr = req.dst_addr,
            .data_type_size = req.data_type_size,
            .num_rows = req.num_rows,
            .num_cols = req.num_cols,
            .src_stride = req.num_cols, // Packed
            .dst_stride = req.num_cols, // Packed
            .is_accum = true,
            .tag = req.tag,
        };
        dma_load_queue_.push(dma_load);
        MatMulReqIn.pop();
    }
    
    // Helper lambda to process queue
    // Process Queue
    if (!DmaLoadReqIn.empty()) {
        auto& dma_load = DmaLoadReqIn.front();
        //std::cout << "Virgo_DMA: Processing dma_load_queue item. src=" << dma_load.src_addr << " dst=" << dma_load.dst_addr << std::endl;
        
        // Determine target adapter based on destination address type
        AddrType dst_type = get_addr_type(dma_load.dst_addr);
        bool is_global_dst = (dst_type != AddrType::Shared);

        auto target_port = dma_load.is_accum ? &accum_mem_adapter_->DmaReqIn.at(0) : (is_global_dst ? &local_mem_adapter_->DmaReqIn.at(0) : &global_mem_adapter_->DmaReqIn.at(0));
        
        DmaReq req(LSU_CHANNELS);
        req.lsuReq.tag = 0;
        static uint64_t uuid_counter = 0;
        req.lsuReq.uuid = ++uuid_counter;
        
        req.lsuReq.write = false; // READ
        
        uint32_t count = 0;
        for (uint32_t i = 0; i < LSU_CHANNELS; ++i) {
            if (dma_state_.row >= dma_load.num_rows) break;
            
            uint32_t src_index = dma_state_.row * dma_load.src_stride + dma_state_.col;
            uint32_t dst_index = dma_state_.row * dma_load.dst_stride + dma_state_.col;
            
            uint64_t src_addr = dma_load.src_addr + src_index * dma_load.data_type_size;
            uint64_t dst_addr = dma_load.dst_addr + dst_index * dma_load.data_type_size;
            
            req.lsuReq.addrs.at(i) = src_addr;
            req.dst_addrs.at(i) = dst_addr;
            req.lsuReq.mask.set(i);
            
            dma_state_.col++;
            if (dma_state_.col >= dma_load.num_cols) {
                dma_state_.col = 0;
                dma_state_.row++;
            }
            count++;
        }
        
        if (dma_state_.row >= dma_load.num_rows) {
            req.last_req = true;
        }
        
        if (count > 0) {
            //std::cout << "Virgo_DMA: Target adapter is " << (dma_load.is_accum ? "AccumMemAdapter" : (is_global_dst ? "LocalMemAdapter" : "GlobalMemAdapter")) << std::endl;
            //std::cout << "Virgo_DMA: Pushing req to adapter. last_req=" << req.last_req << " is_global_dst=" << is_global_dst << " count=" << count << std::endl;
            //for (uint32_t i = 0; i < LSU_CHANNELS; ++i) {
            //    std::cout << "Virgo_DMA: req.addrs.at(" << i << ")=" << req.lsuReq.addrs.at(i) << std::endl;
            //    std::cout << "Virgo_DMA: req.dst_addrs.at(" << i << ")=" << req.dst_addrs.at(i) << std::endl;
            //}
            target_port->push(req, 1); 
        }
    }

    // Handle Responses from GlobalMemAdapter
    if (!global_mem_adapter_->DmaRspOut.at(0).empty()) {
        auto& rsp = global_mem_adapter_->DmaRspOut.at(0).front();
        //std::cout << "Virgo_DMA: Rsp from GlobalMemAdapter last_req=" << rsp.last_req << std::endl;
        if (rsp.last_req) {
            if (!DmaLoadReqIn.empty()) {
                auto dma_load = DmaLoadReqIn.front();
                DmaLoadReqIn.pop();
                dma_state_.reset();
                
                if (dma_load.is_accum) {
                    MatMulDmaRsp rsp = { .tag = dma_load.tag };
                    MatMulRspOut.push(rsp, 1);
                } else {
                    // Decrement read registers for ALL threads in the cluster? 
                    // The original code decremented for all. Assuming this is correct for now.
                    std::cout << "Virgo_DMA: Decrementing read_registers" << std::endl;
                    for (uint32_t i = 0; i < arch_.num_cores()*arch_.num_warps(); i++) {
                        if (read_registers.at(i) > 0)
                            read_registers.at(i)--;
                    }
                }
            }
        }
        global_mem_adapter_->DmaRspOut.at(0).pop();
    }
    
    // Handle Responses from LocalMemAdapter
    if (!local_mem_adapter_->DmaRspOut.at(0).empty()) {
        auto& rsp = local_mem_adapter_->DmaRspOut.at(0).front();
        //std::cout << "Virgo_DMA: Rsp from LocalMemAdapter last_req=" << rsp.last_req << std::endl;
        if (rsp.last_req) {
            if (!DmaLoadReqIn.empty()) {
                auto dma_load = DmaLoadReqIn.front();
                DmaLoadReqIn.pop();
                dma_state_.reset();
                
                if (dma_load.is_accum) {
                    MatMulDmaRsp rsp = { .tag = dma_load.tag };
                    MatMulRspOut.push(rsp, 1);
                } else {
                    // Decrement read registers for ALL threads in the cluster? 
                    // The original code decremented for all. Assuming this is correct for now.
                    std::cout << "Virgo_DMA: Decrementing read_registers" << std::endl;
                    for (uint32_t i = 0; i < arch_.num_cores()*arch_.num_warps(); i++) {
                        if (read_registers.at(i) > 0)
                            read_registers.at(i)--;
                    }
                }
            }
        }
        local_mem_adapter_->DmaRspOut.at(0).pop();
    }

    // Handle Responses from AccumMemAdapter
    if (!accum_mem_adapter_->DmaRspOut.at(0).empty()) {
        auto& rsp = accum_mem_adapter_->DmaRspOut.at(0).front();
        //std::cout << "Virgo_DMA: Rsp from AccumMemAdapter last_req=" << rsp.last_req << std::endl;
        if (rsp.last_req) {
            if (!DmaLoadReqIn.empty()) {
                auto dma_load = DmaLoadReqIn.front();
                DmaLoadReqIn.pop();
                dma_state_.reset();
                
                if (dma_load.is_accum) {
                    MatMulDmaRsp rsp = { .tag = dma_load.tag };
                    MatMulRspOut.push(rsp, 1);
                }
            }
        }
        accum_mem_adapter_->DmaRspOut.at(0).pop();
    }
}

} // namespace vortex