#pragma once

namespace vortex {

class Cluster;

typedef struct {
    uint32_t src_addr_A;
    uint32_t src_addr_B;
    uint32_t dst_addr;
    uint32_t data_type_size;
    uint32_t num_rows_A; // [rows_A x cols_A] x [cols_A x cols_B] = [rows_A x cols_B]
    uint32_t num_cols_A; 
    uint32_t num_cols_B;
    uint8_t  tag;
} virgo_queue_t;

class Virgo_MatMul : public SimObject<Virgo_MatMul> {
public:

Virgo_MatMul(const SimContext& ctx,
            const Arch &arch,               // architecture stats essentially
            Cluster* cluster);

~Virgo_MatMul();

void read(const void* data, uint64_t addr, uint32_t size);
void write(const void* data, uint64_t addr, uint32_t size);
void reset();
void tick();

Cluster* cluster() const {
    return cluster_;
}

private:
    Cluster* cluster_;
    Arch arch_;
    std::vector<uint32_t> write_registers;
    uint8_t read_register;
    std::vector<std::bitset<32>> tag_table;
    std::queue<virgo_queue_t> virgo_compute_queue_;
    
};

}