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

#include "types.h"

using namespace vortex;

LocalMemSwitch::LocalMemSwitch(
  const SimContext& ctx,
  const char* name,
  uint32_t delay
) : SimObject<LocalMemSwitch>(ctx, name)
  , ReqIn(this)
  , RspIn(this)
  , ReqLmem(this)
  , RspLmem(this)
  , ReqMMIORead(this)
  , RspMMIORead(this)
  , ReqMMIOWrite(this) // DMA MMIO
  , RspMMIOWrite(this)
  , ReqVirgoMMIO(this) // Virgo MMIO
  , RspVirgoMMIO(this)
  , ReqDC(this)
  , RspDC(this)
  , delay_(delay)
{}

void LocalMemSwitch::reset() {}

void LocalMemSwitch::tick() {
  // process outgoing responses
  if (!RspLmem.empty()) {
    auto& out_rsp = RspLmem.front();
    DT(4, this->name() << "-lmem-rsp: " << out_rsp);
    RspIn.push(out_rsp, 1);
    RspLmem.pop();
  }
  if (!RspDC.empty()) {
    auto& out_rsp = RspDC.front();
    DT(4, this->name() << "-dc-rsp: " << out_rsp);
    RspIn.push(out_rsp, 1);
    RspDC.pop();
  }
  if (!RspMMIORead.empty()) { // DMA MMIO read response
    auto& out_rsp = RspMMIORead.front();
    //std::cout << "LmemSwitch: Pop MMIO Read Rsp tag=" << out_rsp.tag << " cid=" << out_rsp.cid << " uuid=" << out_rsp.uuid << std::endl;
    DT(4, this->name() << "-mmio-read-rsp: " << out_rsp);
    RspIn.push(out_rsp, 1);
    RspMMIORead.pop();
  }
  if (!RspMMIOWrite.empty()) { // DMA MMIO write response
    auto& out_rsp = RspMMIOWrite.front();
    //std::cout << "LmemSwitch: Pop MMIO Write Rsp tag=" << out_rsp.tag << " cid=" << out_rsp.cid << " uuid=" << out_rsp.uuid << std::endl;
    DT(4, this->name() << "-mmio-write-rsp: " << out_rsp);
    RspIn.push(out_rsp, 1);
    RspMMIOWrite.pop();
  }
  if (!RspVirgoMMIO.empty()) { // Virgo MMIO response
    auto& out_rsp = RspVirgoMMIO.front();
    DT(4, this->name() << "-virgo-mmio-rsp: " << out_rsp);
    RspIn.push(out_rsp, 1);
    RspVirgoMMIO.pop();
  }

  // process incoming requests
  if (!ReqIn.empty()) {
    auto& in_req = ReqIn.front();

    LsuReq out_dc_req(in_req.mask.size());
    out_dc_req.write = in_req.write;
    out_dc_req.tag   = in_req.tag;
    out_dc_req.cid   = in_req.cid;
    out_dc_req.uuid  = in_req.uuid;

    LsuReq out_lmem_req(out_dc_req);

    LsuReq out_mmio_req(out_dc_req);
    LsuRsp out_mmio_rsp(in_req.mask.size());

    LsuReq out_virgo_mmio_req(out_dc_req); // not really sure what out_dc_req is? just in_req.mask.size() maybe
    LsuRsp out_virgo_mmio_rsp(in_req.mask.size());

    for (uint32_t i = 0; i < in_req.mask.size(); ++i) {
      if (in_req.mask.test(i)) {
        auto type = get_addr_type(in_req.addrs.at(i));
        if (type == AddrType::Shared) {
          out_lmem_req.mask.set(i);
          out_lmem_req.addrs.at(i) = in_req.addrs.at(i);
        } else if (type == AddrType::MMIO) {
          out_mmio_req.mask.set(i);
          out_mmio_rsp.mask.set(i);
          out_mmio_req.addrs.at(i) = in_req.addrs.at(i);
          DT(4, "MMIO LocalMemReq at 0x" << std::hex << in_req.addrs.at(i));
        } else if (type == AddrType::MMIO_VIRGO) {
          out_virgo_mmio_req.mask.set(i);
          out_virgo_mmio_req.addrs.at(i) = in_req.addrs.at(i);

          out_virgo_mmio_rsp.mask.set(i);

          has_mmio = true;
          DT(4, "Virgo MMIO LocalMemReq at 0x" << std::hex << in_req.addrs.at(i));
        }
        else {
          out_dc_req.mask.set(i);
          out_dc_req.addrs.at(i) = in_req.addrs.at(i);
        }
      }
    }

    if (!out_dc_req.mask.none()) {
      ReqDC.push(out_dc_req, delay_);
      DT(4, this->name() << "-dc-req: " << out_dc_req);
    }

    if (!out_lmem_req.mask.none()) {
      ReqLmem.push(out_lmem_req, delay_);
      DT(4, this->name() << "-lmem-req: " << out_lmem_req);
    }

    if (!out_mmio_req.mask.none()) {
      if (in_req.write) {
        //std::cout << "LmemSwitch: Pushing MMIO Write Req tag=" << in_req.tag << " cid=" << in_req.cid << " uuid=" << in_req.uuid << std::endl;
        ReqMMIOWrite.push(out_mmio_req, delay_);
        DT(4, this->name() << "-mmio-write-req: " << out_mmio_req);
      } else {
        //std::cout << "LmemSwitch: Pushing MMIO Read Req tag=" << in_req.tag << " cid=" << in_req.cid << " uuid=" << in_req.uuid << std::endl;
        ReqMMIORead.push(out_mmio_req, delay_);
        DT(4, this->name() << "-mmio-read-req: " << out_mmio_req);
      }
    }

    if (!out_virgo_mmio_req.mask.none()) {
      std::cout << "LmemSwitch: Pushing VIRGO MMIO req. tag=" << in_req.tag << ", cid=" << in_req.cid << ", uuid=" << in_req.uuid << std::endl;
      ReqVirgoMMIO.push(out_virgo_mmio_req, delay_);
      DT(4, this->name() << "-virgo-mmio-req: " << out_virgo_mmio_req);
    }

    ReqIn.pop();
  }
}

///////////////////////////////////////////////////////////////////////////////

LsuMemAdapter::LsuMemAdapter(
  const SimContext& ctx,
  const char* name,
  uint32_t num_inputs,
  uint32_t delay
) : SimObject<LsuMemAdapter>(ctx, name)
  , ReqIn(this)
  , RspIn(this)
  , ReqOut(num_inputs, this)
  , RspOut(num_inputs, this)
  , delay_(delay)
{
  std::cout << "LsuMemAdapter: " << this->name() << ", num_inputs=" << num_inputs << std::endl;
}

void LsuMemAdapter::reset() {}

void LsuMemAdapter::tick() {
  uint32_t input_size = ReqOut.size();

  // process outgoing responses
  for (uint32_t i = 0; i < input_size; ++i) {
    if (RspOut.at(i).empty())
      continue;
    auto& out_rsp = RspOut.at(i).front();
    //DT(4, this->name() << "-rsp" << i << ": " << out_rsp);
    // if (this->name() == "cluster0-lsu_lmem_adapter") {
    //   std::cout << "LsuMemAdapter: " << this->name() << ", Processing Rsp from port " << i << " tag=" << out_rsp.tag << std::endl;
    // }

    // build memory response
    LsuRsp in_rsp(input_size);
    in_rsp.mask.set(i);
    in_rsp.tag = out_rsp.tag;
    in_rsp.cid = out_rsp.cid;
    in_rsp.uuid = out_rsp.uuid;

    // include other responses with the same tag
    for (uint32_t j = i + 1; j < input_size; ++j) {
      if (RspOut.at(j).empty())
        continue;
      auto& other_rsp = RspOut.at(j).front();
      if (out_rsp.tag == other_rsp.tag) {
        in_rsp.mask.set(j);
        RspOut.at(j).pop();
      }
    }

    // send memory response
    RspIn.push(in_rsp, 1);

    // remove input
    RspOut.at(i).pop();
    break;
  }

  // process incoming requests
  if (!ReqIn.empty()) {
    auto& in_req = ReqIn.front();
    assert(in_req.mask.size() == input_size);
    for (uint32_t i = 0; i < input_size; ++i) {
      if (in_req.mask.test(i)) {
        // build memory request
        MemReq out_req;
        out_req.write = in_req.write;
        out_req.addr  = in_req.addrs.at(i);
        out_req.type  = get_addr_type(in_req.addrs.at(i));
        out_req.tag   = in_req.tag;
        out_req.cid   = in_req.cid;
        out_req.uuid  = in_req.uuid;
        // send memory request
        ReqOut.at(i).push(out_req, delay_);
        //DT(4, this->name() << "-req" << i << ": " << out_req);
        if (this->name() == "cluster0-lsu_lmem_adapter") {
         std::cout << "LsuMemAdapter: " << this->name() << ", Pushing MemReq to port " << i << " tag=" << out_req.tag << " addr=" << std::hex << out_req.addr << std::dec << std::endl;
        }
        
      }
    }
    ReqIn.pop();
  }
}