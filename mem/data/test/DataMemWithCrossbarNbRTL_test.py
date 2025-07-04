"""
==========================================================================
DataMemWithCrossbarNbRTL_test.py
==========================================================================
Test cases for DataMemWithCrossbarNbRTL.

Author : Yufei Yang
  Date : July 3, 2025
"""

from pymtl3.passes.backends.verilog import (VerilogTranslationPass)
from pymtl3.stdlib.test_utils import config_model_with_cmdline_opts

from ..DataMemWithCrossbarNbRTL import DataMemWithCrossbarNbRTL
from ....lib.basic.val_rdy.SinkRTL import SinkRTL as TestSinkRTL
from ....lib.basic.val_rdy.SourceRTL import SourceRTL as TestSrcRTL
from ....lib.messages import *
from ....lib.opt_type import *


#-------------------------------------------------------------------------
# Test harness
#-------------------------------------------------------------------------

class TestHarness(Component):

  def construct(s, NocPktType, CgraPayloadType, DataType, DataAddrType,
                data_mem_size_global, data_mem_size_per_bank, num_banks,
                rd_tiles, wr_tiles, num_cgra_rows, num_cgra_columns,
                num_tiles,
                read_addr, read_data, write_addr,
                write_data, noc_recv_load,
                send_to_noc_load_request_pkt, send_to_noc_store_pkt,
                preload_data_per_bank,
                NocPktType_NB = None,
                CgraPayloadType_NB = None,
                DataAddrType_NB = None):

    s.num_banks = num_banks
    s.rd_tiles = rd_tiles
    s.wr_tiles = wr_tiles
    s.recv_raddr = [TestSrcRTL(DataAddrType_NB, read_addr[i])
                      for i in range(rd_tiles)]
    s.send_rdata = [TestSinkRTL(DataType, read_data[i])
                    for i in range(rd_tiles)]

    s.recv_waddr = [TestSrcRTL(DataAddrType, write_addr[i])
                    for i in range(wr_tiles)]
    s.recv_wdata = [TestSrcRTL(DataType, write_data[i])
                    for i in range(wr_tiles)]

    s.recv_from_noc = TestSrcRTL(NocPktType_NB, noc_recv_load)
    s.send_to_noc_load_request_pkt = TestSinkRTL(NocPktType_NB, send_to_noc_load_request_pkt)
    s.send_to_noc_store_pkt = TestSinkRTL(NocPktType, send_to_noc_store_pkt)

    s.data_mem = DataMemWithCrossbarNbRTL(NocPktType,
                                          CgraPayloadType,
                                          DataType,
                                          data_mem_size_global,
                                          data_mem_size_per_bank,
                                          num_banks,
                                          rd_tiles,
                                          wr_tiles,
                                          num_cgra_rows,
                                          num_cgra_columns,
                                          num_tiles,
                                          preload_data_per_bank = preload_data_per_bank,
                                          NocPktType_NB = NocPktType_NB,
                                          CgraPayloadType_NB = CgraPayloadType_NB,
                                          DataAddrType_NB = DataAddrType_NB)

    for i in range(rd_tiles):
      s.data_mem.recv_raddr[i] //= s.recv_raddr[i].send
      s.data_mem.send_rdata[i] //= s.send_rdata[i].recv

    for i in range(wr_tiles):
      s.data_mem.recv_waddr[i] //= s.recv_waddr[i].send
      s.data_mem.recv_wdata[i] //= s.recv_wdata[i].send

    s.data_mem.recv_from_noc_load_response_pkt //= s.recv_from_noc.send
    s.data_mem.send_to_noc_load_request_pkt //= s.send_to_noc_load_request_pkt.recv
    s.data_mem.send_to_noc_store_pkt //= s.send_to_noc_store_pkt.recv

    s.data_mem.address_lower //= 0
    s.data_mem.address_upper //= 31

    s.cgra_id = 0

  def done(s):
    for i in range(s.rd_tiles):
      if not s.recv_raddr[i].done() or not s.send_rdata[i].done():
        return False

    for i in range(s.wr_tiles):
      if not s.recv_waddr[i].done() or not s.recv_wdata[i].done():
        return False

    if not s.send_to_noc_load_request_pkt.done() or \
       not s.send_to_noc_store_pkt.done() or \
       not s.recv_from_noc.done():
      return False

    return True

  def line_trace(s):
    return s.data_mem.line_trace()

def run_sim(test_harness, max_cycles = 40):
  test_harness.apply(DefaultPassGroup())
  test_harness.sim_reset()

  # Run simulation

  ncycles = 0
  print()
  print("{}:{}".format(ncycles, test_harness.line_trace()))
  while not test_harness.done() and ncycles < max_cycles:
    test_harness.sim_tick()
    ncycles += 1
    print("{}:{}".format(ncycles, test_harness.line_trace()))

  # Check timeout
  assert ncycles < max_cycles

  test_harness.sim_tick()
  test_harness.sim_tick()
  test_harness.sim_tick()

def test_const_queue_non_blocking(cmdline_opts):
  data_nbits = 32
  predicate_nbits = 1
  kernel_id_nbits = 2 # 4 kernels
  ld_id_nbits = 5 # 32 load operations
  DataType = mk_data(data_nbits, predicate_nbits)
  data_mem_size_global = 64
  data_mem_size_per_bank = 16
  num_banks = 2
  nterminals = 4

  num_registers_per_reg_bank = 16
  num_cgra_columns = 1
  num_cgra_rows = 1
  width = 2
  height = 2
  num_tiles = 4
  ctrl_mem_size = 6
  num_tile_inports  = 4
  num_tile_outports =4
  num_fu_inports = 4
  num_fu_outports = 2

  DataAddrType_NB = mk_nb_addr(clog2(data_mem_size_global), kernel_id_nbits, ld_id_nbits)
  DataAddrType = mk_bits(clog2(data_mem_size_global))
  CtrlAddrType = mk_bits(clog2(ctrl_mem_size))

  CtrlType = mk_ctrl(num_fu_inports,
                     num_fu_outports,
                     num_tile_inports,
                     num_tile_outports,
                     num_registers_per_reg_bank)

  CgraPayloadType = mk_cgra_payload(DataType,
                                    DataAddrType,
                                    CtrlType,
                                    CtrlAddrType)

  CgraPayloadType_NB = mk_cgra_payload(DataType,
                                       DataAddrType_NB,
                                       CtrlType,
                                       CtrlAddrType)

  InterCgraPktType = mk_inter_cgra_pkt(num_cgra_columns,
                                       num_cgra_rows,
                                       num_tiles,
                                       CgraPayloadType)

  InterCgraPktType_NB = mk_inter_cgra_pkt(num_cgra_columns,
                                          num_cgra_rows,
                                          num_tiles,
                                          CgraPayloadType_NB)

  IntraCgraPktType = mk_intra_cgra_pkt(num_cgra_columns,
                                       num_cgra_rows,
                                       num_tiles,
                                       CgraPayloadType)

  test_meta_data = [
      # addr:  0     1     2     3     4     5     6     7     8     9    10    11    12    13    14    15
           [0xa6, 0xa7, 0xa8, 0xa9, 0xb0, 0xb1, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xc0, 0xc1],
      # addr: 16    17    18    19    20    21    22    23    24    25    26    27    28    29    30    31
           [0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xd0, 0xd1, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7]]

  preload_data_per_bank = [[DataType(test_meta_data[j][i], 1)
                            for i in range(data_mem_size_per_bank)]
                           for j in range(num_banks)]

  rd_tiles = 2
  wr_tiles = 2
  # Input data, two memory access addr 44 and  58, one is from kernel 1 ld 1, another is from kernel 2 ld 2.
  # DataAddrType_NB(addr, kernel_id, ld_id) 
  read_addr = [
               #Cycle0, in range, hit         Cycle1, out of range, miss     Cycle2, out of range, hit      Cycle3, in range, hit         Cycle4, in range, hit
               [DataAddrType_NB(0, 0, 0), DataAddrType_NB(44, 1, 1), DataAddrType_NB(44, 1, 1), DataAddrType_NB(1, 0, 0), DataAddrType_NB(2, 0, 0)], # Tile 0
               #Cycle0, out of range, miss     Cycle1, in range, hit          Cycle2, in range, hit          Cycle3, out of range, hit      Cycle4, in range, hit
               [DataAddrType_NB(58, 2, 2), DataAddrType_NB(16, 0, 0), DataAddrType_NB(17, 0, 0), DataAddrType_NB(58, 2, 2), DataAddrType_NB(18, 0, 0)], # Tile 1
              ]
  
  # Expected.
  read_data = [
               #Cycle0             Cycle1             Cycle2               Cycle3             Cycle4
               [DataType(0xa6, 1), DataType(0x00, 0), DataType(0xabcd, 1), DataType(0xa7, 1), DataType(0xa80, 1)], # Tile 0
               [DataType(0x00, 0), DataType(0xc2, 1), DataType(0xc3, 1), DataType(0xdcba, 1), DataType(0xc4, 1)] # Tile 1
              ]

  # Input data.
  write_addr = [
                [DataAddrType(2), DataAddrType(45)],
                [DataAddrType(40), DataAddrType(31)],
                [DataAddrType(2)],
                []
               ]
  write_data = [
                [DataType(0xa80, 1), DataType(0xd545, 1)],
                [DataType(0xd040, 1), DataType(0xd70, 1)],
                [DataType(0xa800, 1)],
                []
               ]


  # Input data.
  send_to_noc_load_request_pkt = [
                        # src  dst src_x src_y dst_x dst_y src_tile dst_tile opq vc
                        InterCgraPktType_NB(0,   0,  0,    0,    0,    0,    0,       0,       0,  0, CgraPayloadType_NB(CMD_LOAD_REQUEST, data_addr = DataAddrType_NB(58, 2, 2))), # Cycle0
                        InterCgraPktType_NB(0,   0,  0,    0,    0,    0,    0,       0,       0,  0, CgraPayloadType_NB(CMD_LOAD_REQUEST, data_addr = DataAddrType_NB(44, 1, 1))), # Cycle1
                        InterCgraPktType_NB(0,   0,  0,    0,    0,    0,    0,       0,       0,  0, CgraPayloadType_NB(CMD_LOAD_REQUEST, data_addr = DataAddrType_NB(44, 1, 1))), # Cycle2
                        InterCgraPktType_NB(0,   0,  0,    0,    0,    0,    0,       0,       0,  0, CgraPayloadType_NB(CMD_LOAD_REQUEST, data_addr = DataAddrType_NB(58, 2, 2))), # Cycle3
  ]

  noc_recv_load_data = [
                        InterCgraPktType_NB(0,   0,  0,    0,    0,    0,    0,       0,       0,  0, CgraPayloadType_NB(CMD_LOAD_RESPONSE, DataType(0xffff, 0), DataAddrType_NB(0, 0, 0))), # Cycle0, nothing
                        InterCgraPktType_NB(0,   0,  0,    0,    0,    0,    0,       0,       0,  0, CgraPayloadType_NB(CMD_LOAD_RESPONSE, DataType(0xabcd, 1), DataAddrType_NB(44, 1, 1))), # Cycle1, assume remote access returns kernel 1 ld 1 at Cycle 1
                        InterCgraPktType_NB(0,   0,  0,    0,    0,    0,    0,       0,       0,  0, CgraPayloadType_NB(CMD_LOAD_RESPONSE, DataType(0xdcba, 1), DataAddrType_NB(58, 2, 2))), # Cycle2, assume remote access returns kernel 2 ld 2 at Cycle 2
                        InterCgraPktType_NB(0,   0,  0,    0,    0,    0,    0,       0,       0,  0, CgraPayloadType_NB(CMD_LOAD_RESPONSE, DataType(0xffff, 0), DataAddrType_NB(0, 0, 0))), # Cycle3, nothing
                        InterCgraPktType_NB(0,   0,  0,    0,    0,    0,    0,       0,       0,  0, CgraPayloadType_NB(CMD_LOAD_RESPONSE, DataType(0xffff, 0), DataAddrType_NB(0, 0, 0))), # Cycle4, nothing
                       ]

  # Expected.
  send_to_noc_store_pkt = [
                     # src  dst src_x src_y dst_x dst_y src_tile dst_tile opq vc                                                         data_addr
      InterCgraPktType(0,   0,  0,    0,    0,    0,    0,       0,       0,  0, CgraPayloadType(CMD_STORE_REQUEST, DataType(0xd040, 1), 40)),
      InterCgraPktType(0,   0,  0,    0,    0,    0,    0,       0,       0,  0, CgraPayloadType(CMD_STORE_REQUEST, DataType(0xd545, 1), 45)),
  ]

  th = TestHarness(InterCgraPktType,
                   CgraPayloadType,
                   DataType,
                   DataAddrType,
                   data_mem_size_global,
                   data_mem_size_per_bank,
                   num_banks,
                   rd_tiles,
                   wr_tiles,
                   num_cgra_rows,
                   num_cgra_columns,
                   num_tiles,
                   read_addr,
                   read_data,
                   write_addr,
                   write_data,
                   noc_recv_load_data,
                   send_to_noc_load_request_pkt,
                   send_to_noc_store_pkt,
                   preload_data_per_bank,
                   NocPktType_NB = InterCgraPktType_NB,
                   CgraPayloadType_NB = CgraPayloadType_NB,
                   DataAddrType_NB = DataAddrType_NB)

  th.elaborate()
  th.data_mem.set_metadata(VerilogTranslationPass.explicit_module_name,
                           f'DataMemWithCrossbarNbRTL_translation')
  th = config_model_with_cmdline_opts( th, cmdline_opts, duts=['data_mem'] )

  run_sim(th)
