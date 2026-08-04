"""
==========================================================================
Im2colEngineRTL_test.py
==========================================================================
Standalone unit test for the runtime-programmable Im2colEngineRTL. Feeds
a preloaded image into the engine's in_mem, sends a CMD_IM2COL_LAUNCH
packet whose fields carry the runtime im2col config, and sinks the
emitted CMD_STORE_REQUEST packets on send_pkt to check them against a
Python golden.
"""

from pymtl3 import *
from pymtl3.stdlib.test_utils import (config_model_with_cmdline_opts,
                                      run_sim)

from ..Im2colEngineRTL import Im2colEngineRTL, pack_im2col_launch
from ....lib.basic.val_rdy.SinkRTL import SinkRTL as TestSinkRTL
from ....lib.basic.val_rdy.SourceRTL import SourceRTL as TestSrcRTL
from ....lib.cmd_type import CMD_IM2COL_LAUNCH, CMD_STORE_REQUEST
from ....lib.messages import (mk_cgra_payload, mk_ctrl, mk_data,
                                mk_intra_cgra_pkt)


#-------------------------------------------------------------------------
# Test parameters (a small CGRA config -- only used to size the packet
# type; the engine is exercised in isolation).
#-------------------------------------------------------------------------

NUM_CGRA_COLUMNS           = 4
NUM_CGRA_ROWS              = 1
X_TILES                    = 3
Y_TILES                    = 3
NUM_TILES                  = X_TILES * Y_TILES
DATA_BITWIDTH              = 32
DATA_MEM_SIZE_GLOBAL       = 128
CTRL_MEM_SIZE              = 6
NUM_TILE_INPORTS           = 4
NUM_TILE_OUTPORTS          = 4
NUM_FU_INPORTS             = 4
NUM_FU_OUTPORTS            = 2
NUM_REGISTERS_PER_REG_BANK = 16

DataAddrType     = mk_bits(clog2(DATA_MEM_SIZE_GLOBAL))
CtrlAddrType     = mk_bits(clog2(CTRL_MEM_SIZE))
DataType         = mk_data(DATA_BITWIDTH, 1)
CtrlType         = mk_ctrl(NUM_FU_INPORTS, NUM_FU_OUTPORTS,
                           NUM_TILE_INPORTS, NUM_TILE_OUTPORTS,
                           NUM_REGISTERS_PER_REG_BANK)
CgraPayloadType  = mk_cgra_payload(DataType, DataAddrType,
                                   CtrlType, CtrlAddrType)
IntraCgraPktType = mk_intra_cgra_pkt(NUM_CGRA_COLUMNS, NUM_CGRA_ROWS,
                                     NUM_TILES, CgraPayloadType)


#-------------------------------------------------------------------------
# Software golden
#-------------------------------------------------------------------------

def golden_im2col(image, H, W, kH, kW, stride):
  Hout = (H - kH) // stride + 1
  Wout = (W - kW) // stride + 1
  out = [0] * (kH * kW * Hout * Wout)
  for oy in range(Hout):
    for ox in range(Wout):
      for ky in range(kH):
        for kx in range(kW):
          row = ky * kW + kx
          col = oy * Wout + ox
          out[row * (Hout * Wout) + col] = image[(oy * stride + ky) * W +
                                                 (ox * stride + kx)]
  return out, Hout, Wout


#-------------------------------------------------------------------------
# Test harness: engine + TestSrcRTL for launch + TestSinkRTL for outputs.
#-------------------------------------------------------------------------

class TestHarness(Component):

  def construct(s, scratch_mem_size, launch_pkts,
                preload_image, expected_packets):

    s.dut = Im2colEngineRTL(DataType, IntraCgraPktType, CgraPayloadType,
                            scratch_mem_size, preload_image)

    s.launch_src = TestSrcRTL(IntraCgraPktType, launch_pkts)
    s.launch_src.send //= s.dut.recv_cmd_pkt

    # Compare only the fields the engine actually populates: cmd, data,
    # data_addr. dst is always 0 (STORE_REQUEST is routed by data_addr).
    cmp_fn = lambda a, b: (a.payload.cmd       == b.payload.cmd and
                           a.payload.data      == b.payload.data and
                           a.payload.data_addr == b.payload.data_addr)
    s.sink = TestSinkRTL(IntraCgraPktType, expected_packets,
                         cmp_fn = cmp_fn)
    s.dut.send_pkt //= s.sink.recv

  def done(s):
    return s.launch_src.done() and s.sink.done()

  def line_trace(s):
    return f"{s.dut.line_trace()} || sink[{s.sink.line_trace()}]"


#-------------------------------------------------------------------------
# Driver
#-------------------------------------------------------------------------

def _build_expected_packets(image, H, W, kH, kW, stride, dest_base):
  values, _, _ = golden_im2col(image, H, W, kH, kW, stride)
  pkts = []
  for i, v in enumerate(values):
    pkts.append(IntraCgraPktType(
        0, 0,                      # src, dst (dst unused on this path)
        0, 0, 0, 0, 0, 0,          # src/dst cgra_id + x/y
        0, 0,                      # opaque, vc_id
        CgraPayloadType(cmd = CMD_STORE_REQUEST,
                        data = DataType(v, 1),
                        data_addr = dest_base + i)))
  return pkts


def run_engine(image, H, W, kH, kW, stride,
               in_base = 0, dest_base = 0,
               scratch_mem_size = 64, cmdline_opts = None):

  launch_pkts = [IntraCgraPktType(payload = CgraPayloadType(
      CMD_IM2COL_LAUNCH,
      data      = DataType(pack_im2col_launch(H = H, W = W,
                                              in_base = in_base,
                                              kH = kH, kW = kW,
                                              stride = stride), 1),
      data_addr = dest_base))]

  expected = _build_expected_packets(image, H, W, kH, kW, stride, dest_base)
  th = TestHarness(scratch_mem_size, launch_pkts, image, expected)
  th.elaborate()
  if cmdline_opts is not None:
    th = config_model_with_cmdline_opts(th, cmdline_opts, duts = ['dut'])
  run_sim(th)


#-------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------

def test_engine_4x4_k2_s1(cmdline_opts):
  # 4x4 image, 2x2 kernel, stride 1 -> 3x3 output grid, 4x9 lowered.
  image = list(range(16))
  run_engine(image, H = 4, W = 4, kH = 2, kW = 2, stride = 1,
             cmdline_opts = cmdline_opts)


def test_engine_4x4_k2_s2(cmdline_opts):
  # Stride-2: 4x4 / 2x2 / s2 -> 2x2 output grid, 4x4 lowered.
  image = [i * 2 + 1 for i in range(16)]
  run_engine(image, H = 4, W = 4, kH = 2, kW = 2, stride = 2,
             cmdline_opts = cmdline_opts)


def test_engine_5x5_k3_s1(cmdline_opts):
  # 5x5 / 3x3 / s1 -> 3x3 output grid, 9x9 lowered (81 outputs).
  image = list(range(25))
  run_engine(image, H = 5, W = 5, kH = 3, kW = 3, stride = 1,
             scratch_mem_size = 128,
             cmdline_opts = cmdline_opts)


def test_engine_smoke_matches_e2e_layout(cmdline_opts):
  # Matches the geometry used by cgra/test/Im2colCgraE2E_test.py's smoke
  # test (image [1,3,2,4], 1x4 / 1x2 / s2 -> lowered [1,2,3,4] stored
  # to SRAM addr 0..3).
  image = [1, 3, 2, 4]
  run_engine(image, H = 1, W = 4, kH = 1, kW = 2, stride = 2,
             cmdline_opts = cmdline_opts)


def test_engine_nonzero_dest_base(cmdline_opts):
  # Verifies that dest_base actually offsets the store address. Same
  # image geometry as test_engine_4x4_k2_s2 but write to addr 16..19.
  image = [i * 2 + 1 for i in range(16)]
  run_engine(image, H = 4, W = 4, kH = 2, kW = 2, stride = 2,
             dest_base = 16,
             cmdline_opts = cmdline_opts)
