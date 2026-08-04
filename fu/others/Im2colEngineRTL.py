"""
==========================================================================
Im2colEngineRTL.py
==========================================================================
Runtime-programmable im2col (image-to-column) engine. Reads a
single-channel input image resident in a local input scratchpad and
streams the lowered (kH*kW) x (Hout*Wout) column matrix into the
enclosing CGRA's shared data memory as a sequence of CMD_STORE_REQUEST
packets on `send_pkt`.

All geometry is configured at runtime via the CMD_IM2COL_LAUNCH packet:
the same synthesized RTL can run any (H, W, kH, kW, stride, in_base,
dest_base) combination that fits in the packed fields. The launch
packet doubles as the trigger -- on the IDLE -> READ edge the engine
latches the packed config and starts.

Packed layout of the CMD_IM2COL_LAUNCH packet:

  payload.data_addr (DataAddrType)  = dest_base   (SRAM store base)
  payload.data.payload (32-bit) bit-fields:
      [ 5: 0]  W             (image width,  <= scratch_mem_size)
      [11: 6]  H             (image height, <= scratch_mem_size)
      [17:12]  in_base       (image base addr in in_mem)
      [20:18]  kW            (kernel width,  1..7)
      [23:21]  kH            (kernel height, 1..7)
      [27:24]  log2_stride   (encoded stride; stride = 1 << log2_stride,
                              so log2_stride 0..15 -> stride 1..32768)
      [31:28]  reserved

Note on stride encoding: stride is stored as log2(stride) so the FSM
can use shifts (>>) instead of divides for computing Hout/Wout, since
the Verilog translator does not support `//`. This restricts stride to
powers of two (1, 2, 4, 8, ...); pack_im2col_launch enforces this.

Emit convention: output i (0-based, ox innermost -> oy -> kx -> ky) is
stored to shared data_mem addr (dest_base + i). Downstream consumers
configure their LD_CONST addresses to match this contiguous range.

Note on packet dst_tile: CMD_STORE_REQUEST packets are routed by the
CGRA controller using the payload.data_addr field alone (see
ControllerRTL.update_sending_to_noc_msg); the packet's dst_tile is
ignored on this path. The engine emits dst=0 for its store packets.

State machine: IDLE -> READ -> EMIT -> DONE
  IDLE: hold rdy=1 for the LAUNCH packet. On handshake, latch config
        from packet fields, precompute Hout/Wout, transition to READ.
  READ: assert the read on in_mem at in_addr(oy, ox, ky, kx).
  EMIT: hold the incoming data, drive send_pkt with
        (data_addr = dest_base + emit_idx, data). On send_pkt fire,
        advance the nested loop counters and either loop back to READ
        or transition to DONE.
  DONE: assert `done` and stay here until reset.
"""

from pymtl3 import *

from ...lib.basic.val_rdy.ifcs import ValRdyRecvIfcRTL as RecvIfcRTL
from ...lib.basic.val_rdy.ifcs import ValRdySendIfcRTL as SendIfcRTL
from ...lib.cmd_type import CMD_IM2COL_LAUNCH, CMD_STORE_REQUEST
from ...lib.util.data_struct_attr import kAttrCmd, kAttrDataAddr, kAttrPayload
from ...mem.data.DataMemRTL import DataMemRTL


# Packed-field bit widths in the CMD_IM2COL_LAUNCH packet's data field.
_W_BITS           = 6
_H_BITS           = 6
_IN_BASE_BITS     = 6
_KW_BITS          = 3
_KH_BITS          = 3
_LOG2_STRIDE_BITS = 4

_W_SHIFT           = 0
_H_SHIFT           = _W_BITS                                            # 6
_IN_BASE_SHIFT     = _H_SHIFT + _H_BITS                                 # 12
_KW_SHIFT          = _IN_BASE_SHIFT + _IN_BASE_BITS                     # 18
_KH_SHIFT          = _KW_SHIFT + _KW_BITS                               # 21
_LOG2_STRIDE_SHIFT = _KH_SHIFT + _KH_BITS                               # 24


def pack_im2col_launch(H, W, in_base, kH, kW, stride):
  """Pack im2col geometry into the CMD_IM2COL_LAUNCH packet's data field.

  Use this helper on the CPU side to keep the bit layout in one place.
  Returns an int suitable for DataType(x, 1). Stride must be a power
  of two (see the module docstring for why).
  """
  assert 0 < H       < (1 << _H_BITS),       f"H={H} out of range"
  assert 0 < W       < (1 << _W_BITS),       f"W={W} out of range"
  assert 0 <= in_base< (1 << _IN_BASE_BITS), f"in_base={in_base} out of range"
  assert 0 < kH      < (1 << _KH_BITS),      f"kH={kH} out of range"
  assert 0 < kW      < (1 << _KW_BITS),      f"kW={kW} out of range"
  assert stride > 0 and (stride & (stride - 1)) == 0, \
      f"stride={stride} must be a power of two"
  log2_stride = stride.bit_length() - 1
  assert log2_stride < (1 << _LOG2_STRIDE_BITS), \
      f"stride={stride} too large"
  return ((log2_stride << _LOG2_STRIDE_SHIFT) |
          (kH          << _KH_SHIFT)          |
          (kW          << _KW_SHIFT)          |
          (in_base     << _IN_BASE_SHIFT)     |
          (H           << _H_SHIFT)           |
          (W           << _W_SHIFT))


class Im2colEngineRTL(Component):

  def construct(s, DataType, IntraCgraPktType, CgraPayloadType,
                scratch_mem_size,
                preload_image):

    # Derive widths from the passed-in packet types so the engine stays
    # generic across CGRA configurations.
    ScratchAddrType = mk_bits(clog2(scratch_mem_size))
    PayloadType     = IntraCgraPktType.get_field_type(kAttrPayload)
    CmdType         = PayloadType.get_field_type(kAttrCmd)
    DataAddrType    = PayloadType.get_field_type(kAttrDataAddr)

    # Narrow types for extracting each packed field from launch.data.
    # A field's slot must be extracted at exactly its own bit width so
    # neighboring slots don't leak into it (e.g. a 6-bit trunc of the
    # kW slot would grab kH's bits too).
    _W_type           = mk_bits(_W_BITS)
    _H_type           = mk_bits(_H_BITS)
    _IN_BASE_type     = mk_bits(_IN_BASE_BITS)
    _KW_type          = mk_bits(_KW_BITS)
    _KH_type          = mk_bits(_KH_BITS)
    _LOG2_STRIDE_type = mk_bits(_LOG2_STRIDE_BITS)

    # emit_idx must fit any valid destination offset within data_mem.
    IdxType   = DataAddrType
    StateType = mk_bits(2)

    S_IDLE = StateType(0)
    S_READ = StateType(1)
    S_EMIT = StateType(2)
    S_DONE = StateType(3)

    # Public I/O.
    s.recv_cmd_pkt = RecvIfcRTL(IntraCgraPktType)
    s.done         = OutPort(b1)
    s.send_pkt     = SendIfcRTL(IntraCgraPktType)

    # Input scratchpad: image preloaded starting at addr 0. Real deploys
    # would fill this via DMA/CPU writes prior to LAUNCH; the preload is
    # a modeling convenience for tests.
    preload_full = [DataType(0, 1) for _ in range(scratch_mem_size)]
    for i, v in enumerate(preload_image):
      preload_full[i] = DataType(v, 1)

    s.in_mem = DataMemRTL(DataType, scratch_mem_size,
                          rd_ports = 1, wr_ports = 1,
                          preload_data = preload_full)

    # FSM state.
    s.state    = Wire(StateType)
    s.oy       = Wire(ScratchAddrType)
    s.ox       = Wire(ScratchAddrType)
    s.ky       = Wire(ScratchAddrType)
    s.kx       = Wire(ScratchAddrType)
    s.emit_idx = Wire(IdxType)
    s.data_reg = Wire(DataType)

    # Runtime config, latched on IDLE -> READ (LAUNCH handshake).
    s.cfg_H           = Wire(ScratchAddrType)
    s.cfg_W           = Wire(ScratchAddrType)
    s.cfg_in_base     = Wire(ScratchAddrType)
    s.cfg_kH          = Wire(ScratchAddrType)
    s.cfg_kW          = Wire(ScratchAddrType)
    s.cfg_log2_stride = Wire(ScratchAddrType)
    s.cfg_dest_base   = Wire(DataAddrType)
    # Precomputed at LAUNCH so the S_EMIT loop-advance can compare
    # against a plain register instead of redoing (H-kH)>>log2_stride
    # each cycle.
    s.cfg_Hout = Wire(ScratchAddrType)
    s.cfg_Wout = Wire(ScratchAddrType)

    # Combinational address computation. Narrowing casts (ScratchAddrType(...))
    # keep pymtl3's AST analyzer happy on binop-result narrowing.
    s.in_row  = Wire(ScratchAddrType)
    s.in_col  = Wire(ScratchAddrType)
    s.in_addr = Wire(ScratchAddrType)

    @update
    def comb_in_addr():
      # ox*stride and oy*stride are shifts because stride is stored as
      # its log2 (Verilog translator has no `//` but supports `>>`/`<<`).
      s.in_row  @= ScratchAddrType((s.oy << s.cfg_log2_stride) + s.ky)
      s.in_col  @= ScratchAddrType((s.ox << s.cfg_log2_stride) + s.kx)
      s.in_addr @= ScratchAddrType(s.cfg_in_base + s.in_row * s.cfg_W + s.in_col)

    # Tie off the unused write port on the input scratchpad.
    @update
    def tie_off_in_mem_wr():
      s.in_mem.recv_waddr[0].val @= b1(0)
      s.in_mem.recv_waddr[0].msg @= ScratchAddrType(0)
      s.in_mem.recv_wdata[0].val @= b1(0)
      s.in_mem.recv_wdata[0].msg @= DataType()

    # Read + emit datapath. Pass integer zeros (rather than type
    # constructors like CtrlType()) for the don't-care fields so the
    # verilator translator can encode them as constants -- it can't
    # handle default constructors of bitstructs that contain list-of-bits
    # fields (e.g. the CtrlType.fu_in array) inside behavioral RTLIR.
    @update
    def drive_read_and_emit():
      s.in_mem.recv_raddr[0].val @= b1(0)
      s.in_mem.recv_raddr[0].msg @= ScratchAddrType(0)
      s.in_mem.send_rdata[0].rdy @= b1(0)

      # LAUNCH packet is accepted only while idle.
      s.recv_cmd_pkt.rdy @= (s.state == S_IDLE)

      s.send_pkt.val @= b1(0)
      s.done         @= b1(0)

      if s.state == S_READ:
        s.in_mem.recv_raddr[0].val @= b1(1)
        s.in_mem.recv_raddr[0].msg @= s.in_addr
        s.in_mem.send_rdata[0].rdy @= b1(1)

      if s.state == S_EMIT:
        s.send_pkt.val @= b1(1)
        s.send_pkt.msg @= IntraCgraPktType(
            0,             # src
            0,             # dst (unused for STORE_REQUEST -- controller
                           # routes by payload.data_addr)
            0, 0,          # src/dst cgra_id
            0, 0,          # src cgra x/y
            0, 0,          # dst cgra x/y
            0,             # opaque
            0,             # vc_id
            PayloadType(
                CmdType(CMD_STORE_REQUEST),
                s.data_reg,
                s.cfg_dest_base + zext(s.emit_idx, DataAddrType),
                0,         # ctrl (zero)
                0,         # ctrl_addr
            ),
        )
      else:
        s.send_pkt.msg @= IntraCgraPktType(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            PayloadType(0, 0, 0, 0, 0),
        )

      if s.state == S_DONE:
        s.done @= b1(1)

    # FSM transitions.
    @update_ff
    def fsm():
      if s.reset:
        s.state    <<= S_IDLE
        s.oy       <<= ScratchAddrType(0)
        s.ox       <<= ScratchAddrType(0)
        s.ky       <<= ScratchAddrType(0)
        s.kx       <<= ScratchAddrType(0)
        s.emit_idx <<= IdxType(0)
        s.data_reg <<= DataType()
      else:
        if s.state == S_IDLE:
          if s.recv_cmd_pkt.val & (
              s.recv_cmd_pkt.msg.payload.cmd == CmdType(CMD_IM2COL_LAUNCH)):
            # Extract each packed field at its native width, then zext
            # into ScratchAddrType so downstream arithmetic is uniform.
            data_bits = s.recv_cmd_pkt.msg.payload.data.payload
            H_lat       = zext(trunc(data_bits >> _H_SHIFT,           _H_type),           ScratchAddrType)
            W_lat       = zext(trunc(data_bits >> _W_SHIFT,           _W_type),           ScratchAddrType)
            in_base_lat = zext(trunc(data_bits >> _IN_BASE_SHIFT,     _IN_BASE_type),     ScratchAddrType)
            kH_lat      = zext(trunc(data_bits >> _KH_SHIFT,          _KH_type),          ScratchAddrType)
            kW_lat      = zext(trunc(data_bits >> _KW_SHIFT,          _KW_type),          ScratchAddrType)
            log2_stride_lat = zext(trunc(data_bits >> _LOG2_STRIDE_SHIFT, _LOG2_STRIDE_type), ScratchAddrType)

            s.cfg_H           <<= H_lat
            s.cfg_W           <<= W_lat
            s.cfg_in_base     <<= in_base_lat
            s.cfg_kH          <<= kH_lat
            s.cfg_kW          <<= kW_lat
            s.cfg_log2_stride <<= log2_stride_lat
            s.cfg_dest_base   <<= s.recv_cmd_pkt.msg.payload.data_addr
            # Hout = (H - kH) >> log2_stride + 1; Wout similarly.
            s.cfg_Hout        <<= ScratchAddrType(((H_lat - kH_lat) >> log2_stride_lat) + 1)
            s.cfg_Wout        <<= ScratchAddrType(((W_lat - kW_lat) >> log2_stride_lat) + 1)

            s.state    <<= S_READ
            s.oy       <<= ScratchAddrType(0)
            s.ox       <<= ScratchAddrType(0)
            s.ky       <<= ScratchAddrType(0)
            s.kx       <<= ScratchAddrType(0)
            s.emit_idx <<= IdxType(0)

        elif s.state == S_READ:
          if s.in_mem.recv_raddr[0].rdy & s.in_mem.send_rdata[0].val:
            s.data_reg <<= s.in_mem.send_rdata[0].msg
            s.state    <<= S_EMIT

        elif s.state == S_EMIT:
          if s.send_pkt.val & s.send_pkt.rdy:
            # Loop order (ox -> oy -> kx -> ky, ox innermost) so emit_idx
            # walks the flat output index monotonically. This lets the
            # store addr be a simple dest_base + emit_idx sum.
            if s.ox + ScratchAddrType(1) < s.cfg_Wout:
              s.ox    <<= s.ox + ScratchAddrType(1)
              s.state <<= S_READ
            elif s.oy + ScratchAddrType(1) < s.cfg_Hout:
              s.ox    <<= ScratchAddrType(0)
              s.oy    <<= s.oy + ScratchAddrType(1)
              s.state <<= S_READ
            elif s.kx + ScratchAddrType(1) < s.cfg_kW:
              s.ox    <<= ScratchAddrType(0)
              s.oy    <<= ScratchAddrType(0)
              s.kx    <<= s.kx + ScratchAddrType(1)
              s.state <<= S_READ
            elif s.ky + ScratchAddrType(1) < s.cfg_kH:
              s.ox    <<= ScratchAddrType(0)
              s.oy    <<= ScratchAddrType(0)
              s.kx    <<= ScratchAddrType(0)
              s.ky    <<= s.ky + ScratchAddrType(1)
              s.state <<= S_READ
            else:
              s.state <<= S_DONE
            s.emit_idx <<= s.emit_idx + IdxType(1)
        # S_DONE is terminal until reset.

  def line_trace(s):
    state_map = {0: "IDLE", 1: "READ", 2: "EMIT", 3: "DONE"}
    st = state_map[int(s.state)]
    return (f"engine[{st} emit_idx={int(s.emit_idx)} "
            f"oy={int(s.oy)} ox={int(s.ox)} ky={int(s.ky)} kx={int(s.kx)} "
            f"done={int(s.done)} "
            f"send_val={int(s.send_pkt.val)} send_rdy={int(s.send_pkt.rdy)}] "
            f"in_addr={int(s.in_addr)} data_reg={s.data_reg}")
