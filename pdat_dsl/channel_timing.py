"""Per-request interface timing for CRISP: wait-time bounds on memory channels.

The original timing template bounds how long a request waits for acceptance
(`data_hit_latency` etc.). It cannot express how long an accepted request waits
for its response, which is what bounds the occupancy of structures that track
outstanding requests (MSHRs, load buffers, write buffers, transaction tables).

Declarations (CRISP):

    grant_latency    <channel> N     # a raised request is accepted within N cycles
    response_latency <channel> N     # each accepted request is answered 1..N cycles later
    response_latency <channel> M N   # ... M..N cycles later (M = N: exact latency)

A channel is described in the core configuration under `signals.channels`:

    channels:
      data_load:
        req_valid: data_load_req_o
        req_ready: data_load_gnt_i       # optional (always accepted if absent)
        req_tag:   data_load_tag_o       # optional: tagged, responses in any order
        tag_width: 4                     # required with req_tag
        rsp_valid: data_load_rvalid_i
        rsp_ready: data_load_rready_o    # optional (always taken if absent)
        rsp_tag:   data_load_tag_i       # required when req_tag is given
        max_outstanding: 4               # untagged channels: monitor capacity

The generated SVA is a set of environment monitors plus immediate assumptions,
injected like every other CRISP assumption. Tagged channels get one age monitor per
tag; untagged channels are in order and get a FIFO of ages. Responses are assumed
only for outstanding requests, and an outstanding request at its deadline is
answered. Requests always have distinct deadlines because a channel accepts at most
one request per cycle.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ChannelTimingRule:
    """`grant_latency <channel> N` or `response_latency <channel> N`."""
    kind: str
    channel: str
    cycles: int
    line: int = 0
    min_cycles: int = 1


def _bits(n: int) -> int:
    return max(1, (n).bit_length())


def _grant(name: str, ch: Dict, n: int, clk: str, rst_n: str) -> str:
    if "req_ready" not in ch:
        return f"  // CRISP grant_latency {name} {n}: channel has no ready signal; nothing to assume\n"
    w = _bits(n + 1)
    v, r = ch["req_valid"], ch["req_ready"]
    return f"""
  // CRISP grant_latency {name} {n}: a raised request is accepted within {n} cycles
  logic [{w-1}:0] crisp_{name}_gwait_q;
  always_ff @(posedge {clk} or negedge {rst_n}) begin
    if (!{rst_n}) crisp_{name}_gwait_q <= '0;
    else if ({v} && !{r}) crisp_{name}_gwait_q <= crisp_{name}_gwait_q + {w}'d1;
    else crisp_{name}_gwait_q <= '0;
  end
  always_comb begin
    if ({rst_n} && {v} && crisp_{name}_gwait_q >= {w}'d{n}) assume ({r});
  end
"""


def _response_tagged(name: str, ch: Dict, n: int, clk: str, rst_n: str, lo: int = 1) -> str:
    w = _bits(n + 1)
    tw = int(ch["tag_width"])
    v, tag = ch["req_valid"], ch["req_tag"]
    req_fire = f"{v} && {ch['req_ready']}" if "req_ready" in ch else v
    rsp_fire = f"{ch['rsp_valid']} && {ch['rsp_ready']}" if "rsp_ready" in ch else ch["rsp_valid"]
    rv, rtag = ch["rsp_valid"], ch["rsp_tag"]
    nt = 1 << tw
    min_tag = f" && crisp_{name}_age_q[{rtag}] >= {w}'d{lo}" if lo > 1 else ""
    decl = f"{lo} {n}" if lo > 1 else f"{n}"
    return f"""
  // CRISP response_latency {name} {decl}: every accepted request (tag {tag}) is
  // answered {lo}..{n} cycles after acceptance; responses only for outstanding tags
  logic [{nt-1}:0] crisp_{name}_out_q;
  logic [{w-1}:0] crisp_{name}_age_q [{nt}];
  always_ff @(posedge {clk} or negedge {rst_n}) begin
    if (!{rst_n}) begin
      crisp_{name}_out_q <= '0;
      for (int t = 0; t < {nt}; t++) crisp_{name}_age_q[t] <= '0;
    end else begin
      for (int t = 0; t < {nt}; t++) begin
        if (({req_fire}) && {tag} == {tw}'(t)) begin
          crisp_{name}_out_q[t] <= 1'b1;
          crisp_{name}_age_q[t] <= {w}'d1;
        end else if (({rsp_fire}) && {rtag} == {tw}'(t)) begin
          crisp_{name}_out_q[t] <= 1'b0;
          crisp_{name}_age_q[t] <= '0;
        end else if (crisp_{name}_out_q[t] && crisp_{name}_age_q[t] < {w}'d{n}) begin
          crisp_{name}_age_q[t] <= crisp_{name}_age_q[t] + {w}'d1;
        end
      end
    end
  end
  always_comb begin
    if ({rst_n}) begin
      if ({rv}) assume (crisp_{name}_out_q[{rtag}]{min_tag});
      for (int t = 0; t < {nt}; t++)
        if (crisp_{name}_out_q[t] && crisp_{name}_age_q[t] >= {w}'d{n})
          assume ({rv} && {rtag} == {tw}'(t));
    end
  end
"""


def _response_inorder(name: str, ch: Dict, n: int, clk: str, rst_n: str, lo: int = 1) -> str:
    m = int(ch.get("max_outstanding", 4))
    w = _bits(n + 1)
    cw = _bits(m)
    v = ch["req_valid"]
    req_fire = f"{v} && {ch['req_ready']}" if "req_ready" in ch else v
    rsp_fire = f"{ch['rsp_valid']} && {ch['rsp_ready']}" if "rsp_ready" in ch else ch["rsp_valid"]
    rv = ch["rsp_valid"]
    min_head = f" && crisp_{name}_age_q[0] >= {w}'d{lo}" if lo > 1 else ""
    decl = f"{lo} {n}" if lo > 1 else f"{n}"
    shift = "\n".join(
        f"    if (crisp_{name}_pop) begin\n"
        f"      if ({i + 1} < crisp_{name}_cnt_q) crisp_{name}_age_d[{i}] = crisp_{name}_inc(crisp_{name}_age_q[{i + 1}]);\n"
        f"    end else if ({i} < crisp_{name}_cnt_q) crisp_{name}_age_d[{i}] = crisp_{name}_inc(crisp_{name}_age_q[{i}]);"
        if i + 1 < m else
        f"    if (!crisp_{name}_pop && {i} < crisp_{name}_cnt_q) crisp_{name}_age_d[{i}] = crisp_{name}_inc(crisp_{name}_age_q[{i}]);"
        for i in range(m))
    return f"""
  // CRISP response_latency {name} {decl}: in-order responses; every accepted request is
  // answered {lo}..{n} cycles after acceptance (the monitor tracks up to {m} outstanding)
  logic [{cw-1}:0] crisp_{name}_cnt_q, crisp_{name}_cnt_d;
  logic [{w-1}:0] crisp_{name}_age_q [{m}];
  logic [{w-1}:0] crisp_{name}_age_d [{m}];
  wire crisp_{name}_push = ({req_fire}) && (crisp_{name}_cnt_q < {cw}'d{m} || ({rsp_fire}));
  wire crisp_{name}_pop  = ({rsp_fire}) && crisp_{name}_cnt_q != '0;
  function automatic logic [{w-1}:0] crisp_{name}_inc(input logic [{w-1}:0] a);
    return (a < {w}'d{n}) ? a + {w}'d1 : a;
  endfunction
  always_comb begin
    for (int i = 0; i < {m}; i++) crisp_{name}_age_d[i] = '0;
{shift}
    crisp_{name}_cnt_d = crisp_{name}_cnt_q - {cw}'(crisp_{name}_pop);
    if (crisp_{name}_push) begin
      crisp_{name}_age_d[crisp_{name}_cnt_d] = {w}'d1;
      crisp_{name}_cnt_d = crisp_{name}_cnt_d + {cw}'d1;
    end
  end
  always_ff @(posedge {clk} or negedge {rst_n}) begin
    if (!{rst_n}) begin
      crisp_{name}_cnt_q <= '0;
      for (int i = 0; i < {m}; i++) crisp_{name}_age_q[i] <= '0;
    end else begin
      crisp_{name}_cnt_q <= crisp_{name}_cnt_d;
      for (int i = 0; i < {m}; i++) crisp_{name}_age_q[i] <= crisp_{name}_age_d[i];
    end
  end
  always_comb begin
    if ({rst_n}) begin
      if ({rv}) assume (crisp_{name}_cnt_q != '0{min_head});
      if (crisp_{name}_cnt_q != '0 && crisp_{name}_age_q[0] >= {w}'d{n}) assume ({rv});
    end
  end
"""


def generate_channel_timing(rules: List[ChannelTimingRule], config) -> str:
    """SystemVerilog monitors and assumptions for the channel timing rules."""
    if not rules:
        return ""
    channels = getattr(config.signals, "channels", {}) or {}
    clk, rst_n = config.signals.clk, config.signals.rst_n
    code = "\n  // ========================================\n"
    code += "  // CRISP interface wait-time bounds (per request)\n"
    code += "  // ========================================\n"
    for rule in rules:
        if rule.channel not in channels:
            raise ValueError(f"line {rule.line}: channel '{rule.channel}' is not defined in the configuration")
        ch = channels[rule.channel]
        if rule.kind == "grant_latency":
            code += _grant(rule.channel, ch, rule.cycles, clk, rst_n)
        elif rule.kind == "response_latency":
            if not 1 <= rule.min_cycles <= rule.cycles:
                raise ValueError(f"line {rule.line}: response_latency needs 1 <= min <= max")
            if "req_tag" in ch:
                if "rsp_tag" not in ch or "tag_width" not in ch:
                    raise ValueError(f"channel '{rule.channel}': req_tag requires rsp_tag and tag_width")
                code += _response_tagged(rule.channel, ch, rule.cycles, clk, rst_n, rule.min_cycles)
            else:
                code += _response_inorder(rule.channel, ch, rule.cycles, clk, rst_n, rule.min_cycles)
        else:
            raise ValueError(f"unknown channel timing rule {rule.kind}")
    return code
