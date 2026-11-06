"""Tests for per-request interface wait-time bounds and the v2 compressed-check option."""
from pathlib import Path
import subprocess
import sys

import pytest

from pdat_dsl.channel_timing import ChannelTimingRule, generate_channel_timing
from pdat_dsl.config import CoreConfig, load_config
from pdat_dsl.parser import parse_dsl

ROOT = Path(__file__).resolve().parents[1]

CONFIG = """core_name: "toy"
architecture: "rv64"
injection:
  module_path: "toy"
  description: "toy"
signals:
  clk: "clk_i"
  rst_n: "rst_ni"
  instruction_data: "instr_i"
  pc: "pc_o"
  has_compressed_check: false
  channels:
    mem:
      req_valid: "req_o"
      req_ready: "gnt_i"
      req_tag: "tag_o"
      tag_width: 3
      rsp_valid: "rvalid_i"
      rsp_tag: "rtag_i"
    imem:
      req_valid: "ireq_o"
      req_ready: "igen_i"
      rsp_valid: "irvalid_i"
      rsp_ready: "irready_o"
      max_outstanding: 2
"""


def _config(tmp_path):
    p = tmp_path / "toy.yaml"
    p.write_text(CONFIG)
    return load_config(p)


def test_parse_channel_rules():
    ast = parse_dsl("version 2\ngrant_latency mem 0\nresponse_latency mem 3\n")
    rules = [r for r in ast.rules if isinstance(r, ChannelTimingRule)]
    assert [(r.kind, r.channel, r.cycles) for r in rules] == [
        ("grant_latency", "mem", 0), ("response_latency", "mem", 3)]


def test_tagged_monitor(tmp_path):
    code = generate_channel_timing([ChannelTimingRule("response_latency", "mem", 3)], _config(tmp_path))
    assert "logic [7:0] crisp_mem_out_q;" in code          # one monitor per tag (3-bit tag)
    assert "if (rvalid_i) assume (crisp_mem_out_q[rtag_i]);" in code
    assert "crisp_mem_age_q[t] >= 3'd3" in code          # age counter saturates at the bound
    assert "$bits" not in code                               # Synlig mis-elaborates $bits(port)


def test_inorder_monitor_and_grant(tmp_path):
    cfg = _config(tmp_path)
    code = generate_channel_timing([ChannelTimingRule("grant_latency", "imem", 0),
                                    ChannelTimingRule("response_latency", "imem", 2)], cfg)
    assert "assume (igen_i);" in code
    assert "crisp_imem_age_q[0] >= 2'd2) assume (irvalid_i);" in code
    assert "irvalid_i && irready_o" in code                   # response handshake uses ready


def test_errors(tmp_path):
    cfg = _config(tmp_path)
    with pytest.raises(ValueError):
        generate_channel_timing([ChannelTimingRule("response_latency", "nope", 2)], cfg)
    with pytest.raises(ValueError):
        generate_channel_timing([ChannelTimingRule("response_latency", "mem", 0)], cfg)


def test_codegen_cli_writes_timing_and_honours_compressed_check(tmp_path):
    cfg = tmp_path / "toy.yaml"
    cfg.write_text(CONFIG)
    dsl = tmp_path / "t.dsl"
    dsl.write_text("version 2\ninclude RV32I\nresponse_latency mem 2\n")
    out = tmp_path / "t.sv"
    subprocess.run([sys.executable, "-m", "pdat_dsl", "codegen", "--config", str(cfg), str(dsl), str(out)],
                   cwd=ROOT, check=True, capture_output=True)
    assert "!= 2'b11" not in out.read_text()                 # no 16-bit exemption without RVC
    assert "crisp_mem_out_q" in (tmp_path / "t_timing.sv").read_text()
