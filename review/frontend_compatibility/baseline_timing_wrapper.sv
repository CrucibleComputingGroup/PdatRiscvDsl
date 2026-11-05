module timing_check(input logic clk_i, input logic rst_ni, input logic instr_req_o, input logic instr_gnt_i, input logic data_req_out, input logic data_gnt_i, input logic data_rvalid_i);

  // ========================================
  // Cache-Aware Timing Constraints
  // ========================================
  // Models realistic cache hit/miss behavior for ABC optimization
  // Injected at ibex_core level for full signal visibility

  // Instruction cache timing tracking
  logic [2:0] instr_stall_counter_q;
  logic instr_likely_miss;

  always_ff @(posedge {clk} or negedge {rst_n}) begin
    if (!{rst_n}) begin
      instr_stall_counter_q <= 3'b0;
      instr_likely_miss <= 1'b0;
    end else begin
      if ({instr_req} && !{instr_gnt}) begin
        instr_stall_counter_q <= instr_stall_counter_q + 1;
        // After 1 cycle of stalling, assume cache miss
        instr_likely_miss <= (instr_stall_counter_q >= 1);
      end else begin
        instr_stall_counter_q <= 3'b0;
        instr_likely_miss <= 1'b0;
      end

    end
  end

  // Data cache timing tracking
  logic [2:0] data_stall_counter_q;
  logic data_likely_miss;

  always_ff @(posedge {clk} or negedge {rst_n}) begin
    if (!{rst_n}) begin
      data_stall_counter_q <= 3'b0;
      data_likely_miss <= 1'b0;
    end else begin
      if ({data_req} && !{data_gnt}) begin
        data_stall_counter_q <= data_stall_counter_q + 1;
        data_likely_miss <= (data_stall_counter_q >= 1);
      end else if ({data_rvalid}) begin
        data_stall_counter_q <= 3'b0;
        data_likely_miss <= 1'b0;
      end

    end
  end

  // Cache-aware timing assumptions
  always_comb begin
    if (rst_ni) begin
      // Instruction cache: different bounds for hit vs miss
      if (!instr_likely_miss) begin
        assume(instr_stall_counter_q <= 3'd1);  // Cache hit: 1 cycle max
      end else begin
        assume(instr_stall_counter_q <= 3'd5);  // Cache miss: up to 5 cycles
      end

      // Force completion at maximum latency
      if (instr_stall_counter_q == 3'd5) begin
        assume(instr_gnt_i);  // Must grant at max stall
      end
      // Data cache: different bounds for hit vs miss
      if (!data_likely_miss) begin
        assume(data_stall_counter_q <= 3'd1);   // Cache hit: 1 cycle max
      end else begin
        assume(data_stall_counter_q <= 3'd4);   // Cache miss: up to 4 cycles
      end

      // Force completion at maximum latency
      if (data_stall_counter_q == 3'd4) begin
        assume(data_gnt_i);  // Must grant at max stall
      end
    end
  end

endmodule
