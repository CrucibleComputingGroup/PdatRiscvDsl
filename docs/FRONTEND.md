# Checked CRISP frontend

The `frontend` command adds reusable specifications and validation before the
existing CRISP generators. It expands into the original DSL and invokes either
the unchanged standalone `codegen` command or the existing hierarchical operand
generator. Known defective standalone paths require explicit compatibility mode. It does not implement
a second SVA emitter or claim additional hardware removal.

Use Python 3.8+ and install the solver dependency:

```sh
pip install -e '.[frontend]'
PYTHONHASHSEED=0 python -m pdat_dsl frontend examples/frontend/operands.crisp \
  /tmp/narrow.sv --target ibex --module-dtype operand_scope --emit-legacy /tmp/narrow.dsl \
  --manifest /tmp/narrow.json
python -m pdat_dsl frontend examples/frontend/narrow.crisp --target ibex --check
```

`--config PATH` selects a custom existing core configuration. `--target` accepts
`ibex`, `boom`, or `rocket`. With neither, the existing default configuration is
used. `--check` performs elaboration and solver checks without SVA generation.
`--module-dtype LABEL` uses the selected configuration's `signals` mapping to
emit only local operand assumptions. `LABEL` names the output scope/comment;
it does not select a `config.modules` entry or infer a module's ports. Supply
`--config` with signal names for the intended injection site when integrating
into RTL. The `--target ibex` command above only demonstrates generation using
the existing Ibex mapping. It requires an operand-only source
and rejects ISA, register, PC, or timing rules instead of silently dropping them.
The configuration must describe the intended module signals; the frontend does
not infer RTL signal correspondence. See `examples/frontend/operands.crisp`.
The manifest records bindings, expansion provenance, exactness checks, and the
hashes of generated artifacts. It is an audit record, not an independent proof
certificate.

## A reusable operand restriction

```text
crisp frontend 1
backend 1
context decode = isa
context ex = alu
predicate small(x: bits[W], n: nat) = fits_u(x, n)

constraint narrow(op: instruction, n: nat) at c: alu:
  operand $op.inputs at $c satisfies small(value, $n) lowers ~(u8 | u$n)
end

at decode: require RV32I
use narrow(ADD, 16) at ex
use narrow(SUB, 8) at ex
```

The expansion is ordinary legacy DSL:

```text
version 1
require RV32I
instruction ADD { dtype = ~(u8 | u16) }
instruction SUB { dtype = ~u8 }
```

`inputs` applies independently to the input operand roles present in the
instruction encoding. `rs1` or `rs2` selects one role. `rd` is rejected because
the existing generator has no output-operand constraint route. The frontend
rejects compressed dtype clauses and dtype widths equal to or larger than the
configured carrier width rather than relying on unsupported legacy behavior.

Predicates take `bits[W]`, fixed-width `bits[32]`, `bool`, and `nat` parameters.
They use typed Boolean/bit-vector expressions, `fits_u`, `fits_s`, `slice`,
`zext`, `sext`, `concat`, `ite`, and signed comparison helpers `slt`/`sle`.
A predicate must return a Boolean at its operand use. Width mismatches require
explicit conversion. Constraint blocks take `instruction`, `nat`, and
`registers` arguments and one explicit context parameter. Blocks may call
blocks; recursion, unbound parameters, and implicit global context capture are
rejected. Expansion retains rule and instantiation order.

A definition library starts with `crisp library 1`. Import it with
`import "relative/path.crisp"`. Libraries contain predicate/block definitions
and imports, never active assumptions or context declarations. Import cycles
and duplicate definitions are rejected.

## What contexts check

| Context kind | Existing route |
|---|---|
| `isa` | Instruction, register, and PC declarations |
| `alu` | Configured ALU input operands |
| `multdiv` | Configured multiply/divide input operands |
| `memory` | Existing memory timing declarations and sidecar |

For example, `operand MUL.rs1 at ex ...` is rejected if `ex = alu`; it requires
a `multdiv` context. These are checked generator routes, not newly introduced
sampling events. The configured signal names and injection hierarchy remain
trusted. No valid guard, reset policy, clock edge, instruction identity, or
pipeline correspondence is inferred or added. The manifest explicitly records
this trust boundary.

## Why the `lowers` clause is checked

A source expression `P(value)` and the declared legacy operand domain `D(value)`
are independently translated into finite-width formulas. The solver must prove
both `P and not D` and `D and not P` unsatisfiable. A mismatch produces a concrete
lost or extra operand value; timeout/unknown is rejected. This prevents a richer
source expression from silently restricting or widening its intended domain.

For example, `fits_u(value, 8) lowers ~u8` passes. Replacing its predicate with
`fits_u(value, 7)` introduces an extra legacy value; replacing it with
`fits_u(value, 9)` loses a source-permitted value. `lowers u8` uses the legacy
negative-domain convention, so its source must describe values *outside* the
unsigned-eight-bit domain.

The exactness guarantee concerns the local dtype predicate and is conditional
on the backend-domain model. It does not prove whole execution-domain admission.
It is not a
proof of general compiler correctness, application compliance, timing semantics,
or safe hardware removal. The solver checks one operand value domain at a time;
relational constraints between multiple operands are not added to the backend.

## Supported legacy declarations

`backend 1` accepts context-bound `require`, `require_registers`,
`require_pc_bits`, and instruction exclusions, plus checked operand clauses.
`backend 2` accepts the supported ordered `include`/`forbid` subset. It rejects
all instruction-field patterns, repeated includes, and operand or
register rules that the legacy backend cannot preserve. Raw patterns are
rejected because their generated descriptions depend on source line numbers.

Memory clauses such as `at bus: data_miss_latency 4` retain the existing
interpretation and artifact output. Unsupported `locality_bits`, duplicate
singleton rules, and bounds outside the existing three-bit counters' range are
rejected. See `examples/frontend/timing.crisp` and `ordered_isa.crisp`.

## Compatibility and inherited limitations

The original backend and experiment inputs are kept unchanged. Paired examples
and a frozen-baseline audit compare primary SVA, timing sidecars, and hierarchical
operand assumptions byte for byte. Run:

```sh
python -m pytest -q tests/test_frontend.py tests/test_frontend_compatibility.py
python scripts/check_sva_compatibility.py --help
```

The legacy generator has Python hash-seed-dependent ordering. Use the same
explicit `PYTHONHASHSEED` for baseline and candidate when comparing bytes; the
frontend does not change that behavior.

The legacy inline generator also excludes every instruction named by a dtype
rule from its allowed instruction set. Thus the inline expansion above does
**not** admit narrow ADD/SUB executions: it emits local operand restrictions
alongside instruction prohibitions. The frontend reports this inherited side
effect on stderr and in `legacy_effects` in the manifest, and rejects standalone
generation unless `--legacy-parity` explicitly requests the old artifact. Its exactness proof
concerns the local operand predicate only. The separate existing
`generate_module_dtype_assertions` API emits conditional operand restrictions
without the instruction-exclusion side effect; the compatibility tests cover
that path independently. The existing full hierarchical synthesis flow also
filters dtype rules from its core ISA constraints, so those rules are handled
only in the module. Existing standalone shell scripts invoking `codegen` do not repair the
inline exclusion. The historical operand-width results require provenance and
reproduction checks before being attributed to this current inline path.

The existing timing template also emits literal `{clk}`, `{rst_n}`, and request
placeholders in counter blocks. Byte parity preserves this defect; it does not
validate those bindings or establish that the historical timing experiment can
be reproduced from the current generator. Repairing the legacy template would
change its SVA and is outside this compatibility-preserving change. Generating
these timing sidecars likewise requires `--legacy-parity`; ordinary generation
rejects the known defective path before writing output.

The separate experimental `contract` command remains available for existing
users. It emits a different representation and is not the compatibility path
or the frontend evaluated in this paper.
