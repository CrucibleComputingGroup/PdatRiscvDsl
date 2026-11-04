# CRISP 3 typed specialization contracts

CRISP 3 extends instruction-set selection with reusable predicates over sampled
operand values, explicit execution scopes, bounded response contracts, and guarded
observations. It generates verification monitors and obligations for a candidate
cut. A successful compilation does **not** prove a program satisfies the contract
or that a proposed RTL transformation preserves behavior.

Run from this repository:

```bash
python3 -m pdat_dsl contract examples/contracts/remainder.crisp \
  --output /tmp/remainder_contract.sv --witnesses
python3 -m pytest -q tests/test_contracts.py
```

The new command requires Python 3.8 or newer. Install optional SMT support with
`pip install -e '.[contracts]'`; validation additionally requires pytest, Yosys,
and Icarus Verilog. The checked environment is recorded in `VALIDATION.md`.

The `contract` command is an opt-in front end. Existing `parse`, `codegen`, and
hierarchical synthesis commands keep their historical behavior. Existing PPA
measurements therefore do not become measurements of this new front end.

## Why the language changes

The manuscript's original semantic domain is a set of instruction encodings.
That domain describes ISA selection but cannot distinguish two executions of
the same instruction with different operand values, interface histories, or
speculative status. Those distinctions need an explicit semantic home.

The active legacy v2 generator also has two independently reproduced problems:
`include ADD {rd = x0}` warns that an instruction pattern is unimplemented and
adds nothing; `include ADD; forbid x16-x31; include ADD` retains the restriction
although the stated union semantics restore all ADD encodings. The new policy
elaborator implements exact Boolean set operations. This is a correction to the
intended semantics, not evidence of bit-for-bit compatibility with those legacy
bugs. The legacy path has not been silently replaced.

## Example and declarations

```text
crisp 3
clock core = clk_i reset rst_ni active_low
event execution on core scope execute when accepted
signal accepted : bool @ execution = accepted_i
signal instruction : bits[32] @ execution = instr_i
signal operand : bits[32] @ execution = operand_i

predicate narrow(x: bits[W], n: nat) = fits_s(x, n) or fits_u(x, n)

policy subset at execution on instruction
include RV32I
include RV32M
forbid DIV
forbid DIVU
end
assume narrow_inputs at execution: narrow(operand, 16)
```

`clock` declares the sampling edge and reset polarity. Events name a clock,
semantic scope (`interface`, `decode`, `execute`, or `retire`), and Boolean
validity guard. Signals have a Boolean or fixed-width bit-vector type and exactly
one sampling event. Bindings identify the corresponding RTL objects. A source
line is one declaration; full-line comments start with `#`.

The event name is a nominal type index: two events with the same clock or scope
remain distinct. Reading a retirement value in an execution contract is rejected.
There is no implicit conversion from architectural behavior to speculative
execution, nor automatic alignment of transactions between stages. A necessary
cross-event relation belongs in a separately proved adapter; this implementation
does not yet synthesize such adapters.

An event declaration is a binding contract, not a discovered fact. Declaring a
retirement-valid signal to be an execution guard would misdescribe the design.
Name/width checking cannot establish transaction identity or the completeness of
the guard. This matters for RSD: the region premise must cover the actual
speculative attempts, including unavailable operands, rather than only retirement
or ready-operand events. `memory_regions.crisp` expresses that premise using
illustrative bindings; it is not an automatically validated RSD integration.

## Expression and refinement types

The type judgment is `Gamma |- e : B @ event`, where `B` is `bool` or
`bits[w]`. Constants have no event and can be used at any event. Combining two
nonconstant expressions requires equal event indices. Equality and arithmetic
require equal widths; Boolean and bitwise operators are distinct.

Signal references synthesize a type. In a comparison or arithmetic operation, a
known bit-vector operand supplies the expected width for an integer literal.
For example, `operand == 7` checks that 7 fits the declared width. `bv(value,w)`
supplies a width when there is no expected type. Negative integer literals and
implicit truncation are absent; write an explicit two's-complement bit pattern.
Addition, subtraction, and multiplication wrap modulo `2**w`.

Available forms are `and`, `or`, `not`, `==`, `!=`, unsigned `<`/`<=`, bitwise
`&`/`|`/`^`, `+`/`-`/`*`, and these functions:

| Form | Meaning |
| --- | --- |
| `slice(x,hi,lo)` | Inclusive bit slice, with checked bounds |
| `concat(a,b,...)` | Concatenation, first argument most significant |
| `zext(x,w)`, `sext(x,w)` | Explicit zero/sign extension |
| `slt(a,b)`, `sle(a,b)` | Signed comparisons of equal-width carriers |
| `ite(c,a,b)` | Boolean choice between equal-width values |
| `fits_u(x,n)` | `x` is the zero extension of its low `n` bits |
| `fits_s(x,n)` | `x` is the sign extension of its low `n` bits |
| `matches(word,"REM")` | Membership in a registered encoding set |

The refinements describe **values**, not recoverable source-language type tags.
For example, the union of unsigned 8- and 16-bit value domains equals the
unsigned 16-bit domain. They overlap without ambiguity. Narrow inputs do not
imply narrow results: two 8-bit operands can produce a 16-bit product. No result
width assumption is inferred from an input refinement.

Predicates declare `bool`, `nat`, or `bits[W]` parameters. Repeated `W` parameters
must instantiate to the same carrier width; different variables need not.
Natural parameters are compile-time integers. Their supported arithmetic is
bounded and cannot depend on an RTL signal. Bodies are expanded hygienically and
checked at instantiation; global signals, recursive definitions, and host-language
execution are excluded. This is a restricted bidirectional elaborator, not full
Hindley-Milner or Liquid Types inference.

Predicate libraries contain `library crisp 3` followed only by definitions.
`import "refinements.crisp"` loads them, without importing hidden assumptions.
For example, alignment and address-region predicates are library definitions:

```text
predicate aligned(x: bits[W], n: nat) = slice(x, n-1, 0) == 0
predicate input_region(x: bits[W]) = slice(x, 21, 16) == 0
predicate output_region(x: bits[W]) = not input_region(x)
```

These families use the existing typed core and all its backends. A genuinely new
temporal primitive would additionally require semantics and lowering validation.

## Exact instruction policies

A `policy name at event on word` block embeds v2 `include`/`forbid` statements.
Its current profile is RV32IM, with an explicit extensible encoding registry.
Other profiles, including compressed encodings, are rejected until their checked
signatures are supplied. A word is exactly 32 bits even when operand widths differ.

Starting from `S = false`, inclusion becomes `S or E`, and exclusion becomes
`S and not E`. A constrained instruction is its opcode predicate conjoined with
**all** its field predicates. This preserves cross-field correlations. In
particular, forbidding `ADD {rd=x3, rs1=x7}` forbids the conjunction, not either
field independently. Re-inclusion can widen the set again. Arbitrary finite
ranges are inclusive comparisons; they are not approximated by a common prefix.
No BDD-to-pattern enumeration or truncation is used.

Field names and widths come from encoding signatures. Integer bounds, mask widths,
and duplicate fields are checked. Immediate fragments retain the registry's
encoded-field meaning; this is not a decoder for all architectural immediates.
Operand refinements and legacy timing directives inside an ISA block are errors,
because they need sampled value or trace contracts. An empty policy denotes
false explicitly. Local satisfiability diagnostics can reveal that no enabled
instruction is admitted.

## Contract semantics and proof obligations

A contract is `(bindings, A, Q, O)`: assumptions, properties to establish, and
observations, under declared clocks/events. For a clause at event `e` with guard
`g_e`, the sampled proposition at cycle `t` is

`not reset(t) and g_e(t) => predicate(t)`.

`assume` contributes to `A`. It declares a usage domain; it does not establish that
the environment or application respects that domain. `assert` contributes to `Q`
and remains an unproved obligation. It is never silently used as an assumption.
Composition conjoins explicitly declared assumptions and obligations. This does
not constitute a circular assume/guarantee proof rule.

`observe name at e: value` defines a pair consisting of the event guard and the
value when that guard holds. Comparing original and transformed designs requires
equal guards on every non-reset sample and equal data on valid samples. Dropping
all valid events therefore fails observation equivalence. Cycle alignment is
preserved; there is no implicit stuttering or latency-insensitive comparison.

`candidate name at e: target -> replacement` checks the replacement's width and
event and records a relational obligation against the observation interface.
It does not rewrite RTL. Its manifest always records `authorized: false` until a
separate verification integration supplies a valid preservation proof; the
current compiler has no proof-import or approval mechanism.

The eventual transformation theorem must have the form

`for every admitted input trace u and related initial states,
 Obs(M,u) = Obs(M',u)`.

For deterministic synchronous models, sufficient obligations are initialization
of a relation `R`, preservation of `R` by both transition functions under `A`,
and equal observations whenever `R` holds. Induction on cycles proves the result.
Reset semantics and observation completeness are premises. The type checker
does not discharge these RTL obligations. Composing cuts requires evidence for
the actual intermediate design and a common observation/environment contract.

## Bounded response semantics

```text
assume response latency at bus: request -> response within 1..4
```

Each enabled request at edge `t` requires an enabled response at some edge
`t+j`, `1 <= j <= 4`. Intervals satisfy `0 <= low <= high <= 64` and count
physical edges of the declared clock. A disabled event does not stop the clock.
Reset cancels pending requests. Monitor history starts empty, independently of
the RTL's initial state. The generated ghost registers record predicates; they
are verification state, not added processor hardware.

This is a bounded-response property. One response can satisfy several requests;
it does not by itself specify transaction matching, queue capacity, ready/valid
stability, response data, or a complete cache protocol. Those need additional
contracts or a richer protocol extension. A finite trace ending before a
deadline reports a pending obligation rather than a successful proof.

The monitor invariant is simple: before edge `t`, history bit `j-1` records the
enabled request/response at `t-j`, unless reset has canceled that history. At
the deadline, the generated implication tests exactly the specified response
window. Initialization establishes this invariant; shifting and reset preserve
it. This is a paper proof of the core lowering, not a mechanized compiler proof.

## Guarantees and implementation boundaries

The expression core has a direct interpreter, SMT encoding, and SystemVerilog DAG
lowering. For well-typed expressions and samples respecting the binding types,
evaluation has the declared Boolean/bit-vector shape. A structural induction
covers literals/references, equal-width operations, checked slices/extensions,
and hygienic template expansion. Each lowering operation has the corresponding
bit-vector meaning; the response-history invariant extends this argument to the
implemented temporal construct. Tests exercise these claims, but are not a
proof of the complete Python implementation.

Generated monitors expose check outputs and observation outputs. Their JSON
manifest records source/library fingerprints, monitor hash, declared bindings,
event scopes, assumption/obligation roles, and pending candidate cuts. This is
provenance, not a proof certificate. Any downstream certificate must additionally
bind both actual RTL models, reset/environment semantics, observations, and the
proof procedure. BMC success and timeout must not be promoted to induction.

Signal input ports are named `crisp_in_<logical_name>`; clock and reset ports are
`crisp_<clock>_clk` and `crisp_<clock>_rst`. The manifest maps these ports to RTL
bindings. Prefixing prevents logical names from colliding with Verilog keywords.
The optional Python emitter module name must start with `crisp_`.

To check bindings against a real elaboration, export a Yosys JSON design at the
binding scope and pass `--rtl-json design.json --top module_name`. Flatten a
hierarchical scope first and bind whole exposed signals. This checks existence
and widths; event meaning remains a trusted integration contract. Without those
options the manifest explicitly labels binding validation as unchecked.

The model is two-state, edge-sampled, and supports independent clocks without
cross-clock expressions. It makes no guarantee about HDL X/Z propagation,
asynchronous reset transients, glitch behavior, transaction correspondence across
events, or arbitrary liveness. Z3 consistency queries exclude transition relations
and temporal obligations, so `sat` is not a reachability or realizability proof.

## Validation

`tests/test_contracts.py` includes negative specifications; all-word SMT checks
of exact ISA semantics; independent Yosys SAT miters for generated arithmetic,
refinements, and cross-field constraints; and Icarus simulation of generated
response monitors against direct trace semantics. It checks all 256 four-cycle
request/response traces for each of four intervals, plus reset/event-gap cases.
It also checks library extension, observation validity, library fingerprints,
real Yosys binding widths, CLI integration, and all four complete examples.

All seven existing `v2_*.dsl` ISA fragments elaborate with a satisfiable policy.
The one file's timing directives are explicitly separated for that test; they
are not silently translated into the new bounded-response semantics. This
coverage establishes a useful migration starting point, not equivalence to every
legacy backend behavior or all experiment contracts.

## Research foundations

The checking/synthesis split follows the design guidance in
[Dunfield and Krishnaswami, Bidirectional Typing](https://research.cs.queensu.ca/home/jana/papers/bidir-survey/Dunfield21_bidir-survey.pdf).
Value refinements draw on
[Rondon, Kawaguchi, and Jhala, Liquid Types](https://ranjitjhala.github.io/static/liquid_types.pdf),
without claiming its inference algorithm or decidability theorem for arbitrary
CRISP temporal reasoning. Event interfaces adapt the idea of explicit temporal
contracts from
[Nigam et al., Modular Hardware Design with Timeline Types](https://people.csail.mit.edu/rachit/files/pubs/filament.pdf);
CRISP uses named sampling events because processor execution can stall or replay.

[Dahlia](https://arxiv.org/pdf/2004.04852) illustrates tying hardware types to a
specific safety property. [Rosette](https://homes.cs.washington.edu/~bodik/ucb/Files/2014/rosette-pldi2014.pdf)
motivates a small shared core supporting language libraries. Neither hardware
typing nor solver-backed DSL extensibility alone is a novelty claim. The research
opportunity is the connection between execution-scoped assumptions, conditional
observations, and reusable specialization contracts on existing processor RTL.
