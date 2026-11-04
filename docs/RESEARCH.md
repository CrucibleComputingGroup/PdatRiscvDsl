# CRISP research assessment and ranked directions

## 1) SETTINGS ECHO

- **Domain:** application-constrained specialization of existing processor RTL.
- **Target output count:** five ranked ideas.
- **Audience:** DAC reviewers; the language supports the pruning contribution.
- **Decisive-test horizon:** one day per idea, with small executable tests first.
- **Resources:** existing Python DSL, Z3, Yosys, Icarus Verilog, processor RTL and
  existing experiments. No FPGA or new physical-design runs are required.
- **Baselines:** original CRISP instruction-set algebra and YAML configuration;
  handwritten SystemVerilog properties; Liquid Types, Dahlia, Filament, Rosette,
  and SecVerilog as related PL mechanisms, not interchangeable pruning systems.
- **Success metrics:** exact denotation, rejection of ill-scoped specifications,
  extension without backend changes, and explicit preservation obligations.
  Existing PPA numbers may remain unchanged.
- **Novelty bar:** a defensible system contribution. Generic types, refinements,
  and temporal assertions are established ideas; combining them requires a
  specialization-specific benefit and evidence. No claim of being first is made.

### What the manuscript and implementation actually needed

The original Section 3 gives a denotation over instruction encodings. Data-width
and interface constraints appear later in the microarchitecture YAML, without a
shared value/trace semantics. Consequently, the claimed coverage of all three
constraint levels is stronger than the language definition. The remainder-only
divider also needs an observation interface: quotient state can vary while
remaining irrelevant, and completion/stall behavior must still be preserved.

The working-tree RSD case study provides an especially useful motivation.
Its address partition must hold on executed speculative/replayed attempts,
including unavailable operands. A property about committed addresses cannot be
lifted to that scope automatically. This case study already existed when this
work began; its experiments were neither changed nor independently reproduced.

The active legacy AIG path has two reproduced semantic gaps:

| Source | Stated meaning | Legacy behavior | New checked path |
|---|---|---|---|
| `include ADD {rd=x0}` | ADD encodings with zero destination | Pattern warning; nothing added | Exact constrained predicate |
| `include ADD; forbid x16-x31; include ADD` | All ADD encodings restored | Existing entry retains restricted fields | Exact union restores the set |
| Range such as `rd in x3-x11` | Every value from 3 through 11 | Original manuscript describes common prefixes, which cannot express this exact range | Inclusive bit-vector bounds |

The first two findings concern `codegen_v2_aig.py` (`add_instruction`,
`process_include_rule`). The third is a manuscript semantics problem, not a
claim that every legacy backend implements the same approximation. The new
`contract` command is opt-in; the historical generator remains unchanged.

### Primary literature and the design lesson

The five linked PDFs were consulted for their language designs and relevant
formal sections. SecVerilog's author abstract was checked; its full proof was
not used as evidence for a CRISP theorem.

| Primary source | Relevant mechanism | Adopt in CRISP | Do not claim |
|---|---|---|---|
| [Dunfield and Krishnaswami, *Bidirectional Typing* (2021)](https://research.cs.queensu.ca/home/jana/papers/bidir-survey/Dunfield21_bidir-survey.pdf), especially §§3–4 and refinements discussion | Separate synthesis from checking; control annotations and error locality | Signals supply widths; literals are checked against them; explicit casts at boundaries | General type inference or hardware correctness from width checking |
| [Rondon, Kawaguchi, and Jhala, *Liquid Types*, PLDI 2008](https://ranjitjhala.github.io/static/liquid_types.pdf), inference construction and soundness theorems | Value predicates refine carrier types; qualifier restrictions matter for inference | Exact signed/unsigned value domains and composable predicates | Liquid inference, automatic RTL invariant discovery, or its theorems for unrestricted temporal properties |
| [Nigam et al., *Predictable Accelerator Design with Time-Sensitive Affine Types*, PLDI 2020](https://arxiv.org/pdf/2004.04852), affine/time typing and formal core | A hardware type discipline is valuable when tied to a concrete resource property | Tie each CRISP check to a specific specification error or obligation | Affine annotations alone justify erasing existing processor state |
| [Nigam, Azevedo de Amorim, and Sampson, *Modular Hardware Design with Timeline Types*, PLDI 2023](https://people.csail.mit.edu/rachit/files/pubs/filament.pdf), timeline interfaces and formal guarantees | Timing availability is part of an interface for statically scheduled pipelines | Make sampling events explicit for dynamic processor traces | Fixed cycle offsets describe arbitrary stalls, replay, or out-of-order transaction identity |
| [Torlak and Bodik, *A Lightweight Symbolic Virtual Machine for Solver-Aided Host Languages*, PLDI 2014](https://homes.cs.washington.edu/~bodik/ucb/Files/2014/rosette-pldi2014.pdf), symbolic-core architecture | Reuse solver translation across DSLs through a small shared core | Library predicates lower through one Boolean/bit-vector core | A new solver-aided DSL architecture, or a need to replace the entire existing backend |
| [Zhang et al., *A Hardware Design Language for Timing-Sensitive Information-Flow Security*, ASPLOS 2015](https://www.cs.cornell.edu/andru/papers/asplos15/), author abstract | Types enforce a relational information-flow property, including timing | Treat observable validity/timing as part of a cut's preservation target | Security noninterference directly proves arbitrary hardware erasure |

These comparisons motivate the proposed mechanisms. They are not an exhaustive
prior-art search establishing novelty against all contract and verification DSLs.
In particular, SVA can express the underlying predicates: the intended gain is a
checked, reusable specialization interface, not greater expressivity than SVA.

## 2) IDEA CARDS (ranked)

### 1. Explicit Sampling Scopes for Specialization Contracts

**A) Title:** Explicit Sampling Scopes for Specialization Contracts.

**B) Hypothesis:** nominal event types reject retirement-to-execution misuse and
unmatched stage combinations while admitting the existing constraint families.

**C) Novelty claim:** new relative to CRISP's encoding-set semantics and ad hoc
YAML/SVA binding. The key mechanism couples each sampled value to a named event
and semantic scope. Relative to Filament, the target is assumptions about
dynamic execution of existing RTL rather than scheduling newly composed hardware.

**D) Why it matters:** a valid architectural fact can fail on speculative
execution. Making the scope visible helps reviewers identify the actual premise
of a pruning result and gives the checker a concrete misuse to reject.

**E) Differentiator:** compare against legacy CRISP, handwritten SVA, and Filament.
The contribution is scoped specialization contracts; event names alone do not
verify that the RTL bindings have the claimed meaning.

**F) Assumptions:**
- The core exposes a meaningful execution guard and the relevant values.
- Transaction identity within an event is established by the adapter.
- Cross-event relations require explicit correspondence evidence.

**G) How this dies:**
- Most dangerous errors merely relabel a wrong RTL guard and evade checking.
- Real constraints require frequent implicit cross-event coercions.
- An existing interface already provides equivalent checked scopes at no cost.

**H) Cheapest decisive test:** hand-code an execution/retirement pair and two
negative contracts. Require both inappropriate lifts and mixed expressions to
fail, while four complete examples compile. One hour, Python and Yosys. If the
idea is wrong, sound cases need unchecked casts or the negative cases pass.
**Observed:** the prototype rejects both scope errors and compiles all four
examples. Real processor adapter correctness remains unproved by this test.

**I) Stop/pivot:** stop a claim of automatic scope certification if it depends
on trusted labels. Pivot to a smaller explicitly trusted adapter interface if
cross-event alignment cannot be established cheaply.

**J) Week-one plan:**
- Inspect actual execution guards and reset behavior in one core.
- Bind the required values and check their elaborated names/widths.
- Add negative speculative traces to test guard completeness.

**K) Risk/confidence:** technical Medium; novelty Medium; execution Low for
the front end, Medium for adapters; test decisiveness Medium.

### 2. Reusable Value Refinements over Exact Encoding Sets

**A) Title:** Reusable Value Refinements over Exact Encoding Sets.

**B) Hypothesis:** a small typed core supports instruction, width, alignment, and
region constraints without per-feature backend modifications or lossy encodings.

**C) Novelty claim:** new relative to the current split between instruction
patterns and specialized datatype/timing generators. The mechanism is exact set
algebra plus hygienic, width-parameterized predicate libraries. Refinements and
shared symbolic cores themselves are established by Liquid Types and Rosette.

**D) Why it matters:** one semantics can support multiple front-end features
and independently checked lowerings. It also exposes concrete implementation
errors, including lost field correlations and failure to widen on re-inclusion.

**E) Differentiator:** compare against legacy CRISP, Liquid Types, and Rosette.
This is foundational language engineering supporting the main novelty claim,
not a standalone novel type system.

**F) Assumptions:**
- The Boolean/bit-vector fragment covers the intended predicates.
- Encoding signatures are audited against the supported ISA profile.
- New libraries add definitions rather than untyped raw HDL snippets.

**G) How this dies:**
- Every useful extension needs a new compiler primitive.
- Typed elaboration silently coerces widths or treats overlapping refinements as tags.
- Exact formulas are too expensive for downstream synthesis.

**H) Cheapest decisive test:** hand-code independent SMT and HDL predicates for
range, cross-field exclusion, re-inclusion, and narrow values; add alignment,
region, and one custom instruction without editing backends. Threshold: no
counterexample for the selected formulas and no backend change for those
extensions. Two hours, Z3 and Yosys. **Observed:** these checks pass. Large-policy
compile/synthesis cost has not been benchmarked.

**I) Stop/pivot:** stop the claim that datatypes encode source-language intent
if only bit patterns are available. Pivot to explicit value-domain refinements;
if expressions grow excessively, normalize the exact formula without approximating it.

**J) Week-one plan:**
- Migrate representative constraints with explicit event bindings.
- Compare the resulting admitted sets, rather than merely testing satisfiability.
- Measure formula size and compilation cost on the largest existing policy.

**K) Risk/confidence:** technical Low; novelty High as a standalone idea;
execution Low; test decisiveness High for the selected semantics.

### 3. Guarded Observations for Relational Hardware Cuts

**A) Title:** Guarded Observations for Relational Hardware Cuts.

**B) Hypothesis:** an explicit observation interface can admit unobservable-state
removal while rejecting transformations that change completion, readiness, or
other declared control behavior.

**C) Novelty claim:** new relative to an instruction-only contract or a test
that asks whether every removed signal is constant. The mechanism links a
candidate replacement to a relational preservation obligation over observable
validity and guarded values. Relational verification is established; the
specialization interface and backend connection must carry the contribution.

**D) Why it matters:** the quotient state in a remainder-only divider need not be
constant. The right question is whether changing it affects declared behavior
under the admitted usage domain, including timing and backpressure.

**E) Differentiator:** compare against constant-only pruning, ordinary RTL
equivalence, and SecVerilog's relational perspective. Observation restriction
requires justification; it cannot be chosen after a failed proof to hide a bug.

**F) Assumptions:**
- The observation boundary covers all behavior required by the environment.
- Both models share inputs and have related initial/reset states.
- Internal assumptions are grounded in the reference model, avoiding circular restrictions on the candidate.

**G) How this dies:**
- Ignoring event validity permits a design that never produces a result.
- A cut changes an omitted stall or readiness signal.
- The proof assumes the very invariant the cut is supposed to establish.

**H) Cheapest decisive test:** use minimal traces and a small paired RTL model.
Require invalid-cycle payload changes to pass, validity suppression to fail,
and an altered completion cycle to fail. Two to four hours, Python and Yosys.
**Observed:** trace comparison handles validity and invalid payloads correctly.
**Pending:** automated generation and discharge of the paired RTL proof from a
CRISP candidate. The compiler currently records an unproved obligation only.

**I) Stop/pivot:** stop any claim that type checking authorizes erasure. Pivot
to explicit local relational lemmas if full-core proof does not scale.

**J) Week-one plan:**
- Connect one candidate to a versioned reference/candidate RTL pair.
- Generate the common-input observation miter and retain reset semantics.
- Include a deliberately unsafe cut and require a counterexample.

**K) Risk/confidence:** technical Medium; novelty Medium; execution Medium;
test decisiveness High for the local example, Low for whole-core completeness.

### 4. Bounded Protocol Contracts with Explicit Monitor Semantics

**A) Title:** Bounded Protocol Contracts with Explicit Monitor Semantics.

**B) Hypothesis:** explicit edge/reset semantics can replace ambiguous latency
parameters for the implemented bounded-response fragment without losing requests.

**C) Novelty claim:** new relative to CRISP's detached latency parameters. The
mechanism compiles a declared trace property to a monitor with an explainable
history invariant. Temporal monitoring is established; a new syntax alone is
not a research contribution.

**D) Why it matters:** counting only enabled events instead of clock edges or
handling reset inconsistently changes the admitted environment. A minimal
well-defined fragment is preferable to an unsupported claim of complete
cache-protocol modeling.

**E) Differentiator:** compare against handwritten SVA, existing timing YAML,
and Filament. Dynamic bounded response is distinct from static scheduling, but
is also weaker than transaction-aware protocol typing.

**F) Assumptions:**
- A two-state, edge-sampled model is appropriate.
- Reset cancels pending obligations under the chosen interface contract.
- The user accepts that one response may satisfy multiple requests.

**G) How this dies:**
- The application needs transaction matching or multiple distinct responses.
- Event gaps accidentally pause the deadline.
- A finite trace is reported as successful despite an open deadline.

**H) Cheapest decisive test:** minimal simulation of all 256 four-cycle
request/response traces for four intervals, plus reset and event gaps. Compare
generated HDL with direct trace semantics, requiring zero mismatches. Under an
hour with Icarus. **Observed:** all 4,128 sampled checks agree; pending deadlines
are reported. This finite test is not a proof of arbitrary protocol correctness.

**I) Stop/pivot:** stop calling bounded response a transaction protocol. Pivot
to explicit IDs, matching rules, or occupancy invariants only if a real cut needs them.

**J) Week-one plan:**
- Map one existing latency assumption to its actual handshake definition.
- Decide whether many-to-one responses are permitted.
- Add a protocol primitive only with a concrete counterexample and lowering test.

**K) Risk/confidence:** technical Low for this fragment; novelty High;
execution Low; test decisiveness High for short traces, Medium for general lowering.

### 5. Proof Artifacts for Composable Specialization Modules

**A) Title:** Proof Artifacts for Composable Specialization Modules.

**B) Hypothesis:** evidence tied to exact RTL versions and contracts can make
sequential composition of independently proved cuts auditable without unsound
reuse of stale or circular assumptions.

**C) Novelty claim:** new relative to an unstructured set of local pruning
results. The mechanism would attach each transformation's preservation theorem
to its environment, observations, reset model, and actual input/output designs.
Proof composition itself is established; a useful scalable processor workflow
would need its own evidence.

**D) Why it matters:** a proof for the original design does not automatically
apply after another transformation. Hashes help identify artifacts but cannot
establish compatibility or prove semantic equivalence.

**E) Differentiator:** compare against manual sequential-equivalence flows and
generic assume/guarantee reasoning. The relevant artifact is a checked proof
dependency, not just a manifest containing the word "proved."

**F) Assumptions:**
- The backend exports evidence with a precisely specified verification status.
- Intermediate RTL and all contract dependencies are identifiable.
- Compatible observations and environments can be established between cuts.

**G) How this dies:**
- Independently valid cuts rely on incompatible assumptions.
- A solver timeout or bounded pass is promoted to an unbounded theorem.
- A changed library or RTL model leaves stale proof status attached.

**H) Cheapest decisive test:** construct two tiny cuts and mutate one dependency.
Require stale evidence rejection and a failed composition when the second cut
breaks the first contract. One day, Python and a local proof engine. **Observed:**
library changes invalidate the source fingerprint and all candidates remain
explicitly unproved. **Pending:** a proof-import checker and the composition test.

**I) Stop/pivot:** do not claim proof-carrying specialization for this revision.
Pivot to transparent provenance until there is a checked proof protocol and a
negative composition example.

**J) Week-one plan:**
- Specify the minimal certificate interface and proof-status taxonomy.
- Bind one local proof to both RTL hashes and its exact assumptions.
- Add stale-artifact and invalid-composition tests before reusing proofs.

**K) Risk/confidence:** technical High; novelty Medium; execution High;
test decisiveness Medium. Defer from the current paper's completed contributions.

## 3) GO / NO-GO SUMMARY

**Pursue:** idea 1, supported by idea 2, is the strongest completed language
revision. Idea 3 is the highest-value next integration step because it connects
the language to the paper's actual pruning mechanism. The manuscript now states
the relational proof target and keeps its discharge separate from typing.

**Kill for this revision:** arbitrary dependent/affine typing added for novelty
alone, and a claim of automatically composable proof-carrying cuts. Neither is
necessary to improve the existing results, and neither is established by the
implemented prototype. Retain idea 4 as a precisely scoped library capability.

**Most useful missing evidence:** one actual core adapter and one generated,
inductively discharged relational cut obligation, with an unsafe-cut negative
control and all relevant interface observations. This is more valuable than
another PPA sweep for demonstrating the language-to-specialization connection.

**Recommended contribution sentence:**

> CRISP organizes instruction policies, reusable value predicates, and sampled
> execution assumptions into typed specialization contracts with explicit
> observation interfaces and hardware-preservation obligations.

This sentence is deliberately narrower than claiming a novel general type
theory or automatic correctness of every pruning result. Existing PPA evidence
supports the pruning framework; separate language checks support the new front end.

## 4) VERIFICATION PASS

| Idea | Falsifiable target and named baseline | Test within one day | Stop/pivot rule | Evidence status |
|---|---|---|---|---|
| 1 | Reject explicit scope mismatch; legacy CRISP/Filament | Two negative cases and four accepted examples | No automatic scope-certification claim | Front-end checks observed; adapter semantics pending |
| 2 | Exact formulas and extension reuse; legacy CRISP/Liquid/Rosette | SMT/HDL comparisons and library/signature additions | Use value domains; avoid approximate formulas | Selected checks observed; full migration/cost pending |
| 3 | Preserve guarded observations; constant pruning/RTL equivalence | Safe/unsafe trace and paired-model tests | Type checking never authorizes a cut | Trace checks observed; generated RTL proof pending |
| 4 | Correct response-window monitoring; timing YAML/SVA | Exhaustive short traces and reset cases | No complete-protocol claim | Short-trace checks observed; broader protocol pending |
| 5 | Reject stale/incompatible proof reuse; manual proof flow | Two cuts and a mutated dependency | No proof-carrying claim yet | Fingerprints observed; proof composition pending |

All five cards have a falsifiable hypothesis, named comparisons, a bounded test,
a measurable acceptance condition, and explicit stop/pivot criteria. Ideas 3
and 5 are not reported as implemented proof systems. See [the language
manual](CRISP3.md), [validation record](VALIDATION.md), and [manuscript
review](REVIEW.md) for the concrete artifacts and remaining limits.
