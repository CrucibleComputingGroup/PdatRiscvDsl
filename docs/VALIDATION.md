# CRISP 3 validation record

Validated on 2026-11-04 using Python 3.13.2, Z3 5.1.0, Yosys 0.33
(`2584903a060`), and Icarus Verilog 12.0. The new command requires Python 3.8+;
older supported versions have not been exercised in this environment.

## Reproduce

```bash
pip install -e '.[dev]'
python3 -m pytest -q tests/test_contracts.py
python3 -m pdat_dsl contract examples/contracts/remainder.crisp \
  --output /tmp/remainder_contract.sv --witnesses
yosys -Q -T -p 'read_verilog -formal -sv /tmp/remainder_contract.sv; prep -top crisp_contract; check -assert'
```

Yosys, Icarus Verilog, and `vvp` must be on PATH. The suite requires these tools;
it does not silently skip HDL checks. Result: **65 passed**.

## What was checked

| Check | Scope and result |
|---|---|
| Malformed specifications | Rejects width mismatch, Boolean/bit-vector confusion, literal overflow, bad slices/extensions, unknown names/fields/profiles, duplicate fields, unsupported host expressions, recursion/free variables, invalid intervals, and invalid candidates |
| Event typing | Rejects explicit retirement-to-execution use and a mixed-event expression |
| Exact policy semantics | Seven independently written SMT equivalences over every 32-bit word: constrained include, re-inclusion, joint exclusion, non-prefix range, wildcard shift field, empty policy, union |
| Reusable refinements | SMT checks of alignment, address region, signed range, overlapping unsigned domains, and full-carrier refinement |
| Concrete core semantics | Exhaustive 4-bit operand pairs for arithmetic, comparisons, and signed/unsigned refinements |
| Generated combinational HDL | Eight Yosys SAT miters for refinements, arithmetic slices, signed comparisons, extension, choice, and constant slicing/extension; one additional miter for cross-field exclusion |
| Generated temporal HDL | Direct trace semantics compared with Icarus output for intervals `[0,0]`, `[0,1]`, `[1,2]`, `[2,2]`; all 256 four-cycle request/response traces plus two reset/event-gap cases per interval, totaling 4,128 sampled comparisons |
| Open deadlines | Finite prefixes report pending obligations, not an unbounded success |
| Observation validity | Suppressing a valid result is a mismatch; differing payloads when both invalid are permitted |
| Assumption/obligation roles | A deliberately false assertion stays a Yosys `$assert`, is not a `$assume`, and does not constrain local feasibility |
| Local feasibility | Detects a contradictory pair of assumptions; explicitly excludes RTL transitions and temporal properties |
| Binding checks | Real Yosys JSON passes for matching signals and rejects missing names and wrong widths |
| Extension | A custom 32-bit instruction signature uses the existing compiler and passes an independent SMT equivalence |
| Complete examples | All four `examples/contracts/*.crisp` programs compile through Yosys; `refinements.crisp` is their definition library |
| Migration sample | All seven existing `v2_*.dsl` ISA fragments elaborate to satisfiable predicates; timing directives are explicitly separated by the test |
| Artifact handling | Library edits change the source fingerprint; the CLI emits a monitor and manifest; unchecked bindings and unproved candidates retain those statuses |
| HDL names | A logical signal named `wire` produces legal, prefixed Verilog ports |

These comparisons include independent golden formulas and a direct temporal
semantics, rather than only snapshots of the compiler's own output. However,
they remain selected tests, not a mechanized proof of the Python compiler.

## What this evidence does not establish

- Full processor equivalence or automatic validity of a candidate cut.
- Correct execution guards or transaction correspondence from name/width checks.
- RTL reachability from a local satisfiable predicate.
- Transaction matching from a bounded-response property.
- Complete compatibility with every historical generator or experiment.
- New PPA results, improved pruning efficiency, or reproduction of the existing PPA runs.
- General validity of arbitrary temporal extensions or HDL X/Z behavior.

The illustrative remainder example observes the result and completion guard.
A real unit adapter must additionally preserve its relevant flow-control
observations. The RSD address-region example uses illustrative bindings and is
not an automatically verified RSD adapter.

## Artifact map

- `pdat_dsl/typed_core.py`: typed core, elaborator, concrete/SMT/HDL meanings.
- `pdat_dsl/typed_isa.py`: exact encoding-set algebra and registry interface.
- `pdat_dsl/contracts.py`: event contracts, monitors, observations, binding audit,
  provenance, and CLI.
- `tests/test_contracts.py`: executable checks described above.
- `docs/CRISP3.md`: language and trust-boundary specification.
- `docs/RESEARCH.md`: literature assessment and ranked follow-up directions.
- `docs/REVIEW.md`: manuscript correctness and novelty audit.

The main manuscript changes live in the separate paper repository's
`src/crisp_language.tex`, with narrow updates to its abstract, introduction,
discussion, and conclusion. Existing numerical results and BibTeX files were
not changed by this revision.
