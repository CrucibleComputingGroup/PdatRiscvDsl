# Frontend compatibility evidence

The historical generator audit compared the working tree with Git commit
`3fe0d81f0d8a2fed326fbd1d4095b1bd9f3b594a`. At each of `PYTHONHASHSEED=0`
and `29`, all 546 generation cases completed, and all 581 artifact files were
byte-identical. Of those files, 435 were nonempty. This is a generation
compatibility result, not processor verification or a new hardware result.

The 78-input corpus consists of 46 PdatDsl sources, 16 ScorrPdat sources, and
16 BoundedDynamicPscorr sources. It includes 74 tracked files and four existing
untracked PdatDsl examples. `corpus.json` freezes their exact text and SHA-256
hashes, including the original repository heads and paths. The original
untracked files are not added to the repository. New frontend/golden pairs are
tested separately; `paired-tests.json` records their validation and source
hashes. The historical 78-source audit does not claim all 78 sources are
accepted by the checked frontend.

The paired suite passes 63 tests: 40 inline CLI comparisons, 10 operand-only
CLI comparisons, four direct module API comparisons, three output-route
rejection checks, four imported-library overwrite checks, one module metadata
check without SVA generation, and one mutation control for the byte comparator.
The complete project test suite also passes: 222 tests, recorded in
`full-suite.json`.

Each source is exercised through the legacy `codegen` command with the default
configuration and four explicit YAML configurations (Ibex, BOOM, Rocket, and
RiscvSingleCycle), plus the separate `generate_module_dtype_assertions` API
with a fixed module binding at 32 and 64 bits. The command cases include
primary output and timing sidecars. The API cases include empty output where
there are no applicable operand rules. The audit does not invoke full
hierarchical synthesis or the dynamic-scheduling experiments' separate formal
RTL generators.

`seed0-report.json` and `seed29-report.json` record the live-workspace audits,
input hashes, Python/dependency versions, 16 unchanged legacy implementation
and configuration hashes, and known timing-template issues. The original
input files did not change during either audit. The matching `*-artifacts.tsv`
files contain both sides' SHA-256 hashes for every output, including empty
outputs. The audit compared actual bytes; it did not just compare hashes.
Temporary absolute paths in these records identify the original run and are
not required for reproduction.

The baseline itself is not byte-reproducible across arbitrary Python hash
seeds: 105 CLI artifacts from 21 inputs differ between seeds 0 and 29 because
instruction extensions are traversed through a Python set. The candidate
matches the baseline at each fixed seed. `seed-comparison.json` inventories
these differences; the compatibility check does not normalize or reorder SVA.

## Reproduce

From this repository, with the recorded baseline commit available in Git and
the project dependencies installed, use fresh output directories:

```sh
python3 scripts/check_sva_compatibility.py --hash-seed 0 --corpus-snapshot review/frontend_compatibility/corpus.json --output /tmp/crisp-audit-seed0
python3 scripts/check_sva_compatibility.py --hash-seed 29 --corpus-snapshot review/frontend_compatibility/corpus.json --output /tmp/crisp-audit-seed29
python3 -m pytest -q tests/test_frontend_compatibility.py
```

The snapshot option checks each saved source against its recorded hash and
does not require the sibling repositories or original untracked files. Omit
it to discover the current live corpus and check that those source files are
unchanged during the audit. A live checkout with different available inputs
can have different counts.

## Known inherited behavior

All 35 timing sidecars in this historical corpus contain unexpanded template
names such as `{clk}` and `{rst_n}`. These braces can parse as SystemVerilog
concatenations, but the identifiers do not match the configured Ibex bindings.
`baseline_timing_wrapper.sv` embeds one frozen baseline sidecar with its seven
configured input signals. Icarus fails elaboration with an unresolved `rst_n`;
the diagnostic and command are recorded in `timing-diagnostic.json` and
`iverilog.log`. Reproduce that expected failure with:

```sh
iverilog -g2012 -s timing_check -o /tmp/crisp-timing-check.vvp review/frontend_compatibility/baseline_timing_wrapper.sv
```

The existing `scripts/inject_core_timing.py` in both experiment repositories
inserts the sidecar verbatim. The live ScorrPdat `synth_core_simplified.sh`
requires `--timing` and defaults to skipping timing injection; the
BoundedDynamicPscorr version has that injection block commented out. Both
`synth_ibex_with_constraints.sh` versions inject a sidecar if it exists. The
audit did not change those scripts or repair the inherited generator.

The legacy inline `codegen` path also treats a dtype-bearing instruction rule
as an instruction exclusion. Standard experiment shell scripts call that path
without filtering. For example, historical case 140 (`rv32im_narrow8.dsl`,
default configuration) excludes ADD, MUL, and the other dtype-constrained
instructions, in addition to emitting conditional operand assumptions. In
contrast, `hierarchical-synth` emits operand restrictions through the module
API and explicitly filters dtype-bearing rules before generating the core's
ISA/PC/register assumptions (`hierarchical_synth.py`, full-core phase). The
separate module API has no instruction-ban side effect. The audit preserves
both legacy paths as they are; equal bytes alone do not establish that a
source specification has its intended meaning.

The checked frontend's normal inline route rejects these inherited timing
and operand side effects. The explicit `--legacy-parity` option is used only
to check preservation of the old inline artifacts in the affected paired
tests. The operand-only frontend example uses `--module-dtype compat_ex` and
is compared to the unchanged baseline module API using the same core signal
configuration, module name, injection path, and data width. That route rejects
mixed ISA/timing/operand specifications so it cannot silently discard rules.
