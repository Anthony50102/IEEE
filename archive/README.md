# archive/

Code, configs, and scripts moved out of the active path during the 2026 refactor
to the framework paper.

This material was relevant to the prior benchmark/critique paper (KS + NS, with
DMD and FNO methods) and to OpInf parameter sweeps that are no longer in scope.
It is preserved here (rather than deleted) so that:

- `git log --follow` still works for files we may need to consult
- code that *might* be useful in future ablations is one `git mv` away
- the active tree (`hw/`, `rom/`, `opinf/`, `disco_lite/`, `eval/`,
  `configs/`) reads cleanly without dead weight

## Contents

| Path | What it is | Why archived |
|------|------------|--------------|
| `dmd/` | BOPDMD pipeline | Method cut from paper |
| `fno/` | FNO pipeline | Method cut from paper |
| `data_ks/`, `data_ns/` | KS, Navier-Stokes data + generators | PDEs cut from paper |
| `analysis_old/` | mani_*, ks_*, dmd_* analysis scripts | Tied to cut methods/PDEs |
| `scripts_old/` | KS/NS/DMD launcher scripts (`pt1_*`, `pt2_*`, `pt3_*`, `mani_*`, `local_dmd_*`, `run_*_ns_*`, `run_fno_ks_*`) | Tied to cut methods/PDEs |
| `configs_old/` | OpInf KS/NS/mani configs and `sweep_configs/` parameter-soup YAMLs | Tied to cut PDEs and old sweep harness |
| `sweep_configs_old/` | Top-level OpInf regularization sweep YAMLs | Old sweep harness; new sweeps will live in `configs/` |

## Policy

Do not edit anything in this directory. If you need to revive a file, `git mv`
it back into the active tree on a feature branch and adapt it. Keep the active
tree small and the framework spine readable.
