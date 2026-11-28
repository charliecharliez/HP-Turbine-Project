# Running and recording the prototype

Run the current single-stage meanline calculation without opening a plot:

```powershell
python src/Runner/run_meanline.py --output runs/baseline.json
```

The optional JSON configuration uses explicit unit-labelled keys; omitted keys retain the current meanline defaults.

```json
{
  "degree_of_reaction": 0.3,
  "shaft_speed_rpm": 13000,
  "radius_2_m": 0.5,
  "radius_3_m": 0.6,
  "stator_exit_angle_deg": 71,
  "rotor_exit_metal_angle_deg": 61,
  "stator_kinetic_loss_fraction": 0.027,
  "rotor_kinetic_loss_fraction": 0.054
}
```

Run a configuration and write a record:

```powershell
python src/Runner/run_meanline.py --config config/meanline.json --output runs/candidate.json
```

The JSON record stores a schema version, run ID, the full boundary-condition and meanline input snapshot, assumptions, warnings, convergence status, and unit-labelled results. `succeeded` means the meanline mass-flow residual is within `1e-6 kg/s`; the result also records this tolerance. `non_converged` contains the solver output, leaves `result.hardware_limits` null, and returns a nonzero process status. If the solver or hardware checks reject invalid values with `ValueError`, the runner writes a `failed` record and re-raises the error. Invalid JSON, unknown fields, and malformed values are rejected before solving.

For converged runs, `result.hardware_limits` reports the stage, measured value, limit, unit, and pass/fail for rotor inlet and exit mean blade speeds and rotor-exit AN². The AN² calculation uses exit annulus area in m² and shaft speed in rpm; values equal to the limits pass. If any check fails, the runner retains the result and `succeeded` solver status, prints `hardware_infeasible`, and returns a nonzero process status. The default configuration exceeds the mean blade-speed limit and is therefore hardware-infeasible.

These runner checks apply to the current single-stage prototype. Direct calls to `solve_meanline` do not enforce them. Passing the checks does not certify other constraints, cooling, or overall design feasibility. The full turbine runner keeps the current one-stage reference design as its no-argument default:

```powershell
python src/Runner/main.py
```

For a comparison, select the stage count at the start of each run in a JSON configuration. `stage_count` must be 1 or 2 and must match the number of entries in `stages`. Omitted stage inputs use the current one-stage reference inputs. Two-stage runs default to an even expansion split; set `expansion_shares` explicitly when comparing a different split.

```json
{
  "stage_count": 1,
  "stages": [
    {
      "degree_of_reaction": 0.27,
      "shaft_speed_rpm": 12790,
      "radius_2_m": 0.3732,
      "radius_3_m": 0.3733,
      "stator_exit_angle_deg": 70,
      "rotor_exit_metal_angle_deg": -70,
      "stator_kinetic_loss_fraction": 0.08,
      "rotor_kinetic_loss_fraction": 0.10
    }
  ]
}
```

Save this as `config/single-stage.json` (create the `config/` directory if needed). Create `config/two-stage.json` with `"stage_count": 2`, two stage-input objects, and an `expansion_shares` array such as `[0.5, 0.5]`. These are starting inputs; check each report's `feasible` field before comparing performance. Run both:

```powershell
python src/Runner/main.py --config config/single-stage.json
python src/Runner/main.py --config config/two-stage.json
```

Optimization objective mode is an explicit run-start choice. Use
`efficiency` for maximum total-to-static efficiency, `power` for maximum
power output, or `pareto` for the efficiency-power Pareto frontier. The
selected mode and stage count are retained in the optimizer run record.

The default output directories are `runs/one-stage/` and `runs/two-stage/`, so the two designs can be inspected together without overwriting each other's files. Rerunning the same stage count replaces that folder's current reports. Pass `--output-directory` to choose another folder.

Each schema-version-2 `turbine_report.json` records overall power, work, efficiencies, stage loading, expansion/work shares, row geometry, convergence, and per-stage/rig constraint results along with station-level geometry and velocity triangles. `turbine_report.csv` has one row per stage-qualified station. Infeasible results are still written and return a nonzero process status. Compare designs using the same inlet conditions and review each run's feasibility before comparing performance. More stages add stator and rotor rows and may affect mass and flow-path length, but the model does not calculate total mass or axial length. The program does not choose between the runs; cooling and metal-temperature feasibility are not assessed.

Run the verification suite with `python -m unittest discover -s tests -v`. The project dependencies are listed in [requirements.txt](./requirements.txt).
