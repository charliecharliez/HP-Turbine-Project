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

These runner checks apply to the current single-stage prototype. Direct calls to `solve_meanline` do not enforce them. Passing the checks does not certify other constraints, cooling, or overall design feasibility. The plotting entry point remains separate:

```powershell
python src/Runner/main.py
```

The turbine runner writes `runs/turbine_report.csv` and `runs/turbine_report.json` on each invocation, replacing the prior files. Both contain stage-qualified station 2 and 3 geometry and velocity data; the JSON nests stations under each stage, while the CSV has one row per station.

Run the verification suite with `python -m unittest discover -s tests -v`. The project dependencies are listed in [requirements.txt](./requirements.txt).
