# Running and recording the prototype

Run the current single-stage meanline calculation without opening a plot:

```powershell
python src/run_meanline.py --output runs/baseline.json
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
python src/run_meanline.py --config config/meanline.json --output runs/candidate.json
```

The JSON record stores a schema version, run ID, the full boundary-condition and meanline input snapshot, assumptions, warnings, convergence status, and unit-labelled results. `succeeded` means the meanline mass-flow residual is within `1e-6 kg/s`; the result also records this tolerance. `non_converged` contains the solver output but returns a nonzero process status. If the solver rejects a case with `ValueError`, the runner writes a `failed` record and re-raises the error. Invalid JSON, unknown fields, and malformed values are rejected before solving.

This runner only records execution of the current single-stage meanline prototype. It explicitly warns that integrated cooling and hardware-limit checks are not part of that run; a successful solver status is not a claim that the design is feasible or validated. The plotting entry point remains separate:

```powershell
python src/main.py
```

Run the verification suite with `python -m unittest discover -s tests -v`. The project dependencies are listed in [requirements.txt](./requirements.txt).
