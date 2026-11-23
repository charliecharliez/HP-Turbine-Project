# Turbine design conventions

These conventions apply to shared inputs, results, tests, and module boundaries. Existing solver variables may retain legacy names internally during migration, but adapters must translate them to these definitions explicitly.

## Stations and rows

Each stage uses local station numbers qualified by the stage index:

| Label | Location |
|---|---|
| `(stage, 1)` | Stage inlet and stator inlet |
| `(stage, 2)` | Stator exit and rotor inlet |
| `(stage, 3)` | Rotor exit and stage exit |

The first-stage turbine inlet total state is `01`, corresponding to `(1, 1)`. For a two-stage design, `(2, 1)` is the state entering the second-stage stator after any interstage coolant mixing. Thus the current single-stage stations 2 and 3 remain recognizable while stage-qualified keys avoid ambiguous numbering in a two-stage result.

## Frames, velocities, and angles

- `C` denotes absolute velocity; `W` denotes rotor-relative velocity; `U = omega * r` denotes blade speed.
- The axial component is positive downstream. The tangential component is positive in the direction of shaft rotation.
- `C_x` is the axial/meridional component, `C_theta` the absolute tangential component, and `W_theta = C_theta - U`. A stationary stator has `U = 0`.
- `alpha = atan2(C_theta, C_x)` is the signed absolute flow angle. `beta = atan2(W_theta, C_x)` is the signed rotor-relative flow angle. Both are in degrees from the positive axial direction; positive angles point in the direction of rotation.
- Keep absolute and rotor-relative total quantities distinct. Use a frame label when a state contains velocity, Mach number, or total properties that depend on the frame.

## Units

Shared models and cross-module interfaces use SI units:

| Quantity | Unit |
|---|---|
| Static/total pressure | Pa |
| Static/total temperature | K |
| Length/radius | m |
| Area | m² |
| Mass flow | kg/s |
| Velocity and blade speed | m/s |
| Angular speed | rad/s |
| Shaft speed input and the AN² constraint | rpm and m²·rpm², respectively |
| Flow/metal angles | degrees |
| Mach number, reaction, flow/loading coefficients, and loss coefficients | dimensionless |

Convert project pressure inputs in bar at the input boundary (`1 bar = 100,000 Pa`). Convert shaft speed from rpm to rad/s before calculating `U`. Names at shared interfaces include unit suffixes where the unit is not self-evident, for example `total_pressure_pa`, `temperature_k`, and `mass_flow_kg_s`. Reports may use a different display unit only when it is labelled and conversion is explicit.

## Coolant bookkeeping

Store coolant mass flow in kg/s for each row and retain the original 30 kg/s main-flow value as the percentage reference. Report each row's coolant percentage as `100 * row_coolant_mass_flow_kg_s / 30`; downstream mixed-stream mass flow includes injected coolant, but the percentage denominator does not change. Track coolant temperature and pressure at injection and mixed-state mass/energy balances explicitly.

## Efficiency definitions

Use turbine work as positive extracted work. For a stage or complete turbine:

- For an uncooled flow, `eta_tt = (h_01 - h_03) / (h_01 - h_03s)`, with the isentropic exit total state at the actual exit total pressure.
- For an uncooled flow, `eta_ts = (h_01 - h_03) / (h_01 - h_3s)`, with the isentropic exit static state at the actual exit static pressure.
- For a cooled flow, calculate actual shaft work from row energy balances that include coolant injection and normalize it by the original main-flow mass: `w_actual = W_dot / mdot_main`. Use `eta_tt,cooled = w_actual / (h_01 - h_03s)` and `eta_ts,cooled = w_actual / (h_01 - h_3s)`, with the isentropic reference states at the actual overall exit total and static pressures, respectively. Report coolant compression/combustor penalties separately from turbine efficiency. Do not mix a main-flow work basis with a mixed-flow enthalpy basis.
- The specified turbine pressure ratio is `P_01 / P_3`: inlet total pressure divided by exit static pressure.

## Compatibility with current modules

The meanline prototype calls its absolute speed `v`, relative speed `w`, stores pressures in bar, and labels the current stations 2 and 3. The velocity-triangle module uses `C` and `W`, with positive tangential direction along blade rotation. Migrations must preserve those results while adapting names, signs, and units at well-defined boundaries; do not silently reinterpret an unsigned legacy angle as a signed angle.
