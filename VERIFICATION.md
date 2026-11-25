# Verification cases

Run the focused automated suite from the repository root:

```powershell
python -m unittest discover -s tests -v
```

The cases are intentionally small and do not depend on selected blade dimensions.

| Case | Check | Reference or expected relation |
|---|---|---|
| Boundary-condition conversion | Convert the project's bar-valued inlet and coolant pressures into shared SI values. | `1 bar = 100,000 Pa`; project values 20 bar and 21 bar become 2,000,000 Pa and 2,100,000 Pa. |
| One/two-stage model validation | Accept consecutive stage indices with work fractions summing to one; reject invalid stage ordering and incomplete work split. | A01/A02 requirements and the `DesignInputs` contract. |
| Velocity-triangle closure | Check that the absolute and relative speed magnitudes satisfy the blade-speed relation. | `C² - W² = 2 U C_theta - U²`, from `W_theta = C_theta - U`. |
| Radial continuity | Recover a uniform axial velocity that passes a specified mass flow through a simple annulus with zero swirl. | `mdot = 2 pi rho integral(Cx r dr)`; the zero-swirl limit has constant `Cx`. |
| Span mapping limits | Map span fractions zero and one to hub and tip radii. | The linear hub-to-tip span coordinate in the radial-equilibrium model. |
| Run-record validity | Reject a success status without a converged result; serialize success and failure records as JSON. | A05 run-status contract; failures retain an error and do not expose a successful result. |
| Rotor hardware limits | Check inlet and exit mean blade speeds and rotor-exit AN², including equality, violations, and the current default. | `U <= 500 m/s`; `A_exit * N_rpm² <= 3.2 x 10^7 m² rpm²` per rotor. |
| Existing foundation regressions | Load each KO chart and check an interpolation node; verify CoolProp pressure inputs are in Pa. | Digitized chart files and the pressure-unit conversion in the project boundary conditions. |

These are foundation checks, not a validation of the complete turbine physics. Meanline, loss, cooling, radial-equilibrium, CFD, and AMBRA models still require their roadmap-specific audits and independent comparisons before their outputs can be treated as final design evidence.
