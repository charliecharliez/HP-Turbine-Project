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
| Meanline thermodynamic and frame invariants | Check gas-property consistency, stator total temperature and continuity, reaction and efficiency pressure relations, rotor rothalpy, and Euler/thermodynamic work at unequal radii. | `cp = gamma R / (gamma - 1)`; `I = h + W²/2 - U²/2`; `work = U2 Ctheta2 - U3 Ctheta3`; project [design conventions](./DESIGN_CONVENTIONS.md) and [ASME rothalpy reference](https://asmedigitalcollection.asme.org/GT/proceedings/GT1992/78934/V001T01A078/238713). |
| Radial continuity | Recover a uniform axial velocity that passes a specified mass flow through a simple annulus with zero swirl. | `mdot = 2 pi rho integral(Cx r dr)`; the zero-swirl limit has constant `Cx`. |
| Radial-equilibrium pressure and rotor work | Check spanwise pressure and entropy under uniform entropy and verify rotor-exit profiles account for radial Euler-work variation. | CoolProp entropy is invariant across a supported vortex-law station; checked for free-vortex, exponential, and first-power laws. |
| Vortex-law limits and meanline anchors | Check the derivative of the axial-velocity shape for supported exponents, free-vortex meanline state matching, and representative flow rates integrated with spanwise ideal-gas density. | Finite-difference radial-equilibrium relation; hub/midspan/tip flow states; integrated mass-flow residual below 0.1%. Continuity calibration itself uses meanline density. |
| Span mapping limits | Map span fractions zero and one to hub and tip radii. | The linear hub-to-tip span coordinate in the radial-equilibrium model. |
| Run-record validity | Reject a success status without a converged result; serialize success and failure records as JSON. | A05 run-status contract; failures retain an error and do not expose a successful result. |
| Rotor hardware limits | Check inlet and exit mean blade speeds and rotor-exit AN², including equality, violations, and the current default. | `U <= 500 m/s`; `A_exit * N_rpm² <= 3.2 x 10^7 m² rpm²` per rotor. |
| Existing foundation regressions | Load each KO chart and check an interpolation node; verify CoolProp pressure inputs are in Pa. | Digitized chart files and the pressure-unit conversion in the project boundary conditions. |
| Temperature-dependent air properties | Check CoolProp-derived inlet, isentropic, stator-exit, rotor-exit, and stagnation states; entropy closure; `cp = gamma R / (gamma - 1)` at local states; row-to-row property variation; and pressure inputs in Pa. | CoolProp `Air` properties and enthalpy/entropy state relations. |
| Kacker–Okapuu loss components | Check profile charts against Aungier correlation points; Fig. 6 hub-Mach interpolation and shock pressure ratio; secondary loading; Reynolds, compressibility and supersonic corrections; Fig. 14 trailing-edge conversion and limits; and the separate unshrouded-rotor tip-efficiency debit. | Kacker and Okapuu (1982) equations and Figs. 1, 2, 6, 14; the [TurboFlow Kacker–Okapuu description](https://turbo-sim.github.io/turboflow/source/model_description/loss_models/kacker_okapuu_1982.html) and [implementation](https://turbo-sim.github.io/turboflow/_modules/turboflow/axial_turbine/loss_model_kacker_okapuu.html) as implementation cross-checks only. |

These are focused regression checks, not a validation of the complete turbine physics. The meanline identities use variable CoolProp air states for enthalpy, entropy, density, viscosity, `cp`, and `gamma` under steady-flow assumptions. Radial-equilibrium checks establish internal consistency for representative vortex laws, but continuity retains the meanline-density calibration and no external benchmark was available. Cooling, CFD, and AMBRA models still require their roadmap-specific audits and independent comparisons before their outputs can be treated as final design evidence.

The hub-to-mean Mach factor now uses linear interpolation of the Fig. 6 digitized values and reports when the hub-to-tip ratio is outside 0.5–1.0; interpolation clamps to the nearest tabulated endpoint in that case. The Fig. 14 pressure-loss conversion retains the Mach-dependent relation from the original method rather than matching TurboFlow's simpler kinetic-energy approximation. At 40 degrees, local Fig. 1/2 profile values differ from the TurboFlow-exposed Aungier correlations by at most about 6.7%/3.9% across the checked curves; at `t_TE/o = 0.2`, local Fig. 14 values are 0.04478/0.02286 versus TurboFlow's coarser 0.045/0.025 values. These are implementation and digitization comparisons, not experimental validation. See [Kacker and Okapuu (1982)](./Kacker,%20Okapuu%20(1982)%20A%20Mean%20Line%20Prediction%20Method%20for%20Axial%20Flow%20Turbine%20Efficiency.pdf).
