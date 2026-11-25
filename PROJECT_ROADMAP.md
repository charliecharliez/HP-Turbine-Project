# Turbine design project roadmap

## Purpose and use

Create a modular, iterative turbine-design program that evaluates and optimizes feasible transonic high-pressure turbine candidates. The active workflow runs from boundary conditions through meanline, row losses, airfoil geometry, selected-candidate 2D CFD, and optimization. Cooling and the AMBRA test proposal remain documented as deferred work, not prerequisites for the current optimizer end goal.

- In a future chat, name the task ID and title, review its prerequisites and discuss the approach before implementing it.
- The task order is a suggested dependency order, not a requirement to finish every item in a phase before starting another. Revisit upstream decisions when later evidence requires it.
- Each task's **Done when** describes its intended deliverable or check, not a claim that it already exists.
- The authoritative requirements are in [Project Instructions Email.pdf](./Project%20Instructions%20Email.pdf). Resolve any conflict in favor of that document.

## Active end goal

The program should take validated boundary conditions and candidate design inputs, then repeatedly evaluate and refine the design:

1. Establish the operating boundary conditions and applicable feasibility constraints.
2. Solve the meanline for the candidate design.
3. Calculate blade-row losses and feed them back into the meanline until the coupled solution converges or is explicitly reported as non-converged.
4. Generate airfoil geometry from the current flow/design state and update it as the design is refined.
5. Use 2D CFD on selected, screened candidates to check and improve the airfoil and meanline predictions, then re-evaluate the candidate.
6. Search the feasible design space and report results for three distinct modes: maximize efficiency, maximize power output, or optimize both and return the efficiency–power Pareto frontier.

All active roadmap tasks should build, verify, or integrate a part of this workflow. The candidate evaluator and optimizer must retain inputs, assumptions, convergence status, infeasible/failed candidates, warnings, and results so comparisons are reproducible. Use lower-cost meanline/loss/geometry evaluations to screen candidates; do not require a CFD solve for every optimizer iteration.

Cooling models and the AMBRA test proposal are deferred and are not blockers for this active end goal. Until cooling work is resumed, the program must not claim thermal feasibility against the metal-temperature limit or full project compliance. The 2D CFD design validation remains in scope.

## Current starting point

The repository has a single-stage meanline prototype and velocity-triangle plotting, a radial-equilibrium module, loss correlations, and parametric airfoil geometry with an interactive visualizer. These are not integrated into a validated optimizer workflow. In particular:

- The loss module now loads charts from `src/Losses/ko_charts`; focused tests confirm all four charts load and representative interpolation works.
- The gas-property code now converts inlet pressure from bar to Pa before calling CoolProp, with focused regression coverage. The default blade speeds still exceed the specified 500 m/s limit.
- The meanline runner checks rotor inlet and exit mean blade speeds and rotor-exit AN2, retaining results while rejecting hardware-infeasible runs. The reusable per-rotor check needs integration into any future multi-stage runner.
- The project instructions also require cooling and an AMBRA test proposal; those workstreams are retained below as deferred scope. The stage-count/flow-coefficient/reaction study and optimizer are part of the active end goal.
- Part A now has shared design conventions, validated data models, engineering-invariant tests, and a single-stage meanline runner with JSON run records. Solver success and hardware-limit checks do not certify overall design feasibility.
- Treat numerical output from the current prototype as preliminary; it must be rechecked after corrections and validation.

## A. Requirements and foundations

- **A05 - Make runs reproducible and errors visible.** Define how to run modules and record inputs, assumptions, warnings, convergence status and results; reject invalid or infeasible designs explicitly. **Status:** runner and run-record foundation and current rotor hardware checks complete; overall design-feasibility rejection remains open pending relevant downstream constraints. **Done when:** another user can repeat a run and distinguish failure from a valid result.

## B. Repair and validate existing foundations

- **B04 - Audit the existing meanline physics.** Check thermodynamic states, relative-frame quantities, rothalpy, Euler work, continuity, angle signs and assumptions against references. **Done when:** identified issues are resolved or explicitly documented and independent checks pass.
- **B05 - Audit existing radial-equilibrium physics.** Check governing assumptions, streamline mapping, pressure integration, mass conservation, meanline matching and vortex-law limits. **Done when:** representative cases and limitations are verified.
- **B06 - Audit Kacker-Okapuu implementation.** Check digitized charts, coefficient definitions, corrections, applicable ranges and tip treatment against the original method and an independent reference. **Done when:** individual loss components and warnings have defensible reference checks.

## C. Gas properties and deferred cooling

- **C01 - Model temperature-dependent air properties.** Evaluate cp and gamma as temperature changes, using an appropriate source; document the approximation used in each module and compare it with any constant-property assumption. **Done when:** inlet, row and exit properties are reported and verified over the design range.
- **C02 - Define coolant and main-flow bookkeeping (deferred).** Decide where each row receives coolant and track coolant mass, temperature, pressure and downstream mixed flow. **Done when:** stage and row mass/energy balances and coolant percentages relative to 30 kg/s are unambiguous.
- **C03 - Select and justify a cooling correlation (deferred).** Choose a literature correlation and define its applicability, required geometry inputs and uncertainty. **Done when:** a reproducible correlation and reference are documented.
- **C04 - Calculate required coolant per row (deferred).** Use cooling effectiveness and the 1250 K metal-temperature limit; use rotor-relative total gas temperature for rotors. **Done when:** each row's coolant requirement and limiting metal temperature can be evaluated and checked.
- **C05 - Model coolant mixing and its loss (deferred).** Determine mixed states and mixing losses using consistent mass, energy and pressure-loss definitions. **Done when:** zero-coolant and conservation checks pass and mixing effects are reported.
- **C06 - Quantify the cooling penalty (deferred).** Compare cooled and uncooled work, efficiencies and total coolant demand; account for coolant that bypasses the combustor. **Done when:** the impact and assumptions are visible for each design candidate.

## D. Meanline and stage configuration

- **D01 - Define design variables and constraints.** Specify speed, stage count, work split, flow and loading coefficients, reaction, exit swirl/Mach, radii and constraints with feasible ranges. **Done when:** candidate designs are explicit and physically invalid inputs are rejected.
- **D02 - Complete the single-stage meanline.** Solve the stator and rotor states, work, annulus and velocity triangles using consistent gas properties and loss inputs. **Done when:** continuity, Euler work and energy balances close and required Mach numbers are reported.
- **D03 - Add a two-stage meanline.** Connect two stages with consistent intermediate states and work split. **Done when:** both stages and all four rows can be evaluated without assuming they are identical.
- **D04 - Couple meanline and losses iteratively.** Feed row geometry and flow into the loss model, update meanline states and repeat to defined tolerances. **Done when:** convergence and non-convergence are explicit and converged mass, energy and loss residuals are checked.
- **D05 - Compute overall performance.** Calculate specific work, power output, total-to-total and total-to-static efficiency consistently, including exit kinetic-energy and swirl penalties. **Done when:** component and overall balances agree with the reported definitions.
- **D06 - Report meanline geometry and triangles.** Produce per-stage annulus dimensions, velocity triangles and vane-exit absolute and rotor-inlet/exit relative Mach numbers. **Done when:** all required stations are identifiable with units and reference frames.
- **D07 - Compare one- and two-stage designs.** Compare feasible candidates on efficiency, power, loading, applicable constraints and qualitative weight/length trade-offs. **Done when:** the chosen stage count and work split are justified with recorded evidence.

## E. Losses, transonic flow and optimization

- **E01 - Connect row geometry to the loss model.** Replace placeholder airfoil parameters with row-specific geometry, angles, Reynolds numbers and Mach numbers. **Done when:** each vane and rotor receives traceable geometry and flow inputs.
- **E02 - Produce row-by-row loss breakdowns.** Report profile, transonic/compressibility and shock corrections, secondary, trailing-edge and tip-clearance effects. **Done when:** the loss and efficiency accounting avoids omissions and double counting.
- **E03 - Assess transonic and shock behavior.** Identify where each row becomes transonic, characterize the trailing-edge shock system where applicable, and quantify modeled shock-loss effects and limitations. **Done when:** Mach and shock evidence is available at the required span locations.
- **E04 - Build the design optimizer.** Search at least flow coefficient, degree of reaction and stage count using the coupled candidate evaluator; provide separate efficiency-maximization, power-maximization and combined Pareto-front modes. Screen candidates with lower-cost models and evaluate selected candidates with 2D CFD. **Done when:** each mode is repeatable and records feasible, infeasible, failed and non-converged candidates instead of silently dropping them.
- **E05 - Select the final design from optimization results.** Compare objective trade-offs, Pareto candidates, sensitivity and active constraints; document why the selected feasible candidate is preferred. **Done when:** inputs and evidence for the design decision are reproducible.

## F. Spanwise design

- **F01 - Compare and justify vortex laws.** Evaluate plausible laws against loading, reaction, Mach distribution and feasibility, including the existing free-vortex option. **Done when:** the chosen law and trade-offs are documented.
- **F02 - Solve the selected spanwise flow.** Apply radial equilibrium for the relevant stations and stages with the selected meanline, gas properties and mass-flow assumptions. **Done when:** radial profiles, boundary conditions and consistency checks are available.
- **F03 - Check hub reaction and spanwise constraints.** Evaluate reaction, Mach numbers, work, temperatures and other limiting quantities at hub, midspan and tip. **Done when:** hub reaction and all identified constraint concerns are reported.
- **F04 - Draw spanwise velocity triangles.** Generate correctly signed absolute and relative triangles at hub, midspan and tip for each stage. **Done when:** plots and the underlying numerical table agree.

## G. Airfoil and blade-row geometry

- **G01 - Size blade rows and counts.** Determine pitch-to-chord ratios using Zweifel, chord, pitch, blade count and aspect ratio, with any rounding effects reconciled. **Done when:** each row's dimensions and assumptions are recorded.
- **G02 - Set metal angles and stagger.** Translate spanwise flow angles into inlet/exit metal angles, incidence/deviation assumptions and justified stagger at each section. **Done when:** all rows have hub, midspan and tip definitions.
- **G03 - Define throat and trailing-edge geometry.** Determine throat, edge thickness and related geometric constraints for the selected flow regime. **Done when:** geometry needed for loss prediction and CFD is available and checked.
- **G04 - Construct section profiles.** Use or extend the existing parameterization to create vane and rotor sections at hub, midspan and tip. **Done when:** each section has reproducible coordinates and geometry parameters.
- **G05 - Validate and export geometry.** Check closed/non-self-intersecting profiles, blade spacing, units and continuity; export the required midspan sections for CFD. **Done when:** geometry checks pass and exported coordinates reproduce the intended sections.
- **G06 - Close the geometry-to-loss loop.** Recalculate row losses and the coupled design using actual profile, throat and pitch data, iterating if necessary. **Done when:** reported geometry and performance refer to the same converged design.

## H. Required two-dimensional CFD

- **H01 - Choose a CFD solver and validation approach.** Select LEO or another suitable solver, document its availability and define a blade-to-blade method for transonic rows. **Done when:** solver choice, governing models and limitations are justified.
- **H02 - Define vane and rotor CFD cases.** Set up midspan passages at design conditions, including periodicity and rotor-relative-frame conditions; for two stages cover at least the first stage. **Done when:** case inputs trace back to the selected design.
- **H03 - Generate at least two grids.** Create and document the meshes for both cases, with appropriate near-wall resolution and quality checks. **Done when:** mesh views, quality metrics and a grid-refinement comparison are available.
- **H04 - Run and assess both CFD cases.** Check convergence history, conservation and y+; resolve solver or boundary-condition issues. **Done when:** vane and rotor results have documented convergence and numerical-quality evidence.
- **H05 - Compare CFD with meanline predictions and feed back results.** Compare isentropic Mach distribution, exit flow angle and loss for each case; use documented differences to refine the airfoil and/or meanline model, then re-evaluate the selected candidate. Explain shock effects and limitations. **Done when:** comparable definitions, quantitative comparisons and traceable refinement decisions are provided.
- **H06 - Consider optional 3D CFD (outside the active end goal).** Assess whether a full-stage 3D calculation is feasible and valuable; keep it separate from the active 2D design loop. **Done when:** the decision and, if pursued, the scope and evidence are recorded.

## I. Deferred: AMBRA test proposal

- **I01 - Define rotor inlet targets (deferred).** From the selected first rotor's spanwise design, provide hub-to-tip relative total pressure, relative flow angle and relative Mach profiles for the inlet gauze. **Done when:** profiles, conventions and operating point are specified.
- **I02 - Scale the rotor to AMBRA (deferred).** Select matching priorities for relative inlet/exit Mach, Reynolds number, hub-to-tip ratio and aspect ratio; check facility ranges. **Done when:** rig geometry and compromises are justified.
- **I03 - Calculate rig operating conditions (deferred).** Determine inlet total pressure and temperature, mass flow and exit static pressure; check vacuum, heated-line and other relevant facility limits. **Done when:** the proposed operating point is feasible or incompatibilities are clearly documented.
- **I04 - Explain rig-to-engine differences (deferred).** Discuss Coriolis and centrifugal effects, tip clearance, coolant ejection, unsteady vane wakes and hot-gas versus rig-air differences. **Done when:** each limitation's effect on comparison is addressed.
- **I05 - Plan measurements and loss reduction (deferred).** Define measurement planes, probes, uncertainty considerations and how blade-row loss will be calculated. **Done when:** a practicable comparison plan is documented.

## J. Final integration and delivery

- **J01 - Integrate the optimizer workflow.** Connect boundary conditions, meanline, row losses, radial equilibrium, airfoil geometry, selected-candidate 2D CFD, optimizer objectives and run records without hiding intermediate results. **Done when:** one- and two-stage candidates can be evaluated, refined and compared through the active pipeline.
- **J02 - Verify the integrated design.** Run conservation and constraint checks, regression cases and independent reference comparisons; inspect warnings and model applicability. **Done when:** the final design's results are traceable and outstanding limitations are explicit.
- **J03 - Assemble in-scope engineering figures and tables.** Collect design choices, geometry, triangles, row-loss breakdowns, efficiency/power comparisons, spanwise results and CFD evidence. **Done when:** active-scope outputs are represented with units and provenance.
- **J04 - Prepare the final report and presentation.** Explain stage/vortex choices, assumptions, trade-offs, computational comparison and limitations; disclose AI use as required. **Done when:** the report follows the instructions' page limit and the presentation summarizes the design and evidence.
- **J05 - Package code and reproduce the final results.** Document dependencies and run instructions, verify outputs from a clean setup, and prepare the report and code together for submission. **Done when:** another team member can reproduce the submitted results.

## Key dependencies and decision points

- Requirements, conventions, shared models, engineering-invariant tests, gas-property units, loss-chart loading and the current rotor hardware checks have established the foundations. The A05 runner and run-record foundation is in place, but results cannot be treated as fully feasible until applicable downstream constraints are enforced. Physics audits **B04-B06** should inform implementation.
- Candidate evaluation depends on validated boundary conditions and gas properties (**A05**, **B02**, **B04**, **C01**), then meanline and row losses (**D02-D05**, **E01-E03**) coupled with actual row geometry (**G01-G06**). Spanwise design (**F**) informs hub, midspan and tip constraints and geometry.
- Optimization (**E04-E05**) follows the validated candidate evaluator. It compares stage count and other design variables in the three objective modes; selected candidates proceed to 2D CFD (**H**) and CFD-informed refinement before final comparison.
- Integrated reporting and verification (**J01-J05**) depend on traceable candidate runs and validated model outputs. Cooling (**C02-C06**) and the AMBRA proposal (**I01-I05**) are deferred and are not dependencies of the active optimizer workflow.
- Decisions such as cooling correlation, CFD solver, vortex law, and optimizer implementation details remain for their dedicated tasks.
