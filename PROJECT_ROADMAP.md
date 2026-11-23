# Turbine design project roadmap

## Purpose and use

Create a modular, iterative pipeline for a cooled, transonic high-pressure turbine, from design inputs through computational verification and a proposed AMBRA test. This roadmap breaks the remaining work into recognizable topics for separate chats; it does not prescribe the detailed implementation of each topic.

- In a future chat, name the task ID and title, review its prerequisites and discuss the approach before implementing it.
- The task order is a suggested dependency order, not a requirement to finish every item in a phase before starting another. Revisit upstream decisions when later evidence requires it.
- Each task's **Done when** describes its intended deliverable or check, not a claim that it already exists.
- The authoritative requirements are in [Project Instructions Email.pdf](./Project%20Instructions%20Email.pdf). Resolve any conflict in favor of that document.

## Current starting point

The repository has a single-stage meanline prototype and velocity-triangle plotting, a radial-equilibrium module, loss correlations, and parametric airfoil geometry with an interactive visualizer. These are not integrated into a validated cooled-design pipeline. In particular:

- The loss module now loads charts from `src/Losses/ko_charts`; focused tests confirm all four charts load and representative interpolation works.
- The gas-property code now converts inlet pressure from bar to Pa before calling CoolProp, with focused regression coverage. The default blade speeds still exceed the specified 500 m/s limit.
- The boundary-condition AN2 constant now matches the specified maximum rotor-exit AN2 of **3.2 x 10^7 m^2 rpm^2**. Enforcing the limit for each rotor and validating passing and failing cases remain outstanding.
- The instructions additionally require metal-temperature-limited cooling, a stage-count/flow-coefficient/reaction study, and an AMBRA test proposal. These are not yet a complete workflow.
- Part A now has a requirements matrix, shared design conventions and validated data models, engineering-invariant tests, and a single-stage meanline runner with JSON run records. The runner reports solver success only; it does not claim that a design meets hardware or cooling constraints.
- Treat numerical output from the current prototype as preliminary; it must be rechecked after corrections and validation.

## A. Requirements and foundations

- **A01 - Translate the project instructions into a requirements checklist.** Record boundary conditions, constraints, mandatory outputs, optional work and acceptance evidence. **Status:** complete. **Done when:** every required report item maps to one or more roadmap tasks and a verification method.
- **A02 - Define stations, frames, sign conventions and units.** Agree on stage/row station numbering, velocity-angle signs, absolute versus relative quantities, pressure units, coolant-flow bases and efficiency definitions. **Status:** complete. **Done when:** the conventions are explicit and usable across modules.
- **A03 - Design shared inputs and results.** Define modular data structures for boundary conditions, stage and row inputs, flow states, geometry, losses, coolant and results; support one or two stages without hard-coding the selected design. **Status:** complete. **Done when:** interfaces can carry the information needed by the downstream tasks without hiding units or assumptions.
- **A04 - Establish verification cases and test workflow.** Choose small, physically interpretable cases and references for conservation, unit conversion, limiting behavior and cross-module checks. **Status:** complete. **Done when:** focused automated tests can be run reproducibly, without relying on final hardware dimensions.
- **A05 - Make runs reproducible and errors visible.** Define how to run modules and record inputs, assumptions, warnings, convergence status and results; reject invalid or infeasible designs explicitly. **Status:** runner and run-record foundation complete; explicit design-feasibility rejection remains open pending constraint enforcement in B02 and relevant downstream tasks. **Done when:** another user can repeat a run and distinguish failure from a valid result.

## B. Repair and validate existing foundations

- **B01 - Fix gas-property pressure units.** Correct the bar-to-Pa conversion and verify CoolProp receives the intended inlet and exit pressures. **Status:** complete; focused regression tests verify the pressure values and a valid property result. **Done when:** a focused regression check confirms the conversion and CoolProp returns valid recalculated properties.
- **B02 - Correct and enforce hardware limits.** The AN2 constant now matches the instructions; enforce each rotor's AN2 and the mean blade-speed limit with explicit units. **Done when:** passing and failing cases are checked, including the present default.
- **B03 - Restore Kacker-Okapuu chart loading.** Point the loss module at the actual digitized charts and verify all required data load from a normal project invocation. **Status:** complete; focused regression tests load all four charts and check representative interpolation. **Done when:** the module imports and representative chart interpolation works.
- **B04 - Audit the existing meanline physics.** Check thermodynamic states, relative-frame quantities, rothalpy, Euler work, continuity, angle signs and assumptions against references. **Done when:** identified issues are resolved or explicitly documented and independent checks pass.
- **B05 - Audit existing radial-equilibrium physics.** Check governing assumptions, streamline mapping, pressure integration, mass conservation, meanline matching and vortex-law limits. **Done when:** representative cases and limitations are verified.
- **B06 - Audit Kacker-Okapuu implementation.** Check digitized charts, coefficient definitions, corrections, applicable ranges and tip treatment against the original method and an independent reference. **Done when:** individual loss components and warnings have defensible reference checks.

## C. Gas properties and cooling

- **C01 - Model temperature-dependent air properties.** Evaluate cp and gamma as temperature changes, using an appropriate source; document the approximation used in each module and compare it with any constant-property assumption. **Done when:** inlet, row and exit properties are reported and verified over the design range.
- **C02 - Define coolant and main-flow bookkeeping.** Decide where each row receives coolant and track coolant mass, temperature, pressure and downstream mixed flow. **Done when:** stage and row mass/energy balances and coolant percentages relative to 30 kg/s are unambiguous.
- **C03 - Select and justify a cooling correlation.** Choose a literature correlation and define its applicability, required geometry inputs and uncertainty. **Done when:** a reproducible correlation and reference are documented.
- **C04 - Calculate required coolant per row.** Use cooling effectiveness and the 1250 K metal-temperature limit; use rotor-relative total gas temperature for rotors. **Done when:** each row's coolant requirement and limiting metal temperature can be evaluated and checked.
- **C05 - Model coolant mixing and its loss.** Determine mixed states and mixing losses using consistent mass, energy and pressure-loss definitions. **Done when:** zero-coolant and conservation checks pass and mixing effects are reported.
- **C06 - Quantify the cooling penalty.** Compare cooled and uncooled work, efficiencies and total coolant demand; account for coolant that bypasses the combustor. **Done when:** the impact and assumptions are visible for each design candidate.

## D. Meanline and stage configuration

- **D01 - Define design variables and constraints.** Specify speed, stage count, work split, flow and loading coefficients, reaction, exit swirl/Mach, radii and constraints with feasible ranges. **Done when:** candidate designs are explicit and physically invalid inputs are rejected.
- **D02 - Complete the single-stage meanline.** Solve the stator and rotor states, work, annulus and velocity triangles using consistent gas properties and loss inputs. **Done when:** continuity, Euler work and energy balances close and required Mach numbers are reported.
- **D03 - Add a two-stage meanline.** Connect two stages with consistent intermediate states, work split, annulus and coolant bookkeeping. **Done when:** both stages and all four rows can be evaluated without assuming they are identical.
- **D04 - Couple meanline, losses and cooling iteratively.** Feed row geometry and flow into losses and cooling, update the states and repeat to defined tolerances. **Done when:** convergence and non-convergence are explicit and converged mass, energy and loss residuals are checked.
- **D05 - Compute overall performance.** Calculate specific work, total-to-total and total-to-static efficiency consistently for cooled and uncooled designs, including exit kinetic-energy and swirl penalties. **Done when:** component and overall balances agree with the reported definitions.
- **D06 - Report meanline geometry and triangles.** Produce per-stage annulus dimensions, velocity triangles and vane-exit absolute and rotor-inlet/exit relative Mach numbers. **Done when:** all required stations are identifiable with units and reference frames.
- **D07 - Compare one- and two-stage designs.** Compare feasible candidates on efficiency, work, loading, cooling, constraints and qualitative weight/length trade-offs. **Done when:** the chosen stage count and work split are justified with recorded evidence.

## E. Losses, transonic flow and parametric study

- **E01 - Connect row geometry to the loss model.** Replace placeholder airfoil parameters with row-specific geometry, angles, Reynolds numbers and Mach numbers. **Done when:** each vane and rotor receives traceable geometry and flow inputs.
- **E02 - Produce row-by-row loss breakdowns.** Report profile, transonic/compressibility and shock corrections, secondary, trailing-edge and tip-clearance effects, plus coolant mixing. **Done when:** the loss and efficiency accounting avoids omissions and double counting.
- **E03 - Assess transonic and shock behavior.** Identify where each row becomes transonic, characterize the trailing-edge shock system where applicable, and quantify modeled shock-loss effects and limitations. **Done when:** Mach and shock evidence is available at the required span locations.
- **E04 - Build the parametric study.** Vary at least flow coefficient, degree of reaction and stage count; record efficiency, coolant flow, feasibility and other relevant outputs. **Done when:** the study can be rerun and exposes failures rather than silently dropping candidates.
- **E05 - Select the final design from the study.** Compare trade-offs, sensitivity and active constraints; document why the selected feasible candidate is preferred. **Done when:** inputs and evidence for the design decision are reproducible.

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
- **H05 - Compare CFD with meanline predictions.** Compare isentropic Mach distribution, exit flow angle and loss for each case; explain differences, shock effects and limitations. **Done when:** comparable definitions, plots and quantitative discussion are provided.
- **H06 - Consider optional 3D CFD.** Assess whether a full-stage 3D calculation is feasible and valuable; keep it separate from required 2D work. **Done when:** the decision and, if pursued, the scope and evidence are recorded.

## I. AMBRA test proposal

- **I01 - Define rotor inlet targets.** From the selected first rotor's spanwise design, provide hub-to-tip relative total pressure, relative flow angle and relative Mach profiles for the inlet gauze. **Done when:** profiles, conventions and operating point are specified.
- **I02 - Scale the rotor to AMBRA.** Select matching priorities for relative inlet/exit Mach, Reynolds number, hub-to-tip ratio and aspect ratio; check facility ranges. **Done when:** rig geometry and compromises are justified.
- **I03 - Calculate rig operating conditions.** Determine inlet total pressure and temperature, mass flow and exit static pressure; check vacuum, heated-line and other relevant facility limits. **Done when:** the proposed operating point is feasible or incompatibilities are clearly documented.
- **I04 - Explain rig-to-engine differences.** Discuss Coriolis and centrifugal effects, tip clearance, coolant ejection, unsteady vane wakes and hot-gas versus rig-air differences. **Done when:** each limitation's effect on comparison is addressed.
- **I05 - Plan measurements and loss reduction.** Define measurement planes, probes, uncertainty considerations and how blade-row loss will be calculated. **Done when:** a practicable comparison plan is documented.

## J. Final integration and delivery

- **J01 - Integrate the complete workflow.** Connect inputs, meanline, row losses, cooling, radial equilibrium, geometry and study/report generation without hiding intermediate results. **Done when:** one- and two-stage candidates can be evaluated through the applicable pipeline.
- **J02 - Verify the integrated design.** Run conservation and constraint checks, regression cases and independent reference comparisons; inspect warnings and model applicability. **Done when:** the final design's results are traceable and outstanding limitations are explicit.
- **J03 - Assemble engineering figures and tables.** Collect design choices, geometry, triangles, loss/cooling breakdowns, efficiencies, spanwise results, CFD evidence and AMBRA proposal. **Done when:** every mandatory deliverable is represented with units and provenance.
- **J04 - Prepare the final report and presentation.** Explain stage/vortex choices, assumptions, trade-offs, computational comparison and limitations; disclose AI use as required. **Done when:** the report follows the instructions' page limit and the presentation summarizes the design and evidence.
- **J05 - Package code and reproduce the final results.** Document dependencies and run instructions, verify outputs from a clean setup, and prepare the report and code together for submission. **Done when:** another team member can reproduce the submitted results.

## Key dependencies and decision points

- **A01-A04** are complete; the A05 runner and run-record foundation is in place, but results cannot be treated as hardware-feasible until **B02** and the applicable downstream constraints are enforced. **B01** and **B03** are complete with focused regression checks. Physics audits **B04-B06** should inform implementation rather than being treated as optional cleanup.
- Gas properties and cooling (**C**) feed the coupled meanline (**D04-D05**); geometry and losses (**E01-E02**, **G**) must converge consistently rather than being treated as a one-way pipeline.
- Compare both stage counts (**D07/E04-E05**) before committing. Spanwise and geometry work may require revisiting the candidate comparison.
- CFD and AMBRA depend on a traceable selected design and section geometry; report and packaging depend on the verified results.
- Decisions such as cooling correlation, CFD solver, vortex law, optimization method and test-scaling priorities are intentionally left for their dedicated future chats.
