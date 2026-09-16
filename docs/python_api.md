# Python API {#python_api}

This page indexes the Python interfaces for the subsystems currently included in
the apoCHARMM documentation build.

Public objects are available from the top-level package:

```{.py}
import apocharmm as apo
```

Select a Python object below to open its class and method reference. Select the
corresponding subsystem page for cross-language behavior, ownership, lifetime,
units, errors, limitations, and implementation details.

## Errors and Enumerations

- [apocharmm.ApoCharmmError](@ref python_apocharmm_error_class) reports nonzero
  native status results as an owned Python exception. See the
  [ApoCharmmError subsystem](@ref apocharmm_error).
- [apocharmm.DistanceRestraintCondition](@ref python_distance_restraint_condition)
  selects no gate, exact `D >= 0` activation, or exact `D <= 0` activation for
  one distance-restraint term. See the
  [DistanceRestraintForce subsystem](@ref distance_restraint_force).
- [apocharmm.PeriodicBoundaryCondition](@ref python_periodic_boundary_condition)
  selects the periodic boundary model used by contexts and force managers. See
  the [CharmmContext subsystem](@ref charmm_context) and
  [ForceManager subsystem](@ref force_manager).
- [apocharmm.CrystalType](@ref python_crystal_type) selects the supported
  crystal symmetry and Langevin-piston degree-of-freedom layout. See the
  [CUDA Integrators subsystem](@ref cuda_integrators).
- [apocharmm.VdwType](@ref python_vdw_type) selects the native van der Waals
  model. See the [ForceManager subsystem](@ref force_manager).

The top-level package also exports the following status constants:

```text
APO_STATUS_OK
APO_STATUS_INVALID_ARGUMENT
APO_STATUS_RUNTIME_ERROR
APO_STATUS_CUDA_ERROR
APO_STATUS_NOT_INITIALIZED
APO_STATUS_NOT_IMPLEMENTED
```

Their meanings and Python exception mapping are documented in the
[ApoCharmmError subsystem](@ref apocharmm_error).

## Molecular Input and Topology

- [apocharmm.CharmmPsf](@ref python_charmm_psf) reads and owns molecular
  topology. See the [CharmmPSF subsystem](@ref charmm_psf).
- [apocharmm.CharmmParameters](@ref python_charmm_parameters) reads and owns
  CHARMM force-field parameters. See the
  [CharmmParameters subsystem](@ref charmm_parameters).
- [apocharmm.CharmmCrd](@ref python_charmm_crd) reads CHARMM coordinate files.
  See the [Coordinates subsystem](@ref coordinates).

## Simulation State and Force Configuration

- [apocharmm.CharmmContext](@ref python_charmm_context) owns mutable molecular
  state and coordinates force evaluation. See the
  [CharmmContext subsystem](@ref charmm_context).
- [apocharmm.DistanceRestraintForce](@ref python_distance_restraint_force)
  configures CHARMM RESDistance/RESD terms made from one or more weighted
  `AtomReference` endpoint pairs. Raw integer endpoint pairs are not accepted.
  See the [DistanceRestraintForce subsystem](@ref distance_restraint_force).
- [apocharmm.ForceManager](@ref python_force_manager) configures and owns native
  force backends. See the [ForceManager subsystem](@ref force_manager).
- [apocharmm.HarmonicCenterOfMassRestraintForce](@ref python_harmonic_center_of_mass_restraint_force)
  configures periodic geometric-center or mass-weighted-center restraints. See
  the [HarmonicCenterOfMassRestraintForce subsystem](@ref harmonic_center_of_mass_restraint_force).
- [apocharmm.HarmonicRestraintForce](@ref python_harmonic_restraint_force)
  configures absolute mass-weighted harmonic restraints. See the
  [HarmonicRestraintForce subsystem](@ref harmonic_restraint_force).
  
## CUDA Integrators

- [apocharmm.CudaIntegrator](@ref python_cuda_integrator) provides the shared
  lifecycle, context, subscriber, restart, and propagation interface.
- [apocharmm.CudaNoseHooverIntegrator](@ref python_cuda_nose_hoover_integrator)
  provides deterministic Nose-Hoover temperature control (NVT).
- [apocharmm.CudaLangevinThermostatIntegrator](@ref python_cuda_langevin_thermostat_integrator)
  provides stochastic Langevin temperature control (NVT).
- [apocharmm.CudaLangevinPistonIntegrator](@ref python_cuda_langevin_piston_integrator)
  provides pressure control with optional Nose-Hoover temperature control
  (NPH/NPT).
  
See the [CUDA Integrators subsystem](@ref cuda_integrators) for construction,
call order, state, ownership, units, restart behavior, and cross-language
errors.

## Atom Selection

- [apocharmm.AtomReference](@ref python_atom_reference) owns one validated,
  topology-aware, zero-based atom identity. Native PSF object identity defines
  topology identity, and equality also requires an equal atom index.
- [apocharmm.AtomSelection](@ref python_atom_selection) owns a set-valued bitset
  result containing zero or more atom indices.
- [apocharmm.AtomSelector](@ref python_atom_selector) evaluates CHARMM-style
  expressions. `select()` returns `AtomSelection`; `selectAtom()` requires
  exactly one match and returns `AtomReference`.

See the [Atom Selection and Atom References subsystem](@ref atom_selection) for
construction, expression syntax, pointer-identity topology semantics, shared
native PSF ownership, equality, lifetime, exact-one errors, and cross-language
behavior.

## Output Subscribers

- [apocharmm.Subscriber](@ref python_subscriber) provides the common operations
  inherited by concrete subscriber wrappers.
- [apocharmm.DcdSubscriber](@ref python_dcd_subscriber) writes coordinate
  trajectories in DCD format.
- [apocharmm.RestartSubscriber](@ref python_restart_subscriber) writes
  CHARMM-style restart state.

See the [Subscriber subsystem](@ref subscriber) for scheduling, lifetime,
synchronization, and file-format behavior.

## Documentation Scope

This index includes only Python wrappers whose subsystem pages and wrapper
sources are part of the curated documentation build. Additional public Python
objects will be added when their corresponding subsystem documentation is
included.
