// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  James E. Gonzales II
//
// ENDLICENSE

#pragma once

#include "Subscriber.h"

#include <filesystem>

/**
 * @brief Writes CHARMM's `DYNA` energy blocks during dynamics.
 *
 * Each block has the layout of CHARMM's `PRINTE` (see `apo::WriteEnergyTable`).
 * A run starts with the first `CudaIntegrator::propagate()` call after
 * subscription, after a restart load, or after the run origin was cleared. Like
 * CHARMM at the start of each `DYNA` command, the subscriber then writes the
 * header and a step-0 block. That block evaluates the energies at the starting
 * coordinates, while its kinetic energy and pressure are the latest that the
 * integrator holds; the piston's pressure is zero before its first step. The
 * subscriber writes another block every `getReportFrequency()` steps of the
 * run, as CHARMM does every `NPRINT` steps. The step column counts from the
 * start of the run, like CHARMM's `ISTEP`, and the time column is the
 * completed-step count times the time step, in picoseconds.
 *
 * A block holds the values that the integrator computed in the step just
 * completed. The Langevin thermostat, Nose-Hoover, Langevin piston, and
 * leapfrog integrators evaluate forces at the coordinates a step starts from,
 * so their block `N` shows the potential energy, kinetic energy, and pressure
 * of step `N - 1`, where CHARMM's line `N` shows step `N`. The block `N` of
 * velocity Verlet shows step `N`. The kinetic energy is that of the context
 * velocities, which are on-step velocities except for
 * `CudaLeapFrogIntegrator`.
 *
 * Restraints appear in CHARMM's lines: CONS HARM and RESD under `CONSTR>`,
 * HMCM under `RESTR>`, and MMFP GEO under `MMFP>`. Other subscribed forces are
 * `USER` energy. Like CHARMM's OpenMM and BLaDE dynamics, which start from
 * zeroed energy properties, the subscriber prints zero for the fields that
 * apoCHARMM does not compute: `GRMS`, the high-frequency corrections, the
 * external virial and pressure, and the crystal energies.
 * `VIRI` is the force virial, without a holonomic-constraint part, and
 * `VIRKe` is `-1.5 * VIRI`. Only `CudaLangevinPistonIntegrator` computes the
 * pressure, so only it fills `PRESSI` and the surface tension, and with it the
 * subscriber also writes CHARMM's crystal-parameter lines.
 *
 * @see subscriber
 */
class DynaSubscriber : public Subscriber {
public:
  /**
   * @brief Constructs a subscriber that writes a block every 10 steps.
   *
   * Ten steps is CHARMM's default `NPRINT`.
   *
   * @param[in] filePath Text output path. See the corresponding @ref Subscriber
   * constructor for validation and file-opening behavior.
   *
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * `filePath` is empty or its parent path does not exist.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Runtime` if the file
   * cannot be opened for writing.
   */
  DynaSubscriber(const std::filesystem::path &filePath);

  /**
   * @brief Constructs a subscriber with an explicit reporting interval.
   *
   * @param[in] filePath Text output path.
   * @param[in] reportFrequency Positive number of steps between blocks.
   *
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * `reportFrequency` is not positive, `filePath` is empty, or its parent path
   * does not exist.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Runtime` if the file
   * cannot be opened for writing.
   */
  DynaSubscriber(const std::filesystem::path &filePath,
                 const int reportFrequency);

  /**
   * @brief Destroys the subscriber and closes its stream.
   */
  ~DynaSubscriber(void) noexcept override = default;

public:
  /**
   * @brief Writes the block of the step just completed.
   *
   * Without a run origin, the call starts a run at the current step, as
   * `DcdSubscriber` does: it writes the header and evaluates the energies at
   * the current coordinates.
   *
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::NotInitialized` if
   * the output file, context, integrator, or initialized force manager is
   * missing, or the stream is closed.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::NotImplemented` if
   * the integrator is not a supported dynamics integrator, or the force manager
   * is composite.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Runtime` if the write
   * fails.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if an energy
   * evaluation or a device-to-host copy fails.
   */
  void update(void) override;

  /**
   * @brief Writes the header and the step-0 block when a run starts.
   *
   * The call does nothing while a run is in progress.
   *
   * @param[in] startStep Completed steps before the propagation.
   * @param[in] numSteps Steps requested by the propagation.
   *
   * @throws ApoCharmmError As @ref update.
   */
  void onPropagationStart(const unsigned long long int startStep,
                          const int numSteps) override;

private:
  /**
   * @brief Writes one block, starting a run when the origin is empty.
   */
  void writeBlock(void);
};
