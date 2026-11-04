// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II, Samarjeet Prasad
//
// ENDLICENSE

#pragma once

#include "Subscriber.h"

#include <filesystem>

/**
 * @brief Writes supported leapfrog state in CHARMM's formatted restart layout.
 *
 * Each successful @ref update writes a version-52 standard restart core with 96
 * property, 192 energy-term, and 50 pressure-statistic entries, followed by an
 * apoCHARMM RNG extension. Supported modes are the ordinary leapfrog paths
 * corresponding to the three integrators listed below, with CUBI, TETR, or ORTH
 * cells and default CHARMM energy-selection flags.
 *
 * Coordinates and box lengths use angstroms; velocities use angstroms per AKMA
 * time unit. HDOT, PNHV, and PNHF retain the integrators' native
 * CHARMM-compatible units. The X/Y/Z section contains leapfrog displacements.
 * Numeric fields use CHARMM D22.15 precision, so restart serialization is not
 * bitwise lossless.
 *
 * Unavailable diagnostic histories are reset. A nonzero completed-step count
 * starts a one-sample temperature window while retaining continuation state.
 * PNH is reset to zero as a thermal-piston reference-energy reset. The standard
 * record advertises no CHARMM RNG state; native Philox data is preserved only
 * in the trailing extension. The header carries the marker `APO` at character
 * offsets 32-34 (Fortran columns 33-35), beyond every field CHARMM's READYN
 * consumes, so CHARMM ignores both the marker and the extension, while
 * apoCHARMM readers require the extension whenever the marker is present.
 * Foreign imports start a new configured RNG stream.
 *
 * Coordinate saving frequency comes from attached DCD subscribers, or is zero
 * if none is attached. Different simultaneous DCD saving frequencies cannot be
 * represented by one CHARMM NSAVC and are rejected. The crystal type comes
 * exclusively from CharmmContext::getCrystalType(). Set the context's crystal
 * type explicitly before writing or loading a restart. Loading requires that
 * the configured type match the file, including an equal-sided ORTH cell.
 *
 * @warning Updates support only CudaNoseHooverIntegrator,
 * CudaLangevinPistonIntegrator, and CudaLangevinThermostatIntegrator.
 * @warning Match topology, potential, constraints, timestep, thermostat and
 * piston settings in the receiving program. A restart does not supply them.
 * @warning Formatting and transfers finish before the update truncates the
 * target, but output is not atomic. A write failure can leave a partial file.
 * Construction also creates or truncates the target immediately.
 *
 * @see Subscriber
 * @see subscriber
 */
class RestartSubscriber : public Subscriber {
public:
  /**
   * @brief Constructs an unattached restart writer without an output file.
   *
   * @post The report frequency is `1000`, the logical file path is empty, and
   * the stream is closed.
   */
  RestartSubscriber(void);

  /**
   * @brief Constructs a restart writer with the default reporting interval.
   *
   * Construction creates or truncates the text output immediately. A complete
   * restart is not written until @ref update is invoked.
   *
   * @param[in] filePath Output file-system path copied by the subscriber. The
   * value must be nonempty; when it has a nonempty parent path, that parent
   * path must exist.
   *
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * `filePath` is empty or its checked parent path does not exist.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Runtime` if the text
   * output file cannot be opened for writing.
   *
   * @post The file has been created or truncated and
   * `getReportFrequency() == 1000`.
   */
  RestartSubscriber(const std::filesystem::path &filePath);

  /**
   * @brief Constructs a restart writer with an explicit reporting interval.
   *
   * Construction validates the frequency, then creates or truncates the text
   * output. A complete restart is not written until @ref update is invoked.
   *
   * @param[in] filePath Output file-system path copied by the subscriber. The
   * value must be nonempty; when it has a nonempty parent path, that parent
   * path must exist.
   * @param[in] reportFrequency Positive, dimensionless number of propagated
   * steps between restart replacements.
   *
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * `reportFrequency` is not positive, `filePath` is empty, or its checked
   * parent path does not exist.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Runtime` if the text
   * output file cannot be opened for writing.
   *
   * @post The file has been created or truncated and
   * `getReportFrequency() == reportFrequency`.
   */
  RestartSubscriber(const std::filesystem::path &filePath,
                    const int reportFrequency);

  /**
   * @brief Destroys the writer and closes its stream.
   *
   * Destruction is non-throwing.
   */
  ~RestartSubscriber(void) noexcept override = default;

  /**
   * @brief Replaces the output with one supported restart snapshot.
   *
   * Required state is transferred, validated, and formatted in memory before
   * the path is reopened for writing. The completed-step counter supplies
   * NPRIV; the requested run length supplies NSTEP.
   *
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::NotInitialized` if a
   * collaborator, atom state, positive degrees of freedom, an explicit context
   * crystal type, or a finite positive box is absent.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::NotImplemented` for
   * unsupported integrators or crystal types, or multiple distinct
   * coordinate-saving frequencies.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` for
   * inconsistent crystal lengths, nonfinite numeric state, or values outside
   * the supported CHARMM integer or D22.15 round-trip ranges.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Runtime` for
   * mismatched collaborator state, inconsistent subscriber lists, formatting
   * failures, or output failures.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` for device
   * transfer or calculation failures.
   * @throws std::bad_alloc If snapshot formatting cannot allocate memory.
   *
   * @post On success, the file is complete and its stream is closed.
   * @warning Output is not atomic; an I/O failure can leave a partial file.
   */
  void update(void) override;
};
