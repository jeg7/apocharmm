// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#pragma once

#include "Subscriber.h"

#include <filesystem>

/**
 * @brief Writes periodic coordinate snapshots in CHARMM DCD format.
 *
 * Construction immediately creates or truncates a binary output file. The first
 * successful @ref update writes a native-endian DCD header, title block, atom
 * count, and one frame. Later updates append frames and patch the header's
 * frame and step counters.
 *
 * Each frame contains one six-`double` unit-cell record followed by separate
 * X, Y, and Z blocks of single-precision coordinates in atom order. Coordinates
 * and box lengths use angstroms. Charges are not written. Coordinate output
 * transfers the complete single-precision coordinate/charge container from the
 * CUDA device to host memory and synchronizes the current CUDA device.
 *
 * Integer fields and record markers are signed 32-bit values. Coordinates use
 * IEEE binary32 and unit-cell entries use IEEE binary64. The byte order is
 * native; the implementation performs no endian conversion.
 *
 * The first saved frame records the integrator's completed step as `START`.
 * `NFILE` counts successfully written frames. For scheduled output, `NSTEP` is
 * the latest announced propagation endpoint minus the run origin, which is the
 * completed step at the first `propagate()` call after subscription (or after
 * reopening). This matches CHARMM, where `NSTEP` is the run length and frames
 * are saved when `MOD(ISTEP,NSAVC) == 0` with `ISTEP` counted from the start of
 * the run; successive `propagate()` calls continue the same run, so a request
 * split into chunks produces the same file as one request. `NSTEP` includes
 * unsaved trailing steps and is updated by later announcements even if they
 * produce no frame.
 *
 * A manual update without a propagation announcement uses the first snapshot's
 * completed step as its origin and the observed step span as `NSTEP`. It does
 * not invent a future run. Direct `propagateOneStep()` calls do not advance the
 * integrator's completed-step metadata. Reopening clears all pending metadata
 * and the run origin. A run origin cleared after frames exist, as a restart
 * load does, is rejected until the file is reopened under a new path.
 *
 * @warning Changes to atom count, degrees of freedom, time step, or reporting
 * frequency after the header is written are rejected. Later frames must be
 * exactly one reporting interval apart, which the integrator's run-relative
 * schedule guarantees; open a new file for another cadence.
 * @warning Writes and propagation announcements are not transactional. A hook
 * can update the planned duration before another subscriber or a later step
 * fails. A failed frame write can leave a partial or inconsistent DCD file.
 *
 * @see Subscriber
 * @see subscriber
 */
class DcdSubscriber : public Subscriber {
public:
  /**
   * @brief Constructs a DCD writer with the default reporting interval.
   *
   * @param[in] filePath File-system path copied by the subscriber. The value
   * must be nonempty; when it has a nonempty paren path, that parent path must
   * exist.
   *
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * `filePath` is empty or its checked parent path does not exist.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Runtime` if the
   * binary output file cannot be opened for writing.
   *
   * @post The file has been created or truncated, no DCD header has been
   * written, and `getReportFrequency() == 1000`.
   */
  DcdSubscriber(const std::filesystem::path &filePath);

  /**
   * @brief Constructs a DCD writer with an explicit reporting interval.
   *
   * @param[in] filePath File-system path copied by the subscriber. The value
   * must be nonempty; when it has a nonempty parent path, that parent path must
   * exist.
   * @param[in] reportFrequency Positive, dimensionless number of propagated
   * steps between scheduled DCD frames.
   *
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * `reportFrequency` is not positive, `filePath` is empty, or its checked
   * parent path does not exist.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Runtime` if the
   * binary output file cannot be opened for writing.
   *
   * @post The file has been created or truncated, no DCD header has been
   * written, and `getReportFrequency() == reportFrequency`.
   */
  DcdSubscriber(const std::filesystem::path &filePath,
                const int reportFrequency);

  /**
   * @brief Destroys the writer and closes its output stream.
   *
   * Destruction is non-throwing.
   */
  ~DcdSubscriber(void) noexcept override = default;

public:
  /**
   * @brief Receives the next propagation request before its first step.
   *
   * Records the planned endpoint, and the run origin if none is set yet. If a
   * header already exists, immediately patches `NSTEP` and flushes the stream;
   * `NFILE` is unchanged. This covers requests shorter than the reporting
   * interval.
   *
   * @param[in] startStep Number of completed steps before the request.
   * @param[in] numSteps Positive number of steps requested for the call.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` for
   * a nonpositive request, unsigned-counter overflow, a backwards integration
   * step, a changed reporting frequency, or a run origin cleared after frames
   * were written (for example by a restart load).
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Runtime` if the
   * planned duration exceeds the signed 32-bit DCD range or metadata output
   * fails.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::NotInitialized` if an
   * existing header's stream has been closed.
   * @warning The planned endpoint describes the accepted request even if a
   * later hook or propagation step fails. This method is not transactional.
   */
  void onPropagationStart(const unsigned long long int startStep,
                          const int numSteps) override;

  /**
   * @brief Appends one DCD coordinate frame and updates header metadata.
   *
   * On the first update, the method writes the header before the unit-cell and
   * coordinate records. It then patches `NFILE` at byte offset 8 and `NSTEP` at
   * byte offset 20, flushes the stream, and increments the in-memory frame
   * count only after the final stream-state check succeeds.
   *
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::NotInitialized` if
   * the logical file path is empty, no context or integrator is attached, the
   * stream is closed, the context does not contain exactly three finite
   * positive box lengths, or the atom count is not positive.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Runtime` if the next
   * frame count, first frame step, derived step count, or coordinate record
   * length exceeds signed 32-bit range, or if header/frame output or flushing
   * fails.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if transferring
   * coordinates from device to host or synchronizing the current CUDA device
   * fails.
   *
   * @pre The context's single-precision coordinate/charge container has one
   * element per atom and valid device storage.
   * @post On success, one complete frame has been appended and the DCD header
   * reflects the new in-memory frame count.
   * @warning On failure, already-written bytes remain observable and the file
   * may require deletion rather than reuse.
   */
  void update(void) override;

  /**
   * @brief Opens the stored path as a truncating binary DCD output stream.
   *
   * An already-open stream is closed first. Successful opening resets the frame
   * count, header metadata, last frame step, planned endpoint, and run origin,
   * and marks the header as unwritten.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if the stored file name is empty.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if the checked parent path does not
   * exist.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Runtime` if the file
   * cannot be opened for binary output.
   *
   * @post On success, prior file contents are truncated and the next update
   * writes a fresh header.
   * @warning A failed reopen can leave the previous stream closed while the
   * stored logical file path remains unchanged.
   */
  void openFile(void) override;

private:
  /**
   * @brief Writes the fixed DCD header, title, and atom-count records.
   *
   * The control record stores the first frame's completed integration step,
   * report frequency, context degree-of-freedom count, integrator time step in
   * AKMA time units, an enabled unit-cell flag, and the reference CHARMM
   * version. It flushes and validates the stream before marking the header
   * written.
   *
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Runtime` if first
   * frame step exceeds signed 32-bit range or writing or flushing the header
   * fails.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * the time step cannot be represented as a positive finite DCD float.
   *
   * @pre The file is open and valid context and integrator attachments exist.
   */
  void writeHeader(void);

  /**
   * @brief Writes one six-value periodic-cell record.
   *
   * The CHARMM symmetric shape matrix has diagonal entries at elements 0, 2,
   * and 5. These receive the context's X, Y, and Z box lengths in angstroms;
   * the other elements are zero for the supported orthogonal cells.
   *
   * @pre The context contains exactly three valid box lengths and the stream is
   * open.
   */
  void writeXtalData(void);

  /**
   * @brief Transfers and writes the X, Y, and Z coordinate blocks.
   *
   * The complete context-owned `CudaContainer<float4>` is copied from device to
   * host and synchronized. Three Fortran-style records then store `float`
   * components in X-all-atoms, Y-all-atoms, Z-all-atoms order.
   *
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if the transfer
   * or device-wide synchronization fails.
   *
   * @pre The context atom count is positive and agrees with both active
   * coordinate/charge mirror lengths.
   */
  void writeCoordData(void);

private:
  /**
   * @brief Stores the number of frames committed by successful updates.
   *
   * The value is encoded into the DCD `NFILE` field and counts complete frames
   * present in the file.
   */
  int m_NumFramesWritten;

  /** @brief Records whether the current open file already contains a header. */
  bool m_IsHeaderWritten;

  /** @brief Stores the report frequency recorded in the current header. */
  int m_HeaderReportFrequency;

  /** @brief Stores the atom count recorded in the current header. */
  int m_HeaderNumAtoms;

  /** @brief Stores the degree-of-freedom count recorded in the header. */
  int m_HeaderNumDegreesOfFreedom;

  /** @brief Stores the time step in picoseconds recorded in the header. */
  double m_HeaderTimeStep;

  /** @brief Stores the completed integration step of the last saved frame. */
  unsigned long long int m_LastStepWritten;

  /**
   * @brief Stores the last announced propagation endpoint for this file.
   *
   * The header's `NSTEP` span is this endpoint minus the run origin kept in
   * `Subscriber::m_StartStep`, which `openFile()` clears so that a new file
   * starts a new run. Manual snapshots without an announcement use the observed
   * span instead.
   */
  unsigned long long int m_PlannedEndStep;
};
