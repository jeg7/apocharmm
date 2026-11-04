// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#pragma once

#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>

class CharmmContext;
class CudaIntegrator;

/**
 * @brief Defines the scheduled output interface used by CUDA integrators.
 *
 * A subscriber owns one host output stream, stores a positive report frequency,
 * and can retain shared ownership of one @ref CharmmContext and one
 * `CudaIntegrator`. `CudaIntegrator::subscribe()` attaches those collaborators,
 * retains the subscriber, and snapshots its current report frequency. During
 * propagation, the integrator invokes @ref update synchronously on the calling
 * thread after each matching dynamics step.
 *
 * The base class is abstract. It is not copyable because it owns a
 * `std::fstream`, and its user-declared destructor prevents implicit move
 * operations. The class performs no internal locking; callers must serialize
 * stream access, attachment, frequency changes, updates, and destruction.
 *
 * @warning A subscribed integrator and subscriber retain each other through
 * `std::shared_ptr`. Call `CudaIntegrator::unsubscribe()` before releasing the
 * final external owners.
 * @warning Configure the file path and report frequency before subscription.
 * The integrator caches the frequency and does not observe later changes.
 *
 * @see subscriber
 */
class Subscriber {
public:
  /**
   * @brief Constructs an unattached subscriber with no output file.
   *
   * @post The report frequency is `1000`, the file path is empty, the stream is
   * closed, and no context or integrator is retained.
   */
  Subscriber(void);

  /**
   * @brief Constructs a subscriber and opens a text output file.
   *
   * The path is copied, any nonempty parent path is checked with
   * `std::filesystem::exists()`, and the file is opened with `std::ios::out`.
   * An existing file is truncated.
   *
   * @param[in] filePath File-system path copied by the subscriber. The value
   * must be nonempty; when it has a nonempty parent path, that parent path must
   * exist.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if `filePath` is empty or its checked
   * parent path does not exist.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Runtime` if the
   * output file cannot be opened for writing.
   *
   * @post On success, the report frequency is `1000` and the text stream is
   * open at `filePath`.
   */
  Subscriber(const std::filesystem::path &filePath);

  /**
   * @brief Constructs a subscriber with an explicit reporting interval.
   *
   * The frequency is validated before the path is stored or the file is opened.
   * The output is opened as a truncating text stream.
   *
   * @param[in] filePath File-system path copied by the subscriber. The value
   * must be nonempty; when it has a nonempty parent path, that parent path must
   * exist.
   * @param[in] reportFrequency Positive, dimensionless number of propagated
   * steps between scheduled updates.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if `reportFrequency` is not positive,
   * `filePath` is empty, or its checked parent path does not exist.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Runtime` if the
   * output file cannot be opened for writing.
   *
   * @post On success, the text stream is open at `filePath` and
   * `getReportFrequency() == reportFrequency`.
   */
  Subscriber(const std::filesystem::path &filePath, const int reportFrequency);

  /**
   * @brief Destroys the stream and releases retained collaborators.
   *
   * Destruction is non-throwing. All references returned by
   * @ref getFilePath are invalid after destruction.
   */
  virtual ~Subscriber(void) noexcept = default;

public:
  /**
   * @brief Sets the number of propagated steps between scheduled updates.
   *
   * @param[in] reportFrequency Positive, dimensionless reporting interval.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if `reportFrequency` is not positive.
   *
   * @post On success, `getReportFrequency() == reportFrequency`. A rejected
   * value leaves the previous frequency unchanged.
   * @warning An attached integrator schedules from the value cached during
   * subscription. Changing this property afterward does not update that cache.
   */
  void setReportFrequency(const int reportFrequency);

  /**
   * @brief Sets the logical output-file path without reopening the stream.
   *
   * @param[in] filePath Output path copied by the subscriber. The value must be
   * nonempty; when it contains `/`, its nonempty parent path must exist.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if `filePath` is empty or its checked
   * parent path does not exist.
   *
   * @post On success, both overloads of @ref getFilePath return `filePath`.
   * The current stream, if any, remains attached to its previous file.
   * @warning Call @ref openFile explicitly after changing the path when output
   * should move to the new path.
   */
  void setFilePath(const std::filesystem::path &filePath);

  /**
   * @brief Retains the context exposed to subscriber implementations.
   *
   * @param[in] ctx Shared context owner to retain. A null shared pointer is
   * accepted and leaves the subscriber unattached.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if a non-null context is already
   * retained.
   *
   * @post On success, the subscriber stores the supplied shared pointer. The
   * native context is not copied, initialized, or otherwise modified.
   */
  void setCharmmContext(std::shared_ptr<CharmmContext> ctx);

  /**
   * @brief Retains the integrator exposed to subscriber implementations.
   *
   * @param[in] integrator Shared integrator owner to retain. A null shared
   * pointer is accepted and leaves the subscriber unattached.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if a non-null integrator is already
   * retained.
   *
   * @post On success, the subscriber stores the supplied shared pointer. The
   * native integrator is not copied or modified.
   */
  void setIntegrator(std::shared_ptr<CudaIntegrator> integrator);

public:
  /**
   * @brief Returns the configured reporting interval.
   *
   * @return Positive, dimensionless number of propagated steps between
   * scheduled updates.
   */
  int getReportFrequency(void) const;

  /**
   * @brief Returns the logical output-file path.
   *
   * @return Borrowed const alias to the subscriber-owned file-system path. The
   * reference remains valid until the path is modified or the subscriber is
   * destroyed. No file operation is performed.
   */
  const std::filesystem::path &getFilePath(void) const;

  /**
   * @brief Returns mutable access to the logical output-file path.
   *
   * @return Borrowed mutable alias to the subscriber-owned file-system path.
   * The reference remains valid until the subscriber is destroyed.
   *
   * @warning Mutation bypasses empty-path and parent-path validation and does
   * not reopen or retarget the current stream.
   */
  std::filesystem::path &getFilePath(void);

public:
  /**
   * @brief Writes one subscriber-specific report.
   *
   * `CudaIntegrator` invokes this virtual operation synchronously after a
   * propagated step whose offset from the subscriber's run origin is
   * divisible by the frequency cached at subscription, matching CHARMM's
   * `MOD(ISTEP,NSAVC)` schedule across successive `propagate()` calls. Direct
   * callers are responsible for establishing every state precondition of the
   * concrete subscriber.
   *
   * @pre The concrete subscriber's required file, context, integrator, and
   * force-manager state has been initialized.
   * @warning Implementations can perform host/device transfers, device-wide
   * synchronization, force or energy evaluation, and irreversible file I/O.
   * Exceptions do not roll back an already propagated step or partial output.
   */
  virtual void update(void) = 0;

  /**
   * @brief Receives an accepted propagation request before its first step.
   *
   * The base implementation does nothing. Subscribers that describe a whole
   * run, such as DCD writers, may retain the supplied plan or update metadata.
   * `CudaIntegrator::propagate()` invokes this hook before assigning the run
   * origin (see @ref getStartStep), so an empty origin marks the first request
   * of a run.
   *
   * @param[in] startStep Number of completed steps before propagation.
   * @param[in] numSteps Positive number of steps requested for the call.
   * @pre The integrator has validated that `startStep + numSteps` is
   * representable.
   * @warning An override may throw or perform file I/O. Earlier subscriber
   * announcements are not rolled back if a later announcement fails.
   */
  virtual void onPropagationStart(const unsigned long long int startStep,
                                  const int numSteps);

  /**
   * @brief Returns the completed step at which the current run began.
   *
   * The run origin is the completed-step count observed by the first
   * `CudaIntegrator::propagate()` call after subscription, after a restart
   * load, or after the subscriber cleared it. Scheduled updates occur when the
   * number of steps completed since this origin is divisible by the cached
   * reporting frequency, which reproduces CHARMM's run-relative
   * `MOD(ISTEP,NSAVC)` schedule across successive `propagate()` calls. The
   * value is empty until a run begins.
   *
   * @return Borrowed reference to the optional run origin.
   */
  const std::optional<unsigned long long int> &getStartStep(void) const;

  /**
   * @brief Sets or clears the run origin used for scheduling.
   *
   * `CudaIntegrator::subscribe()` clears the origin and `propagate()` assigns
   * the completed-step count while the origin is empty, so a restart loaded
   * between subscription and propagation is honored. Subscribers that begin a
   * new output file clear it so the next propagation starts a new run.
   *
   * @param[in] startStep Completed-step origin, or `std::nullopt` to clear it.
   */
  void setStartStep(const std::optional<unsigned long long int> startStep);

  /**
   * @brief Checks whether the parent portion of an output path exists.
   *
   * A path with no parent component returns without a filesystem query.
   * Otherwise, the parent path is checked with `std::filesystem::exists()`.
   * The current implementation checks existence, not whether the result is a
   * directory.
   *
   * @param[in] filePath Path whose parent component is inspected. The path is
   * borrowed for the call and is not retained.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if the parent does not exist or its
   * filesystem status cannot be queried.
   *
   * @post The subscriber and filesystem are unchanged.
   */
  void checkPath(const std::filesystem::path &filePath);

  /**
   * @brief Opens the stored path as a truncating text output stream.
   *
   * An already-open stream is closed before the new open is attempted. The
   * stream state is cleared, and the stored path is revalidated.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if the stored file path is empty.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if the checked parent path does not
   * exist.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Runtime` if the file
   * cannot be opened for writing.
   *
   * @post On success, the stream is open at the stored path and any previous
   * file contents have been truncated.
   * @warning A failed reopen can leave the previous stream closed.
   * @note @ref DcdSubscriber overrides this operation to use binary mode and to
   * reset DCD frame metadata.
   */
  virtual void openFile(void);

  /**
   * @brief Appends and flushes one newline-terminated comment section.
   *
   * The input is copied. A final newline is appended when the supplied string
   * does not already end with one.
   *
   * @param[in] commentLines Nonempty text copied to the current output stream.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if `commentLines` is empty.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if the output stream is not open.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Runtime` if the write
   * or flush leaves the stream in a failed state.
   *
   * @post On success, the file contains the supplied text followed by at least
   * one newline and the stream has been flushed.
   * @warning A runtime failure can leave partial text in the output file.
   */
  void addCommentSection(const std::string &commentLines);

protected:
  /**
   * @brief Stores the positive, dimensionless reporting interval.
   *
   * The default is `1000`. An integrator copies this value into its parallel
   * scheduling array during subscription.
   */
  int m_ReportFrequency;

  /**
   * @brief Stores the completed step at which the current run began.
   *
   * Empty until the first propagation after subscription or after a reset. See
   * @ref getStartStep.
   */
  std::optional<unsigned long long int> m_StartStep;

  /**
   * @brief Stores the logical output path owned by the subscriber.
   *
   * The value can differ from the file currently associated with
   * @ref m_FileStream after a path change that is not followed by
   * @ref openFile.
   */
  std::filesystem::path m_FilePath;

  /**
   * @brief Owns the host file stream used by concrete reporters.
   *
   * The stream has no internal synchronization and can represent text or binary
   * output according to the concrete subscriber's open operation.
   */
  std::fstream m_FileStream;

  /**
   * @brief Retains the context queried by concrete subscriber updates.
   *
   * The pointer can be null. No public operation clears a non-null attachment.
   */
  std::shared_ptr<CharmmContext> m_CharmmContext;

  /**
   * @brief Retains the integrator queried by concrete subscriber updates.
   *
   * The pointer can be null. While subscribed, this strong reference and the
   * integrator's strong subscriber reference form an ownership cycle.
   */
  std::shared_ptr<CudaIntegrator> m_Integrator;
};
