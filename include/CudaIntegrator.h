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

#include "CharmmContext.h"
#include "CudaContainer.h"
#include "CudaHolonomicConstraint.h"
#include "Subscriber.h"

#include <cuda_runtime.h>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

/**
 * @brief Provides shared lifecycle and scheduling for CUDA integrators.
 *
 * `CudaIntegrator` owns common trajectory counters, per-atom displacement
 * storage, CUDA streams, optional holonomic-constraint state, subscriber
 * scheduling, and a retained @ref CharmmContext. Concrete integrators implement
 * the protected initialization, restart, and one-step hooks.
 *
 * The class is a polymorphic base but is not abstract. Direct calls to the
 * base implementation's initialization, restart, descriptor, or one-step hooks
 * throw `ApoCharmmErrorCode::NotImplemented`.
 *
 * Context attachment and propagation are not thread-safe. Serialize all
 * operations involving one integrator, its context, or its subscribers.
 *
 * @warning The current class does not declare copy or move operations. Treat
 * native integrator objects as non-copyable because their resource-owning
 * members do not have safe implicit-copy semantics.
 * @see cuda_integrators
 */
class CudaIntegrator : public std::enable_shared_from_this<CudaIntegrator> {
public:
  /**
   * @brief Constructs an unattached base integrator with no time step.
   *
   * The constructor creates the primary integrator stream and a reserved memcpy
   * stream. The initial public time step is zero, debug printing is disabled,
   * the nonbonded-list interval is 20 steps, and the center-of-mass-removal
   * interval is 1000 steps.
   *
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if either CUDA
   * stream cannot be created.
   * @throws std::bad_alloc If allocation of a stream owner fails.
   *
   * @post No context or subscriber is attached.
   */
  CudaIntegrator(void);

  /**
   * @brief Constructs an unattached integrator with a time step.
   *
   * @param[in] timeStep Finite positive time step in picoseconds.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if `timeStep` is non-finite or not
   * positive.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if a CUDA
   * stream cannot be created.
   * @throws std::bad_alloc If allocation of a stream owner fails.
   *
   * @post The time step is stored internally in AKMA time.
   */
  CudaIntegrator(const double timeStep);

  /**
   * @brief Constructs an integrator with a time step and debug interval.
   *
   * @param[in] timeStep Finite positive time step in picoseconds.
   * @param[in] debugPrintFrequency Non-negative, dimensionless number of
   * propagated steps between concrete-integrator debug calculations. Zero
   * disables the feature.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if `timeStep` is non-finite or not
   * positive, or if `debugPrintFrequency` is negative.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if a CUDA
   * stream cannot be created.
   * @throws std::bad_alloc If allocation of a stream owner fails.
   */
  CudaIntegrator(const double timeStep, const int debugPrintFrequency);

  /**
   * @brief Destroys the base integrator and its CUDA streams.
   *
   * Destruction releases retained context, constraint, and subscriber owners
   * only when the corresponding shared-pointer graph permits it. CUDA stream
   * cleanup failures are discarded.
   */
  virtual ~CudaIntegrator(void) noexcept;

  /**
   * @brief Returns the public integrator time step.
   *
   * @return The configured time step in picoseconds, or zero for a
   * default-constructed base object whose time step has not been set.
   */
  double getTimeStep(void) const;

  /**
   * @brief Sets the public integrator time step.
   *
   * @param[in] dt Finite positive time step in picoseconds.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if `dt` is non-finite or not
   * positive.
   *
   * @post The internal scalar contains `dt / 0.0488882129` AKMA time.
   * @warning This method does not rebuild an attached holonomic-constraint
   * helper or reinitialize concrete-integrator coefficients and displacement
   * history. Configure the time step before attaching a context.
   */
  void setTimeStep(const double dt);

  /**
   * @brief Attaches a context and initializes concrete integrator state.
   *
   * The method retains `ctx`, resizes the three base per-atom work containers,
   * snapshots the context's holonomic-constraint setting, constructs and sets
   * up the constraint helper when required, and invokes the concrete
   * `initializeImpl()` hook.
   *
   * @param[in] ctx Non-null shared context owner. The shared owner is copied
   * and retained on success.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if `ctx` is null or a context has
   * already been attached.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if the context atom count is not
   * initialized or a concrete integrator lacks required state such as a piston
   * crystal type.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotImplemented` if the direct base initialization hook
   * is reached.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if allocation,
   * setup, transfer, kernel launch, or synchronization fails.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Runtime` for a
   * concrete runtime initialization failure.
   * @throws std::bad_alloc If host or shared-owner allocation fails.
   *
   * @pre Configure the time step and all concrete initialization prerequisites
   * before this call. In particular, a Langevin-piston integrator requires a
   * supported crystal type.
   * @post On success, the context is retained and concrete device state is
   * initialized.
   * @post On failure, the retained context and constraint helper are cleared.
   * Per-atom base containers resized before the failure may remain resized.
   * @warning A second attachment is not supported, even when the argument is
   * the same context.
   */
  virtual void setCharmmContext(std::shared_ptr<CharmmContext> ctx) final;

  /**
   * @brief Returns the retained context from a const integrator.
   *
   * @return A copied shared owner of the mutable context, or an empty
   * `std::shared_ptr` before attachment. The returned owner can keep the
   * context alive independently of this integrator.
   */
  const std::shared_ptr<CharmmContext> getCharmmContext(void) const;

  /**
   * @brief Returns the retained context from a mutable integrator.
   *
   * @return A copied shared owner of the context, or an empty
   * `std::shared_ptr` before attachment. The returned owner can keep the
   * context alive independently of this integrator.
   */
  std::shared_ptr<CharmmContext> getCharmmContext(void);

  /**
   * @brief Reinitializes concrete integrator state from the attached context.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if no context is attached or a
   * concrete initialization prerequisite is absent.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotImplemented` if the direct base hook is reached.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument`,
   * `ApoCharmmErrorCode::Runtime`, or `ApoCharmmErrorCode::Cuda` when reported
   * by the concrete initialization hook.
   * @throws std::bad_alloc If host work storage cannot be allocated.
   *
   * @post On success, concrete displacement, kinetic, constraint, and RNG state
   * has been regenerated from the current context and configuration.
   * @warning Reinitialization can overwrite state restored from a restart file
   * or accumulated during earlier propagation.
   */
  virtual void initialize(void) final;

  /**
   * @brief Initializes concrete integrator state from a restart file.
   *
   * @param[in] rstFilePath Borrowed file-system path used only for this call.
   * The path is not canonicalized or retained.
   *
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::NotInitialized` if no
   * context is attached or required concrete state is missing.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * restart state is incompatible with the configured context or piston layout.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::NotImplemented` if
   * the direct base hook is reached.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Runtime` if the file
   * cannot be opened, parsed, or interpreted.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if restored
   * state cannot be transferred or synchronized.
   * @throws std::bad_alloc If parsing or state allocation fails.
   *
   * @warning Restart loading is not transactional. State parsed or transferred
   * before a later failure can remain observable.
   * @note A successful load begins a new CHARMM-style run: every attached
   * subscriber's run origin is cleared and the next `propagate()` fixes it at
   * the imported step count, as a CHARMM `DYNA RESTART` run counts `ISTEP` from
   * the restart point. A `DcdSubscriber` that already wrote frames rejects the
   * next request; give it a new path and call `openFile()` first.
   */
  virtual void
  initializeFromRestartFile(const std::filesystem::path &rstFilePath) final;

  /**
   * @brief Propagates one concrete dynamics step without base-loop bookkeeping.
   *
   * This entry point checks context and force-manager readiness and invokes the
   * concrete one-step hook. It does not reset the neighbor list, increment
   * public step counters, run scheduled subscribers, or perform the base
   * periodic NaN check.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if context or force-manager state is
   * not ready.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotImplemented` if the direct base hook is reached.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument`,
   * `ApoCharmmErrorCode::Runtime`, or `ApoCharmmErrorCode::Cuda` when reported
   * by the concrete step.
   *
   * @post On success, one concrete step has changed context and integrator
   * state, but the base trajectory counters are unchanged.
   */
  virtual void propagateOneStep(void) final;

  /**
   * @brief Propagates dynamics and invokes scheduled subscriber updates.
   *
   * The method validates readiness, resets the context neighbor list and
   * per-call neighbor-list counter, stores `numSteps`, invokes each
   * subscriber's `onPropagationStart()` hook with the completed starting step
   * and requested length, and then assigns that starting step as the run
   * origin of every subscriber that does not have one.
   *
   * After each successful concrete step, the method increments the completed
   * and neighbor-list counters, performs a periodic NaN-energy check, and
   * synchronously invokes subscribers whose cached reporting interval divides
   * the number of steps completed since their run origin. Successive calls
   * therefore continue one CHARMM-style run: a subscriber with frequency 10
   * reports at steps 10, 20, 30, ... regardless of how the steps are split
   * across calls.
   *
   * @param[in] numSteps Positive number of dynamics steps to request.
   *
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * `numSteps` is not positive, the requested final step would overflow the
   * completed-step counter, or a native participant rejects an argument.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::NotInitialized` if
   * the context, force manager, concrete integrator, or scheduled subscriber
   * lacks required state.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::NotImplemented` if a
   * concrete operation or subscriber reports an unsupported operation.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if force
   * evaluation, a kernel, a transfer, or CUDA synchronization fails.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Runtime` if a runtime
   * readiness check, NaN check, subscriber file operation, or other native
   * runtime operation fails.
   * @throws std::invalid_argument If a scheduled legacy `DynaSubscriber`
   * rejects its current integrator.
   * @throws std::runtime_error If a scheduled legacy `XYZSubscriber` has no
   * attached context.
   *
   * @pre An attached context must expose a non-null initialized force manager.
   * @post On success, exactly `numSteps` concrete steps and all matching
   * callbacks have completed.
   * @post The total-step counter has increased by `numSteps`, and the
   * most-recent-request counter equals `numSteps`.
   * @warning The operation is not transactional. The total-step counter records
   * only concrete steps that returned successfully. A failing concrete step may
   * have partly changed state. Earlier steps, subscriber announcements, and
   * output remain observable after a later failure.
   */
  void propagate(const int numSteps);

  /**
   * @brief Sets the concrete-integrator debug-calculation interval.
   *
   * @param[in] freq Non-negative, dimensionless number of propagated steps
   * between debug calculations. Zero disables the feature.
   *
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * `freq` is negative.
   *
   * @note The base class stores this value. Whether and what a concrete
   * integrator calculates at the interval is implementation-specific.
   */
  void setDebugPrintFrequency(const int freq);

  /**
   * @brief Sets the nonbonded neighbor-list update interval.
   *
   * @param[in] freq Positive, dimensionless number of propagated steps between
   * concrete-integrator neighbor-list resets.
   *
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * `freq` is not positive.
   *
   * @note The per-call counter is reset to zero before `propagate()` enters its
   * loop, so concrete implementations test the update condition during the
   * first requested step.
   */
  void setNonbondedListUpdateFrequency(const int freq);

  /**
   * @brief Attaches one subscriber and caches its reporting interval.
   *
   * The method rejects null and duplicate subscribers, obtains a shared owner
   * of this integrator, calls `sub->setCharmmContext(m_Context)`, calls
   * `sub->setIntegrator(...)`, clears the subscriber's run origin so the next
   * `propagate()` starts a new CHARMM-style run, appends the subscriber, and
   * appends its current report frequency to the parallel scheduling array.
   *
   * @param[in] sub Non-null shared subscriber owner retained on success.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if `sub` is null, is already present,
   * or rejects a second context or integrator attachment.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if this integrator is not owned by a
   * `std::shared_ptr` and therefore cannot obtain a shared self-reference.
   * @throws std::bad_alloc If appending either scheduling array cannot
   * allocate.
   *
   * @pre Call @ref setCharmmContext with a non-null context before attaching a
   * subscriber that requires context state.
   * @post On success, the integrator and subscriber retain each other, the
   * subscriber retains the current context, and the current frequency is
   * cached.
   * @warning Attachment is not transactional. Failure after a backlink or first
   * vector append can leave partial state.
   */
  void subscribe(std::shared_ptr<Subscriber> sub);

  /**
   * @brief Attaches subscribers sequentially in vector order.
   *
   * @param[in] sublist Borrowed vector of shared subscriber owners. Each
   * element is passed to the single-subscriber overload and retained on
   * success.
   *
   * @note Exceptions from the single-subscriber overload propagate unchanged,
   * including `ApoCharmmError` and `std::bad_alloc`.
   *
   * @post On success, every element has been attached in input order.
   * @warning The operation is not transactional. Subscribers attached before a
   * failing element remain attached.
   */
  void subscribe(const std::vector<std::shared_ptr<Subscriber>> &sublist);

  /**
   * @brief Removes one subscriber and its cached reporting interval.
   *
   * @param[in] sub Non-null shared subscriber identity to remove. The argument
   * is borrowed for lookup; removing it releases only the integrator's owner.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if `sub` is null or is not present.
   *
   * @post On success, the corresponding entries are erased from both parallel
   * arrays.
   * @warning The removed subscriber still retains its context and integrator.
   * Those backlinks are not cleared, and normal resubscription of the same
   * object therefore fails.
   */
  void unsubscribe(std::shared_ptr<Subscriber> sub);

  /**
   * @brief Removes subscribers sequentially in vector order.
   *
   * @param[in] sublist Borrowed vector of subscriber identities. Each element
   * is passed to the single-subscriber overload.
   *
   * @note Exceptions from the single-subscriber overload propagate unchanged.
   *
   * @post On success, every listed subscriber has been removed.
   * @warning The operation is not transactional. Subscribers removed before a
   * failing element remain removed.
   */
  void unsubscribe(const std::vector<std::shared_ptr<Subscriber>> &sublist);

  /**
   * @brief Returns the attached subscriber array.
   *
   * @return Borrowed const alias to the integrator-owned vector. Each element
   * is a shared owner of a subscriber. The alias remains valid until vector
   * reallocation, integrator destruction, or non-const mutation.
   */
  const std::vector<std::shared_ptr<Subscriber>> &getSubscribers(void) const;

  /**
   * @brief Returns mutable access to the attached subscriber array.
   *
   * @return Borrowed mutable alias to the integrator-owned vector.
   * @warning The vector must remain the same length and order as
   * @ref m_ReportFreqList. Direct mutation can violate that invariant and cause
   * out-of-bounds access or callbacks to the wrong subscriber.
   */
  std::vector<std::shared_ptr<Subscriber>> &getSubscribers(void);

  /**
   * @brief Returns the cached subscriber-frequency array.
   *
   * @return Borrowed const alias to positive, dimensionless intervals. Entry
   * `i` schedules subscriber entry `i`. The alias remains valid until vector
   * reallocation, integrator destruction, or non-const mutation.
   */
  const std::vector<int> &getReportFreqList(void) const;

  /**
   * @brief Returns mutable access to cached subscriber frequencies.
   *
   * @return Borrowed mutable alias to the integrator-owned vector.
   * @warning Every value must remain positive and the vector must remain the
   * same length and order as @ref m_Subscribers. Mutation bypasses validation.
   */
  std::vector<int> &getReportFreqList(void);

  /**
   * @brief Sets the center-of-mass-removal interval.
   *
   * @param[in] freq Positive, dimensionless number of propagated steps between
   * concrete-integrator center-of-mass corrections.
   *
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * `freq` is not positive.
   *
   * @note The completed-step counter starts at zero, so the concrete
   * integrators perform their correction during the first propagated step.
   */
  void setRemoveCenterOfMassFrequency(const int freq);

  /**
   * @brief Returns the current half-step displacement container.
   *
   * @return Borrowed const alias to the integrator-owned `N`-element
   * `CudaContainer<double4>`, where `N` is the attached context atom count.
   * Components `x`, `y`, and `z` contain angstrom-valued displacements. The
   * integrator assigns no public meaning to `w`.
   *
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::NotInitialized` if
   * concrete state no longer matches the context configuration.
   * @note The accessor performs no host/device transfer or synchronization. The
   * alias remains valid until integrator destruction or structural mutation of
   * the container.
   */
  const CudaContainer<double4> &getCoordsDelta(void) const;

  /**
   * @brief Returns mutable current half-step displacement storage.
   *
   * @return Borrowed mutable alias to the integrator-owned `N`-element
   * `CudaContainer<double4>`.
   *
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::NotInitialized` if
   * concrete state no longer matches the context configuration.
   * @warning Preserve the atom-count length and the concrete integrator's
   * host/device state. Resizing or independently changing one mirror can
   * invalidate later kernels.
   */
  CudaContainer<double4> &getCoordsDelta(void);

  /**
   * @brief Returns the previous half-step displacement container.
   *
   * @return Borrowed const alias to the integrator-owned `N`-element
   * `CudaContainer<double4>`. Components `x`, `y`, and `z` contain
   * angstrom-valued displacements; `w` has no public integrator meaning.
   *
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::NotInitialized` if
   * concrete state no longer matches the context configuration.
   * @note The accessor performs no host/device transfer or synchronization.
   */
  const CudaContainer<double4> &getCoordsDeltaPrevious(void) const;

  /**
   * @brief Returns mutable previous half-step displacement storage.
   *
   * @return Borrowed mutable alias to the integrator-owned `N`-element
   * `CudaContainer<double4>`.
   *
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::NotInitialized` if
   * concrete state no longer matches the context configuration.
   * @warning Preserve the atom-count length, component interpretation, and
   * host/device coherence expected by the concrete integrator.
   */
  CudaContainer<double4> &getCoordsDeltaPrevious(void);

  /**
   * @brief Returns key/value descriptors for the concrete integrator.
   *
   * @return A newly allocated map whose strings describe the concrete
   * integrator.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotImplemented` when the base implementation is
   * called.
   * @throws std::bad_alloc If a concrete implementation cannot allocate the
   * result.
   */
  virtual std::map<std::string, std::string> getIntegratorDescriptors(void);

  /**
   * @brief Returns the wrapped completed-step counter.
   *
   * @return A dimensionless value in the range zero through `INT_MAX`.
   *
   * @note This compatibility view is derived from @ref getTotNumSteps modulo
   * `INT_MAX + 1`; it has no independent storage. Internal periodic operations
   * use the full cumulative count. Direct `propagateOneStep()` calls do not
   * advance the cumulative count.
   */
  int getCurrentPropagatedStep(void) const;

  /**
   * @brief Returns the total number of completed base-loop steps.
   *
   * @return The restored restart step plus concrete steps that returned
   * successfully from `propagate()`. Subscribers observe the completed step
   * associated with their snapshot. Direct `propagateOneStep()` calls do not
   * advance this counter.
   */
  unsigned long long int getTotNumSteps(void) const;

  /**
   * @brief Returns the most recently requested propagation length.
   *
   * @return The positive dimensionless `numSteps` value from the most recent
   * accepted `propagate()` call, the saved NSTEP after restart loading, or zero
   * before either operation. This is request metadata, not a second clock.
   */
  int getNumSteps(void) const;

protected:
  /**
   * @brief Initializes concrete state from the attached context.
   *
   * Derived implementations may allocate work arrays, initialize RNG state,
   * calculate forces, and synchronize the primary integrator stream.
   *
   * @pre `m_Context` is non-null.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::NotImplemented` in
   * the base implementation.
   */
  virtual void initializeImpl(void);

  /**
   * @brief Initializes concrete state from a restart file.
   *
   * @param[in] rstFilePath Borrowed restart-file path.
   *
   * @pre `m_Context` is non-null.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::NotImplemented` in
   * the base implementation.
   */
  virtual void
  initializeFromRestartFileImpl(const std::filesystem::path &rstFilePath);

  /**
   * @brief Advances one concrete dynamics step.
   *
   * Derived implementations update context coordinates, velocities, forces,
   * and their own state. They do not update the base public step counters or
   * subscriber schedule.
   *
   * @pre `requirePropagationReady()` has succeeded.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::NotImplemented` in
   * the base implementation.
   */
  virtual void propagateOneStepImpl(void);

  /**
   * @brief Validates shared state required for propagation.
   *
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::NotInitialized` if
   * the context is absent, its force manager is absent, the force manager is
   * not initialized, or concrete state no longer matches the context.
   */
  void requirePropagationReady(void) const;

  /**
   * @brief Checks whether cached concrete state matches the current context.
   *
   * The base implementation does nothing. Concrete integrators may reject stale
   * configuration before propagation, restart loading, or history access.
   * `initialize()` deliberately bypasses this check so it can rebuild state.
   */
  virtual void requireCurrentState(void) const;

protected:
  /**
   * @brief Invokes subscribers scheduled for the just-completed step.
   *
   * The method iterates over the cached frequency array and calls
   * `m_Subscribers[i]->update()` when the number of steps completed since
   * that subscriber's run origin is divisible by `m_ReportFreqList[i]`. This
   * reproduces CHARMM's `MOD(ISTEP,NSAVC)` schedule, where `ISTEP` counts from
   * the start of the run rather than from each `propagate()` call. Calls are
   * synchronous and occur in subscriber-vector order.
   *
   * @pre @ref m_Subscribers and @ref m_ReportFreqList have identical lengths
   * and corresponding order, every cached frequency is positive, and every
   * subscriber's run origin has been assigned by `propagate()`.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::NotInitialized` if a
   * subscriber has no run origin.
   * @note Any exception raised by a subscriber propagates immediately. Later
   * subscribers scheduled for the same step are not called.
   */
  void reportIfNeeded(void);
  void checkForNanEnergy(void);
  static int
  wrapCurrentPropagatedStep(const unsigned long long int propagatedStep);

protected:
  /**
   * @brief Stores the internal time step in AKMA time.
   *
   * Public setters divide picoseconds by TIMFAC before storing this value.
   */
  double m_TimeStep;

  /**
   * @brief Stores the non-negative concrete debug-calculation interval.
   *
   * Zero disables the feature.
   */
  int m_DebugPrintFrequency;

  /**
   * @brief Retains the context attached to this integrator.
   *
   * The owner is empty before attachment and is cleared when attachment
   * initialization fails.
   */
  std::shared_ptr<CharmmContext> m_Context;

  /**
   * @brief Stores the current per-call neighbor-list scheduling counter.
   *
   * `propagate()` resets this value to zero before the loop and increments it
   * after each successful concrete step.
   */
  int m_StepsSinceNeighborListUpdate;

  /**
   * @brief Stores the restored step plus successfully completed base-loop
   * steps.
   */
  unsigned long long int m_TotNumSteps;

  /** @brief Stores the most recently accepted request length or saved NSTEP. */
  int m_NumSteps;

  /**
   * @brief Retains the optional holonomic-constraint helper.
   *
   * The helper is created during context attachment when the context reports
   * that holonomic constraints are enabled.
   */
  std::shared_ptr<CudaHolonomicConstraint> m_HolonomicConstraint;

  /**
   * @brief Stores per-atom reference coordinates used by concrete steps.
   *
   * The container has one `double4` record per context atom. Concrete kernels
   * use the `x`, `y`, and `z` components in angstroms.
   */
  CudaContainer<double4> m_CoordsRef;

  /**
   * @brief Stores current half-step displacements.
   *
   * The container has one `double4` record per context atom. Concrete kernels
   * use the `x`, `y`, and `z` components in angstroms.
   */
  CudaContainer<double4> m_CoordsDelta;

  /**
   * @brief Stores previous half-step displacements.
   *
   * The container has one `double4` record per context atom. Concrete kernels
   * use the `x`, `y`, and `z` components in angstroms.
   */
  CudaContainer<double4> m_CoordsDeltaPrevious;

  /**
   * @brief Owns the primary CUDA stream used by concrete integrator work.
   *
   * Concrete initialization and complete-step paths synchronize this stream at
   * their explicit completion boundaries.
   */
  std::shared_ptr<cudaStream_t> m_IntegratorStream;

  /**
   * @brief Owns a reserved CUDA memcpy stream.
   *
   * The currently documented concrete integrators allocate this stream but do
   * not issue their inspected transfer paths on it.
   */
  std::shared_ptr<cudaStream_t> m_IntegratorMemcpyStream;

  /**
   * @brief Stores the constraint setting captured during context attachment.
   */
  bool m_UsingHolonomicConstraints;

  /**
   * @brief Retains subscribers in callback order.
   *
   * Entry `i` must correspond to entry `i` in @ref m_ReportFreqList. Each
   * shared pointer owns the subscriber until successful unsubscription or
   * integrator destruction.
   */
  std::vector<std::shared_ptr<Subscriber>> m_Subscribers;

  /**
   * @brief Stores the frequency snapshot for each attached subscriber.
   *
   * Entry `i` is the positive, dimensionless reporting interval copied from
   * subscriber `i` during attachment. Later changes to the subscriber property
   * do not update this array.
   */
  std::vector<int> m_ReportFreqList;

  /** @brief Stores the positive nonbonded-list update interval in steps. */
  int m_NonbondedListUpdateFrequency;

  /** @brief Stores the positive center-of-mass-removal interval in steps. */
  int m_RemoveCenterOfMassFrequency;

  /**
   * @brief Stores the concrete integrator label used by subscribers.
   *
   * Restart and descriptor code uses this string to discriminate supported
   * integrator state layouts.
   */
  std::string m_IntegratorTypeName;
};
