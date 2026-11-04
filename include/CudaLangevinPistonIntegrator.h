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

#include "CudaIntegrator.h"

#include "Crystal.h"
#include "CudaContainer.h"

#include <cstdint>
#include <curand_kernel.h>
#include <filesystem>
#include <string>
#include <vector>
#include <vector_types.h>

/**
 * @brief Propagates pressure-controlled dynamics with a Langevin piston.
 *
 * The integrator maintains one to three crystal degrees of freedom, a
 * predictor-corrector pressure loop, optional Nose-Hoover temperature
 * coupling, one Philox RNG state per crystal degree of freedom, pressure
 * tensors, and running temperature and pressure averages.
 *
 * Select a supported @ref CRYSTAL on @ref CharmmContext before attaching it.
 * Attachment derives and allocates piston state from the context crystal.
 * Configure custom Langevin-piston masses and friction after attachment. Change
 * context symmetry only between completed propagation calls, then call
 * initialize() before accessing piston state or continuing dynamics. State
 * getters/setters, restart loading, and propagation reject a stale layout with
 * NotInitialized; getCrystalType() still reports the context selection.
 * Previously borrowed container aliases must not be used across this
 * transition.
 *
 * @warning Treat the class as non-copyable. It owns a raw device RNG-state
 * allocation in addition to base CUDA streams.
 * @see cuda_integrators
 */
class CudaLangevinPistonIntegrator : public CudaIntegrator {
public:
  /**
   * @brief Constructs an unattached Langevin-piston integrator.
   *
   * @param[in] timeStep Finite positive time step in picoseconds.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if `timeStep` is non-finite or not
   * positive.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if stream or
   * fixed-state allocation fails.
   * @throws std::bad_alloc If host or stream-owner allocation fails.
   * @throws std::system_error If the platform random source used for the
   * initial seed fails.
   *
   * @post Nose-Hoover temperature coupling is enabled, the reference
   * temperature is 300 K, the reference pressure is the 3-by-3 identity tensor
   * in atmospheres, no context is attached, the maximum predictor-corrector
   * count is three, and running averages are zero.
   */
  CudaLangevinPistonIntegrator(const double timeStep);

  /**
   * @brief Destroys the piston integrator and its RNG-state allocation.
   *
   * CUDA cleanup failures are discarded and no exception escapes.
   */
  ~CudaLangevinPistonIntegrator(void) noexcept override;

public:
  /**
   * @brief Enables or disables Nose-Hoover temperature coupling.
   *
   * @param[in] usingNoseHooverThermostat `true` includes the Nose-Hoover
   * extended variable in predictor-corrector updates; `false` omits it.
   */
  void useNoseHooverThermostat(const bool usingNoseHooverThermostat);

  /**
   * @brief Sets the reference temperature.
   *
   * @param[in] referenceTemperature Finite non-negative temperature in kelvin.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if `referenceTemperature` is
   * non-finite or negative.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if a selected crystal has inconsistent
   * piston state while friction factors are recomputed.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if updated
   * stochastic factors cannot be transferred.
   *
   * @post When a crystal has been selected, piston-friction factors are
   * recomputed using the new temperature.
   */
  void setReferenceTemperature(const double referenceTemperature);

  /**
   * @brief Sets the scalar Nose-Hoover coupling mass.
   *
   * @param[in] noseHooverPistonMass Finite non-negative native coupling-mass
   * value. The repository does not establish its public dimensional
   * convention.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if the value is non-finite or
   * negative.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if the scalar
   * cannot be copied to device storage.
   *
   * @warning Zero is currently accepted, but propagation divides by this value
   * when Nose-Hoover temperature coupling is enabled.
   */
  void setNoseHooverPistonMass(const double noseHooverPistonMass);

  /**
   * @brief Sets the current Nose-Hoover extended-variable velocity.
   *
   * @param[in] noseHooverPistonVelocity Finite native extended-state value.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if the value is non-finite.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if device
   * update fails.
   */
  void setNoseHooverPistonVelocity(const double noseHooverPistonVelocity);

  /**
   * @brief Sets the previous Nose-Hoover extended-variable velocity.
   *
   * @param[in] noseHooverPistonVelocityPrevious Finite native previous-state
   * value.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if the value is non-finite.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if device
   * update fails.
   */
  void setNoseHooverPistonVelocityPrevious(
      const double noseHooverPistonVelocityPrevious);

  /**
   * @brief Sets the current Nose-Hoover velocity-increment state.
   *
   * @param[in] noseHooverPistonForce Finite native increment value.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if the value is non-finite.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if device
   * update fails.
   *
   * @note The propagation kernel stores a velocity increment despite the
   * historical “force” name.
   */
  void setNoseHooverPistonForce(const double noseHooverPistonForce);

  /**
   * @brief Sets the previous Nose-Hoover velocity-increment state.
   *
   * @param[in] noseHooverPistonForcePrevious Finite native previous-increment
   * value.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if the value is non-finite.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if device
   * update fails.
   */
  void
  setNoseHooverPistonForcePrevious(const double noseHooverPistonForcePrevious);

  /**
   * @brief Sets the number of piston predictor-corrector iterations.
   *
   * @param[in] maxPredictorCorrectorIterations Positive dimensionless
   * iteration count.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if the value is not positive.
   *
   * @note One propagated step advances the recorded piston RNG sequence
   * position once per configured iteration.
   */
  void
  setMaxPredictorCorrectorIterations(const int maxPredictorCorrectorIterations);

  /**
   * @brief Selects the on-step temperature estimator.
   *
   * @param[in] usingOldTemperature `true` selects kinetic-energy element 1 for
   * native feedback and instantaneous temperature; `false` selects element 0.
   */
  void useOldTemperature(const bool usingOldTemperature);

  /**
   * @brief Sets the reference pressure tensor.
   *
   * @param[in] referencePressure Borrowed vector containing exactly nine finite
   * row-major values in atmospheres, ordered `xx`, `xy`, `xz`, `yx`, `yy`,
   * `yz`, `zx`, `zy`, `zz`. Values are copied.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if the vector does not contain
   * exactly nine values or any value is non-finite.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if the copied
   * tensor cannot be transferred to device storage.
   */
  void setReferencePressure(const std::vector<double> &referencePressure);

  /**
   * @brief Enables or disables the constant-surface-tension branch.
   *
   * @param[in] constantSurfaceTensionFlag `true` enables the branch and `false`
   * disables it.
   *
   * @warning The current public API has no target-surface-tension setter, and
   * the target container is not initialized after allocation. Do not enable
   * this mode until that implementation defect is corrected.
   */
  void setConstantSurfaceTension(const bool constantSurfaceTensionFlag);

  /**
   * @brief Sets the Langevin-piston masses.
   *
   * @param[in] mass Borrowed values copied in crystal-degree-of-freedom order.
   * The vector must have exactly the active degree-of-freedom count and every
   * value must be finite and non-negative. The exact dimensional convention is
   * not established by the repository.
   *
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::NotInitialized` if no
   * context with a supported crystal type has been attached.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * the length is wrong or a value is non-finite or negative.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if masses,
   * inverse masses, or derived friction factors cannot be transferred.
   *
   * @post A zero mass stores an inverse mass of zero and disables motion of the
   * corresponding piston degree of freedom.
   */
  void setLangevinPistonMass(const std::vector<double> &mass);

  /**
   * @brief Sets the Langevin-piston RNG seed.
   *
   * @param[in] seed Dimensionless 64-bit seed.
   *
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if an attached
   * context requires RNG allocation or initialization and CUDA work fails.
   *
   * @post The existing sequence position is preserved.
   */
  void setLangevinPistonFrictionSeed(const std::uint64_t seed);

  /**
   * @brief Sets the piston RNG sequence position.
   *
   * @param[in] rngSequencePos Dimensionless Philox sequence position.
   *
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if an attached
   * context requires RNG allocation or initialization and CUDA work fails.
   */
  void setRngSequencePos(const unsigned long long int rngSequencePos);

  /**
   * @brief Restores serialized piston RNG states.
   *
   * @param[in] rngStateString Borrowed opaque string produced by
   * `getRngStates()`.
   *
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::NotInitialized` if no
   * context with a supported crystal type has been attached.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * the decoded state count does not equal the active piston degree-of-freedom
   * count.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Runtime` if the
   * string cannot be decoded.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if device
   * allocation or transfer fails.
   */
  void setRngStates(const std::string &rngStateString);

  /**
   * @brief Sets the Langevin-piston friction coefficient.
   *
   * @param[in] pgamma Finite non-negative friction in inverse picoseconds.
   *
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::NotInitialized` if no
   * context with a supported crystal type has been attached.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::InvalidArgument` if
   * `pgamma` is non-finite or negative.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if stochastic
   * forcing factors cannot be transferred to device storage.
   */
  void setLangevinPistonFriction(const double pgamma);

  /**
   * @brief Resets temperature and pressure running averages.
   *
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if any running
   * average cannot be reset on the device.
   *
   * @post The sample count, both temperature values, all nine pressure-tensor
   * values, and the scalar pressure average are zero.
   */
  void resetAverages(void);

public:
  /** @brief Returns the reference temperature.
   * @return The stored temperature in kelvin.
   */
  double getReferenceTemperature(void) const;

  /** @brief Returns the native temperature-estimator selector.
   * @return `true` for element 1 and `false` for element 0.
   */
  bool usingOldTemperature(void) const;

  /** @brief Returns the scalar Nose-Hoover coupling-mass container.
   * @return Borrowed const alias to a one-element container.
   */
  const CudaContainer<double> &getNoseHooverPistonMass(void) const;

  /** @brief Returns the current Nose-Hoover velocity container.
   * @return Borrowed const alias to a one-element container.
   */
  const CudaContainer<double> &getNoseHooverPistonVelocity(void) const;

  /** @brief Returns the previous Nose-Hoover velocity container.
   * @return Borrowed const alias to a one-element container.
   */
  const CudaContainer<double> &getNoseHooverPistonVelocityPrevious(void) const;

  /** @brief Returns the current Nose-Hoover increment container.
   * @return Borrowed const alias to a one-element container.
   */
  const CudaContainer<double> &getNoseHooverPistonForce(void) const;

  /** @brief Returns the previous Nose-Hoover increment container.
   * @return Borrowed const alias to a one-element container.
   */
  const CudaContainer<double> &getNoseHooverPistonForcePrevious(void) const;

  /** @brief Returns the piston predictor-corrector iteration count.
   * @return A positive dimensionless count.
   */
  int getMaxPredictorCorrectorIterations(void) const;

  /**
   * @brief Returns both kinetic-energy estimators.
   *
   * @return Borrowed const alias to a two-element container in kilocalories per
   * mole. Element 0 is the three-point estimator and element 1 is the on-step
   * estimator.
   */
  const CudaContainer<double> &getKineticEnergy(void) const;

  /**
   * @brief Returns both running-temperature estimators.
   *
   * @return Borrowed const alias to a two-element container in kelvin with the
   * same estimator order as `getKineticEnergy()`.
   */
  const CudaContainer<double> &getAverageTemperature(void) const;

  /**
   * @brief Returns the declared old-temperature average container.
   *
   * @return A borrowed const container alias.
   *
   * @warning The current repository declares this function but provides no
   * definition. ODR-use causes a link failure.
   */
  const CudaContainer<double> &getAverageOldTemperature(void) const;

  /**
   * @brief Returns the reference pressure tensor.
   *
   * @return Borrowed const alias to nine row-major values in atmospheres.
   */
  const CudaContainer<double> &getReferencePressureTensor(void) const;

  /**
   * @brief Returns the attached context's authoritative crystal symmetry.
   *
   * @return `CRYSTAL::UNSET` before context attachment; otherwise the value
   * returned by @ref CharmmContext::getCrystalType. No independent integrator
   * crystal value is stored.
   */
  CRYSTAL getCrystalType(void) const;

  /**
   * @brief Returns the Langevin-piston mass array.
   *
   * @return Borrowed const alias with one, two, or three values in active
   * crystal-degree-of-freedom order, or an empty container before crystal
   * configuration.
   */
  const CudaContainer<double> &getLangevinPistonMass(void) const;

  /** @brief Returns on-step piston positions.
   * @return Borrowed const alias to one, two, or three values in angstroms.
   */
  const CudaContainer<double> &getLangevinPistonOnStepPosition(void) const;

  /** @brief Returns half-step piston positions.
   * @return Borrowed const alias to one, two, or three values in angstroms.
   */
  const CudaContainer<double> &getLangevinPistonHalfStepPosition(void) const;

  /**
   * @brief Returns on-step piston velocities.
   *
   * @return Borrowed const alias to one, two, or three values in angstroms per
   * AKMA time.
   */
  const CudaContainer<double> &getLangevinPistonOnStepVelocity(void) const;

  /**
   * @brief Returns half-step piston velocity storage.
   *
   * @return Borrowed const alias to one, two, or three values.
   *
   * @warning The current propagation path allocates and zeroes this container
   * but does not update it.
   */
  const CudaContainer<double> &getLangevinPistonHalfStepVelocity(void) const;

  /** @brief Returns current piston position increments.
   * @return Borrowed const alias to angstrom-valued increments.
   */
  const CudaContainer<double> &getLangevinPistonDeltaPosition(void) const;

  /** @brief Returns previous piston position increments.
   * @return Borrowed const alias to angstrom-valued increments.
   */
  const CudaContainer<double> &
  getLangevinPistonDeltaPositionPrevious(void) const;

  /** @brief Returns predicted piston position increments.
   * @return Borrowed const alias to angstrom-valued increments.
   */
  const CudaContainer<double> &
  getLangevinPistonDeltaPositionPredicted(void) const;

  /**
   * @brief Returns piston delta-pressure values.
   *
   * @return Borrowed const alias to one, two, or three values in atmospheres
   * per angstrom.
   */
  const CudaContainer<double> &getLangevinPistonDeltaPressure(void) const;

  /**
   * @brief Returns the instantaneous pressure tensor.
   *
   * @return Borrowed const alias to nine row-major values in atmospheres.
   */
  const CudaContainer<double> &getInstantaneousPressureTensor(void) const;

  /**
   * @brief Returns the instantaneous scalar pressure.
   *
   * @return Borrowed const alias to one value in atmospheres.
   */
  const CudaContainer<double> &getInstantaneousPressureScalar(void) const;

  /**
   * @brief Returns the running-average pressure tensor.
   *
   * @return Borrowed const alias to nine row-major values in atmospheres.
   */
  const CudaContainer<double> &getAveragePressureTensor(void) const;

  /**
   * @brief Returns the running-average scalar pressure.
   *
   * @return Borrowed const alias to one value in atmospheres.
   */
  const CudaContainer<double> &getAveragePressureScalar(void) const;

  /** @brief Returns the piston RNG seed.
   * @return The dimensionless 64-bit seed.
   */
  std::uint64_t getLangevinPistonFrictionSeed(void) const;

  /**
   * @brief Returns the recorded piston RNG sequence position.
   *
   * @return A dimensionless position advanced once per predictor-corrector
   * iteration.
   */
  unsigned long long int getRngSequencePos(void) const;

  /**
   * @brief Copies piston RNG state to an opaque string.
   *
   * @return A newly allocated string containing the position and one Philox
   * state per active crystal degree of freedom.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if no crystal is configured or RNG
   * state is absent.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if stream
   * synchronization or device-to-host copying fails.
   * @throws std::bad_alloc If host state or result allocation fails.
   */
  std::string getRngStates(void) const;

  /** @brief Returns mutable Nose-Hoover coupling-mass storage.
   * @return Borrowed mutable alias to the one-element container.
   * @warning Preserve its one-element shape and host/device coherence.
   */
  CudaContainer<double> &getNoseHooverPistonMass(void);

  /** @brief Returns mutable current Nose-Hoover velocity storage.
   * @return Borrowed mutable alias to the one-element container.
   * @warning Preserve its one-element shape and host/device coherence.
   */
  CudaContainer<double> &getNoseHooverPistonVelocity(void);

  /** @brief Returns mutable previous Nose-Hoover velocity storage.
   * @return Borrowed mutable alias to the one-element container.
   * @warning Preserve its one-element shape and host/device coherence.
   */
  CudaContainer<double> &getNoseHooverPistonVelocityPrevious(void);

  /** @brief Returns mutable current Nose-Hoover increment storage.
   * @return Borrowed mutable alias to the one-element container.
   * @warning Preserve its one-element shape and host/device coherence.
   */
  CudaContainer<double> &getNoseHooverPistonForce(void);

  /** @brief Returns mutable previous Nose-Hoover increment storage.
   * @return Borrowed mutable alias to the one-element container.
   * @warning Preserve its one-element shape and host/device coherence.
   */
  CudaContainer<double> &getNoseHooverPistonForcePrevious(void);

  /** @brief Returns mutable two-estimator kinetic-energy storage.
   * @return Borrowed mutable alias to the two-element container.
   * @warning Preserve the estimator layout.
   */
  CudaContainer<double> &getKineticEnergy(void);

  /** @brief Returns mutable two-estimator running-temperature storage.
   * @return Borrowed mutable alias to the two-element container.
   * @warning Preserve the estimator layout.
   */
  CudaContainer<double> &getAverageTemperature(void);

  /**
   * @brief Returns the declared mutable old-temperature average container.
   *
   * @return A borrowed mutable container alias.
   *
   * @warning The current repository declares this function but provides no
   * definition. ODR-use causes a link failure.
   */
  CudaContainer<double> &getAverageOldTemperature(void);

  /**
   * @brief Computes the selected instantaneous temperature.
   *
   * @return Kinetic-energy element 1 in kelvin when old-temperature mode is
   * active, or element 0 in kelvin otherwise.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if no context is attached.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if kinetic
   * energy cannot be transferred to the host.
   */
  double getInstantaneousTemperature(void);

  /** @brief Returns mutable reference-pressure storage.
   * @return Borrowed mutable alias to nine row-major atmosphere values.
   * @warning Preserve the nine-element shape and host/device coherence.
   */
  CudaContainer<double> &getReferencePressureTensor(void);

  /** @brief Returns mutable Langevin-piston mass storage.
   * @return Borrowed mutable alias in active crystal-degree-of-freedom order.
   * @warning Preserve the active length and matching inverse-mass state.
   */
  CudaContainer<double> &getLangevinPistonMass(void);

  /** @brief Returns mutable on-step piston-position storage.
   * @return Borrowed mutable alias to angstrom-valued positions.
   */
  CudaContainer<double> &getLangevinPistonOnStepPosition(void);

  /** @brief Returns mutable half-step piston-position storage.
   * @return Borrowed mutable alias to angstrom-valued positions.
   */
  CudaContainer<double> &getLangevinPistonHalfStepPosition(void);

  /** @brief Returns mutable on-step piston-velocity storage.
   * @return Borrowed mutable alias to values in angstroms per AKMA time.
   */
  CudaContainer<double> &getLangevinPistonOnStepVelocity(void);

  /**
   * @brief Returns mutable half-step piston-velocity storage.
   *
   * @return Borrowed mutable alias to the allocated crystal-sized container.
   * @warning The current propagation path does not update this container.
   */
  CudaContainer<double> &getLangevinPistonHalfStepVelocity(void);

  /** @brief Returns mutable current piston-increment storage.
   * @return Borrowed mutable alias to angstrom-valued increments.
   */
  CudaContainer<double> &getLangevinPistonDeltaPosition(void);

  /** @brief Returns mutable previous piston-increment storage.
   * @return Borrowed mutable alias to angstrom-valued increments.
   */
  CudaContainer<double> &getLangevinPistonDeltaPositionPrevious(void);

  /** @brief Returns mutable predicted piston-increment storage.
   * @return Borrowed mutable alias to angstrom-valued increments.
   */
  CudaContainer<double> &getLangevinPistonDeltaPositionPredicted(void);

  /** @brief Returns mutable piston delta-pressure storage.
   * @return Borrowed mutable alias to atmosphere-per-angstrom values.
   */
  CudaContainer<double> &getLangevinPistonDeltaPressure(void);

  /** @brief Returns mutable instantaneous pressure-tensor storage.
   * @return Borrowed mutable alias to nine row-major atmosphere values.
   */
  CudaContainer<double> &getInstantaneousPressureTensor(void);

  /** @brief Returns mutable instantaneous scalar-pressure storage.
   * @return Borrowed mutable alias to one atmosphere value.
   */
  CudaContainer<double> &getInstantaneousPressureScalar(void);

  /** @brief Returns mutable average pressure-tensor storage.
   * @return Borrowed mutable alias to nine row-major atmosphere values.
   */
  CudaContainer<double> &getAveragePressureTensor(void);

  /** @brief Returns mutable average scalar-pressure storage.
   * @return Borrowed mutable alias to one atmosphere value.
   */
  CudaContainer<double> &getAveragePressureScalar(void);

  /**
   * @brief Computes the instantaneous planar surface tension.
   *
   * @return `0.5 * Lz * (Pzz - 0.5 * (Pxx + Pyy))` converted to dyne per
   * centimeter.
   *
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::NotInitialized` if no
   * context is attached.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if the pressure
   * tensor cannot be transferred to the host.
   */
  double getInstantaneousSurfaceTension(void);

protected:
  /**
   * @brief Validates the context crystal and box and initializes
   * pressure-controlled state.
   *
   * First attachment or a changed crystal layout allocates piston storage,
   * computes default crystal-sized masses, and resets piston/thermostat dynamic
   * history and phase averages. The scalar Nose-Hoover mass, reference
   * settings, friction coefficient, seed, and RNG sequence position are
   * preserved. RNG states and particle displacement history are rebuilt from
   * current state; particle velocities are retained. Configure custom
   * crystal-sized masses after this call. Reinitializing the same layout
   * preserves configured masses, friction, and the existing
   * extended-variable/average behavior.
   * A failed layout rebuild invalidates the derived piston-size marker, so
   * state access remains blocked until a subsequent initialization succeeds; a
   * failed reinitialization of the current layout leaves it usable.
   */
  void initializeImpl(void) override;

  /**
   * @brief Restores box, piston, thermostat, displacement, and RNG state from a
   * restart whose crystal type equals the context's.
   *
   * @param[in] rstFilePath Borrowed restart-file path. The path is not
   * canonicalized or retained.
   */
  void initializeFromRestartFileImpl(
      const std::filesystem::path &rstFilePath) override;

  /** @brief Executes one synchronized pressure-controlled step. */
  void propagateOneStepImpl(void) override;

protected:
  /**
   * @brief Copies decoded piston RNG state to device storage.
   *
   * @param[in] position Dimensionless sequence position.
   * @param[in] states Borrowed states copied during the call.
   */
  void setRngStateData(const unsigned long long int position,
                       const std::vector<curandStatePhilox4_32_10_t> &states);

  /**
   * @brief Computes the default numeric Nose-Hoover coupling mass.
   *
   * @return `0.2` times the sum of context atom masses.
   */
  double computeNoseHooverPistonMass(void);

  /**
   * @brief Computes the default numeric Langevin-piston mass.
   *
   * @return `0.02` times the sum of context atom masses.
   */
  double computeLangevinPistonMass(void);

  /**
   * @brief Allocates and resets every crystal-sized piston container.
   *
   * @pre The piston degree-of-freedom count is one, two, or three.
   */
  void allocateLangevinPistonVariables(void);

  /**
   * @brief Validates context crystal/box and derives the required piston size.
   *
   * @return One, two, or three for cubic, tetragonal, or orthorhombic symmetry.
   * @throws ApoCharmmError If the context is absent, its crystal is unset, or
   * the finite positive box dimensions do not match the declared symmetry.
   */
  int getContextPistonDegreesOfFreedom(void) const;

  /**
   * @brief Rejects state cached for a different context crystal layout.
   *
   * Before attachment this is a no-op, preserving constructor inspection and
   * scalar configuration. A stale or failed layout throws NotInitialized with
   * instructions to reinitialize. Base history and lifecycle entry points
   * dispatch this override, including calls made through a base-class handle.
   */
  void requireCurrentState(void) const override;

  /** @brief Initializes one Philox state per piston degree of freedom. */
  void initializeRng(void);

  /** @brief Removes mass-weighted center-of-mass displacement. */
  void removeCenterOfMassMotion(void);

  /**
   * @brief Allocates raw device storage for `n` piston RNG states.
   *
   * @param[in] n Number of states.
   */
  void alloc(const int n);

  /** @brief Releases raw piston RNG-state storage. */
  void dealloc(void);

protected:
  /** @brief Selects whether predictor-corrector updates include Nose-Hoover. */
  bool m_UsingNoseHooverThermostat;

  /** @brief Stores the reference temperature in kelvin. */
  double m_ReferenceTemperature;

  /** @brief Stores the one-element Nose-Hoover coupling mass. */
  CudaContainer<double> m_NoseHooverPistonMass;

  /** @brief Stores the current one-element Nose-Hoover velocity. */
  CudaContainer<double> m_NoseHooverPistonVelocity;

  /** @brief Stores the previous Nose-Hoover velocity. */
  CudaContainer<double> m_NoseHooverPistonVelocityPrevious;

  /** @brief Stores the current Nose-Hoover velocity increment. */
  CudaContainer<double> m_NoseHooverPistonForce;

  /** @brief Stores the previous Nose-Hoover velocity increment. */
  CudaContainer<double> m_NoseHooverPistonForcePrevious;

  /** @brief Stores one predicted displacement record per context atom. */
  CudaContainer<double4> m_CoordsDeltaPredicted;

  /** @brief Stores per-atom holonomic-constraint forces. */
  CudaContainer<double4> m_HolonomicConstraintForces;

  /**
   * @brief Stores component-major pressure-related reduction partial sums.
   *
   * For a reduction with `numBlocks` first-pass blocks, component `c` and block
   * `b` are stored at `c * numBlocks + b`. The same workspace is reused
   * sequentially for the constraint virial, average kinetic pressure, and delta
   * kinetic pressure.
   */
  CudaContainer<double> m_PressureReductionPartialSums;

  /** @brief Stores the nine-element constraint virial tensor. */
  CudaContainer<double> m_HolonomicConstraintVirial;

  /** @brief Stores the nine-element kinetic pressure tensor. */
  CudaContainer<double> m_KineticPressureTensor;

  /** @brief Stores the instantaneous nine-element pressure tensor in atm. */
  CudaContainer<double> m_PressureTensor;

  /** @brief Stores the instantaneous scalar pressure in atm. */
  CudaContainer<double> m_PressureScalar;

  /** @brief Stores the nine-element reference pressure tensor in atm. */
  CudaContainer<double> m_ReferencePressureTensor;

  /** @brief Stores pressure minus reference pressure in atm. */
  CudaContainer<double> m_DeltaPressureTensor;

  /** @brief Stores the iterative kinetic contribution to delta pressure. */
  CudaContainer<double> m_DeltaKineticPressureTensor;

  /** @brief Stores the predictor-corrector-invariant delta-pressure part. */
  CudaContainer<double> m_StaticDeltaPressureTensor;

  /** @brief Stores the active crystal degree-of-freedom count. */
  int m_LangevinPistonDegreesOfFreedom;

  /** @brief Stores crystal-sized Langevin-piston masses. */
  CudaContainer<double> m_LangevinPistonMass;

  /** @brief Stores reciprocal piston masses, with zero preserved as zero. */
  CudaContainer<double> m_LangevinPistonInverseMass;

  /** @brief Stores on-step piston positions in angstroms. */
  CudaContainer<double> m_LangevinPistonOnStepPosition;

  /** @brief Stores half-step piston positions in angstroms. */
  CudaContainer<double> m_LangevinPistonHalfStepPosition;

  /** @brief Stores on-step piston velocities in angstroms per AKMA time. */
  CudaContainer<double> m_LangevinPistonOnStepVelocity;

  /**
   * @brief Stores allocated half-step piston velocity values.
   *
   * Current propagation does not update this container after reset.
   */
  CudaContainer<double> m_LangevinPistonHalfStepVelocity;

  /** @brief Stores current piston position increments in angstroms. */
  CudaContainer<double> m_LangevinPistonDeltaPosition;

  /** @brief Stores previous piston position increments in angstroms. */
  CudaContainer<double> m_LangevinPistonDeltaPositionPrevious;

  /** @brief Stores predicted piston position increments in angstroms. */
  CudaContainer<double> m_LangevinPistonDeltaPositionPredicted;

  /** @brief Stores piston delta pressures in atmospheres per angstrom. */
  CudaContainer<double> m_LangevinPistonDeltaPressure;

  /** @brief Stores public piston friction in inverse picoseconds. */
  double m_Pgamma;

  /** @brief Stores the dimensionless BBK damping coefficient. */
  double m_Palpha;

  /** @brief Stores the time-step-derived BBK force factor. */
  double m_Pbfact;

  /** @brief Stores crystal-sized stochastic forcing amplitudes. */
  CudaContainer<double> m_Prfwd;

  /** @brief Stores three on-step crystal scaling factors. */
  CudaContainer<double> m_OnStepCrystalFactor;

  /** @brief Stores three half-step crystal scaling factors. */
  CudaContainer<double> m_HalfStepCrystalFactor;

  /** @brief Stores the dimensionless 64-bit Philox seed. */
  std::uint64_t m_Seed;

  /** @brief Stores the recorded dimensionless Philox sequence position. */
  unsigned long long int m_RngSequencePos;

  /** @brief Owns one raw device RNG state per crystal degree of freedom. */
  curandStatePhilox4_32_10_t *m_RngStates;

  /** @brief Selects the constant-surface-tension pressure branch. */
  bool m_ConstantSurfaceTensionFlag;

  /**
   * @brief Stores the one-element target surface tension.
   *
   * The current constructor allocates but does not initialize this value, and
   * no public target setter exists.
   */
  CudaContainer<double> m_SurfaceTension;

  /** @brief Stores the positive predictor-corrector iteration count. */
  int m_MaxPredictorCorrectorIterations;

  /** @brief Stores the number of samples accumulated into running averages. */
  int m_AverageWindowSize;

  /** @brief Stores two kinetic-energy estimators in kcal/mol. */
  CudaContainer<double> m_KineticEnergy;

  /** @brief Stores per-block partial sums for both kinetic estimators. */
  CudaContainer<double> m_KineticEnergyPartialSums;

  /** @brief Stores two running-temperature estimators in kelvin. */
  CudaContainer<double> m_AverageTemperature;

  /** @brief Stores the nine-element running-average pressure tensor in atm. */
  CudaContainer<double> m_AveragePressureTensor;

  /** @brief Stores the running-average scalar pressure in atm. */
  CudaContainer<double> m_AveragePressureScalar;

  /** @brief Selects kinetic and temperature estimator element 1. */
  bool m_UsingOldTemperature;
};
