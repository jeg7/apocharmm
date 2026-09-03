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

#include "ApoCharmmError.h"
#include "CharmmPSF.h"
#include "CharmmParameters.h"
#include "CudaBondedForce.h"
#include "CudaContainer.h"
#include "CudaEnergyVirial.h"
#include "CudaPMEDirectForce.h"
#include "CudaPMEReciprocalForce.h"
#include "Force.h"
#include "PBC.h"

#include <cstddef>
#include <cuda_runtime.h>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

// Forward declaration
class CharmmContext;

/**
 * @brief Dispatches a subscribed force through a non-owning type-erased view.
 *
 * `ForceManager` uses this helper to invoke a force implementation without
 * requiring a common virtual base class. The view stores a raw pointer to the
 * supplied object together with function pointers generated from `ForceType`.
 * It does not own the force.
 *
 * A force accepted by this view must provide:
 *
 * - `static constexpr bool contributesVirial`;
 * - `initialize(int, const std::vector<double> &)`;
 * - `clear()`;
 * - `calcForce(const float4 *, const bool, const bool)`;
 * - `setBoxDimensions(const std::vector<double> &)`;
 * - `getForce()` returning `std::shared_ptr<Force<long long int>>`; and
 * - `getEnergyVirial()` returning `std::shared_ptr<CudaEnergyVirial>`.
 *
 * `ForceManager` preserves the pointee lifetime while a force is subscribed by
 * retaining a shared owner in its parallel subscription state.
 *
 * @warning A standalone `ForceView` does not validate or extend the pointee
 * lifetime. Destroying or moving the pointee while the view remains in use
 * leaves the view invalid.
 * @warning This helper and the object it dispatches provide no internal
 * synchronization for concurrent host access.
 */
class ForceView {
public:
  /**
   * @brief Constructs a view over a concrete force object.
   *
   * The constructor captures `ForceType`'s compile-time virial trait and
   * creates dispatch functions for the required force interface. No data is
   * copied from the force and no ownership is acquired.
   *
   * @tparam ForceType Concrete subscribed-force implementation satisfying the
   * interface documented for `ForceView`.
   * @param[in] inputForce Borrowed non-null pointer to the force. The pointee
   * must remain alive and at the same address for every subsequent operation
   * on this view.
   *
   * @pre `inputForce` is non-null.
   * @pre `ForceType` satisfies the required subscribed-force interface.
   */
  template <typename ForceType>
  ForceView(ForceType *inputForce)
      : m_Force(static_cast<void *>(inputForce)),
        m_ContributesVirial(ForceType::contributesVirial),
        initialize_impl{[](void *force, const int numAtoms,
                           const std::vector<double> &boxDimensions) -> void {
          ForceType *ptr = static_cast<ForceType *>(force);
          return ptr->initialize(numAtoms, boxDimensions);
        }},
        clear_impl{[](void *force) -> void {
          ForceType *ptr = static_cast<ForceType *>(force);
          return ptr->clear();
        }},
        calcForce_impl{[](void *force, const float4 *xyzq,
                          const bool calcEnergy,
                          const bool calcVirial) -> void {
          ForceType *ptr = static_cast<ForceType *>(force);
          return ptr->calcForce(xyzq, calcEnergy, calcVirial);
        }},
        setBoxDimensions_impl{
            [](void *force, const std::vector<double> &boxDimensions) -> void {
              ForceType *ptr = static_cast<ForceType *>(force);
              return ptr->setBoxDimensions(boxDimensions);
            }},
        getForce_impl{[](void *force) -> std::shared_ptr<Force<long long int>> {
          ForceType *ptr = static_cast<ForceType *>(force);
          return ptr->getForce();
        }},
        getEnergyVirial_impl{
            [](void *force) -> std::shared_ptr<CudaEnergyVirial> {
              ForceType *ptr = static_cast<ForceType *>(force);
              return ptr->getEnergyVirial();
            }} {}

  /**
   * @brief Initializes the viewed force for an atom count and periodic box.
   *
   * @param[in] numAtoms Number of atoms represented by subsequent coordinate
   * and force arrays.
   * @param[in] boxDimensions Three box lengths in `[x, y, z]` order, in
   * angstroms.
   *
   * @pre The borrowed force pointer remains valid.
   * @note All validation, allocation, host-to-device transfer, and
   * synchronization behavior is defined by `ForceType::initialize()`.
   */
  void initialize(const int numAtoms,
                  const std::vector<double> &boxDimensions) {
    return this->initialize_impl(m_Force, numAtoms, boxDimensions);
  }

  /**
   * @brief Clears the viewed force's accumulated output.
   *
   * @pre The borrowed force pointer remains valid.
   * @note CUDA stream and synchronization behavior is defined by
   * `ForceType::clear()`.
   */
  void clear(void) { return this->clear_impl(m_Force); }

  /**
   * @brief Computes the viewed force for a device coordinate-charge array.
   *
   * @param[in] xyzq Borrowed device pointer to at least `numAtoms` `float4`
   * records in `[x, y, z, charge]` order. Coordinates are in angstroms and
   * charge is in elementary-charge units. The pointer must remain valid until
   * the viewed force has completed its CUDA work.
   * @param[in] calcEnergy Whether to update the viewed force's energy state.
   * @param[in] calcVirial Whether to update the viewed force's virial state.
   *
   * @pre The borrowed force pointer remains valid.
   * @pre The force has been initialized for the supplied array.
   * @note CUDA launch and synchronization behavior is defined by
   * `ForceType::calcForce()`.
   */
  void calcForce(const float4 *xyzq, const bool calcEnergy,
                 const bool calcVirial) {
    return this->calcForce_impl(m_Force, xyzq, calcEnergy, calcVirial);
  }

  /**
   * @brief Updates the viewed force's box dimensions.
   *
   * @param[in] boxDimensions Three box lengths in `[x, y, z]` order, in
   * angstroms.
   *
   * @pre The borrowed force pointer remains valid.
   * @note Validation, transfer, and synchronization behavior is defined by
   * `ForceType::setBoxDimensions()`.
   */
  void setBoxDimensions(const std::vector<double> &boxDimensions) {
    return this->setBoxDimensions_impl(m_Force, boxDimensions);
  }

  /**
   * @brief Returns the viewed force's fixed-point force storage.
   *
   * @return A copied shared owner of the `Force<long long int>` object returned
   * by `ForceType::getForce()`. Its device array uses structure-of-arrays
   * component layout with the object's reported stride.
   *
   * @pre The borrowed force pointer remains valid.
   */
  std::shared_ptr<Force<long long int>> getForce(void) {
    return this->getForce_impl(m_Force);
  }

  /**
   * @brief Returns the viewed force's energy and virial storage.
   *
   * @return A copied shared owner of the `CudaEnergyVirial` object returned by
   * `ForceType::getEnergyVirial()`.
   *
   * @pre The borrowed force pointer remains valid.
   */
  std::shared_ptr<CudaEnergyVirial> getEnergyVirial(void) const {
    return this->getEnergyVirial_impl(m_Force);
  }

  /**
   * @brief Reports whether the viewed force contributes a virial.
   *
   * @return The value of `ForceType::contributesVirial` captured when the view
   * was constructed.
   */
  bool contributesVirial(void) const { return m_ContributesVirial; }

private:
  /** Borrowed, type-erased pointer to the concrete subscribed force. */
  void *m_Force;

  /** Compile-time virial capability captured from the concrete force type. */
  bool m_ContributesVirial;

  /**
   * Type-erased dispatch table. Every entry expects `m_Force` to still point
   * to an object of the exact `ForceType` used by the constructor.
   */
  void (*initialize_impl)(void *force, const int numAtoms,
                          const std::vector<double> &boxDimensions);
  void (*clear_impl)(void *force);
  void (*calcForce_impl)(void *force, const float4 *xyzq, bool calcEnergy,
                         bool calcVirial);
  void (*setBoxDimensions_impl)(void *force,
                                const std::vector<double> &boxDimensions);
  std::shared_ptr<Force<long long int>> (*getForce_impl)(void *force);
  std::shared_ptr<CudaEnergyVirial> (*getEnergyVirial_impl)(void *force);
};

/**
 * @brief Owns and coordinates native GPU force-evaluation backends.
 *
 * A `ForceManager` combines a @ref CharmmPSF and @ref CharmmParameters with
 * box, Ewald, cutoff, FFT-grid, spline, periodic-boundary, and van der Waals
 * configuration. Initialization constructs the bonded, direct-space, and
 * reciprocal-space CUDA backends, creates their streams and force storage,
 * initializes already subscribed forces, and allocates aggregate force,
 * energy, virial, and holonomic-constraint state.
 *
 * The manager retains shared ownership of its PSF and parameter set. Its
 * association with a @ref CharmmContext is weak and does not extend the
 * context's lifetime. Subscribed force objects and their associated streams,
 * force arrays, and energy-virial objects are retained through shared
 * ownership until they are unsubscribed or the manager is destroyed.
 *
 * Before initialization, callers must supply a PSF, a parameter set, and three
 * positive box lengths. Configuration setters should normally be called before
 * the first initialization. Several setters update stored configuration
 * without rebuilding an already active backend.
 *
 * Force calculations consume a device-resident `float4` coordinate-charge
 * array and produce device-resident structure-of-arrays force storage.
 * Component calculations use separate CUDA streams and are synchronized before
 * the aggregate force is returned as complete.
 *
 * The base class is non-composite. Derived managers may override the child,
 * initialization, force, and potential-energy extension points.
 *
 * @warning `ForceManager` provides no internal synchronization for concurrent
 * host access. Configuration, subscription, initialization, force evaluation,
 * and mutable accessor use must be externally serialized.
 * @warning Initialization is not transactional. A failure can leave partially
 * allocated native or CUDA state while `isInitialized()` remains `false`.
 * @see force_manager
 */
class ForceManager : public std::enable_shared_from_this<ForceManager> {
public:
  /**
   * @brief Constructs an uninitialized manager with default configuration.
   *
   * The manager initially has no PSF, no parameter set, no context, no CUDA
   * force backends, and sentinel box dimensions. Its defaults are:
   *
   * - Ewald splitting parameter `0.34` inverse angstroms;
   * - pair-list cutoff `14.0` angstroms;
   * - outer switching distance exposed as `ctonnb`, `12.0` angstroms;
   * - inner switching distance exposed as `ctofnb`, `10.0` angstroms;
   * - automatic FFT dimensions, represented by `-1` on each axis;
   * - PME spline order `4`;
   * - periodic boundary condition @ref PBC::P1;
   * - van der Waals model `VDW_VFSW`; and
   * - disabled energy-decomposition printing.
   *
   * Five nine-element virial work containers are allocated and initialized to
   * zero during construction.
   *
   * @post `isInitialized()` returns `false`.
   * @throws ApoCharmmError With code @ref ApoCharmmErrorCode::Cuda if virial
   * storage allocation or initialization fails in CUDA.
   * @throws std::bad_alloc If host-side object or container allocation fails.
   */
  ForceManager(void);

  /**
   * @brief Constructs an uninitialized manager from a PSF and parameter set.
   *
   * The manager retains shared ownership of both collaborators; it does not
   * copy either object. CUDA force backends are not created until
   * `initialize()` is called directly or through a `CharmmContext`.
   *
   * @param[in] psf Shared owner of a non-null PSF. The manager retains a copy
   * of the shared pointer.
   * @param[in] prm Shared owner of a non-null parameter set. The manager
   * retains a copy of the shared pointer.
   *
   * @post `getPsf()` and `getPrm()` share ownership with the supplied objects.
   * @post `isInitialized()` returns `false`.
   * @throws ApoCharmmError With code
   * @ref ApoCharmmErrorCode::InvalidArgument if `psf` or `prm` is null.
   * @throws ApoCharmmError With code @ref ApoCharmmErrorCode::Cuda if default
   * virial storage construction fails.
   * @throws std::bad_alloc If host-side allocation fails.
   */
  ForceManager(std::shared_ptr<CharmmPSF> psf,
               std::shared_ptr<CharmmParameters> prm);

  /**
   * @brief Constructs a configuration-only deep copy of another manager.
   *
   * A non-null PSF and parameter set are copied into independent
   * `CharmmPSF` and `CharmmParameters` objects. Box dimensions, Ewald and
   * cutoff values, FFT dimensions, spline order, periodic-boundary condition,
   * and van der Waals model are copied.
   *
   * The context association, initialized state, CUDA streams, CUDA backends,
   * force arrays, energy and virial values, subscribed forces, children, CUDA
   * clear graph, and print-energy flag are not copied. They retain the default
   * constructor state in the new manager.
   *
   * @param[in] other Manager whose configuration is copied. The source remains
   * unchanged.
   *
   * @post The new manager is uninitialized and has no associated context.
   * @post Any copied PSF and parameter set are independent of `other`.
   * @throws ApoCharmmError With code @ref ApoCharmmErrorCode::Cuda if default
   * virial storage construction or a copied CUDA-backed collaborator fails.
   * @throws std::bad_alloc If collaborator or container allocation fails.
   */
  ForceManager(const ForceManager &other);

  /**
   * @brief Destroys the manager and releases its owned native resources.
   *
   * CUDA streams and cached graph objects are destroyed through non-throwing
   * cleanup paths. Cleanup failures are discarded. Shared ownership of the PSF,
   * parameter set, subscribed-force resources, and force arrays is released.
   *
   * A separately retained PSF, parameter set, subscribed force, or force-array
   * shared pointer may outlive the manager. A shared pointer returned by a
   * stream getter may keep the host `cudaStream_t` value allocated, but the
   * manager still destroys the underlying CUDA stream, so that handle must not
   * be used after manager destruction.
   *
   * @post Borrowed references returned by this manager are invalid.
   */
  virtual ~ForceManager(void) noexcept;

public:
  /**
   * @brief Stores a non-owning association with a `CharmmContext`.
   *
   * The manager stores `ctx` as a `std::weak_ptr`. Passing `nullptr` clears the
   * association. This method does not modify the supplied context and does not
   * call `CharmmContext::setForceManager()`.
   *
   * @param[in] ctx Context whose control block is observed. The pointer may be
   * null and is not retained through shared ownership.
   *
   * @post `getContext()` returns a shared owner when the context is still
   * alive, or an empty pointer after the association is cleared or expires.
   */
  void setContext(std::shared_ptr<CharmmContext> ctx);

  /**
   * @brief Sets the PSF retained by the manager.
   *
   * The supplied object is retained through shared ownership and the manager's
   * initialized flag is cleared. Existing CUDA streams, backends, cached graph
   * state, and force arrays are not deallocated or rebuilt by this call.
   *
   * @param[in] psf Shared owner of a non-null PSF.
   *
   * @post `getPsf()` shares ownership with `psf`.
   * @post `isInitialized()` returns `false`.
   * @throws ApoCharmmError With code
   * @ref ApoCharmmErrorCode::InvalidArgument if `psf` is null. The existing
   * PSF and initialized flag remain unchanged on this validation failure.
   *
   * @warning Configure the PSF before the first initialization. The current
   * implementation does not provide a complete in-place backend rebuild after
   * replacing an initialized manager's PSF.
   */
  void setPsf(std::shared_ptr<CharmmPSF> psf);

  /**
   * @brief Sets the parameter set retained by the manager.
   *
   * The supplied object is retained through shared ownership and the manager's
   * initialized flag is cleared. Existing CUDA streams, backends, cached graph
   * state, and force arrays are not deallocated or rebuilt by this call.
   *
   * @param[in] prm Shared owner of a non-null parameter set.
   *
   * @post `getPrm()` shares ownership with `prm`.
   * @post `isInitialized()` returns `false`.
   * @throws ApoCharmmError With code
   * @ref ApoCharmmErrorCode::InvalidArgument if `prm` is null. The existing
   * parameter set and initialized flag remain unchanged on this validation
   * failure.
   *
   * @warning Configure the parameter set before the first initialization.
   */
  void setPrm(std::shared_ptr<CharmmParameters> prm);

  /**
   * @brief Loads and installs a PSF from a file.
   *
   * A new `CharmmPSF` is constructed from `psfFile`. The existing PSF is
   * replaced only after construction succeeds, then the initialized flag is
   * cleared.
   *
   * @param[in] psfPath Non-empty file-system path to the CHARMM PSF file.
   *
   * @post On success, `getPsf()` owns the newly parsed PSF and
   * `isInitialized()` returns `false`.
   * @throws ApoCharmmError With code @ref ApoCharmmErrorCode::Runtime if the
   * file cannot be read or its PSF records are invalid.
   * @throws ApoCharmmError With code @ref ApoCharmmErrorCode::Cuda if
   * construction of CUDA-backed PSF data fails.
   * @throws std::bad_alloc If parsing or collaborator allocation fails.
   *
   * @warning Existing initialized CUDA force state is not deallocated by this
   * call.
   */
  void addPsf(const std::filesystem::path &psfPath);

  /**
   * @brief Loads and installs parameters from one file.
   *
   * A new `CharmmParameters` is constructed from `prmFile`. The existing
   * parameter set is replaced only after construction succeeds, then the
   * initialized flag is cleared.
   *
   * @param[in] prmPath Non-empty file-system path to a CHARMM parameter or
   * stream file.
   *
   * @post On success, `getPrm()` owns the newly parsed parameter set and
   * `isInitialized()` returns `false`.
   * @throws ApoCharmmError With code @ref ApoCharmmErrorCode::Runtime if the
   * file cannot be read or a parameter record is invalid.
   * @throws std::bad_alloc If parsing or collaborator allocation fails.
   *
   * @warning Existing initialized CUDA force state is not deallocated by this
   * call.
   */
  void addPrm(const std::filesystem::path &prmPath);

  /**
   * @brief Loads and installs parameters from an ordered list of files.
   *
   * Files are parsed in `prmList` order into one new `CharmmParameters` object.
   * The existing parameter set is replaced only after all files are parsed
   * successfully, then the initialized flag is cleared.
   *
   * @param[in] prmList Non-empty ordered list of file-system paths to CHARMM
   * parameter or stream files.
   *
   * @post On success, `getPrm()` owns the newly parsed parameter set and
   * `isInitialized()` returns `false`.
   * @throws ApoCharmmError With code
   * @ref ApoCharmmErrorCode::InvalidArgument if `prmList` is empty.
   * @throws ApoCharmmError With code @ref ApoCharmmErrorCode::Runtime if a file
   * cannot be read or a parameter record is invalid.
   * @throws std::bad_alloc If parsing or collaborator allocation fails.
   *
   * @warning Existing initialized CUDA force state is not deallocated by this
   * call.
   */
  void addPrm(const std::vector<std::filesystem::path> &prmList);

  /**
   * @brief Sets the orthorhombic box dimensions.
   *
   * The three values are copied into the manager's double-precision box vector
   * and narrowed to the float scalar fields used by the native force backends.
   * Existing bonded, reciprocal, direct, and subscribed force objects receive
   * the new box in that order.
   *
   * This call does not recompute an automatic FFT grid, reset the direct-space
   * neighbor list, or clear the initialized flag.
   *
   * @param[in] size Exactly three finite positive lengths in `[x, y, z]` order,
   * in angstroms.
   *
   * @post On successful return, `getBoxDimensions()` equals `size`.
   * @throws ApoCharmmError With code
   * @ref ApoCharmmErrorCode::InvalidArgument if `size` does not contain exactly
   * three values, or if any value is non-finite or not positive.
   * @throws ApoCharmmError Propagates a categorized error from an already
   * constructed native or subscribed force receiving the box update.
   *
   * @warning Propagation is not transactional. A failure from a downstream
   * force occurs after the manager's stored box has changed and after any
   * earlier force objects have accepted the update.
   */
  virtual void setBoxDimensions(const std::vector<double> &size);

  /**
   * @brief Sets the Ewald splitting parameter.
   *
   * @param[in] kappa Finite non-negative Ewald parameter in inverse angstroms.
   *
   * @post `getKappa()` returns `kappa`, subject to float representation.
   * @throws ApoCharmmError With code
   * @ref ApoCharmmErrorCode::InvalidArgument if `kappa` is non-finite or
   * negative.
   *
   * @warning This setter updates stored configuration only. It does not
   * reconfigure an already constructed reciprocal or direct backend.
   */
  virtual void setKappa(const float kappa);

  /**
   * @brief Sets the direct-space pair-list cutoff.
   *
   * @param[in] cutoff Finite positive distance in angstroms.
   *
   * @post `getCutoff()` returns `cutoff`.
   * @throws ApoCharmmError With code
   * @ref ApoCharmmErrorCode::InvalidArgument if `cutoff` is non-finite or not
   * positive.
   *
   * @note `initialize()` additionally requires this value not to exceed half
   * the X box length.
   * @warning This setter does not update an already constructed direct-space
   * backend or neighbor list.
   */
  virtual void setCutoff(const float cutoff);

  /**
   * @brief Sets the outer nonbonded switching distance exposed as `ctonnb`.
   *
   * The current backend forwarding uses this value as the distance at which
   * the switching function reaches zero.
   *
   * @param[in] ctonnb Finite positive distance in angstroms.
   *
   * @post `getCtonnb()` returns `ctonnb`.
   * @throws ApoCharmmError With code
   * @ref ApoCharmmErrorCode::InvalidArgument if `ctonnb` is non-finite or not
   * positive.
   *
   * @note The implementation does not validate the ordering of `ctonnb` and
   * `ctofnb`.
   * @warning This setter does not reconfigure an already constructed
   * direct-space backend.
   */
  virtual void setCtonnb(const float ctonnb);

  /**
   * @brief Sets the inner nonbonded switching distance exposed as `ctofnb`.
   *
   * The current backend forwarding uses this value as the distance at which
   * switching begins.
   *
   * @param[in] ctofnb Finite positive distance in angstroms.
   *
   * @post `getCtofnb()` returns `ctofnb`.
   * @throws ApoCharmmError With code
   * @ref ApoCharmmErrorCode::InvalidArgument if `ctofnb` is non-finite or not
   * positive.
   *
   * @note The implementation does not validate the ordering of `ctonnb` and
   * `ctofnb`.
   * @warning This setter does not reconfigure an already constructed
   * direct-space backend.
   */
  virtual void setCtofnb(const float ctofnb);

  /**
   * @brief Sets the three PME FFT grid dimensions.
   *
   * @param[in] nfftx Positive dimensionless grid size along X.
   * @param[in] nffty Positive dimensionless grid size along Y.
   * @param[in] nfftz Positive dimensionless grid size along Z.
   *
   * @post `getFFTGrid()` returns `{nfftx, nffty, nfftz}`.
   * @throws ApoCharmmError With code
   * @ref ApoCharmmErrorCode::InvalidArgument if any dimension is not positive.
   *
   * @warning This setter does not rebuild an already constructed reciprocal
   * backend.
   */
  virtual void setFFTGrid(const int nfftx, const int nffty, const int nfftz);

  /**
   * @brief Sets the PME interpolation spline order.
   *
   * @param[in] pmeSplineOrder Positive dimensionless spline order.
   *
   * @post `getPmeSplineOrder()` returns `pmeSplineOrder`.
   * @throws ApoCharmmError With code
   * @ref ApoCharmmErrorCode::InvalidArgument if `pmeSplineOrder` is not
   * positive.
   *
   * @warning This setter does not rebuild an already constructed reciprocal
   * backend.
   */
  virtual void setPmeSplineOrder(const int pmeSplineOrder);

  /**
   * @brief Sets the periodic boundary condition.
   *
   * @param[in] _pbc Declared @ref PBC value to retain.
   *
   * @post `getPeriodicBoundaryCondition()` returns `_pbc`.
   * @post `isInitialized()` returns `false`.
   *
   * @warning The native C++ setter does not validate values produced by casting
   * an arbitrary integer to `PBC`.
   * @warning Existing CUDA backend state is not deallocated by this call.
   */
  virtual void setPeriodicBoundaryCondition(const PBC pbc);

  /**
   * @brief Sets the native van der Waals model code.
   *
   * @param[in] vdwType Integer model code from `VDW_VSH` through `VDW_DBEXP`,
   * inclusive, corresponding to values `1` through `6`.
   *
   * @post `getVdwType()` returns `vdwType`.
   * @throws ApoCharmmError With code
   * @ref ApoCharmmErrorCode::InvalidArgument if `vdwType` is outside
   * `[1, 6]`.
   *
   * @warning This setter does not rebuild an already constructed direct-space
   * backend.
   */
  virtual void setVdwType(const int vdwType);

  /**
   * @brief Enables or disables energy-decomposition printing.
   *
   * When enabled, `calcForce()` writes standard and subscribed energy
   * components to `std::cout` whenever `calcEnergy` is `true`.
   *
   * @param[in] printEnergyDecomposition Whether to print energy terms. Omitting
   * the argument enables printing.
   *
   * @post The selected flag is used by subsequent energy-producing force
   * calculations.
   */
  void setPrintEnergyDecomposition(const bool printEnergyDecomposition = true);

  /**
   * @brief Rejects child managers in the non-composite base implementation.
   *
   * Derived composite managers override this extension point.
   *
   * @param[in] fm Proposed child manager.
   *
   * @throws ApoCharmmError With code
   * @ref ApoCharmmErrorCode::InvalidArgument if `fm` is null.
   * @throws ApoCharmmError With code @ref ApoCharmmErrorCode::Runtime for every
   * non-null `fm`, because the base manager does not support children.
   */
  virtual void addForceManager(std::shared_ptr<ForceManager> fm);

public:
  /**
   * @brief Returns the associated `CharmmContext` when it is still alive.
   *
   * @return A newly acquired shared owner of the associated context, or an
   * empty `std::shared_ptr` when the weak association is unset or expired.
   */
  std::shared_ptr<CharmmContext> getContext(void);

  /**
   * @brief Reports whether the associated `CharmmContext` is still alive.
   *
   * @return `true` when the stored weak reference is not expired; otherwise
   * `false`.
   *
   * @note The result can become stale immediately if another thread releases
   * the final context owner. `ForceManager` provides no internal locking.
   */
  bool hasCharmmContext(void) const;

  /**
   * @brief Returns the retained PSF.
   *
   * @return A copied shared owner of the current PSF, or an empty pointer when
   * no PSF has been set. The returned owner may outlive the manager.
   */
  virtual std::shared_ptr<CharmmPSF> getPsf(void);

  /**
   * @brief Returns the retained parameter set.
   *
   * @return A copied shared owner of the current parameter set, or an empty
   * pointer when no parameters have been set. The returned owner may outlive
   * the manager.
   */
  std::shared_ptr<CharmmParameters> getPrm(void);

  /**
   * @brief Reports whether native force initialization completed.
   *
   * @return `true` only after `initialize()` reaches its successful final
   * assignment; otherwise `false`.
   *
   * @note A `false` result does not imply that no partial CUDA resources were
   * allocated by an earlier failed initialization.
   */
  virtual bool isInitialized(void) const;

  /**
   * @brief Returns read-only SHAKE atom-index records.
   *
   * Each `int4` record represents one heavy-atom-centered constrained group:
   * `(heavy, hydrogen1, hydrogen2, hydrogen3)`. Unused hydrogen positions are
   * `-1`. The container is populated by `initialize()` and may be empty.
   *
   * @return A borrowed read-only alias of manager-owned host/device mirrored
   * storage. It remains valid until the manager is destroyed or the container
   * is reassigned by another initialization.
   */
  const CudaContainer<int4> &getShakeAtoms(void) const;

  /**
   * @brief Returns mutable SHAKE atom-index records.
   *
   * @return A borrowed mutable alias with the layout documented by the const
   * overload.
   *
   * @warning Direct mutation bypasses topology validation and can make
   * host/device constraint state inconsistent with the PSF.
   */
  CudaContainer<int4> &getShakeAtoms(void);

  /**
   * @brief Returns read-only SHAKE parameter records.
   *
   * Each `float4` contains the solver fields
   * `(inverse-heavy-mass, average-mass-field, squared-bond-length,
   * inverse-hydrogen-mass)`. Reciprocal masses use inverse atomic mass units,
   * the average-mass field uses atomic mass units, and the squared bond length
   * uses square angstroms. The representation is solver-internal and is not a
   * stable interchange format.
   *
   * @return A borrowed read-only alias of manager-owned host/device mirrored
   * storage. It remains valid until the manager is destroyed or the container
   * is reassigned by another initialization.
   */
  const CudaContainer<float4> &getShakeParams(void) const;

  /**
   * @brief Returns mutable SHAKE parameter records.
   *
   * @return A borrowed mutable alias with the layout documented by the const
   * overload.
   *
   * @warning Direct mutation bypasses mass, topology, and bond-parameter
   * validation.
   */
  CudaContainer<float4> &getShakeParams(void);

  /**
   * @brief Returns read-only bonded energy and virial state.
   *
   * @return A borrowed read-only alias valid until manager destruction.
   *
   * @note The call performs no transfer or synchronization. Host energy values
   * are current only after an energy-producing calculation has copied them.
   */
  const CudaEnergyVirial &getBondedEnergyVirial(void) const;

  /**
   * @brief Returns mutable bonded energy and virial state.
   *
   * @return A borrowed mutable alias valid until manager destruction.
   * @warning Direct mutation can invalidate force-manager energy invariants.
   */
  CudaEnergyVirial &getBondedEnergyVirial(void);

  /**
   * @brief Returns read-only reciprocal energy and virial state.
   *
   * @return A borrowed read-only alias valid until manager destruction.
   *
   * @note The call performs no transfer or synchronization.
   */
  const CudaEnergyVirial &getReciprocalEnergyVirial(void) const;

  /**
   * @brief Returns mutable reciprocal energy and virial state.
   *
   * @return A borrowed mutable alias valid until manager destruction.
   * @warning Direct mutation can invalidate force-manager energy invariants.
   */
  CudaEnergyVirial &getReciprocalEnergyVirial(void);

  /**
   * @brief Returns read-only direct-space energy and virial state.
   *
   * @return A borrowed read-only alias valid until manager destruction.
   *
   * @note The call performs no transfer or synchronization.
   */
  const CudaEnergyVirial &getDirectEnergyVirial(void) const;

  /**
   * @brief Returns mutable direct-space energy and virial state.
   *
   * @return A borrowed mutable alias valid until manager destruction.
   * @warning Direct mutation can invalidate force-manager energy invariants.
   */
  CudaEnergyVirial &getDirectEnergyVirial(void);

  /**
   * @brief Returns the current host-side energy decomposition.
   *
   * The returned map contains the keys `bond`, `angle`, `ureyb`, `dihe`,
   * `imdihe`, `cmap`, `ewks`, `ewse`, `ewex`, `elec`, `vdw`, and `user`.
   * `user` is the sum of the default energy component from every subscribed
   * energy-virial object. Values use kilocalories per mole.
   *
   * This method reads existing host mirrors. It does not copy energy values
   * from the device and does not synchronize a CUDA stream.
   *
   * @return A new map containing copied energy labels and values.
   * @throws ApoCharmmError With code
   * @ref ApoCharmmErrorCode::NotInitialized if the manager is not initialized.
   * @throws std::bad_alloc If map or string allocation fails.
   *
   * @pre An energy-producing force calculation has completed if current values
   * are required.
   */
  std::map<std::string, double> getEnergyComponents(void);

  /**
   * @brief Returns the bonded-force CUDA stream holder.
   *
   * @return A copied shared owner of the host `cudaStream_t` value, or an empty
   * pointer before initialization.
   *
   * @warning The manager owns the underlying CUDA stream and destroys it during
   * cleanup regardless of external copies of this shared pointer.
   */
  std::shared_ptr<cudaStream_t> getBondedStream(void);

  /**
   * @brief Returns the reciprocal-force CUDA stream holder.
   *
   * @return A copied shared owner of the host `cudaStream_t` value, or an empty
   * pointer before initialization.
   *
   * @warning The manager owns the underlying CUDA stream.
   */
  std::shared_ptr<cudaStream_t> getReciprocalStream(void);

  /**
   * @brief Returns the direct-force CUDA stream holder.
   *
   * @return A copied shared owner of the host `cudaStream_t` value, or an empty
   * pointer before initialization.
   *
   * @warning The manager owns the underlying CUDA stream.
   */
  std::shared_ptr<cudaStream_t> getDirectStream(void);

  /**
   * @brief Returns the aggregate-force CUDA stream holder.
   *
   * @return A copied shared owner of the host `cudaStream_t` value, or an empty
   * pointer before initialization.
   *
   * @warning The manager owns the underlying CUDA stream.
   */
  std::shared_ptr<cudaStream_t> getForceManagerStream(void);

  /**
   * @brief Returns synchronized bonded fixed-point force storage.
   *
   * @return A copied shared owner of device-resident
   * `Force<long long int>` storage. Components use structure-of-arrays layout
   * with X at offset `0`, Y at `stride`, and Z at `2 * stride`. Raw values must
   * be multiplied by `INV_FORCE_SCALE` to recover force values in
   * kilocalories per mole per angstrom.
   *
   * @throws ApoCharmmError With code
   * @ref ApoCharmmErrorCode::NotInitialized if the manager is not initialized.
   * @throws ApoCharmmError With code @ref ApoCharmmErrorCode::Cuda if
   * synchronizing the aggregate stream fails.
   */
  std::shared_ptr<Force<long long int>> getBondedForcevalues(void);

  /**
   * @brief Returns synchronized reciprocal fixed-point force storage.
   *
   * @return A copied shared owner with the same layout and scaling as
   * `getBondedForcevalues()`.
   * @throws ApoCharmmError With code
   * @ref ApoCharmmErrorCode::NotInitialized if the manager is not initialized.
   * @throws ApoCharmmError With code @ref ApoCharmmErrorCode::Cuda if
   * synchronization fails.
   */
  std::shared_ptr<Force<long long int>> getReciprocalForcevalues(void);

  /**
   * @brief Returns synchronized direct-space fixed-point force storage.
   *
   * @return A copied shared owner with the same layout and scaling as
   * `getBondedForcevalues()`.
   * @throws ApoCharmmError With code
   * @ref ApoCharmmErrorCode::NotInitialized if the manager is not initialized.
   * @throws ApoCharmmError With code @ref ApoCharmmErrorCode::Cuda if
   * synchronization fails.
   */
  std::shared_ptr<Force<long long int>> getDirectForcevalues(void);

  /**
   * @brief Returns synchronized aggregate double-precision force storage.
   *
   * @return A copied shared owner of device-resident `Force<double>` storage
   * using structure-of-arrays layout and force units of kilocalories per mole
   * per angstrom.
   * @throws ApoCharmmError With code
   * @ref ApoCharmmErrorCode::NotInitialized if the manager is not initialized.
   * @throws ApoCharmmError With code @ref ApoCharmmErrorCode::Cuda if
   * synchronization fails.
   */
  std::shared_ptr<Force<double>> getTotalForcevalues(void);

  /**
   * @brief Returns the aggregate double-precision force storage.
   *
   * This is an aliasing API for `getTotalForcevalues()`.
   *
   * @return A copied shared owner of the same `Force<double>` object returned
   * by `getTotalForcevalues()`.
   * @throws ApoCharmmError With code
   * @ref ApoCharmmErrorCode::NotInitialized if the manager is not initialized.
   * @throws ApoCharmmError With code @ref ApoCharmmErrorCode::Cuda if
   * synchronization fails.
   */
  virtual std::shared_ptr<Force<double>> getForces(void);

  /**
   * @brief Returns the component stride of aggregate force storage.
   *
   * @return Number of `double` elements between corresponding X, Y, and Z
   * components. The value is dimensionless and may exceed the atom count due to
   * alignment.
   * @throws ApoCharmmError With code
   * @ref ApoCharmmErrorCode::NotInitialized if the manager is not initialized.
   */
  int getForceStride(void) const;

  /**
   * @brief Returns read-only stored box dimensions.
   *
   * @return A borrowed read-only reference to exactly three values in
   * `[x, y, z]` order, in angstroms. Before a box is set, the base constructor
   * stores three sentinel values near `-9999.9999`.
   */
  virtual const std::vector<double> &getBoxDimensions(void) const;

  /**
   * @brief Returns mutable stored box dimensions.
   *
   * @return A borrowed mutable reference to the manager's box vector.
   *
   * @warning Mutating this vector bypasses length and value validation, does
   * not update the float box fields, does not propagate to force backends, and
   * does not reset a neighbor list or FFT grid. Prefer
   * `setBoxDimensions()`.
   */
  virtual std::vector<double> &getBoxDimensions(void);

  /**
   * @brief Returns the stored Ewald splitting parameter.
   *
   * @return Kappa in inverse angstroms.
   */
  float getKappa(void) const;

  /**
   * @brief Returns the stored pair-list cutoff.
   *
   * @return Cutoff distance in angstroms.
   */
  float getCutoff(void) const;

  /**
   * @brief Returns the stored outer switching distance exposed as `ctonnb`.
   *
   * @return Distance in angstroms.
   */
  float getCtonnb(void) const;

  /**
   * @brief Returns the stored inner switching distance exposed as `ctofnb`.
   *
   * @return Distance in angstroms.
   */
  float getCtofnb(void) const;

  /**
   * @brief Returns the three stored PME FFT grid dimensions.
   *
   * @return A new `{nfftx, nffty, nfftz}` vector of dimensionless grid sizes.
   * Before an explicit grid or successful initialization, each value is `-1`.
   */
  std::vector<int> getFFTGrid(void) const;

  /**
   * @brief Returns the PME interpolation spline order.
   *
   * @return Positive dimensionless spline order; the default is `4`.
   */
  int getPmeSplineOrder(void) const;

  /**
   * @brief Returns the stored periodic boundary condition.
   *
   * @return Current @ref PBC value.
   */
  PBC getPeriodicBoundaryCondition(void) const;

  /**
   * @brief Returns aggregate potential-energy storage.
   *
   * @return A borrowed mutable alias of a one-element `CudaContainer<double>`
   * in kilocalories per mole.
   *
   * @throws ApoCharmmError With code
   * @ref ApoCharmmErrorCode::NotInitialized if the manager is not initialized.
   *
   * @note The current device value is updated only when `calcForce()` is called
   * with `calcEnergy == true`. The host mirror is transferred only by code that
   * explicitly requests a transfer or prints the decomposition.
   * @note This accessor performs no synchronization or transfer.
   */
  virtual CudaContainer<double> &getPotentialEnergy(void);

  /**
   * @brief Returns the current standard potential-energy sum on the host.
   *
   * The current implementation copies the bonded, reciprocal, and direct
   * energy-virial objects to host storage, synchronizes the aggregate stream,
   * and sums `bond`, `angle`, `ureyb`, `dihe`, `imdihe`, `ewks`, `ewse`,
   * `ewex`, `elec`, and `vdw`.
   *
   * @return The listed energy-term sum, narrowed to `float`, in kilocalories
   * per mole.
   * @throws ApoCharmmError With code
   * @ref ApoCharmmErrorCode::NotInitialized if the manager is not initialized.
   * @throws ApoCharmmError With code @ref ApoCharmmErrorCode::Cuda if a
   * device-to-host copy or stream synchronization fails.
   *
   * @warning The current implementation omits the `cmap` term and all
   * subscribed-force energies. It is therefore not equivalent to the
   * one-element total produced by `calcForce(..., calcEnergy = true, ...)`.
   */
  virtual float getPotentialEnergies(void);

  /**
   * @brief Sums standard and eligible subscribed virial contributions.
   *
   * The method obtains the bonded, reciprocal, and direct nine-element virial
   * arrays, transfers them to the host, halves the reciprocal contribution for
   * @ref PBC::P21, adds subscribed virials whose force type reports
   * `contributesVirial`, and transfers the aggregate back to the device.
   *
   * @return A borrowed mutable alias of a nine-element
   * `CudaContainer<double>`. Values use energy units of kilocalories per mole.
   * Both host and device representations contain the aggregate on successful
   * return.
   * @throws ApoCharmmError With code
   * @ref ApoCharmmErrorCode::NotInitialized if the manager is not initialized.
   * @throws ApoCharmmError With code @ref ApoCharmmErrorCode::Cuda if a virial
   * copy or transfer fails.
   *
   * @note The public component-to-matrix mapping and sign convention are not
   * established by the current repository; see the unresolved documentation
   * question in the ForceManager documentation work item.
   */
  virtual CudaContainer<double> &getVirial(void);

  /**
   * @brief Returns the native van der Waals model code.
   *
   * @return Integer model code in `[1, 6]` for manager state created through
   * the validated setter.
   */
  int getVdwType(void) const;

  /**
   * @brief Reports whether this manager is composite.
   *
   * @return `false` for the base `ForceManager`.
   */
  virtual bool isComposite(void) const;

  /**
   * @brief Returns read-only child-manager storage.
   *
   * @return A borrowed read-only reference to the base manager's child vector.
   * The vector is empty unless modified through the mutable accessor or by a
   * derived implementation.
   */
  virtual const std::vector<std::shared_ptr<ForceManager>> &
  getChildren(void) const;

  /**
   * @brief Returns mutable child-manager storage.
   *
   * @return A borrowed mutable reference to the base manager's child vector.
   *
   * @warning Direct mutation bypasses `addForceManager()` and can make
   * `isComposite()` disagree with the stored children.
   */
  virtual std::vector<std::shared_ptr<ForceManager>> &getChildren(void);

public:
  /**
   * @brief Initializes all native force-evaluation state.
   *
   * Initialization validates required collaborators and configuration, selects
   * an automatic FFT grid when any stored FFT dimension is not positive, and
   * constructs resources in this order:
   *
   * 1. bonded stream, lists, coefficients, fixed-point force storage, and
   *    bonded backend;
   * 2. reciprocal stream, fixed-point force storage, and PME reciprocal
   *    backend;
   * 3. direct stream, exclusions, neighbor-list state, fixed-point force
   *    storage, and PME direct backend;
   * 4. every force already registered through `subscribe()`;
   * 5. aggregate stream and double-precision force storage;
   * 6. one-element aggregate potential-energy storage; and
   * 7. SHAKE atom and parameter records.
   *
   * The method performs a device-wide synchronization before creating the
   * aggregate potential-energy container and marks the manager initialized only
   * after every phase succeeds.
   *
   * @pre A non-null PSF and parameter set have been installed.
   * @pre Three finite positive box dimensions have been set.
   * @pre The pair-list cutoff is positive and does not exceed half the X box
   * length.
   *
   * @post On success, `isInitialized()` returns `true`; all four native streams
   * and force arrays are allocated; and already subscribed forces have received
   * the atom count and box.
   * @throws ApoCharmmError With code
   * @ref ApoCharmmErrorCode::NotInitialized if the PSF, parameter set, or box
   * has not been set.
   * @throws ApoCharmmError With code
   * @ref ApoCharmmErrorCode::InvalidArgument if the cutoff is invalid for the X
   * box length or automatic-grid validation fails.
   * @throws ApoCharmmError With code @ref ApoCharmmErrorCode::Runtime if the
   * PSF atom count is not positive or a topology/parameter operation reports a
   * runtime failure.
   * @throws ApoCharmmError With code @ref ApoCharmmErrorCode::Cuda if CUDA
   * stream creation, force allocation, backend setup, transfer, or
   * synchronization fails.
   * @throws std::bad_alloc If native object or container allocation fails.
   *
   * @warning Initialization is not transactional. Failure can leave partially
   * allocated resources while `isInitialized()` remains `false`.
   * @warning Repeated initialization of the same instance is not a supported
   * rebuild mechanism in the current implementation.
   */
  virtual void initialize(void);

  /**
   * @brief Rebuilds the direct-space neighbor list for current coordinates.
   *
   * @param[in] xyzq Borrowed device pointer to at least the retained PSF's atom
   * `float4` records in `[x, y, z, charge]` order. Coordinates are in
   * angstroms and charges are in elementary-charge units.
   *
   * @pre The manager is initialized.
   * @pre `xyzq` is non-null, device-accessible, and remains valid until the
   * direct backend has completed its work.
   * @throws ApoCharmmError Propagates a categorized direct-backend or CUDA
   * failure.
   *
   * @note This method performs no explicit manager-level validation or stream
   * synchronization.
   */
  virtual void resetNeighborList(const float4 *xyzq);

  /**
   * @brief Clears force and requested energy-virial state.
   *
   * On first use, the method captures and instantiates a CUDA graph that clears
   * the three built-in fixed-point force arrays. Later calls launch the cached
   * graph. Requested energy and virial state is cleared on the component
   * streams, subscribed forces are cleared, and the aggregate stream is
   * synchronized before return.
   *
   * @param[in] reset Must be `false`; neighbor-list reset through this argument
   * is not implemented.
   * @param[in] calcEnergy Whether energy state will be computed and therefore
   * must be cleared.
   * @param[in] calcVirial Whether virial state will be computed and therefore
   * must be cleared.
   *
   * @pre The manager is initialized.
   * @throws ApoCharmmError With code
   * @ref ApoCharmmErrorCode::NotImplemented if `reset` is `true`.
   * @throws ApoCharmmError With code @ref ApoCharmmErrorCode::Cuda if graph
   * capture, instantiation, launch, force clearing, or synchronization fails.
   * @throws ApoCharmmError Propagates a categorized subscribed-force clear
   * failure.
   */
  void calcForcePart1(const bool reset, const bool calcEnergy,
                      const bool calcVirial);

  /**
   * @brief Enqueues all built-in and subscribed force calculations.
   *
   * Bonded, reciprocal, and direct calculations are enqueued on their
   * respective streams. Each subscribed force is then invoked, with its virial
   * flag enabled only when both `calcVirial` is true and the force declares
   * that it contributes a virial.
   *
   * @param[in] xyzq Borrowed device pointer to at least `getNumAtoms()`
   * `float4` coordinate-charge records.
   * @param[in] calcEnergy Whether each force should compute energy.
   * @param[in] calcVirial Whether eligible forces should compute virial.
   *
   * @pre `calcForcePart1()` has completed for this evaluation.
   * @pre The manager is initialized and `xyzq` is valid device storage.
   * @throws ApoCharmmError Propagates categorized backend, subscribed-force, or
   * CUDA failures.
   *
   * @note This phase does not wait for the enqueued calculations to finish.
   */
  void calcForcePart2(const float4 *xyzq, const bool calcEnergy,
                      const bool calcVirial);

  /**
   * @brief Synchronizes component work and assembles aggregate outputs.
   *
   * The method clears aggregate force storage, waits for each component and
   * subscribed stream, and adds every fixed-point force into the aggregate
   * double-precision force array. When requested, it converts force storage for
   * virial calculation, computes eligible virials, copies energies to host
   * storage, sums standard and subscribed energy terms on the aggregate stream,
   * and optionally prints the decomposition.
   *
   * @param[in] xyzq Borrowed device pointer to the same coordinate-charge array
   * passed to `calcForcePart2()`.
   * @param[in] calcEnergy Whether to update aggregate potential-energy state.
   * @param[in] calcVirial Whether to update virial state.
   *
   * @pre `calcForcePart2()` has enqueued all component calculations.
   * @pre The manager is initialized and `xyzq` remains valid.
   * @post Aggregate force storage is complete before return.
   * @post If `calcEnergy` is true, aggregate device energy is complete and all
   * component host energy mirrors have been copied.
   * @post If `calcVirial` is true, requested virial calculations have
   * completed.
   * @throws ApoCharmmError With code @ref ApoCharmmErrorCode::Cuda if
   * synchronization, conversion, aggregation, transfer, or a CUDA launch
   * reports failure.
   * @throws ApoCharmmError Propagates categorized subscribed-force failures.
   *
   * @note If `calcEnergy` or `calcVirial` is false, the corresponding previous
   * output remains observable and must be treated as stale.
   */
  void calcForcePart3(const float4 *xyzq, const bool calcEnergy,
                      const bool calcVirial);

  /**
   * @brief Computes aggregate forces and optional energy and virial outputs.
   *
   * This convenience method calls `calcForcePart1()`, `calcForcePart2()`, and
   * `calcForcePart3()` in that order.
   *
   * @param[in] xyzq Borrowed device pointer to at least the retained PSF's atom
   * `float4` records in `[x, y, z, charge]` order.
   * @param[in] reset Must be `false`; `true` is currently unsupported.
   * @param[in] calcEnergy Whether to compute and aggregate potential energy.
   * @param[in] calcVirial Whether to compute virial contributions.
   *
   * @pre The manager is initialized.
   * @pre `xyzq` is non-null, device-accessible, and remains valid through
   * completion.
   * @post Aggregate force storage is complete before return.
   * @throws ApoCharmmError With code
   * @ref ApoCharmmErrorCode::NotImplemented if `reset` is `true`.
   * @throws ApoCharmmError With code @ref ApoCharmmErrorCode::Cuda if any CUDA
   * operation in the three phases fails.
   * @throws ApoCharmmError Propagates categorized built-in or subscribed-force
   * failures.
   *
   * @note The method does not implicitly rebuild the direct-space neighbor list
   * when `reset` is false.
   * @callergraph
   */
  virtual void calcForce(const float4 *xyzq, const bool reset = false,
                         const bool calcEnergy = false,
                         const bool calcVirial = false);

  /**
   * @brief Subscribes an additional force and its CUDA resources.
   *
   * The manager retains shared ownership of `force`, `forceStream`,
   * `forceValues`, and `energyVirial`. A type-erased @ref ForceView is stored
   * at the same index as those resources and `forceTag`. Successful operation
   * therefore preserves the invariant that all six subscription vectors have
   * equal length and corresponding indices describe one force.
   *
   * If the manager is already initialized, `force->initialize()` is called
   * with the current PSF atom count and box before any manager subscription
   * vector is modified.
   *
   * @tparam ForceType Concrete force type satisfying the `ForceView` interface.
   * @param[in] force Shared owner of the non-null force. The exact object
   * address must not already be subscribed.
   * @param[in] forceTag Non-empty diagnostic and energy-print label. Tags are
   * not required to be unique.
   * @param[in] forceStream Shared owner of the non-null CUDA stream holder used
   * by this force.
   * @param[in] forceValues Shared owner of non-null fixed-point device force
   * storage corresponding to `force`.
   * @param[in] energyVirial Shared owner of non-null energy and virial storage
   * corresponding to `force`.
   *
   * @post On success, the manager owns one shared reference to every supplied
   * object and includes the force in box propagation, clearing, force
   * evaluation, force aggregation, energy aggregation, and eligible virial
   * aggregation.
   * @throws ApoCharmmError With code
   * @ref ApoCharmmErrorCode::InvalidArgument if any shared pointer is null,
   * `forceTag` is empty, or the same force object is already subscribed.
   * @throws ApoCharmmError Propagates a categorized error from
   * `ForceType::initialize()` when the manager is already initialized.
   * @throws std::bad_alloc If subscription-vector growth fails.
   *
   * @warning The supplied stream, force storage, and energy-virial object must
   * belong to `force`; the manager does not verify this relationship.
   * @warning Host allocation failure during the sequence of `push_back()`
   * operations can leave the parallel subscription vectors with different
   * lengths.
   */
  template <typename ForceType>
  void subscribe(std::shared_ptr<ForceType> force, const std::string &forceTag,
                 std::shared_ptr<cudaStream_t> forceStream,
                 std::shared_ptr<Force<long long int>> forceValues,
                 std::shared_ptr<CudaEnergyVirial> energyVirial) {
    APOCHARMM_REQUIRE(force != nullptr, ApoCharmmErrorCode::InvalidArgument,
                      "Subscribed force must not be null");

    APOCHARMM_REQUIRE(!forceTag.empty(), ApoCharmmErrorCode::InvalidArgument,
                      "Force tag must not be empty");

    APOCHARMM_REQUIRE(forceStream != nullptr,
                      ApoCharmmErrorCode::InvalidArgument,
                      "Subscribed force stream must not be null");

    APOCHARMM_REQUIRE(forceValues != nullptr,
                      ApoCharmmErrorCode::InvalidArgument,
                      "Subscribed force storage must not be null");

    APOCHARMM_REQUIRE(energyVirial != nullptr,
                      ApoCharmmErrorCode::InvalidArgument,
                      "Subscribed energy-virial storage must not be null");

    for (const std::shared_ptr<void> &subscribedForce : m_ForcePtrs) {
      APOCHARMM_REQUIRE(subscribedForce.get() !=
                            static_cast<void *>(force.get()),
                        ApoCharmmErrorCode::InvalidArgument,
                        "Force is already subscribed to this ForceManager");
    }

    if (m_IsInitialized == true) {
      force->initialize(m_Psf->getNumAtoms(), {static_cast<double>(m_BoxX),
                                               static_cast<double>(m_BoxY),
                                               static_cast<double>(m_BoxZ)});
    }

    m_ForcePtrs.push_back(static_cast<std::shared_ptr<void>>(force));
    m_ForceViews.push_back(ForceView(force.get()));
    m_ForceTags.push_back(forceTag);
    m_ForceStreams.push_back(forceStream);
    m_ForceValues.push_back(forceValues);
    m_EnergyVirials.push_back(energyVirial);

    return;
  }

  /**
   * @brief Unsubscribes a force by object identity.
   *
   * The first subscription whose retained object address equals `force.get()`
   * is removed from every parallel subscription vector. The manager releases
   * its shared references but does not clear or deinitialize the force.
   *
   * @tparam ForceType Concrete force type previously passed to `subscribe()`.
   * @param[in] force Shared owner identifying the subscribed object.
   *
   * @post On success, the force no longer participates in manager operations.
   * The force and its resources remain alive while any external shared owners
   * remain.
   * @throws ApoCharmmError With code
   * @ref ApoCharmmErrorCode::InvalidArgument if `force` is null or its object
   * address is not subscribed.
   *
   * @note A validation failure leaves all subscription vectors unchanged.
   */
  template <typename ForceType>
  void unsubscribe(std::shared_ptr<ForceType> force) {
    APOCHARMM_REQUIRE(force != nullptr, ApoCharmmErrorCode::InvalidArgument,
                      "Subscribed force must not be null");

    for (std::size_t i = 0; i < m_ForceViews.size(); i++) {
      if (static_cast<void *>(force.get()) ==
          static_cast<void *>(m_ForcePtrs[i].get())) {
        m_ForcePtrs.erase(m_ForcePtrs.begin() + i);
        m_ForceViews.erase(m_ForceViews.begin() + i);
        m_ForceTags.erase(m_ForceTags.begin() + i);
        m_ForceStreams.erase(m_ForceStreams.begin() + i);
        m_ForceValues.erase(m_ForceValues.begin() + i);
        m_EnergyVirials.erase(m_EnergyVirials.begin() + i);
        return;
      }
    }

    APOCHARMM_THROW(ApoCharmmErrorCode::InvalidArgument,
                    "Force is not subscribed to this ForceManager");
  }

  /**
   * @brief Unsubscribes the first force with a matching tag.
   *
   * Tags are not required to be unique. When multiple subscriptions use the
   * same tag, only the lowest-index match is removed.
   *
   * @param[in] forceTag Non-empty tag to find.
   *
   * @post On success, the matched force and corresponding resource entries are
   * removed from every parallel subscription vector.
   * @throws ApoCharmmError With code
   * @ref ApoCharmmErrorCode::InvalidArgument if `forceTag` is empty or no
   * subscription has that tag.
   *
   * @note A validation or lookup failure leaves all subscription vectors
   * unchanged.
   */
  void unsubscribe(const std::string &forceTag) {
    APOCHARMM_REQUIRE(!forceTag.empty(), ApoCharmmErrorCode::InvalidArgument,
                      "Force tag must not be empty");

    for (std::size_t i = 0; i < m_ForceTags.size(); i++) {
      if (m_ForceTags[i] == forceTag) {
        m_ForcePtrs.erase(m_ForcePtrs.begin() + i);
        m_ForceViews.erase(m_ForceViews.begin() + i);
        m_ForceTags.erase(m_ForceTags.begin() + i);
        m_ForceStreams.erase(m_ForceStreams.begin() + i);
        m_ForceValues.erase(m_ForceValues.begin() + i);
        m_EnergyVirials.erase(m_EnergyVirials.begin() + i);
        return;
      }
    }

    APOCHARMM_THROW(ApoCharmmErrorCode::InvalidArgument,
                    "Force tag is not subscribed to this ForceManager");
  }

  /**
   * @brief Rejects child-energy evaluation in the base manager.
   *
   * Derived composite managers override this extension point.
   *
   * @param[in] xyzq Proposed device coordinate-charge array.
   *
   * @return The base implementation never returns successfully.
   * @throws ApoCharmmError With code
   * @ref ApoCharmmErrorCode::InvalidArgument if `xyzq` is null.
   * @throws ApoCharmmError With code @ref ApoCharmmErrorCode::Runtime for every
   * non-null `xyzq`, because the base manager does not support child
   * evaluation.
   */
  virtual CudaContainer<double>
  computeAllChildrenPotentialEnergy(const float4 *xyzq);

protected:
  /**
   * @brief Builds GPU SHAKE records from PSF bonds and parameter data.
   *
   * Hydrogen-containing bonds are grouped by heavy atom, except the currently
   * excluded `OT`-`HT` and hydrogen-hydrogen cases. Each selected group creates
   * one `int4` atom record and one `float4` parameter record as documented by
   * the public SHAKE accessors. The generated vectors are assigned to
   * `m_ShakeAtoms` and `m_ShakeParams`, which copies their contents into the
   * corresponding CUDA containers.
   *
   * @pre `m_Psf` and `m_Prm` are non-null and describe compatible atom,
   * topology, mass, atom-type, and bond-parameter data.
   * @post SHAKE containers contain one record per selected heavy-atom group.
   * @throws ApoCharmmError Propagates categorized PSF, parameter, container, or
   * CUDA failures.
   * @throws std::bad_alloc If grouping or record allocation fails.
   *
   * @warning The current implementation assumes at most three selected
   * hydrogens per heavy atom and non-empty atom-type strings.
   */
  void initializeHolonomicConstraintsVariables(void);

  /**
   * @brief Computes automatic even PME FFT grid dimensions.
   *
   * Each result is the largest even integer not greater than the corresponding
   * box length after truncation to `int`, with a minimum value of `2`. A
   * warning is written to `std::cout` when a minimum is applied.
   *
   * @return A new three-element `{nfftx, nffty, nfftz}` vector of dimensionless
   * grid sizes.
   * @throws ApoCharmmError With code
   * @ref ApoCharmmErrorCode::InvalidArgument if the stored box vector is not
   * exactly three finite positive values.
   * @throws std::bad_alloc If return-vector allocation fails.
   */
  std::vector<int> computeFFTGridSize(void);

  /**
   * @brief Validates an orthorhombic box-dimension vector.
   *
   * @param[in] boxDimensions Vector that must contain exactly three finite
   * positive values in angstroms.
   *
   * @throws ApoCharmmError With code
   * @ref ApoCharmmErrorCode::InvalidArgument if the length is not three or any
   * value is non-finite or not positive.
   */
  void checkBoxDimensions(const std::vector<double> &boxDimensions);

private:
  /**
   * @brief Constructs or reconstructs the bonded-force backend.
   *
   * The manager-owned CUDA stream, force storage, and energy-virial registry
   * must already exist and remain unchanged.
   */
  void rebuildBondedForce(void);

  /**
   * @brief Constructs or reconstructs the reciprocal-space backend.
   *
   * The manager-owned CUDA stream, force storage, and energy-virial registry
   * must already exist and remain unchanged.
   */
  void rebuildReciprocalForce(void);

  /**
   * @brief Constructs or reconstructs the direct-space backend.
   *
   * The manager-owned CUDA stream, force storage, and energy-virial registry
   * must already exist and remain unchanged.
   */
  void rebuildDirectForce(void);

  /**
   * @brief Reconstructs every dirty built-in backend before force evaluation.
   *
   * When the direct backend is reconstructed, its neighbor list is also built
   * from `xyzq` before this function returns.
   *
   * @param[in] xyzq Current device-resident coordinate-charge array.
   */
  void rebuildDirtyForces(const float4 *xyzq);

  /**
   * @brief Requires successful manager initialization.
   *
   * @throws ApoCharmmError With code
   * @ref ApoCharmmErrorCode::NotInitialized if `m_IsInitialized` is false.
   */
  void checkInitialized(void) const;

  /**
   * @brief Releases manager-owned CUDA streams and cached graph objects.
   *
   * Stream cleanup uses the non-throwing CUDA destruction helper. Graph and
   * graph-executable destruction statuses are discarded. Shared stream holders
   * are reset and the graph-created flag is cleared.
   *
   * @post Native stream members are empty and cached graph handles are null.
   */
  void dealloc(void) noexcept;

protected:
  /**
   * Non-owning backlink to the associated context. The context is never kept
   * alive solely by its force manager.
   */
  std::weak_ptr<CharmmContext> m_Context;

  /** Shared owner of the topology and atom metadata used by initialization. */
  std::shared_ptr<CharmmPSF> m_Psf;

  /** Shared owner of the force-field parameter set used by initialization. */
  std::shared_ptr<CharmmParameters> m_Prm;

  /**
   * Successful-initialization flag. It changes to `true` only at the final
   * statement of `initialize()` and can be cleared by selected collaborator or
   * PBC setters without deallocating existing resources.
   */
  bool m_IsInitialized;

  /** Whether the bonded-force backend must be reconstructed before use. */
  bool m_BondedForceDirty;

  /** Whether the reciprocal-space backend must be reconstructed before use. */
  bool m_ReciprocalForceDirty;

  /** Whether the direct-space backend must be reconstructed before use. */
  bool m_DirectForceDirty;

  /**
   * Host/device mirrored SHAKE atom groups. Each `int4` stores one heavy atom
   * and up to three hydrogen atom indices.
   */
  CudaContainer<int4> m_ShakeAtoms;

  /**
   * Host/device mirrored solver parameters corresponding one-to-one with
   * `m_ShakeAtoms`.
   */
  CudaContainer<float4> m_ShakeParams;

  // TODO : these should not be directly here

  /**
   * Built-in energy and virial registries. Their device values are written on
   * separate component streams, while host values are updated only by explicit
   * copies.
   */
  CudaEnergyVirial m_BondedEnergyVirial;
  CudaEnergyVirial m_ReciprocalEnergyVirial;
  CudaEnergyVirial m_DirectEnergyVirial;

  /**
   * Host allocations containing the bonded, reciprocal, and direct CUDA stream
   * handles. `ForceManager` owns and explicitly destroys the underlying CUDA
   * streams.
   */
  std::shared_ptr<cudaStream_t> m_BondedStream;
  std::shared_ptr<cudaStream_t> m_ReciprocalStream;
  std::shared_ptr<cudaStream_t> m_DirectStream;

  /**
   * Host allocation containing the CUDA stream used for force clearing,
   * aggregation, and total-energy kernels.
   */
  std::shared_ptr<cudaStream_t> m_ForceManagerStream;

  /**
   * Device-resident fixed-point force arrays for the three built-in force
   * components. Each uses X/Y/Z structure-of-arrays layout and
   * `Force<long long int>` scaling.
   */
  std::shared_ptr<Force<long long int>> m_BondedForceValues;
  std::shared_ptr<Force<long long int>> m_ReciprocalForceValues;
  std::shared_ptr<Force<long long int>> m_DirectForceValues;

  /**
   * Device-resident double-precision aggregate force array. The exact
   * force-versus-energy-gradient sign convention requires separate
   * clarification; the current integration paths consume this object as the
   * manager's force output.
   */
  std::shared_ptr<Force<double>> m_TotalForceValues;

  /**
   * Float box lengths passed to selected native backends, in angstroms. They
   * duplicate `m_BoxDimensions` and are updated only by
   * `setBoxDimensions()`.
   */
  float m_BoxX;
  float m_BoxY;
  float m_BoxZ;

  /**
   * Double-precision `[x, y, z]` orthorhombic box lengths in angstroms.
   */
  std::vector<double> m_BoxDimensions;

  // Long range and PME options

  /** Ewald splitting parameter in inverse angstroms; default `0.34`. */
  float m_Kappa;

  /**
   * Direct-space pair-list cutoff in angstroms, corresponding to CHARMM CUTNB;
   * default `14.0`.
   */
  float m_Cutoff;

  /**
   * Outer switching distance in angstroms as currently forwarded to the
   * direct backend; default `12.0`.
   */
  float m_Ctonnb;

  /**
   * Inner switching distance in angstroms as currently forwarded to the
   * direct backend; default `10.0`.
   */
  float m_Ctofnb;

  /**
   * Dimensionless PME FFT dimensions. The `-1` constructor sentinel selects
   * automatic grid generation during initialization.
   */
  int m_NfftX;
  int m_NfftY;
  int m_NfftZ;

  /** Positive dimensionless PME interpolation spline order; default `4`. */
  int m_PmeSplineOrder;

  /** Periodic-boundary mode; default @ref PBC::P1. */
  PBC m_Pbc;

  /**
   * Manager-owned built-in CUDA backend objects. They are null before
   * initialization and are replaced by `initialize()`.
   */
  std::unique_ptr<CudaBondedForce<long long int, float>> m_BondedForcePtr;
  std::unique_ptr<CudaPMEReciprocalForce> m_ReciprocalForcePtr;
  std::unique_ptr<CudaPMEDirectForce<long long int, float>> m_DirectForcePtr;

  /**
   * One-element aggregate potential-energy storage in kilocalories per mole.
   */
  CudaContainer<double> m_TotalPotentialEnergy;

  /**
   * Nine-element virial work arrays for built-in, subscribed, and aggregate
   * contributions. The arrays maintain host and device storage.
   */
  CudaContainer<double> m_BondedVirial;
  CudaContainer<double> m_ReciprocalVirial;
  CudaContainer<double> m_DirectVirial;
  CudaContainer<double> m_SubscribedForceVirial;
  CudaContainer<double> m_TotalVirial;

  /**
   * Cached CUDA graph used to clear the three built-in fixed-point force
   * arrays. `m_ClearGraphCreated` is the lifecycle discriminator for the graph
   * and executable handles.
   */
  bool m_ClearGraphCreated;
  cudaGraph_t m_ClearGraph;
  cudaGraphExec_t m_ClearGraphInstance;

  /**
   * Parallel subscription state. For every valid index:
   *
   * - `m_ForcePtrs` owns the concrete force;
   * - `m_ForceViews` dispatches to that force;
   * - `m_ForceTags` stores its label;
   * - `m_ForceStreams` owns its stream holder;
   * - `m_ForceValues` owns its fixed-point force storage; and
   * - `m_EnergyVirials` owns its energy and virial storage.
   *
   * Successful subscription and unsubscription require all vectors to have
   * equal length.
   */
  std::vector<std::shared_ptr<void>> m_ForcePtrs;
  std::vector<ForceView> m_ForceViews;
  std::vector<std::string> m_ForceTags;
  std::vector<std::shared_ptr<cudaStream_t>> m_ForceStreams;
  std::vector<std::shared_ptr<Force<long long int>>> m_ForceValues;
  std::vector<std::shared_ptr<CudaEnergyVirial>> m_EnergyVirials;

  /**
   * Legacy direct-space computation-selection flag used by specialized manager
   * implementations. The base constructor initializes it to `true`.
   */
  bool m_ComputeDirectSpaceForces;

  /** Native van der Waals model code; default `VDW_VFSW`. */
  int m_VdwType;

  /**
   * Child-manager storage used by derived composite implementations. The base
   * manager leaves it empty and reports `isComposite() == false`.
   */
  std::vector<std::shared_ptr<ForceManager>> m_Children;

  /**
   * Selects standard and subscribed energy output to `std::cout` during an
   * energy-producing force calculation.
   */
  bool m_PrintEnergyDecomposition;
};
