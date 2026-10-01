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

#include "CharmmPSF.h"
#include "CharmmParameters.h"
#include "Coordinates.h"
#include "CudaContainer.h"
#include "Force.h"
#include "ForceManager.h"
#include "PBC.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <vector_types.h>

/**
 * @brief Represents mutable molecular state used by apoCHARMM force
 * calculations.
 *
 * A context stores coordinates, charges, velocities, inverse masses,
 * thermodynamic bookkeeping, periodic-box state, and force-evaluation output.
 * It shares ownership of an attached @ref CharmmPSF,
 * @ref CharmmParameters, and @ref ForceManager.
 *
 * Coordinates and charges are maintained as both single-precision and
 * double-precision @ref CudaContainer mirrors. Velocities and inverse masses
 * are maintained in a double-precision `double4` container. Host and device
 * storage are coherent only after the explicit transfers performed by the
 * documented mutators or by a caller using the mutable container accessors.
 *
 * The force manager stores only a weak backlink to the context. Construct a
 * context through `std::make_shared<CharmmContext>` when the backlink is
 * required.
 *
 * The class performs no internal locking. Concurrent access requires external
 * synchronization whenever any participating operation can mutate the context,
 * an attached collaborator, a returned container, or device storage.
 *
 * Copy construction deep-copies the context-owned containers and shares the
 * PSF and parameter objects, but does not copy the force-manager association.
 * Copy assignment remains compiler-generated and therefore performs memberwise
 * assignment, including the force-manager member. No implicit move operations
 * are generated because a copy constructor is explicitly declared.
 *
 * @warning Operations that update host state and then perform CUDA work are not
 * transactional. A CUDA or backend failure can leave already-applied host-side
 * changes observable.
 *
 * @see charmm_context
 */
class CharmmContext : public std::enable_shared_from_this<CharmmContext> {
public:
  /**
   * @brief Constructs an incomplete context without molecular collaborators.
   *
   * The atom count and degree-of-freedom count are initialized to `-1`, the box
   * is unset, the stored random seed is zero, and coordinate, charge, velocity,
   * and inverse-mass storage is empty. Scalar energy and tensor bookkeeping
   * containers are allocated with their implementation-defined initial sizes.
   *
   * @post The context cannot perform force calculations until a PSF,
   * parameters, force manager, valid box, and coordinates have been supplied.
   *
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if allocation
   * of context-owned CUDA storage fails.
   */
  CharmmContext(void);

  /**
   * @brief Constructs a context from a PSF and CHARMM parameters.
   *
   * The context retains shared ownership of both arguments, creates and retains
   * a new `ForceManager`, copies atom charges and masses from the PSF, enables
   * holonomic-constraint degree-of-freedom accounting, and selects a seed with
   * `std::random_device`. The force manager remains uninitialized until a valid
   * three-dimensional box is also available.
   *
   * @param[in] psf Shared PSF owner. The pointer must be non-null and is
   * retained by the context.
   * @param[in] prm Shared parameter-set owner. The pointer must be non-null and
   * is retained by the context.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if either pointer is null or imported
   * state violates a native validation rule.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if GPU runtime
   * setup, storage allocation, or a required transfer fails.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Runtime` if native
   * topology or parameter preparation fails.
   */
  CharmmContext(std::shared_ptr<CharmmPSF> psf,
                std::shared_ptr<CharmmParameters> prm);

  /**
   * @brief Constructs a context from an existing ForceManager.
   *
   * The context retains shared ownership of `fm` and imports any PSF,
   * parameters, periodic-boundary condition, and box already stored by the
   * manager. When the imported state is complete and the manager is not already
   * initialized, construction initializes the manager and computes the
   * constrained degree-of-freedom count.
   *
   * An incomplete force manager is accepted. Operations requiring initialized
   * force storage continue to report a not-initialized error until the missing
   * state is supplied.
   *
   * @param[in] fm Shared force-manager owner. The pointer must be non-null and
   * is retained by the context.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if `fm` is null or imported
   * configuration is invalid.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if GPU setup or
   * force-manager initialization encounters a CUDA failure.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Runtime` if native
   * force-manager initialization fails for another verified runtime condition.
   */
  CharmmContext(std::shared_ptr<ForceManager> fm);

  /**
   * @brief Destroys context-owned storage and releases shared collaborators.
   *
   * Destruction is non-throwing. Cleanup failures from nested CUDA-owning
   * objects are not propagated from this destructor.
   */
  ~CharmmContext(void) noexcept = default;

  /**
   * @brief Constructs a context from another context's stored state.
   *
   * Context-owned coordinate, velocity, inverse-mass, energy, and pressure
   * containers are deep-copied. PSF and parameter objects remain shared. Energy
   * table history is reset.
   *
   * The force-manager association is not copied, so the result cannot evaluate
   * forces until a force manager is attached separately. The kinetic-energy
   * partial-sum workspace and kinetic-virial tensor member are
   * default-constructed rather than copied by the current implementation.
   *
   * @param[in] other Context whose current state is copied. The source remains
   * unchanged.
   *
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if copying a
   * context-owned CUDA container fails.
   */
  CharmmContext(const CharmmContext &other);

public: // Setters
  /**
   * @brief Sets the CHARMM parameter set used by the context.
   *
   * The context retains shared ownership of `prm` and forwards the same native
   * object to an attached force manager. Supplying the final missing piece of a
   * complete configuration can initialize the force manager.
   *
   * @param[in] prm Shared parameter-set owner. The pointer must be non-null and
   * is retained.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if `prm` is null or the completed
   * force configuration is invalid.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if triggered
   * force-manager initialization encounters a CUDA failure.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Runtime` if triggered
   * parameter or force initialization fails for another runtime condition.
   *
   * @warning The operation is not transactional when it triggers force-manager
   * initialization.
   */
  void setPrm(std::shared_ptr<CharmmParameters> prm);

  /**
   * @brief Sets the molecular PSF and imports its atom data.
   *
   * The context retains shared ownership of `psf`, sets the atom count, copies
   * charges in elementary-charge units, copies masses in atomic mass units, and
   * forwards the PSF to an attached force manager. Changing from one positive
   * atom count to a different positive atom count is rejected.
   *
   * Resizing atom storage invalidates previously borrowed element references,
   * iterators, and device pointers from the coordinate/charge and
   * velocity/inverse-mass containers.
   *
   * @param[in] psf Shared PSF owner. The pointer must be non-null and is
   * retained.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if `psf` is null or its atom count
   * differs from the context's established positive atom count.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if resizing,
   * copying, or triggered force-manager initialization fails in CUDA.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Runtime` if PSF or
   * force initialization fails for another runtime condition.
   *
   * @warning A failure after resizing or copying atom state does not roll the
   * context back to its previous state.
   */
  void setPsf(std::shared_ptr<CharmmPSF> psf);

  /**
   * @brief Sets the ForceManager associated with the context.
   *
   * The context retains shared ownership of `fm`. Existing context PSF,
   * parameter, box, and periodic-boundary state take precedence and are pushed
   * into the manager. State absent from the context is imported from the
   * manager. A complete uninitialized manager is initialized automatically.
   *
   * When the context is owned by `std::shared_ptr`, the manager receives a weak
   * backlink to the context. No ownership cycle is created.
   *
   * @param[in] fm Shared force-manager owner. The pointer must be non-null and
   * is retained.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if `fm` is null or synchronized
   * configuration is invalid.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if
   * synchronization or initialization encounters a CUDA failure.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Runtime` if native
   * force initialization fails for another runtime condition.
   *
   * @warning Replacing the manager can invalidate force, energy, virial,
   * constraint, and stride references obtained from the previous manager.
   */
  void setForceManager(std::shared_ptr<ForceManager> fm);

  /**
   * @brief Sets the atom count and sizes per-atom state storage.
   *
   * `numAtoms` must be positive. When a size changes, the single-precision and
   * double-precision coordinate/charge arrays and the
   * velocity/inverse-mass array are resized to `numAtoms` and initialized to
   * zero. A coordinate-size change clears the internal
   * coordinates-present flag.
   *
   * This function does not import charges, masses, coordinates, or velocities.
   *
   * @param[in] numAtoms Number of atoms. The value is dimensionless and must be
   * greater than zero.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if `numAtoms` is not positive.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if CUDA storage
   * resizing or initialization fails.
   *
   * @post Every per-atom container has active length `numAtoms`.
   * @warning A reallocation invalidates borrowed device pointers and element
   * references into the resized containers.
   */
  void setNumAtoms(const int numAtoms);

  /**
   * @brief Sets coordinates and charges from `double4` records.
   *
   * The vector must contain exactly one record per atom. Components `x`, `y`,
   * and `z` are Cartesian coordinates in angstroms; component `w` is charge in
   * elementary-charge units. Values are copied into double-precision storage,
   * converted to `float4` for single-precision storage, and transferred to both
   * device mirrors.
   *
   * If the force manager was already initialized, a successful update
   * image-centers the coordinates and rebuilds its neighbor list.
   *
   * @param[in] coordinatesCharges Contiguous host records in atom order. The
   * input is borrowed for the call and is not retained.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if the atom count is not positive.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if the record count differs from the
   * context atom count or an initialized non-composite force manager's count.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if a transfer,
   * image-centering operation, synchronization, or neighbor-list rebuild fails.
   *
   * @post On success, the host and device coordinate/charge mirrors represent
   * the supplied values, subject to any periodic image-centering performed
   * during neighbor-list rebuilding.
   */
  void setCoordinatesCharges(const std::vector<double4> &coordinatesCharges);

  /**
   * @brief Sets coordinates and charges from nested scalar rows.
   *
   * Every row must contain exactly four values in `[x, y, z, charge]` order.
   * Coordinates use angstroms and charges use elementary-charge units. The
   * nested input is converted to `double4` records and then processed by the
   * primary coordinate/charge setter.
   *
   * @param[in] coordinatesCharges Host rows in atom order. The input is copied
   * and is not retained.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if any row does not contain exactly
   * four values or the number of rows does not match the atom count.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if the atom count is not positive.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` for failures
   * propagated by the primary setter.
   */
  void setCoordinatesCharges(
      const std::vector<std::vector<double>> &coordinatesCharges);

  /**
   * @brief Sets coordinates and charges from a flat scalar array.
   *
   * The array is interpreted as `N` consecutive
   * `[x, y, z, charge]` records with stride four. Coordinates use angstroms and
   * charges use elementary-charge units.
   *
   * @param[in] coordinatesCharges Flat host array. Its length must be exactly
   * `4 * getNumAtoms()`. The input is copied and is not retained.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if the length is not a multiple of
   * four or the derived record count differs from the atom count.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if the atom count is not positive.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` for failures
   * propagated by the primary setter.
   */
  void setCoordinatesCharges(const std::vector<double> &coordinatesCharges);

  /**
   * @brief Sets Cartesian coordinates while preserving stored charges.
   *
   * The vector must contain exactly one `double3` per atom. Components are
   * Cartesian coordinates in angstroms. Coordinates are copied to the
   * double-precision array, converted for the single-precision array, and
   * transferred to both device mirrors.
   *
   * If the force manager was already initialized, a successful update
   * image-centers coordinates and rebuilds its neighbor list.
   *
   * @param[in] coordinates Contiguous host coordinates in atom order. The input
   * is borrowed for the call and is not retained.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if the atom count is not positive.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if the coordinate count differs from
   * the atom count.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if a transfer,
   * image-centering operation, synchronization, or neighbor-list rebuild fails.
   */
  void setCoordinates(const std::vector<double3> &coordinates);

  /**
   * @brief Sets Cartesian coordinates from nested scalar rows.
   *
   * Every row must contain exactly three values in `[x, y, z]` order and uses
   * angstroms. Existing charge components are preserved.
   *
   * @param[in] coordinates Host rows in atom order. The input is copied and is
   * not retained.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if a row does not contain exactly
   * three values or the number of rows differs from the atom count.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if the atom count is not positive.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` for failures
   * propagated by the primary coordinate setter.
   */
  void setCoordinates(const std::vector<std::vector<double>> &coordinates);

  /**
   * @brief Sets Cartesian coordinates from a flat scalar array.
   *
   * The array is interpreted as `N` consecutive `[x, y, z]` records with
   * stride three and units of angstroms. Existing charge components are
   * preserved.
   *
   * @param[in] coordinates Flat host array. Its length must be exactly
   * `3 * getNumAtoms()`. The input is copied and is not retained.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if the length is not a multiple of
   * three or the derived record count differs from the atom count.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if the atom count is not positive.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` for failures
   * propagated by the primary coordinate setter.
   */
  void setCoordinates(const std::vector<double> &coordinates);

  /**
   * @brief Sets coordinates from a Coordinates object.
   *
   * Double-precision coordinates are read from `crd` and copied into the
   * context. The context does not retain the supplied object. Existing charges
   * are preserved.
   *
   * @param[in] crd Shared owner of the coordinate object to read. The pointer
   * must be non-null and is borrowed only for this call.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if `crd` is null or its coordinate
   * count differs from the atom count.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if the atom count is not positive.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` for failures
   * propagated by the primary coordinate setter.
   */
  void setCoordinates(const std::shared_ptr<Coordinates> crd);

  /**
   * @brief Sets per-atom charges while preserving coordinates.
   *
   * Charges are copied in atom order, converted to single precision for the
   * `float4` mirror, and transferred to both device mirrors. Charge values use
   * elementary-charge units. This call does not rebuild the neighbor list.
   *
   * @param[in] charges One charge per atom. The input is borrowed for the call
   * and is not retained.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if the atom count is not positive.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if `charges.size()` differs from the
   * atom count.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if either
   * host-to-device transfer fails.
   */
  void setCharges(const std::vector<double> &charges);

  /**
   * @brief Sets velocities and inverse masses from `double4` records.
   *
   * The vector must contain one record per atom. Components `x`, `y`, and `z`
   * are velocity components in angstroms per AKMA time unit; component `w` is
   * inverse mass in reciprocal atomic mass units. Values are copied and
   * transferred to device storage.
   *
   * @param[in] velocitiesInverseMasses Host records in atom order. The input is
   * borrowed for the call and is not retained.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if the atom count is not positive.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if the record count differs from the
   * atom count.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if the
   * host-to-device transfer fails.
   */
  void setVelocitiesInverseMasses(
      const std::vector<double4> &velocitiesInverseMasses);

  /**
   * @brief Sets velocities and inverse masses from nested scalar rows.
   *
   * Every row must contain `[vx, vy, vz, inverse_mass]`. Velocity components
   * use angstroms per AKMA time unit and inverse masses use reciprocal atomic
   * mass units.
   *
   * @param[in] velocitiesInverseMasses Host rows in atom order. The input is
   * copied and is not retained.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if a row does not contain exactly
   * four values or the row count differs from the atom count.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if the atom count is not positive.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` for failures
   * propagated by the primary setter.
   */
  void setVelocitiesInverseMasses(
      const std::vector<std::vector<double>> &velocitiesInverseMasses);

  /**
   * @brief Sets velocities and inverse masses from a flat scalar array.
   *
   * The array is interpreted as `N` consecutive
   * `[vx, vy, vz, inverse_mass]` records with stride four.
   *
   * @param[in] velocitiesInverseMasses Flat host array with length exactly
   * `4 * getNumAtoms()`. The input is copied and is not retained.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if the length is not a multiple of
   * four or the derived record count differs from the atom count.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if the atom count is not positive.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` for failures
   * propagated by the primary setter.
   */
  void setVelocitiesInverseMasses(
      const std::vector<double> &velocitiesInverseMasses);

  /**
   * @brief Sets velocity components while preserving inverse masses.
   *
   * The vector must contain one `double3` per atom. Components use angstroms
   * per AKMA time unit. Values are copied to the host mirror and transferred to
   * the device.
   *
   * @param[in] velocities Host velocities in atom order. The input is borrowed
   * for the call and is not retained.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if the atom count is not positive.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if the velocity count differs from
   * the atom count.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if the
   * host-to-device transfer fails.
   */
  void setVelocities(const std::vector<double3> &velocities);

  /**
   * @brief Sets velocities from nested scalar rows.
   *
   * Every row must contain `[vx, vy, vz]` in angstroms per AKMA time unit.
   * Existing inverse masses are preserved.
   *
   * @param[in] velocities Host rows in atom order. The input is copied and is
   * not retained.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if a row does not contain exactly
   * three values or the row count differs from the atom count.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if the atom count is not positive.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` for failures
   * propagated by the primary velocity setter.
   */
  void setVelocities(const std::vector<std::vector<double>> &velocities);

  /**
   * @brief Sets velocities from a flat scalar array.
   *
   * The array is interpreted as `N` consecutive `[vx, vy, vz]` records with
   * stride three and units of angstroms per AKMA time unit. Existing inverse
   * masses are preserved.
   *
   * @param[in] velocities Flat host array with length exactly
   * `3 * getNumAtoms()`. The input is copied and is not retained.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if the length is not a multiple of
   * three or the derived record count differs from the atom count.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if the atom count is not positive.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` for failures
   * propagated by the primary velocity setter.
   */
  void setVelocities(const std::vector<double> &velocities);

  /**
   * @brief Loads velocities from a CHARMM coordinate-format velocity file.
   *
   * Lines beginning with `*` are skipped as title lines. The next non-title
   * line must contain the atom count, followed by exactly one parseable atom
   * record per context atom. Only the three velocity fields are copied;
   * existing inverse masses are preserved. Parsed velocity values use the
   * units encoded by the source file and are consumed by apoCHARMM as
   * angstroms per AKMA time unit.
   *
   * @param[in] fileName Nonempty path to the file. The string is borrowed for
   * the call and is not retained.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if the atom count is not positive.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if `fileName` is empty or the file's
   * atom count differs from the context atom count.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Runtime` if the file
   * cannot be opened or its atom count or a velocity record cannot be read or
   * parsed.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if the final
   * host-to-device transfer fails.
   */
  void setVelocitiesFromCHARMMVelocityFile(const std::string &fileName);

  /**
   * @brief Sets masses and stores their reciprocals with the velocities.
   *
   * The vector must contain one mass per atom in atomic mass units. For each
   * atom, the context stores `1.0 / masses[i]` in the `w` component and
   * preserves the existing velocity components.
   *
   * @param[in] masses Host masses in atom order. The input is borrowed for the
   * call and is not retained.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if the atom count is not positive.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if the number of masses differs from
   * the atom count.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if the
   * host-to-device transfer fails.
   *
   * @warning The current implementation does not reject zero, negative,
   * infinite, or NaN masses before computing their reciprocals.
   */
  void setMasses(const std::vector<double> &masses);

  /**
   * @brief Sets the stored target temperature.
   *
   * This call updates only the context's stored temperature in kelvin. It does
   * not generate, rescale, or otherwise modify velocities.
   *
   * @param[in] temperature Temperature in kelvin. The value must be finite and
   * non-negative.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if `temperature` is NaN, infinite, or
   * negative.
   */
  void setTemperature(const double temperature);

  /**
   * @brief Sets the periodic boundary condition.
   *
   * The selected value is stored in the context and forwarded to an attached
   * force manager. Supplying this state can initialize a complete force
   * manager. If the manager was already initialized and coordinates are
   * present, a successful update image-centers the coordinates and rebuilds the
   * neighbor list.
   *
   * @param[in] pbc Runnable periodic boundary condition. The accepted values
   * are `PBC::P1` and `PBC::P21`. `PBC::UNSET` is an invalid sentinel and does
   * not select an aperiodic system.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if `pbc` is `PBC::UNSET`, is not a
   * recognized enumerator, or the completed force configuration is rejected.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if
   * initialization, image centering, synchronization, or neighbor-list
   * rebuilding fails in CUDA.
   */
  void setPeriodicBoundaryCondition(const PBC pbc);

  /**
   * @brief Sets the orthorhombic box dimensions.
   *
   * `boxDimensions` must contain `[x, y, z]` lengths in angstroms. The context
   * requires exactly three positive values. An attached force manager
   * additionally rejects non-finite dimensions and can reject a configuration
   * whose direct-space cutoff exceeds half the X dimension.
   *
   * Supplying the final missing setup state can initialize the force manager.
   * If it was already initialized and coordinates are present, a successful
   * update image-centers coordinates and rebuilds the neighbor list.
   *
   * @param[in] boxDimensions Three host values in X, Y, Z order. The input is
   * copied and is not retained.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if the vector does not contain
   * exactly three positive values or attached force-manager validation rejects
   * the dimensions or cutoff.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if triggered
   * initialization, image centering, synchronization, or neighbor-list
   * rebuilding fails.
   */
  void setBoxDimensions(const std::vector<double> &boxDimensions);

  /**
   * @brief Sets the deterministic random seed used for velocity generation.
   *
   * @param[in] randomSeed Unsigned 64-bit seed. The value is dimensionless.
   *
   * @post A later call to `assignVelocitiesAtTemperature()` initializes a new
   * `std::mt19937` from this value.
   */
  void setRandomSeed(const std::uint64_t randomSeed);

  /**
   * @brief Selects holonomic-constraint degree-of-freedom accounting.
   *
   * The method recomputes the degree-of-freedom count from the atom count,
   * periodic-boundary condition, PSF water molecules, and the force manager's
   * SHAKE atom records. It controls whether those constraints are subtracted
   * from the reported degree-of-freedom count; it does not itself execute a
   * constraint solver.
   *
   * @param[in] usingHolonomicConstraints `true` to subtract recognized water
   * and SHAKE constraints; `false` to retain only periodic-boundary
   * reductions.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if no PSF or no force manager is
   * attached.
   *
   * @post `getNumDegreesOfFreedom()` returns the newly computed count.
   */
  void useHolonomicConstraints(const bool usingHolonomicConstraints);

  /**
   * @brief Sets the Ewald splitting parameter.
   *
   * @param[in] kappa Ewald splitting parameter in inverse angstroms. The value
   * must be finite and non-negative.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if no force manager is attached.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if `kappa` is non-finite or negative.
   *
   * @warning Configure this value before force-manager initialization. The
   * current setter updates stored configuration but does not rebuild already
   * initialized reciprocal- or direct-space force objects.
   */
  void setKappa(const float kappa);

  /**
   * @brief Sets the direct-space neighbor-list cutoff.
   *
   * @param[in] cutoff Cutoff distance in angstroms. The value must be finite
   * and positive. Force-manager initialization additionally requires it not to
   * exceed half the X box length.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if no force manager is attached.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if `cutoff` is non-finite or not
   * positive.
   *
   * @warning Configure this value before force-manager initialization. The
   * current setter does not rebuild an already initialized neighbor-list
   * implementation.
   */
  void setCutoff(const float cutoff);

  /**
   * @brief Sets the nonbonded distance exposed as `ctonnb`.
   *
   * @param[in] ctonnb Distance in angstroms. The value must be finite and
   * positive.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if no force manager is attached.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if `ctonnb` is non-finite or not
   * positive.
   *
   * @warning The current direct-force initialization forwards this stored value
   * to the backend parameter named `roff`. Configure it before force-manager
   * initialization.
   */
  void setCtonnb(const float ctonnb);

  /**
   * @brief Sets the nonbonded distance exposed as `ctofnb`.
   *
   * @param[in] ctofnb Distance in angstroms. The value must be finite and
   * positive.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if no force manager is attached.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if `ctofnb` is non-finite or not
   * positive.
   *
   * @warning The current direct-force initialization forwards this stored value
   * to the backend parameter named `ron`. Configure it before force-manager
   * initialization.
   */
  void setCtofnb(const float ctofnb);

  /**
   * @brief Sets the three-dimensional PME FFT grid.
   *
   * @param[in] nfftx Positive number of grid points along X.
   * @param[in] nffty Positive number of grid points along Y.
   * @param[in] nfftz Positive number of grid points along Z.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if no force manager is attached.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if any dimension is not positive.
   *
   * @warning Configure the grid before force-manager initialization. The
   * current setter does not rebuild an initialized PME object.
   */
  void setFFTGrid(const int nfftx, const int nffty, const int nfftz);

  /**
   * @brief Sets the PME interpolation spline order.
   *
   * @param[in] pmeSplineOrder Positive dimensionless spline order.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if no force manager is attached.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if `pmeSplineOrder` is not positive.
   *
   * @warning Configure the order before force-manager initialization. The
   * current setter does not rebuild an initialized PME object.
   */
  void setPmeSplineOrder(const int pmeSplineOrder);

  /**
   * @brief Sets the van der Waals model identifier.
   *
   * @param[in] vdwType One of the native constants from `VDW_VSH` through
   * `VDW_DBEXP`, with numeric range `[1, 6]`.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if no force manager is attached.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if `vdwType` is outside `[1, 6]`.
   *
   * @warning Configure the model before force-manager initialization. The
   * current setter does not rebuild an initialized direct-space force object.
   */
  void setVdwType(const int vdwType);

public: // Getters
  /**
   * @brief Returns the configured atom count.
   *
   * @return Positive atom count after initialization, or `-1` in a
   * default-constructed context before `setNumAtoms()` or `setPsf()`.
   */
  int getNumAtoms(void) const;

  /**
   * @brief Returns the current number of degrees of freedom.
   *
   * @return Dimensionless degree-of-freedom count, or `-1` before it has been
   * computed.
   */
  int getNumDegreesOfFreedom(void) const;

  /**
   * @brief Returns the single-precision coordinate/charge container.
   *
   * Each `float4` is `[x, y, z, charge]` in atom order. Coordinates use
   * angstroms and charges use elementary-charge units.
   *
   * @return Borrowed const alias to context-owned storage. No transfer or CUDA
   * synchronization is performed. The reference remains valid until context
   * destruction, but element addresses can be invalidated by resizing.
   */
  const CudaContainer<float4> &getCoordinatesChargesSP(void) const;

  /**
   * @brief Returns the double-precision coordinate/charge container.
   *
   * Each `double4` is `[x, y, z, charge]` in atom order. Coordinates use
   * angstroms and charges use elementary-charge units.
   *
   * @return Borrowed const alias to context-owned storage. No transfer or CUDA
   * synchronization is performed. The reference remains valid until context
   * destruction, but element addresses can be invalidated by resizing.
   */
  const CudaContainer<double4> &getCoordinatesChargesDP(void) const;

  /**
   * @brief Returns the velocity/inverse-mass container.
   *
   * Each `double4` is `[vx, vy, vz, inverse_mass]` in atom order. Velocity uses
   * angstroms per AKMA time unit and inverse mass uses reciprocal atomic mass
   * units.
   *
   * @return Borrowed const alias to context-owned storage. No transfer or CUDA
   * synchronization is performed. The reference remains valid until context
   * destruction, but element addresses can be invalidated by resizing.
   */
  const CudaContainer<double4> &getVelocitiesInverseMasses(void) const;

  /**
   * @brief Returns the stored target temperature.
   *
   * @return Temperature in kelvin. This is the value set by `setTemperature()`
   * or `assignVelocitiesAtTemperature()`, not a newly computed kinetic
   * temperature.
   */
  double getTemperature(void) const;

  /**
   * @brief Returns the stored periodic boundary condition.
   *
   * @return Current `PBC` value. A default-constructed context stores
   * `PBC::P1` even before periodic-boundary state has been explicitly marked
   * present.
   */
  PBC getPeriodicBoundaryCondition(void) const;

  /**
   * @brief Returns the stored box dimensions.
   *
   * @return Borrowed const alias to the context-owned X, Y, Z vector in
   * angstroms. Before valid dimensions are set, the vector contains the
   * constructor sentinel values.
   */
  const std::vector<double> &getBoxDimensions(void) const;

  /**
   * @brief Returns the random seed used for generated velocities.
   *
   * @return Unsigned 64-bit dimensionless seed.
   */
  std::uint64_t getRandomSeed(void) const;

  /**
   * @brief Reports whether holonomic constraints are included in DOF
   * accounting.
   *
   * @return `true` when water and SHAKE constraints are subtracted from the
   * degree-of-freedom count; otherwise `false`.
   */
  bool usingHolonomicConstraints(void) const;

  /**
   * @brief Returns the orthorhombic box volume.
   *
   * @return Product of the three box lengths in cubic angstroms.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if valid positive box dimensions have
   * not been set.
   */
  double getVolume(void) const;

  /**
   * @brief Returns the context-owned pressure storage.
   *
   * @return Borrowed const alias to the nine-element pressure container. No
   * transfer or synchronization is performed.
   *
   * @warning `computePressure()` is not implemented, so the current subsystem
   * does not provide a supported operation that populates this container.
   */
  const CudaContainer<double> &getPressure(void) const;

  /**
   * @brief Returns the PSF bond records.
   *
   * @return Borrowed const alias to the attached PSF's bond vector. The
   * reference is valid while the retained PSF remains alive and is not
   * structurally modified.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if no PSF is attached.
   */
  const std::vector<Bond> &getBonds(void) const;

  /**
   * @brief Returns the PSF water-molecule index records.
   *
   * @return Borrowed const alias to the attached PSF's `int4` CUDA container.
   * No transfer or synchronization is performed. The component interpretation
   * is defined by the PSF representation.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if no PSF is attached.
   */
  const CudaContainer<int4> &getWaterMolecules(void) const;

  /**
   * @brief Returns the initialized SHAKE atom records.
   *
   * @return Borrowed const alias to the force manager's `int4` SHAKE atom
   * container. No transfer or synchronization is performed.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if the force manager is missing or not
   * initialized.
   */
  const CudaContainer<int4> &getShakeAtoms(void) const;

  /**
   * @brief Returns the initialized SHAKE parameter records.
   *
   * @return Borrowed const alias to the force manager's `float4` SHAKE
   * parameter container. No transfer or synchronization is performed.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if the force manager is missing or not
   * initialized.
   */
  const CudaContainer<float4> &getShakeParams(void) const;

  /**
   * @brief Returns the stored Ewald splitting parameter.
   *
   * @return Kappa in inverse angstroms.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if no force manager is attached.
   */
  float getKappa(void) const;

  /**
   * @brief Returns the stored direct-space cutoff.
   *
   * @return Cutoff distance in angstroms.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if no force manager is attached.
   */
  float getCutoff(void) const;

  /**
   * @brief Returns the distance exposed as `ctonnb`.
   *
   * @return Stored `ctonnb` value in angstroms.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if no force manager is attached.
   */
  float getCtonnb(void) const;

  /**
   * @brief Returns the distance exposed as `ctofnb`.
   *
   * @return Stored `ctofnb` value in angstroms.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if no force manager is attached.
   */
  float getCtofnb(void) const;

  /**
   * @brief Returns the stored PME FFT grid.
   *
   * @return New three-element vector in X, Y, Z order. Values are
   * dimensionless grid-point counts and can be non-positive before automatic
   * or explicit grid selection.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if no force manager is attached.
   */
  std::vector<int> getFFTGrid(void) const;

  /**
   * @brief Returns the stored PME interpolation spline order.
   *
   * @return Positive dimensionless spline order for a configured manager.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if no force manager is attached.
   */
  int getPmeSplineOrder(void) const;

  /**
   * @brief Returns the stored van der Waals model identifier.
   *
   * @return Native model code in the `VDW_VSH` through `VDW_DBEXP` range.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if no force manager is attached.
   */
  int getVdwType(void) const;

  /**
   * @brief Returns the component stride of the total-force storage.
   *
   * @return Number of scalar elements between the X, Y, and Z component
   * segments in the force manager's structure-of-arrays representation.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if the force manager is missing or not
   * initialized.
   */
  int getForceStride(void) const;

  /**
   * @brief Returns mutable single-precision coordinate/charge storage.
   *
   * @return Borrowed mutable alias to the context-owned `float4` container.
   *
   * @warning Mutations through this alias bypass coordinate-presence tracking,
   * double-precision mirror updates, host/device transfer, image centering, and
   * neighbor-list rebuilding.
   */
  CudaContainer<float4> &getCoordinatesChargesSP(void);

  /**
   * @brief Returns mutable double-precision coordinate/charge storage.
   *
   * @return Borrowed mutable alias to the context-owned `double4` container.
   *
   * @warning Mutations through this alias bypass coordinate-presence tracking,
   * single-precision mirror updates, host/device transfer, image centering, and
   * neighbor-list rebuilding.
   */
  CudaContainer<double4> &getCoordinatesChargesDP(void);

  /**
   * @brief Returns mutable velocity/inverse-mass storage.
   *
   * @return Borrowed mutable alias to the context-owned `double4` container.
   *
   * @warning Mutations through this alias bypass unit, shape, and atom-count
   * validation and do not perform host/device transfer.
   */
  CudaContainer<double4> &getVelocitiesInverseMasses(void);

  /**
   * @brief Returns mutable box-dimension storage.
   *
   * @return Borrowed mutable alias to the context-owned X, Y, Z vector in
   * angstroms.
   *
   * @warning Mutations through this alias bypass validation, do not update the
   * internal box-present flag, are not forwarded to the force manager, and do
   * not rebuild the neighbor list.
   */
  std::vector<double> &getBoxDimensions(void);

  /**
   * @brief Computes and returns the current kinetic energy.
   *
   * The calculation performs a deterministic CUDA reduction over velocity and
   * inverse-mass storage, synchronizes the current CUDA device, and transfers
   * the resulting scalar to host memory.
   *
   * @return Kinetic energy in kilocalories per mole.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if the atom count or
   * velocity/inverse-mass storage is not initialized consistently.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if workspace
   * allocation, kernel launch, synchronization, or result transfer fails.
   */
  double getKineticEnergy(void);

  /**
   * @brief Returns mutable pressure storage.
   *
   * @return Borrowed mutable alias to the context-owned nine-element pressure
   * container.
   *
   * @warning `computePressure()` is not implemented. Mutations through this
   * alias have no validation or defined integration with force calculations.
   */
  CudaContainer<double> &getPressure(void);

  /**
   * @brief Returns the attached CharmmPSF.
   *
   * @return A copied shared owner of the PSF, or an empty pointer when no PSF
   * is attached. The returned owner keeps the native PSF alive independently
   * of the context.
   */
  std::shared_ptr<CharmmPSF> getPsf(void);

  /**
   * @brief Returns the attached CharmmParameters.
   *
   * @return A copied shared owner of the parameter set, or an empty pointer
   * when no parameters are attached. The returned owner keeps the native
   * object alive independently of the context.
   */
  std::shared_ptr<CharmmParameters> getPrm(void);

  /**
   * @brief Returns the attached ForceManager.
   *
   * @return A copied shared owner of the manager, or an empty pointer when no
   * manager is attached. The returned manager can outlive the context; its
   * backlink to the context is weak.
   */
  std::shared_ptr<ForceManager> getForceManager(void);

  /**
   * @brief Returns the total potential-energy container.
   *
   * @return Borrowed mutable alias to the initialized force manager's
   * single-element potential-energy container in kilocalories per mole. No
   * host transfer is performed by this accessor.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if the force manager is missing or not
   * initialized.
   */
  CudaContainer<double> &getPotentialEnergy(void);

  /**
   * @brief Returns the legacy host-summed potential-energy value.
   *
   * The current implementation copies selected standard bonded,
   * direct-space, and reciprocal energy components to host memory and sums
   * them.
   *
   * @return Selected potential-energy sum in kilocalories per mole.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if the force manager is missing or not
   * initialized.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if component
   * transfer or stream synchronization fails.
   *
   * @warning The current sum omits CMAP and subscribed-force energy
   * contributions.
   */
  float getPotentialEnergies(void);

  /**
   * @brief Returns the total force storage.
   *
   * The force manager synchronizes its aggregation stream before returning.
   *
   * @return A copied shared owner of the initialized total `Force<double>`
   * object. Force values use kilocalories per mole per angstrom and follow the
   * force object's X/Y/Z stride layout.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if the force manager is missing or not
   * initialized.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if stream
   * synchronization fails.
   */
  std::shared_ptr<Force<double>> getForces(void);

  /**
   * @brief Aggregates and returns the total virial storage.
   *
   * The force manager combines bonded, reciprocal, direct, and eligible
   * subscribed-force virials into its nine-element total container.
   *
   * @return Borrowed mutable alias to the force manager's total virial
   * container.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if the force manager is missing or not
   * initialized.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if a required
   * transfer or synchronization fails.
   */
  CudaContainer<double> &getVirial(void);

  /**
   * @brief Returns mutable PSF bond records.
   *
   * @return Borrowed mutable alias to the attached PSF's bond vector.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if no PSF is attached.
   *
   * @warning Structural mutation can invalidate references and can make the
   * initialized force manager inconsistent with its topology.
   */
  std::vector<Bond> &getBonds(void);

  /**
   * @brief Returns mutable PSF water-molecule records.
   *
   * @return Borrowed mutable alias to the attached PSF's `int4` container.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if no PSF is attached.
   *
   * @warning Mutations bypass PSF validation and do not rebuild initialized
   * holonomic-constraint state.
   */
  CudaContainer<int4> &getWaterMolecules(void);

  /**
   * @brief Returns mutable SHAKE atom records.
   *
   * @return Borrowed mutable alias to the initialized force manager's `int4`
   * SHAKE atom container.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if the force manager is missing or not
   * initialized.
   *
   * @warning Mutations bypass constraint validation and degree-of-freedom
   * recomputation.
   */
  CudaContainer<int4> &getShakeAtoms(void);

  /**
   * @brief Returns mutable SHAKE parameter records.
   *
   * @return Borrowed mutable alias to the initialized force manager's `float4`
   * SHAKE parameter container.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if the force manager is missing or not
   * initialized.
   *
   * @warning Mutations bypass constraint validation and host/device coherence
   * management.
   */
  CudaContainer<float4> &getShakeParams(void);

public: // Specialized functions
  /**
   * @brief Assigns Maxwell-Boltzmann velocity components at a temperature.
   *
   * A new `std::mt19937` is initialized from the stored random seed on every
   * call. For atom `i`, each Cartesian component is drawn independently from a
   * zero-mean normal distribution with standard deviation
   * `sqrt(k_B * temperature * inverse_mass[i])`. The resulting velocities are
   * transferred to device storage.
   *
   * The operation does not remove center-of-mass motion or rescale the sampled
   * kinetic temperature.
   *
   * @param[in] temperature Sampling temperature in kelvin. The value must be
   * finite and non-negative.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if the atom count is not positive.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::InvalidArgument` if `temperature` is NaN, infinite, or
   * negative.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if the
   * host-to-device transfer fails.
   *
   * @post The stored target temperature equals `temperature`.
   */
  void assignVelocitiesAtTemperature(const double temperature);

  /**
   * @brief Computes the instantaneous kinetic temperature.
   *
   * The method computes kinetic energy on the GPU and evaluates
   * `K / (0.5 * ndegf * k_B)`.
   *
   * @return Instantaneous temperature in kelvin.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if atom and velocity storage are
   * inconsistent or the degree-of-freedom count is not positive.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if kinetic
   * energy calculation, synchronization, or transfer fails.
   */
  double computeTemperature(void);

  /**
   * @brief Reports that pressure computation is not implemented.
   *
   * @throws ApoCharmmError Always, with code
   * `ApoCharmmErrorCode::NotImplemented`.
   */
  void computePressure(void);

  /**
   * @brief Images complete PSF groups into the central periodic box.
   *
   * The operation computes each contiguous PSF group's arithmetic coordinate
   * center and translates the full group by box lengths as needed. Under
   * `PBC::P21`, crossing an X boundary also reflects Y and Z coordinates,
   * velocity components, and force components.
   *
   * The kernel mutates the device copies of single- and double-precision
   * coordinates, velocity components, and forces. It launches on the default
   * stream and finishes with `cudaDeviceSynchronize()`. Host mirrors are not
   * updated.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if the force manager is missing or not
   * initialized, the PSF is missing, or valid box dimensions are absent.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if the kernel
   * launch or device synchronization fails.
   *
   * @warning Host coordinate and velocity mirrors can be stale after this
   * method returns. Transfer from device before reading them on the host.
   */
  void imageCentering(void);

  /**
   * @brief Images coordinates and rebuilds the direct-space neighbor list.
   *
   * `imageCentering()` is called first, after which the force manager rebuilds
   * its neighbor list from the single-precision device coordinate/charge
   * array.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if required force-manager, PSF, or box
   * state is missing.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if imaging,
   * synchronization, or neighbor-list rebuilding fails.
   */
  void resetNeighborList(void);

  /**
   * @brief Computes kinetic energy into the context-owned device scalar.
   *
   * The implementation uses a fixed deterministic reduction hierarchy. The
   * first pass calculates one block-indexed partial sum for each contiguous
   * atom range. When multiple partial sums exist, a single-block second pass
   * reduces them in a fixed index order and directly overwrites the result
   * scalar. A one-block input writes the result directly and skips the second
   * kernel.
   *
   * No floating-point atomic accumulation is used. The method synchronizes the
   * current CUDA device before returning.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if atom and
   * velocity/inverse-mass storage are inconsistent.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if partial-sum
   * storage allocation, kernel launch, or synchronization fails.
   *
   * @note Run-to-run arithmetic order is fixed for a given executable and
   * launch configuration. Changing the block size, items per thread, compiler
   * options, or CUDA target can change floating-point rounding.
   */
  void calculateKineticEnergy(void);

  /**
   * @brief Computes forces, potential energy, and virial.
   *
   * The call delegates to `ForceManager::calcForce()` with energy and virial
   * calculation enabled. When `print` is `true`, it subsequently prints a
   * CHARMM-style energy table to `std::cout` and updates the table's evaluation
   * and delta-energy history.
   *
   * @param[in] reset Requests the force-manager reset mode. The current
   * `ForceManager` implementation rejects `true` as not implemented.
   * @param[in] print `true` to print the CHARMM-style energy table; otherwise
   * `false`.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if the force manager is missing or not
   * initialized.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotImplemented` if `reset` is true or energy-table
   * printing is requested for a composite force manager.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if force,
   * energy, virial, transfer, or synchronization work fails.
   * @throws std::out_of_range If a required named energy component is absent
   * while formatting the printed energy table.
   */
  void calculatePotentialEnergy(const bool reset = false,
                                const bool print = false);

  /**
   * @brief Computes forces with selectable energy and virial accumulation.
   *
   * @param[in] reset Requests the force-manager reset mode. The current
   * `ForceManager` implementation rejects `true` as not implemented.
   * @param[in] calcEnergy `true` to calculate energy components.
   * @param[in] calcVirial `true` to calculate virial components.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if the force manager is missing or not
   * initialized.
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotImplemented` if `reset` is true.
   * @throws ApoCharmmError With code `ApoCharmmErrorCode::Cuda` if force,
   * energy, virial, or stream work fails.
   */
  void calculateForces(const bool reset = false, const bool calcEnergy = false,
                       const bool calcVirial = false);

  /**
   * @brief Stores a weak context backlink in the attached ForceManager.
   *
   * The operation uses `weak_from_this().lock()`. It has no effect when this
   * object is not currently owned by a `std::shared_ptr`.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if no force manager is attached.
   *
   * @post When shared ownership exists, the force manager observes this context
   * through a non-owning weak pointer.
   */
  void linkBackForceManager(void);

protected:
  /**
   * @brief Formats and prints the current CHARMM-style energy table.
   *
   * The helper transfers total potential energy to host, obtains host force
   * components, computes gradient RMS, prints named standard energy
   * components, and advances the context-local print-history counters.
   *
   * @pre The force manager is initialized and exposes all component names read
   * by the formatter.
   */
  void printEnergyTable(void);

  /**
   * @brief Imports collaborator state from the attached ForceManager.
   *
   * Existing PSF atom storage is resized and initialized when the manager's PSF
   * changes. Parameters, box dimensions, and periodic-boundary state are then
   * copied into the context.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if no force manager is attached.
   */
  void syncStateFromForceManager(void);

  /**
   * @brief Reconciles context state with the attached ForceManager.
   *
   * Context PSF, parameters, box, and periodic-boundary state take precedence
   * when present. Missing context state is imported from the manager.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if no force manager is attached.
   */
  void syncForceManagerFromState(void);

  /**
   * @brief Reports whether force-manager initialization prerequisites exist.
   *
   * @return `true` only when a force manager, PSF, parameters, and exactly
   * three positive box dimensions are present.
   */
  bool hasCompleteForceManagerState(void) const;

  /**
   * @brief Initializes the force manager when all required state is available.
   *
   * The helper returns without effect when the manager is missing, already
   * initialized, or incomplete. After initialization it reimports manager
   * state into the context.
   */
  void initializeForceManagerIfReady(void);

  /**
   * @brief Requires an attached PSF.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if no PSF is attached.
   */
  void requirePsf(void) const;

  /**
   * @brief Requires an attached ForceManager.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if no manager is attached.
   */
  void requireForceManager(void) const;

  /**
   * @brief Requires an attached and initialized ForceManager.
   *
   * @throws ApoCharmmError With code
   * `ApoCharmmErrorCode::NotInitialized` if the manager is missing or not
   * initialized.
   */
  void requireInitializedForceManager(void) const;

  /**
   * @brief Advances the context/force-manager setup state when possible.
   *
   * The helper establishes the weak backlink, initializes a complete manager,
   * recomputes the degree-of-freedom count for initialized state, and rebuilds
   * the neighbor list when initialization occurred after coordinates were
   * supplied.
   */
  void finalizeSetupIfReady(void);

protected:
  /** Seed used to initialize the per-call velocity generator. */
  std::uint64_t m_RandomSeed;

  /** Shared PSF owner retained by the context. */
  std::shared_ptr<CharmmPSF> m_Psf;
  /** Shared parameter-set owner retained by the context. */
  std::shared_ptr<CharmmParameters> m_Prm;
  /** Orthorhombic X, Y, Z box lengths in angstroms. */
  std::vector<double> m_BoxDimensions;
  /** Records whether valid positive box dimensions were established. */
  bool m_HasBoxDimensions;

  /**
   * Shared force-manager owner. The reverse manager-to-context association is
   * weak.
   */
  std::shared_ptr<ForceManager> m_ForceManager;

  /** Number of atoms represented by every per-atom context array. */
  int m_NumAtoms;
  /** Current mechanical degree-of-freedom count. */
  int m_NumDegreesOfFreedom;

  /** Stored periodic-boundary condition. */
  PBC m_Pbc;
  /** Records whether periodic-boundary state was explicitly imported or set. */
  bool m_HasPbc;

  /** Single-precision `[x, y, z, charge]` host/device mirrors. */
  CudaContainer<float4> m_CoordinatesChargesSP;
  /** Double-precision `[x, y, z, charge]` host/device mirrors. */
  CudaContainer<double4> m_CoordinatesChargesDP;
  /** Records whether coordinates were supplied after the latest resize. */
  bool m_HasCoordinates;

  /** Double-precision `[vx, vy, vz, inverse_mass]` host/device mirrors. */
  CudaContainer<double4> m_VelocitiesInverseMasses;

  /** Stores one kinetic-energy partial sum per first-pass CUDA block. */
  CudaContainer<double> m_KineticEnergyPartialSums;
  /** Single-element kinetic-energy result container. */
  CudaContainer<double> m_KineticEnergy;
  /** Nine-element pressure storage; pressure computation is not implemented. */
  CudaContainer<double> m_Pressure;
  /** Nine-element kinetic virial tensor storage. */
  CudaContainer<double> m_VirialKineticEnergyTensor;

  /** Number printed in the next CHARMM-style energy table. */
  std::uint64_t m_EnergyTableEvaluationCount;
  /** Total energy printed by the previous table. */
  double m_PreviousPrintedPotentialEnergy;
  /** Records whether a previous printed energy exists for delta calculation. */
  bool m_HasPreviousPrintedPotentialEnergy;

  /** Stored target temperature in kelvin. */
  double m_Temperature;
  /** Selects constrained degree-of-freedom accounting. */
  bool m_UsingHolonomicConstraints;
};
