// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "AtomSelection.h"
#include "CharmmPSF.h"
#include "CudaContainer.h"
#include "ForceManager.h"
#include "GeometricRestraintForce.h"
#include "apo_test_helpers.h"
#include "apocharmm_c/AtomReference.h"
#include "apocharmm_c/AtomSelection.h"
#include "apocharmm_c/CharmmContext.h"
#include "apocharmm_c/CharmmParameters.h"
#include "apocharmm_c/CharmmPsf.h"
#include "apocharmm_c/Error.h"
#include "apocharmm_c/ForceManager.h"
#include "apocharmm_c/GeometricRestraintForce.h"
#include "apocharmm_c/Status.h"
#include "apocharmm_c/detail/AtomReferenceHandle.h"
#include "apocharmm_c/detail/AtomSelectionHandle.h"
#include "apocharmm_c/detail/CharmmPsfHandle.h"
#include "apocharmm_c/detail/ForceManagerHandle.h"
#include "apocharmm_c/detail/GeometricRestraintForceHandle.h"
#include "catch.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

constexpr int NUM_ATOMS = 4;
constexpr char CREATE_FUNCTION_NAME[] = "apo_geometric_restraint_force_create";
constexpr char SET_MASSES_FUNCTION_NAME[] =
    "apo_geometric_restraint_force_set_masses";
constexpr char MMFP_SOURCE_FILE[] = "src/GeometricRestraintForce.cu";
constexpr double QUIET_NAN = std::numeric_limits<double>::quiet_NaN();
constexpr double INF = std::numeric_limits<double>::infinity();
constexpr double ORIGIN[3] = {0.0, 0.0, 0.0};
constexpr double X_DIRECTION[3] = {1.0, 0.0, 0.0};
constexpr double ZERO_DIRECTION[3] = {0.0, 0.0, 0.0};
constexpr apo_geometric_restraint_activation SYMMETRIC =
    APO_GEOMETRIC_RESTRAINT_ACTIVATION_SYMMETRIC;
constexpr apo_geometric_restraint_activation INSIDE =
    APO_GEOMETRIC_RESTRAINT_ACTIVATION_INSIDE;
constexpr apo_geometric_restraint_activation OUTSIDE =
    APO_GEOMETRIC_RESTRAINT_ACTIVATION_OUTSIDE;
constexpr apo_geometric_restraint_selection_mode ATOMWISE =
    APO_GEOMETRIC_RESTRAINT_SELECTION_MODE_ATOMWISE;
constexpr apo_geometric_restraint_selection_mode RCM =
    APO_GEOMETRIC_RESTRAINT_SELECTION_MODE_RCM;

constexpr int CHARMM_NUM_ATOMS = 12;
constexpr char PSF_FILE[] = "tmp_capi_geometric_restraint_force_manager.psf";
constexpr char PARAMETER_FILE[] =
    "tmp_capi_geometric_restraint_force_manager.prm";
constexpr char SUBSCRIBE_FUNCTION_NAME[] =
    "apo_force_manager_subscribe_geometric_restraint_force";
constexpr char UNSUBSCRIBE_FUNCTION_NAME[] =
    "apo_force_manager_unsubscribe_geometric_restraint_force";
constexpr char SET_PBC_FUNCTION_NAME[] =
    "apo_force_manager_set_periodic_boundary_condition";
constexpr char FORCE_MANAGER_HEADER_FILE[] = "include/ForceManager.h";
constexpr char FORCE_MANAGER_SOURCE_FILE[] = "src/ForceManager.cu";
constexpr double BOX_DIMENSIONS[3] = {20.0, 30.0, 40.0};
constexpr double INV_FORCE_SCALE_TEST = 1.0 / static_cast<double>(1LL << 40);
constexpr double ARITHMETIC_TOLERANCE =
    64.0 * std::numeric_limits<double>::epsilon();

constexpr char ZERO_INTERACTION_PSF_TEXT[] = R"PSF(PSF

       1 !NTITLE
 REMARKS C ABI MMFP zero-interaction system
      12 !NATOM
       1 MMFP     1 DUM  DUM  DUM    0.000000   1.0000           0
       2 MMFP     2 DUM  DUM  DUM    0.000000   1.5000           0
       3 MMFP     3 DUM  DUM  DUM    0.000000   2.0000           0
       4 MMFP     4 DUM  DUM  DUM    0.000000   2.5000           0
       5 MMFP     5 DUM  DUM  DUM    0.000000   3.0000           0
       6 MMFP     6 DUM  DUM  DUM    0.000000   3.5000           0
       7 MMFP     7 DUM  DUM  DUM    0.000000   4.0000           0
       8 MMFP     8 DUM  DUM  DUM    0.000000   4.5000           0
       9 MMFP     9 DUM  DUM  DUM    0.000000   5.0000           0
      10 MMFP    10 DUM  DUM  DUM    0.000000   5.5000           0
      11 MMFP    11 DUM  DUM  DUM    0.000000   6.0000           0
      12 MMFP    12 DUM  DUM  DUM    0.000000   6.5000           0
       0 !NBOND: bonds
       0 !NTHETA: angles
       0 !NPHI: dihedrals
       0 !NIMPHI: impropers
       0 !NDON: donors
       0 !NACC: acceptors
       0 !NCRTERM: cross-terms
)PSF";

constexpr char ZERO_INTERACTION_PARAMETER_TEXT[] = "NONBONDED\n"
                                                   "DUM 0.0 0.0 1.0\n"
                                                   "END\n";

constexpr std::array<double, CHARMM_NUM_ATOMS> STANDARD_MASSES = {
    1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5};
constexpr std::array<double, CHARMM_NUM_ATOMS> UNEQUAL_MASSES = {
    1.0, 3.0, 2.0, 4.0, 1.5, 5.0, 2.5, 6.0, 3.5, 7.0, 4.5, 8.0};
constexpr std::array<double, CHARMM_NUM_ATOMS> EQUAL_MASSES = {
    2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
constexpr std::array<double, CHARMM_NUM_ATOMS> NEGATIVE_TOTAL_MASSES = {
    1.0, -3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
constexpr std::array<double, CHARMM_NUM_ATOMS> ZERO_TOTAL_MASSES = {
    1.0, -1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};

enum class Geometry { PLANE, CYLINDER, SPHERE };

enum class AtomRoute { SELECTION, ATOM_REFERENCE };

struct CharmmTerm {
  Geometry geometry;
  apo_geometric_restraint_activation activation;
  apo_geometric_restraint_selection_mode selectionMode;
  std::vector<int> atoms;
  std::array<double, 3> origin;
  std::array<double, 3> direction;
  double forceConstant;
  double offset;
};

struct CharmmForce {
  int atom;
  std::array<double, 3> force;
};

struct CharmmPhase {
  std::string_view name;
  bool resetFirst;
  std::vector<CharmmTerm> terms;
  bool energyReported;
  double energy;
  std::vector<CharmmForce> forces;
  std::array<double, 9> virial;
  double energyTolerance;
  double forceTolerance;
  double virialTolerance;
};

struct CharmmCase {
  std::string_view id;
  std::vector<std::array<double, 3>> coordinatePrefix;
  std::array<double, CHARMM_NUM_ATOMS> masses;
  bool translateFirstAtomToOrigin;
  std::vector<CharmmPhase> phases;
};

struct ManagerOutput {
  std::array<std::array<double, 3>, CHARMM_NUM_ATOMS> gradients;
  double energy;
  std::array<double, 9> virial;
};

struct GeometricRestraintForceDeleter {
  void operator()(apo_geometric_restraint_force *restraint) const noexcept {
    apo_geometric_restraint_force_destroy(restraint);
    return;
  }
};

struct CharmmPsfDeleter {
  void operator()(apo_charmm_psf *psf) const noexcept {
    apo_charmm_psf_destroy(psf);
    return;
  }
};

struct AtomSelectionDeleter {
  void operator()(apo_atom_selection *selection) const noexcept {
    apo_atom_selection_destroy(selection);
    return;
  }
};

struct AtomReferenceDeleter {
  void operator()(apo_atom_reference *reference) const noexcept {
    apo_atom_reference_destroy(reference);
    return;
  }
};

struct CharmmParametersDeleter {
  void operator()(apo_charmm_parameters *parameters) const noexcept {
    apo_charmm_parameters_destroy(parameters);
    return;
  }
};

struct ForceManagerDeleter {
  void operator()(apo_force_manager *forceManager) const noexcept {
    apo_force_manager_destroy(forceManager);
    return;
  }
};

struct CharmmContextDeleter {
  void operator()(apo_charmm_context *context) const noexcept {
    apo_charmm_context_destroy(context);
    return;
  }
};

using GeometricRestraintForceHandle =
    std::unique_ptr<apo_geometric_restraint_force,
                    GeometricRestraintForceDeleter>;
using CharmmPsfHandle = std::unique_ptr<apo_charmm_psf, CharmmPsfDeleter>;
using AtomSelectionHandle =
    std::unique_ptr<apo_atom_selection, AtomSelectionDeleter>;
using AtomReferenceHandle =
    std::unique_ptr<apo_atom_reference, AtomReferenceDeleter>;
using CharmmParametersHandle =
    std::unique_ptr<apo_charmm_parameters, CharmmParametersDeleter>;
using ForceManagerHandle =
    std::unique_ptr<apo_force_manager, ForceManagerDeleter>;
using CharmmContextHandle =
    std::unique_ptr<apo_charmm_context, CharmmContextDeleter>;

struct ZeroInteractionSystem {
  CharmmPsfHandle psf;
  CharmmParametersHandle parameters;
  ForceManagerHandle forceManager;
  CharmmContextHandle context;
};

using TermAddition = std::function<apo_status(
    apo_geometric_restraint_force *, apo_geometric_restraint_activation,
    apo_geometric_restraint_selection_mode)>;

struct NamedTermAddition {
  std::string_view functionName;
  TermAddition add;
};

template <typename Function, typename Atom>
constexpr bool ACCEPTS_DIRECTED_TERM_ATOM = std::is_invocable_v<
    Function, apo_geometric_restraint_force *, Atom, const double *,
    std::size_t, const double *, std::size_t, double, double,
    apo_geometric_restraint_activation, apo_geometric_restraint_selection_mode>;

template <typename Function, typename Atom>
constexpr bool ACCEPTS_SPHERE_TERM_ATOM =
    std::is_invocable_v<Function, apo_geometric_restraint_force *, Atom,
                        const double *, std::size_t, double, double,
                        apo_geometric_restraint_activation,
                        apo_geometric_restraint_selection_mode>;

CharmmPsfHandle MakePsf(const int numAtoms) {
  CharmmPsfHandle psf(new apo_charmm_psf());
  psf->object = std::make_shared<CharmmPSF>();
  psf->object->setNumAtoms(numAtoms);
  for (int atom = 0; atom < numAtoms; atom++)
    psf->object->getMasses()[atom] = static_cast<double>(atom) + 1.0;
  return psf;
}

GeometricRestraintForceHandle MakeRestraint(const apo_charmm_psf *psf) {
  apo_geometric_restraint_force *restraint = nullptr;

  REQUIRE(apo_geometric_restraint_force_create(&restraint, psf) ==
          APO_STATUS_OK);
  REQUIRE(restraint != nullptr);

  return GeometricRestraintForceHandle(restraint);
}

AtomSelectionHandle MakeSelection(const int numAtoms,
                                  const std::vector<int> &selectedAtoms) {
  AtomSelectionHandle selection(new apo_atom_selection());
  selection->object = std::make_shared<AtomSelection>(numAtoms);

  for (const int atomIndex : selectedAtoms)
    selection->object->set(atomIndex);

  return selection;
}

AtomReferenceHandle MakeReference(const apo_charmm_psf *psf,
                                  const int atomIndex) {
  apo_atom_reference *reference = nullptr;

  REQUIRE(apo_atom_reference_create(&reference, psf, atomIndex) ==
          APO_STATUS_OK);
  REQUIRE(reference != nullptr);

  return AtomReferenceHandle(reference);
}

std::vector<NamedTermAddition>
MakeTermAdditions(const apo_atom_selection *selection,
                  const apo_atom_reference *atom) {
  return {
      {"apo_geometric_restraint_force_add_plane_from_selection",
       [selection](apo_geometric_restraint_force *restraint,
                   const apo_geometric_restraint_activation activation,
                   const apo_geometric_restraint_selection_mode selectionMode)
           -> apo_status {
         return apo_geometric_restraint_force_add_plane_from_selection(
             restraint, selection, ORIGIN, 3, X_DIRECTION, 3, 2.0, 1.0,
             activation, selectionMode);
       }},
      {"apo_geometric_restraint_force_add_plane_from_atom_reference",
       [atom](apo_geometric_restraint_force *restraint,
              const apo_geometric_restraint_activation activation,
              const apo_geometric_restraint_selection_mode selectionMode)
           -> apo_status {
         return apo_geometric_restraint_force_add_plane_from_atom_reference(
             restraint, atom, ORIGIN, 3, X_DIRECTION, 3, 2.0, 1.0, activation,
             selectionMode);
       }},
      {"apo_geometric_restraint_force_add_cylinder_from_selection",
       [selection](apo_geometric_restraint_force *restraint,
                   const apo_geometric_restraint_activation activation,
                   const apo_geometric_restraint_selection_mode selectionMode)
           -> apo_status {
         return apo_geometric_restraint_force_add_cylinder_from_selection(
             restraint, selection, ORIGIN, 3, X_DIRECTION, 3, 2.0, 1.0,
             activation, selectionMode);
       }},
      {"apo_geometric_restraint_force_add_cylinder_from_atom_reference",
       [atom](apo_geometric_restraint_force *restraint,
              const apo_geometric_restraint_activation activation,
              const apo_geometric_restraint_selection_mode selectionMode)
           -> apo_status {
         return apo_geometric_restraint_force_add_cylinder_from_atom_reference(
             restraint, atom, ORIGIN, 3, X_DIRECTION, 3, 2.0, 1.0, activation,
             selectionMode);
       }},
      {"apo_geometric_restraint_force_add_sphere_from_selection",
       [selection](apo_geometric_restraint_force *restraint,
                   const apo_geometric_restraint_activation activation,
                   const apo_geometric_restraint_selection_mode selectionMode)
           -> apo_status {
         return apo_geometric_restraint_force_add_sphere_from_selection(
             restraint, selection, ORIGIN, 3, 2.0, 1.0, activation,
             selectionMode);
       }},
      {"apo_geometric_restraint_force_add_sphere_from_atom_reference",
       [atom](apo_geometric_restraint_force *restraint,
              const apo_geometric_restraint_activation activation,
              const apo_geometric_restraint_selection_mode selectionMode)
           -> apo_status {
         return apo_geometric_restraint_force_add_sphere_from_atom_reference(
             restraint, atom, ORIGIN, 3, 2.0, 1.0, activation, selectionMode);
       }},
  };
}

template <typename Function>
void CheckRestraintHandleValidation(const std::string_view functionName,
                                    Function function) {
  apo_test::CheckStatusAndDiagnostic(
      function(nullptr), APO_STATUS_INVALID_ARGUMENT,
      std::string(functionName) + ": GeometricRestraintForce is NULL");

  apo_geometric_restraint_force emptyRestraint;
  apo_test::CheckStatusAndDiagnostic(
      function(&emptyRestraint), APO_STATUS_INVALID_ARGUMENT,
      std::string(functionName) + ": GeometricRestraintForce object is NULL");

  return;
}

template <typename AtomHandle, typename Function>
void CheckAtomHandleValidation(const std::string_view functionName,
                               const std::string_view handleName,
                               const apo_geometric_restraint_force *restraint,
                               Function function) {
  apo_test::CheckStatusAndDiagnostic(
      function(nullptr), APO_STATUS_INVALID_ARGUMENT,
      std::string(functionName) + ": " + std::string(handleName) + " is NULL");

  AtomHandle emptyHandle;
  apo_test::CheckStatusAndDiagnostic(
      function(&emptyHandle), APO_STATUS_INVALID_ARGUMENT,
      std::string(functionName) + ": " + std::string(handleName) +
          " object is NULL");

  CHECK(restraint->object->getNumTerms() == 0);

  return;
}

template <typename Function>
void CheckCartesianArrayValidation(
    const std::string_view functionName, const std::string_view argumentName,
    const apo_geometric_restraint_force *restraint, Function function) {
  const double values[4] = {0.0, 0.0, 1.0, 0.0};

  apo_test::CheckStatusAndDiagnostic(
      function(nullptr, 3), APO_STATUS_INVALID_ARGUMENT,
      std::string(functionName) + ": " + std::string(argumentName) +
          " is NULL");

  for (const std::size_t length :
       {std::size_t{0}, std::size_t{2}, std::size_t{4}}) {
    INFO("argument length: " << length);
    apo_test::CheckStatusAndDiagnostic(
        function(values, length), APO_STATUS_INVALID_ARGUMENT,
        std::string(functionName) + ": " + std::string(argumentName) +
            " must contain exactly 3 elements");
  }

  CHECK(restraint->object->getNumTerms() == 0);

  return;
}

void CheckMmfpNativeError(const apo_status status,
                          const std::string_view functionName,
                          const std::string_view expectedMessage,
                          const std::string_view sourceFunction) {
  apo_test::CheckNativeError(status, APO_STATUS_INVALID_ARGUMENT,
                             "InvalidArgument", functionName, expectedMessage,
                             MMFP_SOURCE_FILE, sourceFunction);
  return;
}

std::vector<CharmmCase> MakeCharmmCases(void) {
  const std::vector<CharmmCase> cases = {
      {"Positive harmonic constant above the plane",
       {{3.0, 0.0, 0.0}},
       STANDARD_MASSES,
       false,
       {{"Restraint evaluated",
         false,
         {{Geometry::PLANE,
           SYMMETRIC,
           ATOMWISE,
           {0},
           {0.0, 0.0, 0.0},
           {1.0, 0.0, 0.0},
           2.0,
           2.0}},
         true,
         1.0,
         {{0, {-2.0, 0.0, 0.0}}},
         {6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
         5e-07,
         1e-10,
         4.7000000000599994e-09}}},
      {"Symmetric plane on its negative side",
       {{-3.0, 0.0, 0.0}},
       STANDARD_MASSES,
       false,
       {{"Restraint evaluated",
         false,
         {{Geometry::PLANE,
           SYMMETRIC,
           ATOMWISE,
           {0},
           {0.0, 0.0, 0.0},
           {1.0, 0.0, 0.0},
           2.0,
           2.0}},
         true,
         25.0,
         {{0, {10.0, 0.0, 0.0}}},
         {30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
         5e-06,
         1e-10,
         5.100000000059999e-09}}},
      {"Plane with a nonunit oblique normal",
       {{3.0, 2.0, 4.0}},
       STANDARD_MASSES,
       false,
       {{"Restraint evaluated",
         false,
         {{Geometry::PLANE,
           SYMMETRIC,
           ATOMWISE,
           {0},
           {0.5, -1.0, 1.5},
           {2.0, 4.0, 6.0},
           2.0,
           2.0}},
         true,
         5.1809948,
         {{0, {-1.2166693181, -2.4333386361, -3.6500079542}}},
         {3.6500079542999995, 7.300015908299999, 10.9500238626, 2.4333386362,
          4.8666772722, 7.3000159084, 4.8666772724, 9.7333545444,
          14.6000318168},
         5e-08,
         1e-10,
         5.08250039777e-09}}},
      {"INSIDE plane with negative offset at zero distance",
       {},
       STANDARD_MASSES,
       false,
       {{"Restraint evaluated",
         false,
         {{Geometry::PLANE,
           INSIDE,
           ATOMWISE,
           {0},
           {0.0, 0.0, 0.0},
           {1.0, 2.0, 3.0},
           2.0,
           -2.0}},
         true,
         4.0,
         {{0, {-1.0690449676, -2.1380899353, -3.2071349029}}},
         {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
         5e-07,
         1e-10,
         4.660356745205e-09}}},
      {"OUTSIDE plane at zero signed distance",
       {},
       STANDARD_MASSES,
       false,
       {{"Restraint evaluated",
         false,
         {{Geometry::PLANE,
           OUTSIDE,
           ATOMWISE,
           {0},
           {0.0, 0.0, 0.0},
           {1.0, 2.0, 3.0},
           2.0,
           2.0}},
         true,
         4.0,
         {{0, {1.0690449676, 2.1380899353, 3.2071349029}}},
         {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
         5e-07,
         1e-10,
         4.660356745205e-09}}},
      {"Default harmonic potential and INSIDE activation",
       {{4.0, 0.0, 0.0}},
       STANDARD_MASSES,
       false,
       {{"Restraint evaluated",
         false,
         {{Geometry::PLANE,
           INSIDE,
           ATOMWISE,
           {0},
           {0.0, 0.0, 0.0},
           {1.0, 0.0, 0.0},
           2.0,
           2.0}},
         true,
         4.0,
         {{0, {-4.0, 0.0, 0.0}}},
         {16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
         5e-07,
         1e-10,
         4.900000000059999e-09}}},
      {"Cylinder with a nonunit oblique axis",
       {{4.0, 1.0, -1.0}},
       STANDARD_MASSES,
       false,
       {{"Restraint evaluated",
         false,
         {{Geometry::CYLINDER,
           SYMMETRIC,
           ATOMWISE,
           {0},
           {1.0, -2.0, 3.0},
           {2.0, 4.0, 6.0},
           2.0,
           2.0}},
         true,
         14.2548857,
         {{0, {-4.2024438653, -4.4826067897, 4.3892191482}}},
         {16.8097754612, 17.9304271588, -17.5568765928, 4.2024438653,
          4.4826067897, -4.3892191482, -4.2024438653, -4.4826067897,
          4.3892191482},
         5e-08,
         1e-10,
         4.924130339544999e-09}}},
      {"INSIDE cylinder off its axis with a negative offset",
       {{1.0, 0.0, 5.0}},
       STANDARD_MASSES,
       false,
       {{"Restraint evaluated",
         false,
         {{Geometry::CYLINDER,
           INSIDE,
           ATOMWISE,
           {0},
           {0.0, 0.0, 0.0},
           {0.0, 0.0, 1.0},
           2.0,
           -2.0}},
         true,
         9.0,
         {{0, {-6.0, 0.0, 0.0}}},
         {6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 30.0, 0.0, 0.0},
         5e-07,
         1e-10,
         5.30000000006e-09}}},
      {"OUTSIDE cylinder active inside its offset",
       {{1.0, 0.0, 5.0}},
       STANDARD_MASSES,
       false,
       {{"Restraint evaluated",
         false,
         {{Geometry::CYLINDER,
           OUTSIDE,
           ATOMWISE,
           {0},
           {0.0, 0.0, 0.0},
           {0.0, 0.0, 1.0},
           2.0,
           2.0}},
         true,
         1.0,
         {{0, {2.0, 0.0, 0.0}}},
         {-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, -10.0, 0.0, 0.0},
         5e-07,
         1e-10,
         5.10000000006e-09}}},
      {"Sphere distance just below the radial cutoff",
       {{1.2500099999, -2.5, 3.75}},
       STANDARD_MASSES,
       true,
       {{"Restraint evaluated",
         false,
         {{Geometry::SPHERE,
           SYMMETRIC,
           ATOMWISE,
           {0},
           {1.25, -2.5, 3.75},
           {0.0, 0.0, 0.0},
           2.0,
           2.0}},
         true,
         0.0,
         {},
         {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
         5e-07,
         1e-10,
         4.87500000006e-09}}},
      {"Sphere distance just above the radial cutoff",
       {{1.25001, -2.5, 3.75}},
       STANDARD_MASSES,
       true,
       {{"Restraint evaluated",
         false,
         {{Geometry::SPHERE,
           SYMMETRIC,
           ATOMWISE,
           {0},
           {1.25, -2.5, 3.75},
           {0.0, 0.0, 0.0},
           2.0,
           2.0}},
         true,
         3.99996,
         {{0, {3.99998, 0.0, 0.0}}},
         {-5.0000149998, 0.0, 0.0, 9.99995, 0.0, 0.0, -14.999925, 0.0, 0.0},
         5e-07,
         1e-10,
         5.07499900006e-09}}},
      {"Two sphere terms on different atoms",
       {{3.0, 2.0, 1.0}, {4.0, 1.0, 2.0}, {5.0, 3.0, 1.0}},
       UNEQUAL_MASSES,
       false,
       {{"Restraint evaluated",
         false,
         {{Geometry::SPHERE,
           SYMMETRIC,
           ATOMWISE,
           {0},
           {0.0, 0.0, 0.0},
           {0.0, 0.0, 0.0},
           1.0,
           1.0},
          {Geometry::SPHERE,
           SYMMETRIC,
           ATOMWISE,
           {1},
           {1.0, 1.0, 1.0},
           {0.0, 0.0, 0.0},
           1.5,
           1.0}},
         true,
         7.26492612,
         {{0, {-2.1982162743, -1.4654775162, -0.7327387581}},
          {1, {-3.0769750529, 0.0, -1.025658351}}},
         {18.902549034499998, 4.3964325486, 6.3008496783, 7.4734076015,
          2.9309550324, 2.4911358671999997, 8.3521663801, 1.4654775162,
          2.7840554601},
         5e-09,
         1e-10,
         5.663759566419999e-09}}},
      {"Sphere at the negative P1 image coordinate",
       {{-10.5, 0.0, 0.0}},
       STANDARD_MASSES,
       false,
       {{"Restraint evaluated",
         false,
         {{Geometry::SPHERE,
           SYMMETRIC,
           ATOMWISE,
           {0},
           {9.0, 0.0, 0.0},
           {0.0, 0.0, 0.0},
           2.0,
           2.0}},
         true,
         306.25,
         {{0, {35.0, 0.0, 0.0}}},
         {367.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
         5e-05,
         1e-10,
         7.1000000000599994e-09}}},
      {"Large terms cancel before the small term",
       {{3.0, 0.0, 0.0}},
       STANDARD_MASSES,
       false,
       {{"Restraint evaluated",
         false,
         {{Geometry::PLANE,
           SYMMETRIC,
           ATOMWISE,
           {0},
           {0.0, 0.0, 0.0},
           {1.0, 0.0, 0.0},
           1e+16,
           2.0},
          {Geometry::PLANE,
           SYMMETRIC,
           ATOMWISE,
           {0},
           {0.0, 0.0, 0.0},
           {1.0, 0.0, 0.0},
           -1e+16,
           2.0},
          {Geometry::PLANE,
           SYMMETRIC,
           ATOMWISE,
           {0},
           {0.0, 0.0, 0.0},
           {1.0, 0.0, 0.0},
           1.0,
           2.0}},
         true,
         0.5,
         {{0, {-1.0, 0.0, 0.0}}},
         {3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
         5e-08,
         1e-10,
         4.650000000059999e-09}}},
      {"Small term added before large cancellation",
       {{3.0, 0.0, 0.0}},
       STANDARD_MASSES,
       false,
       {{"Restraint evaluated",
         false,
         {{Geometry::PLANE,
           SYMMETRIC,
           ATOMWISE,
           {0},
           {0.0, 0.0, 0.0},
           {1.0, 0.0, 0.0},
           1e+16,
           2.0},
          {Geometry::PLANE,
           SYMMETRIC,
           ATOMWISE,
           {0},
           {0.0, 0.0, 0.0},
           {1.0, 0.0, 0.0},
           1.0,
           2.0},
          {Geometry::PLANE,
           SYMMETRIC,
           ATOMWISE,
           {0},
           {0.0, 0.0, 0.0},
           {1.0, 0.0, 0.0},
           -1e+16,
           2.0}},
         true,
         0.0,
         {},
         {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
         5e-07,
         1e-10,
         4.600000000059999e-09}}},
      {"Three atoms restrained independently",
       {{3.0, 0.0, 0.0}, {4.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {8.0, 8.0, 8.0}},
       STANDARD_MASSES,
       false,
       {{"Restraint evaluated",
         false,
         {{Geometry::PLANE,
           SYMMETRIC,
           ATOMWISE,
           {0, 1, 2},
           {0.0, 0.0, 0.0},
           {1.0, 0.0, 0.0},
           2.0,
           2.0}},
         true,
         6.0,
         {{0, {-2.0, 0.0, 0.0}}, {1, {-4.0, 0.0, 0.0}}, {2, {2.0, 0.0, 0.0}}},
         {20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
         5e-07,
         1e-10,
         6.20000000006e-09}}},
      {"Terms with overlapping selections",
       {{3.0, 0.0, 0.0}, {4.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {8.0, 8.0, 8.0}},
       STANDARD_MASSES,
       false,
       {{"Restraint evaluated",
         false,
         {{Geometry::PLANE,
           SYMMETRIC,
           ATOMWISE,
           {0, 1},
           {0.0, 0.0, 0.0},
           {1.0, 0.0, 0.0},
           1.0,
           2.0},
          {Geometry::SPHERE,
           SYMMETRIC,
           ATOMWISE,
           {1, 2},
           {0.0, 0.0, 0.0},
           {0.0, 0.0, 0.0},
           1.5,
           2.0}},
         true,
         6.25,
         {{0, {-1.0, 0.0, 0.0}}, {1, {-5.0, 0.0, 0.0}}, {2, {1.5, 0.0, 0.0}}},
         {21.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
         5e-07,
         1e-10,
         6.1750000000599994e-09}}},
      {"RCM with unequal positive masses",
       {{1.0, 0.0, 0.0}, {5.0, 0.0, 0.0}, {9.0, 0.0, 0.0}},
       UNEQUAL_MASSES,
       false,
       {{"Restraint evaluated",
         false,
         {{Geometry::PLANE,
           SYMMETRIC,
           RCM,
           {0, 1},
           {0.0, 0.0, 0.0},
           {1.0, 0.0, 0.0},
           2.0,
           2.0}},
         true,
         4.0,
         {{0, {-1.0, 0.0, 0.0}}, {1, {-3.0, 0.0, 0.0}}},
         {16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
         5e-07,
         1e-10,
         5.9000000000599994e-09}}},
      {"RCM with a negative selected total",
       {{1.0, 0.0, 0.0}, {5.0, 0.0, 0.0}, {9.0, 0.0, 0.0}},
       NEGATIVE_TOTAL_MASSES,
       false,
       {{"Restraint evaluated",
         false,
         {{Geometry::PLANE,
           SYMMETRIC,
           RCM,
           {0, 1},
           {0.0, 0.0, 0.0},
           {1.0, 0.0, 0.0},
           2.0,
           2.0}},
         true,
         25.0,
         {{0, {5.0, 0.0, 0.0}}, {1, {-15.0, 0.0, 0.0}}},
         {70.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
         5e-06,
         1e-10,
         6.70000000006e-09}}},
      {"RCM sphere acrosss a P1 boundary",
       {{-10.0, 0.0, 0.0}, {-11.0, 0.0, 0.0}},
       EQUAL_MASSES,
       false,
       {{"Restraint evaluated",
         false,
         {{Geometry::SPHERE,
           SYMMETRIC,
           RCM,
           {0, 1},
           {9.0, 0.0, 0.0},
           {0.0, 0.0, 0.0},
           2.0,
           2.0}},
         true,
         306.25,
         {{0, {17.5, 0.0, 0.0}}, {1, {17.5, 0.0, 0.0}}},
         {367.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
         5e-05,
         1e-10,
         8.05000000006e-09}}},
      {"External tensor for an RCM cylinder",
       {{1.0, 0.0, 0.0}, {5.0, 0.0, 0.0}, {9.0, 0.0, 0.0}},
       UNEQUAL_MASSES,
       false,
       {{"Restraint evaluated",
         false,
         {{Geometry::CYLINDER,
           SYMMETRIC,
           RCM,
           {0, 1, 2},
           {0.0, 0.0, 0.0},
           {0.0, 0.0, 1.0},
           2.0,
           2.0}},
         true,
         13.4444444,
         {{0, {-1.2222222222, 0.0, 0.0}},
          {1, {-3.6666666667, 0.0, 0.0}},
          {2, {-2.4444444444, 0.0, 0.0}}},
         {41.5555555553, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
         5e-08,
         1e-10,
         6.066666666725e-09}}},
      {"External tensor for mixed atomwise and RCM terms",
       {{1.0, 0.0, 0.0}, {5.0, 0.0, 0.0}, {9.0, 0.0, 0.0}},
       UNEQUAL_MASSES,
       false,
       {{"Restraint evaluated",
         false,
         {{Geometry::PLANE,
           SYMMETRIC,
           ATOMWISE,
           {0, 1},
           {0.0, 0.0, 0.0},
           {1.0, 0.0, 0.0},
           1.0,
           2.0},
          {Geometry::CYLINDER,
           OUTSIDE,
           ATOMWISE,
           {1, 2},
           {0.0, 0.0, 0.0},
           {0.0, 0.0, 1.0},
           1.5,
           2.0},
          {Geometry::SPHERE,
           SYMMETRIC,
           RCM,
           {0, 1, 2},
           {1.0, 1.0, 1.0},
           {0.0, 0.0, 0.0},
           0.75,
           2.5}},
         true,
         7.11745489,
         {{0, {0.7157355097, 0.0609138193, 0.0609138193}},
          {1, {-3.8527934708, 0.182741458, 0.182741458}},
          {2, {-0.5685289805, 0.1218276387, 0.1218276387}}},
         {23.664992668800004, -2.0710698576, -2.0710698576, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0},
         5e-09,
         1e-10,
         5.95685289811e-09}}},
      {"Reset then reuse with an RCM term",
       {{1.0, 0.0, 0.0}, {5.0, 0.0, 0.0}},
       UNEQUAL_MASSES,
       false,
       {{"Before reset",
         false,
         {{Geometry::PLANE,
           SYMMETRIC,
           ATOMWISE,
           {0},
           {0.0, 0.0, 0.0},
           {1.0, 0.0, 0.0},
           2.0,
           2.0}},
         true,
         1.0,
         {{0, {2.0, 0.0, 0.0}}},
         {-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
         5e-07,
         1e-10,
         4.9000000000599996e-09},
        {"After reset",
         true,
         {},
         false,
         0.0,
         {},
         {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
         0.0,
         1e-10,
         4.8000000000599995e-09},
        {"New RCM term after reset",
         false,
         {{Geometry::PLANE,
           SYMMETRIC,
           RCM,
           {0, 1},
           {0.0, 0.0, 0.0},
           {1.0, 0.0, 0.0},
           2.0,
           2.0}},
         true,
         4.0,
         {{0, {-1.0, 0.0, 0.0}}, {1, {-3.0, 0.0, 0.0}}},
         {16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
         5e-07,
         1e-10,
         5.000000000059999e-09}}}};

  return cases;
}

CharmmCase FindCharmmCase(const std::string_view id) {
  for (const CharmmCase &test : MakeCharmmCases()) {
    if (test.id == id)
      return test;
  }

  FAIL("Unknown CHARMM case: " << id);
  return {};
}

std::array<std::array<double, 3>, CHARMM_NUM_ATOMS>
EvaluatedCoordinates(const CharmmCase &test) {
  std::array<std::array<double, 3>, CHARMM_NUM_ATOMS> coordinates = {{
      {0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0},
      {0.0, 1.0, 0.0},
      {0.0, 0.0, 1.0},
      {3.0, 4.0, 5.0},
      {-3.0, -4.0, -5.0},
      {7.0, 1.0, -2.0},
      {-7.0, -1.0, 2.0},
      {9.0, 9.0, 9.0},
      {-9.0, -9.0, -9.0},
      {2.0, 5.0, -6.0},
      {-2.0, -5.0, 6.0},
  }};
  std::copy(test.coordinatePrefix.begin(), test.coordinatePrefix.end(),
            coordinates.begin());

  if (test.translateFirstAtomToOrigin)
    coordinates[0] = {0.0, 0.0, 0.0};

  return coordinates;
}

std::array<double, 3> OriginShift(const CharmmCase &test) {
  if (!test.translateFirstAtomToOrigin)
    return {0.0, 0.0, 0.0};

  REQUIRE(test.coordinatePrefix.empty() == false);
  return test.coordinatePrefix[0];
}

ZeroInteractionSystem MakeSystem(void) {
  apo_test::WriteTextFile(PSF_FILE, ZERO_INTERACTION_PSF_TEXT);
  apo_test::WriteTextFile(PARAMETER_FILE, ZERO_INTERACTION_PARAMETER_TEXT);

  apo_charmm_psf *psf = nullptr;
  const apo_status psfStatus = apo_charmm_psf_create(&psf, PSF_FILE);
  CharmmPsfHandle psfHandle(psf);

  apo_charmm_parameters *parameters = nullptr;
  const apo_status parametersStatus =
      apo_charmm_parameters_create(&parameters, PARAMETER_FILE);
  CharmmParametersHandle parametersHandle(parameters);

  apo_test::RemoveIfExists(PSF_FILE);
  apo_test::RemoveIfExists(PARAMETER_FILE);

  REQUIRE(psfStatus == APO_STATUS_OK);
  REQUIRE(parametersStatus == APO_STATUS_OK);

  apo_force_manager *forceManager = nullptr;
  REQUIRE(apo_force_manager_create(&forceManager, psfHandle.get(),
                                   parametersHandle.get()) == APO_STATUS_OK);
  ForceManagerHandle forceManagerHandle(forceManager);

  REQUIRE(apo_force_manager_set_box_dimensions(forceManager, BOX_DIMENSIONS,
                                               3) == APO_STATUS_OK);
  REQUIRE(apo_force_manager_set_cutoff(forceManager, 9.0) == APO_STATUS_OK);
  REQUIRE(apo_force_manager_set_ctonnb(forceManager, 8.5) == APO_STATUS_OK);
  REQUIRE(apo_force_manager_set_ctofnb(forceManager, 7.5) == APO_STATUS_OK);
  REQUIRE(apo_force_manager_set_periodic_boundary_condition(
              forceManager, APO_PBC_P1) == APO_STATUS_OK);

  return {std::move(psfHandle), std::move(parametersHandle),
          std::move(forceManagerHandle), CharmmContextHandle()};
}

void InitializeSystem(ZeroInteractionSystem &system) {
  bool initialized = true;
  REQUIRE(apo_force_manager_is_initialized(
              &initialized, system.forceManager.get()) == APO_STATUS_OK);
  REQUIRE(initialized == false);

  apo_charmm_context *context = nullptr;
  REQUIRE(apo_charmm_context_create(&context, system.forceManager.get()) ==
          APO_STATUS_OK);
  system.context.reset(context);

  REQUIRE(apo_force_manager_is_initialized(
              &initialized, system.forceManager.get()) == APO_STATUS_OK);
  REQUIRE(initialized == true);

  return;
}

apo_status AddCharmmTerm(apo_geometric_restraint_force *restraint,
                         const apo_charmm_psf *psf, const CharmmTerm &term,
                         const std::array<double, 3> &originShift,
                         const AtomRoute route) {
  const double origin[3] = {term.origin[0] - originShift[0],
                            term.origin[1] - originShift[1],
                            term.origin[2] - originShift[2]};
  const double direction[3] = {term.direction[0], term.direction[1],
                               term.direction[2]};

  if (route == AtomRoute::ATOM_REFERENCE) {
    REQUIRE(term.atoms.size() == 1);
    AtomReferenceHandle atom = MakeReference(psf, term.atoms[0]);

    if (term.geometry == Geometry::PLANE) {
      return apo_geometric_restraint_force_add_plane_from_atom_reference(
          restraint, atom.get(), origin, 3, direction, 3, term.forceConstant,
          term.offset, term.activation, term.selectionMode);
    }

    if (term.geometry == Geometry::CYLINDER) {
      return apo_geometric_restraint_force_add_cylinder_from_atom_reference(
          restraint, atom.get(), origin, 3, direction, 3, term.forceConstant,
          term.offset, term.activation, term.selectionMode);
    }

    return apo_geometric_restraint_force_add_sphere_from_atom_reference(
        restraint, atom.get(), origin, 3, term.forceConstant, term.offset,
        term.activation, term.selectionMode);
  }

  AtomSelectionHandle selection = MakeSelection(CHARMM_NUM_ATOMS, term.atoms);

  if (term.geometry == Geometry::PLANE) {
    return apo_geometric_restraint_force_add_plane_from_selection(
        restraint, selection.get(), origin, 3, direction, 3, term.forceConstant,
        term.offset, term.activation, term.selectionMode);
  }

  if (term.geometry == Geometry::CYLINDER) {
    return apo_geometric_restraint_force_add_cylinder_from_selection(
        restraint, selection.get(), origin, 3, direction, 3, term.forceConstant,
        term.offset, term.activation, term.selectionMode);
  }

  return apo_geometric_restraint_force_add_sphere_from_selection(
      restraint, selection.get(), origin, 3, term.forceConstant, term.offset,
      term.activation, term.selectionMode);
}

void AddCharmmTerms(apo_geometric_restraint_force *restraint,
                    const apo_charmm_psf *psf, const CharmmCase &test,
                    const CharmmPhase &phase, const AtomRoute route) {
  const std::array<double, 3> originShift = OriginShift(test);

  for (const CharmmTerm &term : phase.terms) {
    if (test.translateFirstAtomToOrigin)
      REQUIRE(term.atoms == std::vector<int>{0});

    const apo_status status =
        AddCharmmTerm(restraint, psf, term, originShift, route);
    INFO("term-addition diagnostic: " << apo_last_error());
    REQUIRE(status == APO_STATUS_OK);
  }

  return;
}

ManagerOutput EvaluateManager(
    const apo_force_manager *forceManager,
    const std::array<std::array<double, 3>, CHARMM_NUM_ATOMS> &coordinates) {
  std::vector<float4> xyzq;
  xyzq.reserve(coordinates.size());

  for (const std::array<double, 3> &position : coordinates) {
    for (const double value : position)
      REQUIRE(static_cast<double>(static_cast<float>(value)) == value);

    xyzq.push_back(make_float4(static_cast<float>(position[0]),
                               static_cast<float>(position[1]),
                               static_cast<float>(position[2]), 0.0f));
  }

  CudaContainer<float4> deviceCoordinates(xyzq);
  ForceManager &manager = *forceManager->object;
  manager.resetNeighborList(deviceCoordinates.getDeviceArray().data());
  manager.calcForce(deviceCoordinates.getDeviceArray().data(), false, true,
                    true);

  const std::shared_ptr<Force<double>> forces = manager.getForces();
  REQUIRE(forces->size() == CHARMM_NUM_ATOMS);

  std::vector<double> x(CHARMM_NUM_ATOMS);
  std::vector<double> y(CHARMM_NUM_ATOMS);
  std::vector<double> z(CHARMM_NUM_ATOMS);
  forces->getXYZ(x.data(), y.data(), z.data());

  ManagerOutput result;
  for (int atom = 0; atom < CHARMM_NUM_ATOMS; atom++)
    result.gradients[atom] = {x[atom], y[atom], z[atom]};

  CudaContainer<double> &energy = manager.getPotentialEnergy();
  energy.transferToHost();
  result.energy = energy[0];

  CudaContainer<double> &virial = manager.getVirial();
  virial.transferToHost();
  for (std::size_t i = 0; i < result.virial.size(); i++)
    result.virial[i] = virial[i];

  return result;
}

void CheckManagerOutput(const CharmmCase &test, const CharmmPhase &phase,
                        const ManagerOutput &baseline,
                        const ManagerOutput &observed) {
  constexpr std::array<const char *, 9> components = {
      "XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"};

  std::array<std::array<double, 3>, CHARMM_NUM_ATOMS> expectedForces{};
  double forceMagnitude = 0.0;
  for (const CharmmForce &force : phase.forces) {
    expectedForces[force.atom] = force.force;
    for (const double component : force.force)
      forceMagnitude = std::max(forceMagnitude, std::abs(component));
  }

  const double energyTolerance =
      phase.energyTolerance + ARITHMETIC_TOLERANCE * std::abs(phase.energy);
  const double forceTolerance =
      phase.forceTolerance + 0.5 * INV_FORCE_SCALE_TEST +
      0.5 * std::numeric_limits<float>::epsilon() * forceMagnitude +
      ARITHMETIC_TOLERANCE * std::max(1.0, forceMagnitude);
  const std::array<double, 3> originShift = OriginShift(test);

  INFO(test.id << ", phase " << phase.name);

  const double energy = observed.energy - baseline.energy;
  {
    INFO("aggregate energy " << energy << ", expected " << phase.energy
                             << ", tolerance " << energyTolerance);
    CHECK(std::abs(energy - phase.energy) <= energyTolerance);
  }

  for (int atom = 0; atom < CHARMM_NUM_ATOMS; atom++) {
    for (std::size_t k = 0; k < 3; k++) {
      const double force =
          -(observed.gradients[atom][k] - baseline.gradients[atom][k]);
      INFO("atom " << atom << ", physical force component " << k << ": "
                   << force << ", expected " << expectedForces[atom][k]
                   << ", tolerance " << forceTolerance);
      CHECK(std::abs(force - expectedForces[atom][k]) <= forceTolerance);
    }
  }

  for (std::size_t a = 0; a < 3; a++) {
    for (std::size_t b = 0; b < 3; b++) {
      const std::size_t index = 3 * a + b;
      const double firstAtomGradient =
          observed.gradients[0][b] - baseline.gradients[0][b];
      const double virial = observed.virial[index] - baseline.virial[index] +
                            originShift[a] * firstAtomGradient;
      const double tolerance =
          phase.virialTolerance +
          ARITHMETIC_TOLERANCE * (1.0 + std::abs(phase.virial[index])) +
          std::abs(originShift[a]) * forceTolerance;
      INFO("virial component " << components[index] << ": " << virial
                               << ", expected " << phase.virial[index]
                               << ", tolerance " << tolerance);
      CHECK(std::isfinite(virial));
      CHECK(std::abs(virial - phase.virial[index]) <= tolerance);
    }
  }

  return;
}

void CheckNoManagerContribution(const ManagerOutput &baseline,
                                const ManagerOutput &observed) {
  CHECK(observed.energy == baseline.energy);
  CHECK(observed.gradients == baseline.gradients);
  CHECK(observed.virial == baseline.virial);
  return;
}

void RunCharmmCase(const ZeroInteractionSystem &system, const CharmmCase &test,
                   const AtomRoute route) {
  INFO(test.id << ((route == AtomRoute::SELECTION)
                       ? " through AtomSelection handles"
                       : " through AtomReference handles"));

  const std::array<std::array<double, 3>, CHARMM_NUM_ATOMS> coordinates =
      EvaluatedCoordinates(test);
  GeometricRestraintForceHandle restraint = MakeRestraint(system.psf.get());

  REQUIRE(apo_force_manager_subscribe_geometric_restraint_force(
              system.forceManager.get(), restraint.get(),
              std::string(test.id).c_str()) == APO_STATUS_OK);

  const ManagerOutput baseline =
      EvaluateManager(system.forceManager.get(), coordinates);

  REQUIRE(apo_geometric_restraint_force_set_masses(
              restraint.get(), test.masses.data(), test.masses.size()) ==
          APO_STATUS_OK);

  for (const CharmmPhase &phase : test.phases) {
    if (phase.resetFirst)
      REQUIRE(apo_geometric_restraint_force_reset(restraint.get()) ==
              APO_STATUS_OK);

    AddCharmmTerms(restraint.get(), system.psf.get(), test, phase, route);
    CheckManagerOutput(test, phase, baseline,
                       EvaluateManager(system.forceManager.get(), coordinates));
  }

  REQUIRE(apo_force_manager_unsubscribe_geometric_restraint_force(
              system.forceManager.get(), restraint.get()) == APO_STATUS_OK);

  return;
}

template <typename Function>
void CheckForceManagerHandleValidation(const std::string_view functionName,
                                       Function function) {
  apo_test::CheckStatusAndDiagnostic(
      function(nullptr), APO_STATUS_INVALID_ARGUMENT,
      std::string(functionName) + ": ForceManager is NULL");

  apo_force_manager emptyForceManager;
  apo_test::CheckStatusAndDiagnostic(
      function(&emptyForceManager), APO_STATUS_INVALID_ARGUMENT,
      std::string(functionName) + ": ForceManager object is NULL");

  return;
}

void CheckForceManagerNativeError(const apo_status status,
                                  const std::string_view functionName,
                                  const std::string_view expectedMessage,
                                  const std::string_view sourceFunction) {
  apo_test::CheckNativeError(status, APO_STATUS_INVALID_ARGUMENT,
                             "InvalidArgument", functionName, expectedMessage,
                             FORCE_MANAGER_HEADER_FILE, sourceFunction);
  return;
}

} // namespace

TEST_CASE("CapiGeometricRestraintForceCreatesAndDestroys") {
  CharmmPsfHandle psf = MakePsf(NUM_ATOMS);

  SECTION("NullOutputPointer") {
    apo_test::CheckStatusAndDiagnostic(
        apo_geometric_restraint_force_create(nullptr, psf.get()),
        APO_STATUS_INVALID_ARGUMENT,
        std::string(CREATE_FUNCTION_NAME) + ": out pointer is NULL");
  }

  SECTION("NullPsfHandleClearsOutput") {
    apo_geometric_restraint_force staleRestraint;
    apo_geometric_restraint_force *restraint = &staleRestraint;

    apo_test::CheckStatusAndDiagnostic(
        apo_geometric_restraint_force_create(&restraint, nullptr),
        APO_STATUS_INVALID_ARGUMENT,
        std::string(CREATE_FUNCTION_NAME) + ": CharmmPsf is NULL");
    CHECK(restraint == nullptr);
  }

  SECTION("NullPsfObjectClearsOutput") {
    apo_charmm_psf emptyPsf;
    apo_geometric_restraint_force staleRestraint;
    apo_geometric_restraint_force *restraint = &staleRestraint;

    apo_test::CheckStatusAndDiagnostic(
        apo_geometric_restraint_force_create(&restraint, &emptyPsf),
        APO_STATUS_INVALID_ARGUMENT,
        std::string(CREATE_FUNCTION_NAME) + ": CharmmPsf object is NULL");
    CHECK(restraint == nullptr);
  }

  SECTION("NativeConstructionErrorsClearOutput") {
    apo_geometric_restraint_force staleRestraint;
    apo_geometric_restraint_force *restraint = &staleRestraint;
    apo_status status = APO_STATUS_OK;

    CharmmPsfHandle uninitializedPsf(new apo_charmm_psf());
    uninitializedPsf->object = std::make_shared<CharmmPSF>();
    CHECK_NOTHROW((status = apo_geometric_restraint_force_create(
                       &restraint, uninitializedPsf.get())));
    apo_test::CheckNativeError(
        status, APO_STATUS_NOT_INITIALIZED, "NotInitialized",
        CREATE_FUNCTION_NAME,
        "CharmmPSF atom count is not initialized; observed -1",
        MMFP_SOURCE_FILE, "GeometricRestraintForce");
    CHECK(restraint == nullptr);

    CharmmPsfHandle emptyAtomPsf = MakePsf(0);
    restraint = &staleRestraint;
    CHECK_NOTHROW((status = apo_geometric_restraint_force_create(
                       &restraint, emptyAtomPsf.get())));
    CheckMmfpNativeError(status, CREATE_FUNCTION_NAME,
                         "Atom count must be positive; observed 0",
                         "GeometricRestraintForce");
    CHECK(restraint == nullptr);

    psf->object->getMasses().pop_back();
    restraint = &staleRestraint;
    CHECK_NOTHROW(
        (status = apo_geometric_restraint_force_create(&restraint, psf.get())));
    apo_test::CheckNativeError(
        status, APO_STATUS_RUNTIME_ERROR, "Runtime", CREATE_FUNCTION_NAME,
        "PSF mass count mismatch; expected 4, observed 3", MMFP_SOURCE_FILE,
        "GeometricRestraintForce");
    CHECK(restraint == nullptr);
  }

  SECTION("NonfinitePsfMassesClearOutput") {
    for (const double mass : {QUIET_NAN, INF, -INF}) {
      psf->object->getMasses()[0] = mass;
      apo_geometric_restraint_force staleRestraint;
      apo_geometric_restraint_force *restraint = &staleRestraint;

      CheckMmfpNativeError(
          apo_geometric_restraint_force_create(&restraint, psf.get()),
          CREATE_FUNCTION_NAME,
          "Mass at atom index 0 must be finite; observed " +
              std::to_string(mass),
          "validateMasses");
      CHECK(restraint == nullptr);
    }
  }

  SECTION("SuccessReturnsOwnedNonNullHandle") {
    apo_geometric_restraint_force *restraint = nullptr;

    REQUIRE(apo_geometric_restraint_force_create(&restraint, psf.get()) ==
            APO_STATUS_OK);
    REQUIRE(restraint != nullptr);
    GeometricRestraintForceHandle ownedRestraint(restraint);

    CHECK(std::string(apo_last_error()).empty());
    REQUIRE(ownedRestraint->object != nullptr);
    CHECK(ownedRestraint->object.use_count() == 1);
    CHECK(ownedRestraint->object->getNumAtoms() == NUM_ATOMS);
    CHECK(ownedRestraint->object->getNumTerms() == 0);
    CHECK(ownedRestraint->object->getMasses() == psf->object->getMasses());
  }

  SECTION("DestructionReleasesNativeObjectAndPreservesDiagnostic") {
    apo_geometric_restraint_force *restraint = nullptr;
    REQUIRE(apo_geometric_restraint_force_create(&restraint, psf.get()) ==
            APO_STATUS_OK);
    REQUIRE(restraint != nullptr);
    const std::weak_ptr<GeometricRestraintForce<long long int, float>>
        weakRestraint = restraint->object;

    REQUIRE(apo_geometric_restraint_force_reset(nullptr) ==
            APO_STATUS_INVALID_ARGUMENT);
    const std::string diagnostic = apo_last_error();
    REQUIRE(diagnostic.empty() == false);

    CHECK_NOTHROW(apo_geometric_restraint_force_destroy(restraint));
    CHECK(weakRestraint.expired());
    CHECK(std::string(apo_last_error()) == diagnostic);

    CHECK_NOTHROW(apo_geometric_restraint_force_destroy(nullptr));
    CHECK(std::string(apo_last_error()) == diagnostic);
  }

  SECTION("NullObjectDestructionDoesNotThrow") {
    auto *restraint = new apo_geometric_restraint_force();

    CHECK_NOTHROW(apo_geometric_restraint_force_destroy(restraint));
  }
}

TEST_CASE("CapiGeometricRestraintForceCopiesPsfSnapshotWithoutRetainingIt") {
  CharmmPsfHandle psf = MakePsf(NUM_ATOMS);
  psf->object->getMasses()[0] = -2.0;
  psf->object->getMasses()[1] = 0.0;
  const std::vector<double> masses = psf->object->getMasses();
  const std::weak_ptr<CharmmPSF> weakPsf = psf->object;

  GeometricRestraintForceHandle restraint = MakeRestraint(psf.get());

  psf->object->getMasses()[0] = 99.0;
  psf->object->setNumAtoms(NUM_ATOMS + 1);
  psf.reset();

  CHECK(weakPsf.expired());
  CHECK(restraint->object->getNumAtoms() == NUM_ATOMS);
  CHECK(restraint->object->getMasses() == masses);

  AtomSelectionHandle selection = MakeSelection(NUM_ATOMS, {0, 2});
  REQUIRE(apo_geometric_restraint_force_add_plane_from_selection(
              restraint.get(), selection.get(), ORIGIN, 3, X_DIRECTION, 3, 2.0,
              1.0, SYMMETRIC, RCM) == APO_STATUS_OK);
  CHECK(restraint->object->getNumTerms() == 1);

  const double replacement[NUM_ATOMS] = {1.0, 1.0, 1.0, 1.0};
  REQUIRE(apo_geometric_restraint_force_set_masses(restraint.get(), replacement,
                                                   NUM_ATOMS) == APO_STATUS_OK);
  CHECK(restraint->object->getMasses() ==
        std::vector<double>(replacement, replacement + NUM_ATOMS));

  REQUIRE(apo_geometric_restraint_force_reset(restraint.get()) ==
          APO_STATUS_OK);
  CHECK(restraint->object->getNumTerms() == 0);
  CHECK(std::string(apo_last_error()).empty());
}

TEST_CASE("CapiGeometricRestraintForceValidatesEveryRestraintHandle") {
  CharmmPsfHandle psf = MakePsf(NUM_ATOMS);
  AtomSelectionHandle selection = MakeSelection(NUM_ATOMS, {0});
  AtomReferenceHandle atom = MakeReference(psf.get(), 0);
  const double masses[NUM_ATOMS] = {1.0, 2.0, 3.0, 4.0};

  for (const NamedTermAddition &addition :
       MakeTermAdditions(selection.get(), atom.get())) {
    CheckRestraintHandleValidation(
        addition.functionName,
        [&addition](apo_geometric_restraint_force *restraint) -> apo_status {
          return addition.add(restraint, INSIDE, ATOMWISE);
        });
  }

  CheckRestraintHandleValidation(
      SET_MASSES_FUNCTION_NAME,
      [&masses](apo_geometric_restraint_force *restraint) -> apo_status {
        return apo_geometric_restraint_force_set_masses(restraint, masses,
                                                        NUM_ATOMS);
      });

  CheckRestraintHandleValidation(
      "apo_geometric_restraint_force_reset",
      [](apo_geometric_restraint_force *restraint) -> apo_status {
        return apo_geometric_restraint_force_reset(restraint);
      });
}

TEST_CASE("CapiGeometricRestraintForceValidatesTypedAtomHandles") {
  CharmmPsfHandle psf = MakePsf(NUM_ATOMS);
  GeometricRestraintForceHandle restraint = MakeRestraint(psf.get());

  CheckAtomHandleValidation<apo_atom_selection>(
      "apo_geometric_restraint_force_add_plane_from_selection", "AtomSelection",
      restraint.get(),
      [&restraint](const apo_atom_selection *selection) -> apo_status {
        return apo_geometric_restraint_force_add_plane_from_selection(
            restraint.get(), selection, ORIGIN, 3, X_DIRECTION, 3, 2.0, 1.0,
            INSIDE, ATOMWISE);
      });

  CheckAtomHandleValidation<apo_atom_selection>(
      "apo_geometric_restraint_force_add_cylinder_from_selection",
      "AtomSelection", restraint.get(),
      [&restraint](const apo_atom_selection *selection) -> apo_status {
        return apo_geometric_restraint_force_add_cylinder_from_selection(
            restraint.get(), selection, ORIGIN, 3, X_DIRECTION, 3, 2.0, 1.0,
            INSIDE, ATOMWISE);
      });

  CheckAtomHandleValidation<apo_atom_selection>(
      "apo_geometric_restraint_force_add_sphere_from_selection",
      "AtomSelection", restraint.get(),
      [&restraint](const apo_atom_selection *selection) -> apo_status {
        return apo_geometric_restraint_force_add_sphere_from_selection(
            restraint.get(), selection, ORIGIN, 3, 2.0, 1.0, INSIDE, ATOMWISE);
      });

  CheckAtomHandleValidation<apo_atom_reference>(
      "apo_geometric_restraint_force_add_plane_from_atom_reference",
      "AtomReference", restraint.get(),
      [&restraint](const apo_atom_reference *atom) -> apo_status {
        return apo_geometric_restraint_force_add_plane_from_atom_reference(
            restraint.get(), atom, ORIGIN, 3, X_DIRECTION, 3, 2.0, 1.0, INSIDE,
            ATOMWISE);
      });

  CheckAtomHandleValidation<apo_atom_reference>(
      "apo_geometric_restraint_force_add_cylinder_from_atom_reference",
      "AtomReference", restraint.get(),
      [&restraint](const apo_atom_reference *atom) -> apo_status {
        return apo_geometric_restraint_force_add_cylinder_from_atom_reference(
            restraint.get(), atom, ORIGIN, 3, X_DIRECTION, 3, 2.0, 1.0, INSIDE,
            ATOMWISE);
      });

  CheckAtomHandleValidation<apo_atom_reference>(
      "apo_geometric_restraint_force_add_sphere_from_atom_reference",
      "AtomReference", restraint.get(),
      [&restraint](const apo_atom_reference *atom) -> apo_status {
        return apo_geometric_restraint_force_add_sphere_from_atom_reference(
            restraint.get(), atom, ORIGIN, 3, 2.0, 1.0, INSIDE, ATOMWISE);
      });
}

TEST_CASE("CapiGeometricRestraintForceValidatesCartesianArrays") {
  CharmmPsfHandle psf = MakePsf(NUM_ATOMS);
  GeometricRestraintForceHandle restraint = MakeRestraint(psf.get());
  AtomSelectionHandle selection = MakeSelection(NUM_ATOMS, {0});
  AtomReferenceHandle atom = MakeReference(psf.get(), 0);

  CheckCartesianArrayValidation(
      "apo_geometric_restraint_force_add_plane_from_selection", "origin",
      restraint.get(),
      [&](const double *values, const std::size_t length) -> apo_status {
        return apo_geometric_restraint_force_add_plane_from_selection(
            restraint.get(), selection.get(), values, length, X_DIRECTION, 3,
            2.0, 1.0, INSIDE, ATOMWISE);
      });

  CheckCartesianArrayValidation(
      "apo_geometric_restraint_force_add_plane_from_selection", "direction",
      restraint.get(),
      [&](const double *values, const std::size_t length) -> apo_status {
        return apo_geometric_restraint_force_add_plane_from_selection(
            restraint.get(), selection.get(), ORIGIN, 3, values, length, 2.0,
            1.0, INSIDE, ATOMWISE);
      });

  CheckCartesianArrayValidation(
      "apo_geometric_restraint_force_add_plane_from_atom_reference", "origin",
      restraint.get(),
      [&](const double *values, const std::size_t length) -> apo_status {
        return apo_geometric_restraint_force_add_plane_from_atom_reference(
            restraint.get(), atom.get(), values, length, X_DIRECTION, 3, 2.0,
            1.0, INSIDE, ATOMWISE);
      });

  CheckCartesianArrayValidation(
      "apo_geometric_restraint_force_add_plane_from_atom_reference",
      "direction", restraint.get(),
      [&](const double *values, const std::size_t length) -> apo_status {
        return apo_geometric_restraint_force_add_plane_from_atom_reference(
            restraint.get(), atom.get(), ORIGIN, 3, values, length, 2.0, 1.0,
            INSIDE, ATOMWISE);
      });

  CheckCartesianArrayValidation(
      "apo_geometric_restraint_force_add_cylinder_from_selection", "origin",
      restraint.get(),
      [&](const double *values, const std::size_t length) -> apo_status {
        return apo_geometric_restraint_force_add_cylinder_from_selection(
            restraint.get(), selection.get(), values, length, X_DIRECTION, 3,
            2.0, 1.0, INSIDE, ATOMWISE);
      });

  CheckCartesianArrayValidation(
      "apo_geometric_restraint_force_add_cylinder_from_selection", "direction",
      restraint.get(),
      [&](const double *values, const std::size_t length) -> apo_status {
        return apo_geometric_restraint_force_add_cylinder_from_selection(
            restraint.get(), selection.get(), ORIGIN, 3, values, length, 2.0,
            1.0, INSIDE, ATOMWISE);
      });

  CheckCartesianArrayValidation(
      "apo_geometric_restraint_force_add_cylinder_from_atom_reference",
      "origin", restraint.get(),
      [&](const double *values, const std::size_t length) -> apo_status {
        return apo_geometric_restraint_force_add_cylinder_from_atom_reference(
            restraint.get(), atom.get(), values, length, X_DIRECTION, 3, 2.0,
            1.0, INSIDE, ATOMWISE);
      });

  CheckCartesianArrayValidation(
      "apo_geometric_restraint_force_add_cylinder_from_atom_reference",
      "direction", restraint.get(),
      [&](const double *values, const std::size_t length) -> apo_status {
        return apo_geometric_restraint_force_add_cylinder_from_atom_reference(
            restraint.get(), atom.get(), ORIGIN, 3, values, length, 2.0, 1.0,
            INSIDE, ATOMWISE);
      });

  CheckCartesianArrayValidation(
      "apo_geometric_restraint_force_add_sphere_from_selection", "center",
      restraint.get(),
      [&](const double *values, const std::size_t length) -> apo_status {
        return apo_geometric_restraint_force_add_sphere_from_selection(
            restraint.get(), selection.get(), values, length, 2.0, 1.0, INSIDE,
            ATOMWISE);
      });

  CheckCartesianArrayValidation(
      "apo_geometric_restraint_force_add_sphere_from_atom_reference", "center",
      restraint.get(),
      [&](const double *values, const std::size_t length) -> apo_status {
        return apo_geometric_restraint_force_add_sphere_from_atom_reference(
            restraint.get(), atom.get(), values, length, 2.0, 1.0, INSIDE,
            ATOMWISE);
      });
}

TEST_CASE("CapiGeometricRestraintForceValidatesEnumerations") {
  using NativeActivation =
      GeometricRestraintForce<long long int, float>::Activation;
  using NativeSelectionMode =
      GeometricRestraintForce<long long int, float>::SelectionMode;

  STATIC_REQUIRE(static_cast<int>(SYMMETRIC) ==
                 static_cast<int>(NativeActivation::SYMMETRIC));
  STATIC_REQUIRE(static_cast<int>(INSIDE) ==
                 static_cast<int>(NativeActivation::INSIDE));
  STATIC_REQUIRE(static_cast<int>(OUTSIDE) ==
                 static_cast<int>(NativeActivation::OUTSIDE));
  STATIC_REQUIRE(static_cast<int>(ATOMWISE) ==
                 static_cast<int>(NativeSelectionMode::ATOMWISE));
  STATIC_REQUIRE(static_cast<int>(RCM) ==
                 static_cast<int>(NativeSelectionMode::RCM));

  CharmmPsfHandle psf = MakePsf(NUM_ATOMS);
  GeometricRestraintForceHandle restraint = MakeRestraint(psf.get());
  AtomSelectionHandle selection = MakeSelection(NUM_ATOMS, {0, 1});
  AtomReferenceHandle atom = MakeReference(psf.get(), 2);
  const std::vector<NamedTermAddition> additions =
      MakeTermAdditions(selection.get(), atom.get());

  SECTION("UndeclaredValuesAreRejected") {
    for (const NamedTermAddition &addition : additions) {
      for (const int value : {2, -2}) {
        INFO("activation value: " << value);
        apo_test::CheckStatusAndDiagnostic(
            addition.add(restraint.get(),
                         static_cast<apo_geometric_restraint_activation>(value),
                         ATOMWISE),
            APO_STATUS_INVALID_ARGUMENT,
            std::string(addition.functionName) +
                ": activation is not a declared "
                "apo_geometric_restraint_activation value");
      }

      for (const int value : {2, -1}) {
        INFO("selection-mode value: " << value);
        apo_test::CheckStatusAndDiagnostic(
            addition.add(
                restraint.get(), INSIDE,
                static_cast<apo_geometric_restraint_selection_mode>(value)),
            APO_STATUS_INVALID_ARGUMENT,
            std::string(addition.functionName) +
                ": selection_mode is not a declared "
                "apo_geometric_restraint_selection_mode value");
      }
    }

    CHECK(restraint->object->getNumTerms() == 0);
  }

  SECTION("EveryDeclaredCombinationIsAccepted") {
    std::size_t expectedTerms = 0;

    for (const NamedTermAddition &addition : additions) {
      for (const apo_geometric_restraint_activation activation :
           {SYMMETRIC, INSIDE, OUTSIDE}) {
        for (const apo_geometric_restraint_selection_mode selectionMode :
             {ATOMWISE, RCM}) {
          INFO(addition.functionName
               << ", activation " << static_cast<int>(activation)
               << ", selection mode " << static_cast<int>(selectionMode));
          REQUIRE(addition.add(restraint.get(), activation, selectionMode) ==
                  APO_STATUS_OK);
          CHECK(std::string(apo_last_error()).empty());
          expectedTerms++;
          CHECK(restraint->object->getNumTerms() == expectedTerms);
        }
      }
    }
  }

  SECTION("SelectionModeReachesNativeMassValidation") {
    const double masses[NUM_ATOMS] = {1.0, -1.0, 0.0, 4.0};
    REQUIRE(apo_geometric_restraint_force_set_masses(
                restraint.get(), masses, NUM_ATOMS) == APO_STATUS_OK);

    for (const NamedTermAddition &addition : additions) {
      INFO(addition.functionName);

      const std::size_t numTerms = restraint->object->getNumTerms();
      CheckMmfpNativeError(addition.add(restraint.get(), INSIDE, RCM),
                           addition.functionName,
                           "RCM total mass must be finite and nonzero; "
                           "observed " +
                               std::to_string(0.0),
                           "validateRcmMasses");
      CHECK(restraint->object->getNumTerms() == numTerms);

      REQUIRE(addition.add(restraint.get(), INSIDE, ATOMWISE) == APO_STATUS_OK);
      CHECK(restraint->object->getNumTerms() == numTerms + 1);
    }
  }
}

TEST_CASE("CapiGeometricRestraintForceAddsEachGeometryFromTypedAtoms") {
  CharmmPsfHandle psf = MakePsf(NUM_ATOMS);
  GeometricRestraintForceHandle restraint = MakeRestraint(psf.get());
  AtomSelectionHandle selection = MakeSelection(NUM_ATOMS, {0, 2, 3});
  AtomReferenceHandle atom = MakeReference(psf.get(), 1);
  const double origin[3] = {0.5, -1.0, 1.5};
  const double direction[3] = {1.0, 2.0, 3.0};
  const double center[3] = {2.0, 1.0, -0.5};

  REQUIRE(apo_geometric_restraint_force_add_plane_from_selection(
              restraint.get(), selection.get(), origin, 3, direction, 3, 2.0,
              2.0, SYMMETRIC, ATOMWISE) == APO_STATUS_OK);
  CHECK(std::string(apo_last_error()).empty());
  CHECK(restraint->object->getNumTerms() == 1);

  REQUIRE(apo_geometric_restraint_force_add_plane_from_atom_reference(
              restraint.get(), atom.get(), origin, 3, direction, 3, -2.0, -1.0,
              INSIDE, RCM) == APO_STATUS_OK);
  CHECK(std::string(apo_last_error()).empty());
  CHECK(restraint->object->getNumTerms() == 2);

  REQUIRE(apo_geometric_restraint_force_add_cylinder_from_selection(
              restraint.get(), selection.get(), origin, 3, direction, 3, 0.0,
              2.0, OUTSIDE, RCM) == APO_STATUS_OK);
  CHECK(std::string(apo_last_error()).empty());
  CHECK(restraint->object->getNumTerms() == 3);

  REQUIRE(apo_geometric_restraint_force_add_cylinder_from_atom_reference(
              restraint.get(), atom.get(), origin, 3, direction, 3, 2.0, 0.0,
              SYMMETRIC, ATOMWISE) == APO_STATUS_OK);
  CHECK(std::string(apo_last_error()).empty());
  CHECK(restraint->object->getNumTerms() == 4);

  REQUIRE(apo_geometric_restraint_force_add_sphere_from_selection(
              restraint.get(), selection.get(), center, 3, 2.0, 1.0, INSIDE,
              ATOMWISE) == APO_STATUS_OK);
  CHECK(std::string(apo_last_error()).empty());
  CHECK(restraint->object->getNumTerms() == 5);

  REQUIRE(apo_geometric_restraint_force_add_sphere_from_atom_reference(
              restraint.get(), atom.get(), center, 3, 2.0, 1.0, OUTSIDE, RCM) ==
          APO_STATUS_OK);
  CHECK(std::string(apo_last_error()).empty());
  CHECK(restraint->object->getNumTerms() == 6);

  REQUIRE(apo_geometric_restraint_force_add_sphere_from_atom_reference(
              restraint.get(), atom.get(), center, 3, 2.0, 1.0, OUTSIDE, RCM) ==
          APO_STATUS_OK);
  CHECK(restraint->object->getNumTerms() == 7);
}

TEST_CASE("CapiGeometricRestraintForceMapsNativeTermValidation") {
  CharmmPsfHandle psf = MakePsf(NUM_ATOMS);
  GeometricRestraintForceHandle restraint = MakeRestraint(psf.get());
  AtomSelectionHandle selection = MakeSelection(NUM_ATOMS, {0, 1});
  AtomReferenceHandle atom = MakeReference(psf.get(), 1);

  SECTION("SelectionAtomCountMismatch") {
    AtomSelectionHandle largerSelection =
        MakeSelection(NUM_ATOMS + 1, {0, NUM_ATOMS});
    const std::string expectedMessage =
        "Selection atom count mismatch; expected 4, observed 5";

    CheckMmfpNativeError(
        apo_geometric_restraint_force_add_plane_from_selection(
            restraint.get(), largerSelection.get(), ORIGIN, 3, X_DIRECTION, 3,
            2.0, 1.0, INSIDE, ATOMWISE),
        "apo_geometric_restraint_force_add_plane_from_selection",
        expectedMessage, "getSelectionIndices");
    CheckMmfpNativeError(
        apo_geometric_restraint_force_add_cylinder_from_selection(
            restraint.get(), largerSelection.get(), ORIGIN, 3, X_DIRECTION, 3,
            2.0, 1.0, INSIDE, ATOMWISE),
        "apo_geometric_restraint_force_add_cylinder_from_selection",
        expectedMessage, "getSelectionIndices");
    CheckMmfpNativeError(
        apo_geometric_restraint_force_add_sphere_from_selection(
            restraint.get(), largerSelection.get(), ORIGIN, 3, 2.0, 1.0, INSIDE,
            ATOMWISE),
        "apo_geometric_restraint_force_add_sphere_from_selection",
        expectedMessage, "getSelectionIndices");
  }

  SECTION("EmptySelectionIsAcceptedWithoutAddingTerms") {
    AtomSelectionHandle emptySelection = MakeSelection(NUM_ATOMS, {});
    const double zeroMasses[NUM_ATOMS] = {0.0, 0.0, 0.0, 0.0};
    REQUIRE(apo_geometric_restraint_force_set_masses(
                restraint.get(), zeroMasses, NUM_ATOMS) == APO_STATUS_OK);

    for (const apo_geometric_restraint_selection_mode selectionMode :
         {ATOMWISE, RCM}) {
      INFO("selection mode: " << static_cast<int>(selectionMode));
      CHECK(apo_geometric_restraint_force_add_plane_from_selection(
                restraint.get(), emptySelection.get(), ORIGIN, 3, X_DIRECTION,
                3, 2.0, 1.0, INSIDE, selectionMode) == APO_STATUS_OK);
      CHECK(apo_geometric_restraint_force_add_cylinder_from_selection(
                restraint.get(), emptySelection.get(), ORIGIN, 3, X_DIRECTION,
                3, 2.0, 1.0, INSIDE, selectionMode) == APO_STATUS_OK);
      CHECK(apo_geometric_restraint_force_add_sphere_from_selection(
                restraint.get(), emptySelection.get(), ORIGIN, 3, 2.0, 1.0,
                INSIDE, selectionMode) == APO_STATUS_OK);
      CHECK(std::string(apo_last_error()).empty());
    }

    CHECK(restraint->object->getNumTerms() == 0);

    CheckMmfpNativeError(
        apo_geometric_restraint_force_add_plane_from_selection(
            restraint.get(), emptySelection.get(), ORIGIN, 3, ZERO_DIRECTION, 3,
            2.0, 1.0, INSIDE, ATOMWISE),
        "apo_geometric_restraint_force_add_plane_from_selection",
        "Direction must have a positive finite norm", "appendTerm");
    CHECK(restraint->object->getNumTerms() == 0);
  }

  SECTION("AtomReferenceOutsideForceLocalRange") {
    CharmmPsfHandle largerPsf = MakePsf(NUM_ATOMS + 1);
    AtomReferenceHandle outsideAtom = MakeReference(largerPsf.get(), NUM_ATOMS);
    const std::string expectedMessage =
        "Atom index is out of range; expected [0, 4), observed 4";

    CheckMmfpNativeError(
        apo_geometric_restraint_force_add_plane_from_atom_reference(
            restraint.get(), outsideAtom.get(), ORIGIN, 3, X_DIRECTION, 3, 2.0,
            1.0, INSIDE, ATOMWISE),
        "apo_geometric_restraint_force_add_plane_from_atom_reference",
        expectedMessage, "appendTerm");
    CheckMmfpNativeError(
        apo_geometric_restraint_force_add_cylinder_from_atom_reference(
            restraint.get(), outsideAtom.get(), ORIGIN, 3, X_DIRECTION, 3, 2.0,
            1.0, INSIDE, ATOMWISE),
        "apo_geometric_restraint_force_add_cylinder_from_atom_reference",
        expectedMessage, "appendTerm");
    CheckMmfpNativeError(
        apo_geometric_restraint_force_add_sphere_from_atom_reference(
            restraint.get(), outsideAtom.get(), ORIGIN, 3, 2.0, 1.0, INSIDE,
            ATOMWISE),
        "apo_geometric_restraint_force_add_sphere_from_atom_reference",
        expectedMessage, "appendTerm");
    CHECK(restraint->object->getNumTerms() == 0);
  }

  SECTION("AtomReferenceFromAnotherTopology") {
    std::weak_ptr<CharmmPSF> weakForeignPsf;

    {
      CharmmPsfHandle foreignPsf = MakePsf(NUM_ATOMS + 5);
      weakForeignPsf = foreignPsf->object;
      AtomReferenceHandle foreignAtom = MakeReference(foreignPsf.get(), 3);

      REQUIRE(apo_geometric_restraint_force_add_plane_from_atom_reference(
                  restraint.get(), foreignAtom.get(), ORIGIN, 3, X_DIRECTION, 3,
                  2.0, 1.0, INSIDE, ATOMWISE) == APO_STATUS_OK);
      REQUIRE(apo_geometric_restraint_force_add_cylinder_from_atom_reference(
                  restraint.get(), foreignAtom.get(), ORIGIN, 3, X_DIRECTION, 3,
                  2.0, 1.0, INSIDE, RCM) == APO_STATUS_OK);
      REQUIRE(apo_geometric_restraint_force_add_sphere_from_atom_reference(
                  restraint.get(), foreignAtom.get(), ORIGIN, 3, 2.0, 1.0,
                  INSIDE, ATOMWISE) == APO_STATUS_OK);
    }

    CHECK(weakForeignPsf.expired());
    CHECK(restraint->object->getNumTerms() == 3);
  }

  SECTION("DirectedGeometriesRejectZeroDirection") {
    const std::string expectedMessage =
        "Direction must have a positive finite norm";

    CheckMmfpNativeError(
        apo_geometric_restraint_force_add_plane_from_selection(
            restraint.get(), selection.get(), ORIGIN, 3, ZERO_DIRECTION, 3, 2.0,
            1.0, INSIDE, ATOMWISE),
        "apo_geometric_restraint_force_add_plane_from_selection",
        expectedMessage, "appendTerm");
    CheckMmfpNativeError(
        apo_geometric_restraint_force_add_plane_from_atom_reference(
            restraint.get(), atom.get(), ORIGIN, 3, ZERO_DIRECTION, 3, 2.0, 1.0,
            INSIDE, ATOMWISE),
        "apo_geometric_restraint_force_add_plane_from_atom_reference",
        expectedMessage, "appendTerm");
    CheckMmfpNativeError(
        apo_geometric_restraint_force_add_cylinder_from_selection(
            restraint.get(), selection.get(), ORIGIN, 3, ZERO_DIRECTION, 3, 2.0,
            1.0, INSIDE, ATOMWISE),
        "apo_geometric_restraint_force_add_cylinder_from_selection",
        expectedMessage, "appendTerm");
    CheckMmfpNativeError(
        apo_geometric_restraint_force_add_cylinder_from_atom_reference(
            restraint.get(), atom.get(), ORIGIN, 3, ZERO_DIRECTION, 3, 2.0, 1.0,
            INSIDE, ATOMWISE),
        "apo_geometric_restraint_force_add_cylinder_from_atom_reference",
        expectedMessage, "appendTerm");
    CHECK(restraint->object->getNumTerms() == 0);
  }

  SECTION("NonfiniteParameters") {
    const double nonfiniteOrigin[3] = {QUIET_NAN, 0.0, 0.0};
    const double nonfiniteDirection[3] = {0.0, 0.0, QUIET_NAN};
    const double nonfiniteCenter[3] = {0.0, INF, 0.0};

    CheckMmfpNativeError(
        apo_geometric_restraint_force_add_plane_from_selection(
            restraint.get(), selection.get(), nonfiniteOrigin, 3, X_DIRECTION,
            3, 2.0, 1.0, INSIDE, ATOMWISE),
        "apo_geometric_restraint_force_add_plane_from_selection",
        "Origin component must be finite; index 0", "appendTerm");
    CheckMmfpNativeError(
        apo_geometric_restraint_force_add_cylinder_from_atom_reference(
            restraint.get(), atom.get(), ORIGIN, 3, nonfiniteDirection, 3, 2.0,
            1.0, INSIDE, ATOMWISE),
        "apo_geometric_restraint_force_add_cylinder_from_atom_reference",
        "Direction component must be finite; index 2", "appendTerm");
    CheckMmfpNativeError(
        apo_geometric_restraint_force_add_sphere_from_selection(
            restraint.get(), selection.get(), nonfiniteCenter, 3, 2.0, 1.0,
            INSIDE, ATOMWISE),
        "apo_geometric_restraint_force_add_sphere_from_selection",
        "Origin component must be finite; index 1", "appendTerm");
    CheckMmfpNativeError(
        apo_geometric_restraint_force_add_sphere_from_atom_reference(
            restraint.get(), atom.get(), ORIGIN, 3, INF, 1.0, INSIDE, ATOMWISE),
        "apo_geometric_restraint_force_add_sphere_from_atom_reference",
        "Force constant must be finite; observed " + std::to_string(INF),
        "appendTerm");
    CheckMmfpNativeError(
        apo_geometric_restraint_force_add_plane_from_atom_reference(
            restraint.get(), atom.get(), ORIGIN, 3, X_DIRECTION, 3, 2.0,
            QUIET_NAN, INSIDE, ATOMWISE),
        "apo_geometric_restraint_force_add_plane_from_atom_reference",
        "Offset must be finite; observed " + std::to_string(QUIET_NAN),
        "appendTerm");
    CHECK(restraint->object->getNumTerms() == 0);
  }

  SECTION("FailedAdditionPreservesInstalledTerms") {
    REQUIRE(apo_geometric_restraint_force_add_plane_from_selection(
                restraint.get(), selection.get(), ORIGIN, 3, X_DIRECTION, 3,
                2.0, 1.0, INSIDE, ATOMWISE) == APO_STATUS_OK);
    REQUIRE(restraint->object->getNumTerms() == 1);

    CHECK(apo_geometric_restraint_force_add_cylinder_from_selection(
              restraint.get(), selection.get(), ORIGIN, 3, ZERO_DIRECTION, 3,
              2.0, 1.0, INSIDE, ATOMWISE) == APO_STATUS_INVALID_ARGUMENT);
    CHECK(apo_geometric_restraint_force_add_sphere_from_atom_reference(
              restraint.get(), atom.get(), ORIGIN, 2, 2.0, 1.0, INSIDE,
              ATOMWISE) == APO_STATUS_INVALID_ARGUMENT);
    CHECK(restraint->object->getNumTerms() == 1);
  }
}

TEST_CASE("CapiGeometricRestraintForceRetainsNoSelectionOrReference") {
  CharmmPsfHandle psf = MakePsf(NUM_ATOMS);
  GeometricRestraintForceHandle restraint = MakeRestraint(psf.get());
  std::weak_ptr<AtomSelection> weakSelection;
  std::weak_ptr<CharmmPSF> weakReferencePsf;

  {
    AtomSelectionHandle selection = MakeSelection(NUM_ATOMS, {0, 1});
    weakSelection = selection->object;
    CharmmPsfHandle referencePsf = MakePsf(NUM_ATOMS);
    weakReferencePsf = referencePsf->object;
    AtomReferenceHandle atom = MakeReference(referencePsf.get(), 2);

    REQUIRE(apo_geometric_restraint_force_add_sphere_from_selection(
                restraint.get(), selection.get(), ORIGIN, 3, 2.0, 1.0, INSIDE,
                RCM) == APO_STATUS_OK);
    REQUIRE(apo_geometric_restraint_force_add_plane_from_atom_reference(
                restraint.get(), atom.get(), ORIGIN, 3, X_DIRECTION, 3, 2.0,
                1.0, INSIDE, RCM) == APO_STATUS_OK);
  }

  CHECK(weakSelection.expired());
  CHECK(weakReferencePsf.expired());
  CHECK(restraint->object->getNumTerms() == 2);

  const double zeroSelectedTotal[NUM_ATOMS] = {1.0, -1.0, 3.0, 4.0};
  CheckMmfpNativeError(apo_geometric_restraint_force_set_masses(
                           restraint.get(), zeroSelectedTotal, NUM_ATOMS),
                       SET_MASSES_FUNCTION_NAME,
                       "RCM total mass must be finite and nonzero; observed " +
                           std::to_string(0.0),
                       "validateRcmMasses");

  const double zeroReferencedMass[NUM_ATOMS] = {1.0, 2.0, 0.0, 4.0};
  CheckMmfpNativeError(apo_geometric_restraint_force_set_masses(
                           restraint.get(), zeroReferencedMass, NUM_ATOMS),
                       SET_MASSES_FUNCTION_NAME,
                       "RCM total mass must be finite and nonzero; observed " +
                           std::to_string(0.0),
                       "validateRcmMasses");

  const double zeroUnselectedMass[NUM_ATOMS] = {1.0, 2.0, 3.0, 0.0};
  CHECK(apo_geometric_restraint_force_set_masses(
            restraint.get(), zeroUnselectedMass, NUM_ATOMS) == APO_STATUS_OK);
  CHECK(
      restraint->object->getMasses() ==
      std::vector<double>(zeroUnselectedMass, zeroUnselectedMass + NUM_ATOMS));
}

TEST_CASE("CapiGeometricRestraintForceReplacesMasses") {
  CharmmPsfHandle psf = MakePsf(NUM_ATOMS);
  GeometricRestraintForceHandle restraint = MakeRestraint(psf.get());

  SECTION("CopiesCallerBuffer") {
    double masses[NUM_ATOMS] = {2.0, 4.0, 6.0, 8.0};
    const std::vector<double> expected(masses, masses + NUM_ATOMS);

    REQUIRE(apo_geometric_restraint_force_set_masses(
                restraint.get(), masses, NUM_ATOMS) == APO_STATUS_OK);
    CHECK(std::string(apo_last_error()).empty());

    masses[0] = 99.0;
    masses[3] = -99.0;

    CHECK(restraint->object->getMasses() == expected);
  }

  SECTION("RejectsWrongLength") {
    const double masses[NUM_ATOMS + 1] = {1.0, 2.0, 3.0, 4.0, 5.0};

    CheckMmfpNativeError(apo_geometric_restraint_force_set_masses(
                             restraint.get(), masses, NUM_ATOMS - 1),
                         SET_MASSES_FUNCTION_NAME,
                         "Mass count mismatch; expected 4, observed 3",
                         "validateMasses");
    CheckMmfpNativeError(apo_geometric_restraint_force_set_masses(
                             restraint.get(), masses, NUM_ATOMS + 1),
                         SET_MASSES_FUNCTION_NAME,
                         "Mass count mismatch; expected 4, observed 5",
                         "validateMasses");
  }

  SECTION("RejectsNullPointer") {
    for (const std::size_t length :
         {static_cast<std::size_t>(NUM_ATOMS), std::size_t{0}}) {
      INFO("masses_len: " << length);
      apo_test::CheckStatusAndDiagnostic(
          apo_geometric_restraint_force_set_masses(restraint.get(), nullptr,
                                                   length),
          APO_STATUS_INVALID_ARGUMENT,
          std::string(SET_MASSES_FUNCTION_NAME) + ": masses is NULL");
    }
  }

  SECTION("RejectsNonfiniteMasses") {
    for (const double mass : {QUIET_NAN, INF, -INF}) {
      const double masses[NUM_ATOMS] = {1.0, mass, 3.0, 4.0};

      CheckMmfpNativeError(apo_geometric_restraint_force_set_masses(
                               restraint.get(), masses, NUM_ATOMS),
                           SET_MASSES_FUNCTION_NAME,
                           "Mass at atom index 1 must be finite; observed " +
                               std::to_string(mass),
                           "validateMasses");
    }
  }

  SECTION("AcceptsUnusualMassesThatCharmmAccepts") {
    AtomSelectionHandle tinyPartner = MakeSelection(NUM_ATOMS, {0, 2});
    AtomSelectionHandle hugePartner = MakeSelection(NUM_ATOMS, {0, 3});
    AtomReferenceHandle negativeAtom = MakeReference(psf.get(), 1);
    const std::vector<std::vector<double>> massSets = {
        {0.0, -2.0, 1.0e-300, 1.0e300},
        {0.0, -2.0, -1.0e-300, -1.0e300},
        {-1.0, 3.0, 1.0 + 1.0e-10, 2.0},
    };

    REQUIRE(apo_geometric_restraint_force_set_masses(
                restraint.get(), massSets[0].data(), massSets[0].size()) ==
            APO_STATUS_OK);
    REQUIRE(apo_geometric_restraint_force_add_plane_from_selection(
                restraint.get(), tinyPartner.get(), ORIGIN, 3, X_DIRECTION, 3,
                2.0, 1.0, SYMMETRIC, RCM) == APO_STATUS_OK);
    REQUIRE(apo_geometric_restraint_force_add_sphere_from_selection(
                restraint.get(), hugePartner.get(), ORIGIN, 3, 2.0, 1.0,
                OUTSIDE, RCM) == APO_STATUS_OK);
    REQUIRE(apo_geometric_restraint_force_add_cylinder_from_atom_reference(
                restraint.get(), negativeAtom.get(), ORIGIN, 3, X_DIRECTION, 3,
                2.0, 1.0, INSIDE, RCM) == APO_STATUS_OK);
    REQUIRE(restraint->object->getNumTerms() == 3);

    for (const std::vector<double> &masses : massSets) {
      INFO("masses: " << masses[0] << ", " << masses[1] << ", " << masses[2]
                      << ", " << masses[3]);
      CHECK(apo_geometric_restraint_force_set_masses(
                restraint.get(), masses.data(), masses.size()) ==
            APO_STATUS_OK);
      CHECK(std::string(apo_last_error()).empty());
      CHECK(restraint->object->getMasses() == masses);
      CHECK(restraint->object->getNumTerms() == 3);
    }
  }

  SECTION("FailedReplacementPreservesActiveMasses") {
    AtomSelectionHandle selection = MakeSelection(NUM_ATOMS, {0, 1});
    const std::vector<double> active = {2.0, -1.0, 0.0, 5.0};
    REQUIRE(apo_geometric_restraint_force_set_masses(
                restraint.get(), active.data(), active.size()) ==
            APO_STATUS_OK);
    REQUIRE(apo_geometric_restraint_force_add_plane_from_selection(
                restraint.get(), selection.get(), ORIGIN, 3, X_DIRECTION, 3,
                2.0, 1.0, INSIDE, RCM) == APO_STATUS_OK);

    const double nonfinite[NUM_ATOMS] = {2.0, QUIET_NAN, 0.0, 5.0};
    CHECK(apo_geometric_restraint_force_set_masses(restraint.get(), nonfinite,
                                                   NUM_ATOMS) ==
          APO_STATUS_INVALID_ARGUMENT);
    CHECK(restraint->object->getMasses() == active);

    CHECK(apo_geometric_restraint_force_set_masses(
              restraint.get(), active.data(), NUM_ATOMS - 1) ==
          APO_STATUS_INVALID_ARGUMENT);
    CHECK(restraint->object->getMasses() == active);

    CHECK(apo_geometric_restraint_force_set_masses(restraint.get(), nullptr,
                                                   NUM_ATOMS) ==
          APO_STATUS_INVALID_ARGUMENT);
    CHECK(restraint->object->getMasses() == active);

    const double zeroSelectedTotal[NUM_ATOMS] = {1.0, -1.0, 0.0, 5.0};
    CheckMmfpNativeError(apo_geometric_restraint_force_set_masses(
                             restraint.get(), zeroSelectedTotal, NUM_ATOMS),
                         SET_MASSES_FUNCTION_NAME,
                         "RCM total mass must be finite and nonzero; "
                         "observed " +
                             std::to_string(0.0),
                         "validateRcmMasses");
    CHECK(restraint->object->getMasses() == active);
    CHECK(restraint->object->getNumTerms() == 1);
  }
}

TEST_CASE("CapiGeometricRestraintForceResetRemovesTermsAndPreservesMasses") {
  CharmmPsfHandle psf = MakePsf(NUM_ATOMS);
  GeometricRestraintForceHandle restraint = MakeRestraint(psf.get());
  AtomSelectionHandle selection = MakeSelection(NUM_ATOMS, {0, 1, 2});
  AtomReferenceHandle atom = MakeReference(psf.get(), 0);

  SECTION("RemovesEveryTermAndAcceptsRepeatedReset") {
    for (const NamedTermAddition &addition :
         MakeTermAdditions(selection.get(), atom.get())) {
      REQUIRE(addition.add(restraint.get(), INSIDE, RCM) == APO_STATUS_OK);
    }
    REQUIRE(restraint->object->getNumTerms() == 6);

    CHECK(apo_geometric_restraint_force_reset(restraint.get()) ==
          APO_STATUS_OK);
    CHECK(std::string(apo_last_error()).empty());
    CHECK(restraint->object->getNumTerms() == 0);

    CHECK(apo_geometric_restraint_force_reset(restraint.get()) ==
          APO_STATUS_OK);
    CHECK(restraint->object->getNumTerms() == 0);

    REQUIRE(apo_geometric_restraint_force_add_sphere_from_selection(
                restraint.get(), selection.get(), ORIGIN, 3, 2.0, 1.0,
                SYMMETRIC, ATOMWISE) == APO_STATUS_OK);
    CHECK(restraint->object->getNumTerms() == 1);
  }

  SECTION("PreservesActiveMasses") {
    const std::vector<double> active = {0.0, 2.0, -3.0, 4.0};
    REQUIRE(apo_geometric_restraint_force_set_masses(
                restraint.get(), active.data(), active.size()) ==
            APO_STATUS_OK);
    REQUIRE(apo_geometric_restraint_force_add_cylinder_from_selection(
                restraint.get(), selection.get(), ORIGIN, 3, X_DIRECTION, 3,
                2.0, 1.0, INSIDE, RCM) == APO_STATUS_OK);

    REQUIRE(apo_geometric_restraint_force_reset(restraint.get()) ==
            APO_STATUS_OK);
    CHECK(restraint->object->getNumTerms() == 0);
    CHECK(restraint->object->getMasses() == active);

    CheckMmfpNativeError(
        apo_geometric_restraint_force_add_plane_from_atom_reference(
            restraint.get(), atom.get(), ORIGIN, 3, X_DIRECTION, 3, 2.0, 1.0,
            INSIDE, RCM),
        "apo_geometric_restraint_force_add_plane_from_atom_reference",
        "RCM total mass must be finite and nonzero; observed " +
            std::to_string(0.0),
        "validateRcmMasses");
    CHECK(apo_geometric_restraint_force_add_plane_from_atom_reference(
              restraint.get(), atom.get(), ORIGIN, 3, X_DIRECTION, 3, 2.0, 1.0,
              INSIDE, ATOMWISE) == APO_STATUS_OK);
    CHECK(restraint->object->getNumTerms() == 1);
  }
}

TEST_CASE("CapiGeometricRestraintForceExposesNoRawIndexEntryPoint") {
  using PlaneFromSelection =
      decltype(&apo_geometric_restraint_force_add_plane_from_selection);
  using PlaneFromAtomReference =
      decltype(&apo_geometric_restraint_force_add_plane_from_atom_reference);
  using CylinderFromSelection =
      decltype(&apo_geometric_restraint_force_add_cylinder_from_selection);
  using CylinderFromAtomReference =
      decltype(&apo_geometric_restraint_force_add_cylinder_from_atom_reference);
  using SphereFromSelection =
      decltype(&apo_geometric_restraint_force_add_sphere_from_selection);
  using SphereFromAtomReference =
      decltype(&apo_geometric_restraint_force_add_sphere_from_atom_reference);

  STATIC_REQUIRE(ACCEPTS_DIRECTED_TERM_ATOM<PlaneFromSelection,
                                            const apo_atom_selection *>);
  STATIC_REQUIRE_FALSE(ACCEPTS_DIRECTED_TERM_ATOM<PlaneFromSelection, int>);
  STATIC_REQUIRE_FALSE(
      ACCEPTS_DIRECTED_TERM_ATOM<PlaneFromSelection, const int *>);
  STATIC_REQUIRE_FALSE(
      ACCEPTS_DIRECTED_TERM_ATOM<PlaneFromSelection, const void *>);
  STATIC_REQUIRE_FALSE(ACCEPTS_DIRECTED_TERM_ATOM<PlaneFromSelection,
                                                  const apo_atom_reference *>);

  STATIC_REQUIRE(ACCEPTS_DIRECTED_TERM_ATOM<PlaneFromAtomReference,
                                            const apo_atom_reference *>);
  STATIC_REQUIRE_FALSE(ACCEPTS_DIRECTED_TERM_ATOM<PlaneFromAtomReference, int>);
  STATIC_REQUIRE_FALSE(
      ACCEPTS_DIRECTED_TERM_ATOM<PlaneFromAtomReference, const int *>);
  STATIC_REQUIRE_FALSE(
      ACCEPTS_DIRECTED_TERM_ATOM<PlaneFromAtomReference, const void *>);
  STATIC_REQUIRE_FALSE(ACCEPTS_DIRECTED_TERM_ATOM<PlaneFromAtomReference,
                                                  const apo_atom_selection *>);

  STATIC_REQUIRE(ACCEPTS_DIRECTED_TERM_ATOM<CylinderFromSelection,
                                            const apo_atom_selection *>);
  STATIC_REQUIRE_FALSE(ACCEPTS_DIRECTED_TERM_ATOM<CylinderFromSelection, int>);
  STATIC_REQUIRE_FALSE(
      ACCEPTS_DIRECTED_TERM_ATOM<CylinderFromSelection, const int *>);
  STATIC_REQUIRE_FALSE(
      ACCEPTS_DIRECTED_TERM_ATOM<CylinderFromSelection, const void *>);
  STATIC_REQUIRE_FALSE(ACCEPTS_DIRECTED_TERM_ATOM<CylinderFromSelection,
                                                  const apo_atom_reference *>);

  STATIC_REQUIRE(ACCEPTS_DIRECTED_TERM_ATOM<CylinderFromAtomReference,
                                            const apo_atom_reference *>);
  STATIC_REQUIRE_FALSE(
      ACCEPTS_DIRECTED_TERM_ATOM<CylinderFromAtomReference, int>);
  STATIC_REQUIRE_FALSE(
      ACCEPTS_DIRECTED_TERM_ATOM<CylinderFromAtomReference, const int *>);
  STATIC_REQUIRE_FALSE(
      ACCEPTS_DIRECTED_TERM_ATOM<CylinderFromAtomReference, const void *>);
  STATIC_REQUIRE_FALSE(ACCEPTS_DIRECTED_TERM_ATOM<CylinderFromAtomReference,
                                                  const apo_atom_selection *>);

  STATIC_REQUIRE(ACCEPTS_SPHERE_TERM_ATOM<SphereFromSelection,
                                          const apo_atom_selection *>);
  STATIC_REQUIRE_FALSE(ACCEPTS_SPHERE_TERM_ATOM<SphereFromSelection, int>);
  STATIC_REQUIRE_FALSE(
      ACCEPTS_SPHERE_TERM_ATOM<SphereFromSelection, const int *>);
  STATIC_REQUIRE_FALSE(
      ACCEPTS_SPHERE_TERM_ATOM<SphereFromSelection, const void *>);
  STATIC_REQUIRE_FALSE(ACCEPTS_SPHERE_TERM_ATOM<SphereFromSelection,
                                                const apo_atom_reference *>);

  STATIC_REQUIRE(ACCEPTS_SPHERE_TERM_ATOM<SphereFromAtomReference,
                                          const apo_atom_reference *>);
  STATIC_REQUIRE_FALSE(ACCEPTS_SPHERE_TERM_ATOM<SphereFromAtomReference, int>);
  STATIC_REQUIRE_FALSE(
      ACCEPTS_SPHERE_TERM_ATOM<SphereFromAtomReference, const int *>);
  STATIC_REQUIRE_FALSE(
      ACCEPTS_SPHERE_TERM_ATOM<SphereFromAtomReference, const void *>);
  STATIC_REQUIRE_FALSE(ACCEPTS_SPHERE_TERM_ATOM<SphereFromAtomReference,
                                                const apo_atom_selection *>);
}

TEST_CASE("CapiGeometricRestraintForceSuccessClearsStaleDiagnostic") {
  CharmmPsfHandle psf = MakePsf(NUM_ATOMS);
  GeometricRestraintForceHandle restraint = MakeRestraint(psf.get());

  REQUIRE(apo_geometric_restraint_force_reset(nullptr) ==
          APO_STATUS_INVALID_ARGUMENT);
  REQUIRE(std::string(apo_last_error()).empty() == false);

  CHECK(apo_geometric_restraint_force_reset(restraint.get()) == APO_STATUS_OK);
  CHECK(std::string(apo_last_error()).empty() == true);
}

TEST_CASE("CapiGeometricRestraintForceManagerValidatesHandlesAndTags") {
  ZeroInteractionSystem system = MakeSystem();
  GeometricRestraintForceHandle restraint = MakeRestraint(system.psf.get());

  CheckForceManagerHandleValidation(
      SUBSCRIBE_FUNCTION_NAME,
      [&restraint](apo_force_manager *candidate) -> apo_status {
        return apo_force_manager_subscribe_geometric_restraint_force(
            candidate, restraint.get(), "mmfp");
      });

  CheckForceManagerHandleValidation(
      UNSUBSCRIBE_FUNCTION_NAME,
      [&restraint](apo_force_manager *candidate) -> apo_status {
        return apo_force_manager_unsubscribe_geometric_restraint_force(
            candidate, restraint.get());
      });

  CheckRestraintHandleValidation(
      SUBSCRIBE_FUNCTION_NAME,
      [&system](apo_geometric_restraint_force *candidate) -> apo_status {
        return apo_force_manager_subscribe_geometric_restraint_force(
            system.forceManager.get(), candidate, "mmfp");
      });

  CheckRestraintHandleValidation(
      UNSUBSCRIBE_FUNCTION_NAME,
      [&system](apo_geometric_restraint_force *candidate) -> apo_status {
        return apo_force_manager_unsubscribe_geometric_restraint_force(
            system.forceManager.get(), candidate);
      });

  apo_test::CheckStatusAndDiagnostic(
      apo_force_manager_subscribe_geometric_restraint_force(
          system.forceManager.get(), restraint.get(), nullptr),
      APO_STATUS_INVALID_ARGUMENT,
      std::string(SUBSCRIBE_FUNCTION_NAME) + ": force_tag is NULL");

  CheckForceManagerNativeError(
      apo_force_manager_subscribe_geometric_restraint_force(
          system.forceManager.get(), restraint.get(), ""),
      SUBSCRIBE_FUNCTION_NAME, "Force tag must not be empty", "subscribe");

  CheckForceManagerNativeError(
      apo_force_manager_unsubscribe_geometric_restraint_force(
          system.forceManager.get(), restraint.get()),
      UNSUBSCRIBE_FUNCTION_NAME, "Force is not subscribed to this ForceManager",
      "unsubscribe");

  CHECK(restraint->object.use_count() == 1);

  CHECK(apo_force_manager_subscribe_geometric_restraint_force(
            system.forceManager.get(), restraint.get(), "mmfp") ==
        APO_STATUS_OK);
  CHECK(std::string(apo_last_error()).empty());
  CHECK(restraint->object.use_count() == 2);

  CHECK(apo_force_manager_unsubscribe_geometric_restraint_force(
            system.forceManager.get(), restraint.get()) == APO_STATUS_OK);
  CHECK(std::string(apo_last_error()).empty());
  CHECK(restraint->object.use_count() == 1);
}

TEST_CASE("CapiGeometricRestraintForceManagerCopiesTagsAndRejectsDuplicates") {
  ZeroInteractionSystem system = MakeSystem();
  GeometricRestraintForceHandle first = MakeRestraint(system.psf.get());
  GeometricRestraintForceHandle second = MakeRestraint(system.psf.get());
  ForceManager &manager = *system.forceManager->object;

  char tag[] = "mmfp_copied_tag";
  REQUIRE(apo_force_manager_subscribe_geometric_restraint_force(
              system.forceManager.get(), first.get(), tag) == APO_STATUS_OK);
  std::fill(std::begin(tag), std::end(tag) - 1, 'x');

  CheckForceManagerNativeError(
      apo_force_manager_subscribe_geometric_restraint_force(
          system.forceManager.get(), first.get(), "mmfp_copied_tag"),
      SUBSCRIBE_FUNCTION_NAME,
      "Force is already subscribed to this ForceManager", "subscribe");

  CheckForceManagerNativeError(
      apo_force_manager_subscribe_geometric_restraint_force(
          system.forceManager.get(), first.get(), "mmfp_other_tag"),
      SUBSCRIBE_FUNCTION_NAME,
      "Force is already subscribed to this ForceManager", "subscribe");

  CHECK(first->object.use_count() == 2);

  CHECK(apo_force_manager_subscribe_geometric_restraint_force(
            system.forceManager.get(), second.get(), "mmfp_copied_tag") ==
        APO_STATUS_OK);
  CHECK(second->object.use_count() == 2);

  CHECK(apo_force_manager_unsubscribe_geometric_restraint_force(
            system.forceManager.get(), second.get()) == APO_STATUS_OK);
  CHECK(first->object.use_count() == 2);
  CHECK(second->object.use_count() == 1);

  apo_test::CheckApoCharmmError(
      [&manager, &tag](void) -> void {
        manager.unsubscribe(std::string(tag));
        return;
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Force tag is not subscribed to this ForceManager");

  CHECK_NOTHROW(manager.unsubscribe(std::string("mmfp_copied_tag")));
  CHECK(first->object.use_count() == 1);

  CheckForceManagerNativeError(
      apo_force_manager_unsubscribe_geometric_restraint_force(
          system.forceManager.get(), first.get()),
      UNSUBSCRIBE_FUNCTION_NAME, "Force is not subscribed to this ForceManager",
      "unsubscribe");

  CheckForceManagerNativeError(
      apo_force_manager_unsubscribe_geometric_restraint_force(
          system.forceManager.get(), second.get()),
      UNSUBSCRIBE_FUNCTION_NAME, "Force is not subscribed to this ForceManager",
      "unsubscribe");
}

TEST_CASE("CapiGeometricRestraintForceManagerRequiresP1") {
  ZeroInteractionSystem system = MakeSystem();
  GeometricRestraintForceHandle restraint = MakeRestraint(system.psf.get());
  apo_force_manager *forceManager = system.forceManager.get();
  apo_pbc pbc = APO_PBC_UNSET;

  REQUIRE(apo_force_manager_subscribe_geometric_restraint_force(
              forceManager, restraint.get(), "mmfp") == APO_STATUS_OK);
  CHECK(apo_force_manager_set_periodic_boundary_condition(
            forceManager, APO_PBC_P1) == APO_STATUS_OK);

  apo_test::CheckNativeError(
      apo_force_manager_set_periodic_boundary_condition(forceManager,
                                                        APO_PBC_P21),
      APO_STATUS_INVALID_ARGUMENT, "InvalidArgument", SET_PBC_FUNCTION_NAME,
      "Subscribed force does not support periodic boundary condition value 2",
      FORCE_MANAGER_SOURCE_FILE, "checkPBCCompatibility");

  apo_test::CheckNativeError(
      apo_force_manager_set_periodic_boundary_condition(forceManager,
                                                        APO_PBC_UNSET),
      APO_STATUS_INVALID_ARGUMENT, "InvalidArgument", SET_PBC_FUNCTION_NAME,
      "Periodic boundary condition must be PBC::P1 or PBC::P21; observed 0",
      FORCE_MANAGER_SOURCE_FILE, "checkPBCCompatibility");

  apo_test::CheckStatusAndDiagnostic(
      apo_force_manager_set_periodic_boundary_condition(
          forceManager, static_cast<apo_pbc>(99)),
      APO_STATUS_INVALID_ARGUMENT,
      std::string(SET_PBC_FUNCTION_NAME) +
          ": invalid periodic boundary condition");

  REQUIRE(apo_force_manager_get_periodic_boundary_condition(
              &pbc, forceManager) == APO_STATUS_OK);
  CHECK(pbc == APO_PBC_P1);

  CheckForceManagerNativeError(
      apo_force_manager_subscribe_geometric_restraint_force(
          forceManager, restraint.get(), "mmfp"),
      SUBSCRIBE_FUNCTION_NAME,
      "Force is already subscribed to this ForceManager", "subscribe");

  REQUIRE(apo_force_manager_unsubscribe_geometric_restraint_force(
              forceManager, restraint.get()) == APO_STATUS_OK);
  CHECK(apo_force_manager_set_periodic_boundary_condition(
            forceManager, APO_PBC_P21) == APO_STATUS_OK);
  REQUIRE(apo_force_manager_get_periodic_boundary_condition(
              &pbc, forceManager) == APO_STATUS_OK);
  CHECK(pbc == APO_PBC_P21);

  CheckForceManagerNativeError(
      apo_force_manager_subscribe_geometric_restraint_force(
          forceManager, restraint.get(), "mmfp"),
      SUBSCRIBE_FUNCTION_NAME,
      "Subscribed force does not support periodic boundary condition value 2",
      "subscribe");

  CheckForceManagerNativeError(
      apo_force_manager_unsubscribe_geometric_restraint_force(forceManager,
                                                              restraint.get()),
      UNSUBSCRIBE_FUNCTION_NAME, "Force is not subscribed to this ForceManager",
      "unsubscribe");
  CHECK(restraint->object.use_count() == 1);

  CHECK(apo_force_manager_set_periodic_boundary_condition(
            forceManager, APO_PBC_P1) == APO_STATUS_OK);
  CHECK(apo_force_manager_subscribe_geometric_restraint_force(
            forceManager, restraint.get(), "mmfp") == APO_STATUS_OK);
  CHECK(apo_force_manager_unsubscribe_geometric_restraint_force(
            forceManager, restraint.get()) == APO_STATUS_OK);
}

TEST_CASE("CapiGeometricRestraintForceManagerSubscribesBeforeAndAfterInit") {
  const CharmmCase test =
      FindCharmmCase("Cylinder with a nonunit oblique axis");
  const std::array<std::array<double, 3>, CHARMM_NUM_ATOMS> coordinates =
      EvaluatedCoordinates(test);

  ZeroInteractionSystem system = MakeSystem();
  GeometricRestraintForceHandle restraint = MakeRestraint(system.psf.get());
  AddCharmmTerms(restraint.get(), system.psf.get(), test, test.phases[0],
                 AtomRoute::ATOM_REFERENCE);

  SECTION("SubscribeBeforeInitialization") {
    REQUIRE(apo_force_manager_subscribe_geometric_restraint_force(
                system.forceManager.get(), restraint.get(), "mmfp") ==
            APO_STATUS_OK);
    InitializeSystem(system);
  }

  SECTION("SubscribeAfterInitialization") {
    InitializeSystem(system);
    REQUIRE(apo_force_manager_subscribe_geometric_restraint_force(
                system.forceManager.get(), restraint.get(), "mmfp") ==
            APO_STATUS_OK);
  }

  const ManagerOutput restrained =
      EvaluateManager(system.forceManager.get(), coordinates);

  REQUIRE(apo_force_manager_unsubscribe_geometric_restraint_force(
              system.forceManager.get(), restraint.get()) == APO_STATUS_OK);

  CheckManagerOutput(test, test.phases[0],
                     EvaluateManager(system.forceManager.get(), coordinates),
                     restrained);
}

TEST_CASE("CapiGeometricRestraintForceManagerAppliesMutationsWhileSubscribed") {
  const CharmmCase unequal = FindCharmmCase("RCM with unequal positive masses");
  const CharmmCase negative =
      FindCharmmCase("RCM with a negative selected total");
  const std::array<std::array<double, 3>, CHARMM_NUM_ATOMS> coordinates =
      EvaluatedCoordinates(unequal);

  REQUIRE(EvaluatedCoordinates(negative) == coordinates);
  REQUIRE(unequal.phases[0].terms.size() == 1);
  REQUIRE(negative.phases[0].terms.size() == 1);
  REQUIRE(negative.phases[0].terms[0].atoms ==
          unequal.phases[0].terms[0].atoms);
  REQUIRE(negative.phases[0].terms[0].origin ==
          unequal.phases[0].terms[0].origin);
  REQUIRE(negative.phases[0].terms[0].direction ==
          unequal.phases[0].terms[0].direction);
  REQUIRE(negative.phases[0].terms[0].forceConstant ==
          unequal.phases[0].terms[0].forceConstant);
  REQUIRE(negative.phases[0].terms[0].offset ==
          unequal.phases[0].terms[0].offset);

  ZeroInteractionSystem system = MakeSystem();
  InitializeSystem(system);
  GeometricRestraintForceHandle restraint = MakeRestraint(system.psf.get());

  REQUIRE(apo_force_manager_subscribe_geometric_restraint_force(
              system.forceManager.get(), restraint.get(), "mmfp") ==
          APO_STATUS_OK);

  const ManagerOutput baseline =
      EvaluateManager(system.forceManager.get(), coordinates);

  REQUIRE(apo_geometric_restraint_force_set_masses(
              restraint.get(), unequal.masses.data(), unequal.masses.size()) ==
          APO_STATUS_OK);
  AddCharmmTerms(restraint.get(), system.psf.get(), unequal, unequal.phases[0],
                 AtomRoute::SELECTION);
  CheckManagerOutput(unequal, unequal.phases[0], baseline,
                     EvaluateManager(system.forceManager.get(), coordinates));

  REQUIRE(apo_geometric_restraint_force_set_masses(
              restraint.get(), negative.masses.data(),
              negative.masses.size()) == APO_STATUS_OK);
  CheckManagerOutput(negative, negative.phases[0], baseline,
                     EvaluateManager(system.forceManager.get(), coordinates));

  REQUIRE(apo_geometric_restraint_force_reset(restraint.get()) ==
          APO_STATUS_OK);
  CheckNoManagerContribution(
      baseline, EvaluateManager(system.forceManager.get(), coordinates));
  CheckForceManagerNativeError(
      apo_force_manager_subscribe_geometric_restraint_force(
          system.forceManager.get(), restraint.get(), "mmfp"),
      SUBSCRIBE_FUNCTION_NAME,
      "Force is already subscribed to this ForceManager", "subscribe");

  REQUIRE(apo_geometric_restraint_force_set_masses(
              restraint.get(), unequal.masses.data(), unequal.masses.size()) ==
          APO_STATUS_OK);
  AddCharmmTerms(restraint.get(), system.psf.get(), unequal, unequal.phases[0],
                 AtomRoute::SELECTION);
  CheckManagerOutput(unequal, unequal.phases[0], baseline,
                     EvaluateManager(system.forceManager.get(), coordinates));

  apo_test::CheckNativeError(
      apo_force_manager_set_periodic_boundary_condition(
          system.forceManager.get(), APO_PBC_P21),
      APO_STATUS_INVALID_ARGUMENT, "InvalidArgument", SET_PBC_FUNCTION_NAME,
      "Subscribed force does not support periodic boundary condition value 2",
      FORCE_MANAGER_SOURCE_FILE, "checkPBCCompatibility");
  CheckManagerOutput(unequal, unequal.phases[0], baseline,
                     EvaluateManager(system.forceManager.get(), coordinates));

  REQUIRE(apo_force_manager_unsubscribe_geometric_restraint_force(
              system.forceManager.get(), restraint.get()) == APO_STATUS_OK);
  CheckNoManagerContribution(
      baseline, EvaluateManager(system.forceManager.get(), coordinates));
  CHECK(restraint->object->getNumTerms() == 1);

  REQUIRE(apo_force_manager_subscribe_geometric_restraint_force(
              system.forceManager.get(), restraint.get(), "mmfp") ==
          APO_STATUS_OK);
  CheckManagerOutput(unequal, unequal.phases[0], baseline,
                     EvaluateManager(system.forceManager.get(), coordinates));
}

TEST_CASE("CapiGeometricRestraintForceManagerRejectsAtomCountMismatch") {
  ZeroInteractionSystem system = MakeSystem();
  InitializeSystem(system);

  CharmmPsfHandle smallPsf = MakePsf(3);
  GeometricRestraintForceHandle restraint = MakeRestraint(smallPsf.get());

  CheckMmfpNativeError(apo_force_manager_subscribe_geometric_restraint_force(
                           system.forceManager.get(), restraint.get(), "mmfp"),
                       SUBSCRIBE_FUNCTION_NAME,
                       "Initialization atom count mismatch; expected 3, "
                       "observed 12",
                       "initialize");

  CheckForceManagerNativeError(
      apo_force_manager_unsubscribe_geometric_restraint_force(
          system.forceManager.get(), restraint.get()),
      UNSUBSCRIBE_FUNCTION_NAME, "Force is not subscribed to this ForceManager",
      "unsubscribe");
  CHECK(restraint->object.use_count() == 1);
}

TEST_CASE("CapiGeometricRestraintForceManagerRetainsNativeOwnership") {
  const CharmmCase test =
      FindCharmmCase("OUTSIDE plane at zero signed distance");
  const std::array<std::array<double, 3>, CHARMM_NUM_ATOMS> coordinates =
      EvaluatedCoordinates(test);

  ZeroInteractionSystem system = MakeSystem();
  InitializeSystem(system);
  const ManagerOutput baseline =
      EvaluateManager(system.forceManager.get(), coordinates);

  GeometricRestraintForceHandle restraint = MakeRestraint(system.psf.get());
  const std::weak_ptr<GeometricRestraintForce<long long int, float>>
      weakRestraint = restraint->object;
  const std::weak_ptr<cudaStream_t> weakStream = restraint->object->getStream();
  const std::weak_ptr<Force<long long int>> weakForce =
      restraint->object->getForce();
  const std::weak_ptr<CudaEnergyVirial> weakEnergyVirial =
      restraint->object->getEnergyVirial();
  const std::weak_ptr<ForceManager> weakManager = system.forceManager->object;

  REQUIRE(apo_force_manager_subscribe_geometric_restraint_force(
              system.forceManager.get(), restraint.get(), "mmfp") ==
          APO_STATUS_OK);
  AddCharmmTerms(restraint.get(), system.psf.get(), test, test.phases[0],
                 AtomRoute::SELECTION);
  CheckManagerOutput(test, test.phases[0], baseline,
                     EvaluateManager(system.forceManager.get(), coordinates));

  SECTION("RestraintHandleDestroyedBeforeForceManager") {
    apo_geometric_restraint_force_destroy(restraint.release());

    CHECK(weakRestraint.expired() == false);
    CHECK(weakStream.expired() == false);
    CHECK(weakForce.expired() == false);
    CHECK(weakEnergyVirial.expired() == false);
    CHECK(weakRestraint.use_count() == 1);
    CheckManagerOutput(test, test.phases[0], baseline,
                       EvaluateManager(system.forceManager.get(), coordinates));

    system.context.reset();
    system.forceManager.reset();
    CHECK(weakManager.expired());
  }

  SECTION("ForceManagerDestroyedBeforeRestraintHandle") {
    system.context.reset();
    system.forceManager.reset();
    CHECK(weakManager.expired());
    CHECK(restraint->object.use_count() == 1);

    ZeroInteractionSystem replacement = MakeSystem();
    InitializeSystem(replacement);
    const ManagerOutput replacementBaseline =
        EvaluateManager(replacement.forceManager.get(), coordinates);

    REQUIRE(apo_force_manager_subscribe_geometric_restraint_force(
                replacement.forceManager.get(), restraint.get(), "mmfp") ==
            APO_STATUS_OK);
    CheckManagerOutput(
        test, test.phases[0], replacementBaseline,
        EvaluateManager(replacement.forceManager.get(), coordinates));
    REQUIRE(apo_force_manager_unsubscribe_geometric_restraint_force(
                replacement.forceManager.get(), restraint.get()) ==
            APO_STATUS_OK);

    restraint.reset();
  }

  CHECK(weakRestraint.expired());
  CHECK(weakStream.expired());
  CHECK(weakForce.expired());
  CHECK(weakEnergyVirial.expired());
}

TEST_CASE("CapiGeometricRestraintForceManagerReleasesBorrowedHandles") {
  const CharmmCase mixed =
      FindCharmmCase("External tensor for mixed atomwise and RCM terms");
  const CharmmCase plane =
      FindCharmmCase("Plane with a nonunit oblique normal");
  const std::array<std::array<double, 3>, CHARMM_NUM_ATOMS> coordinates =
      EvaluatedCoordinates(mixed);

  ZeroInteractionSystem system = MakeSystem();
  InitializeSystem(system);
  const ManagerOutput baseline =
      EvaluateManager(system.forceManager.get(), coordinates);

  std::weak_ptr<CharmmPSF> weakMassPsf;
  GeometricRestraintForceHandle restraint;
  {
    CharmmPsfHandle massPsf = MakePsf(CHARMM_NUM_ATOMS);
    std::copy(mixed.masses.begin(), mixed.masses.end(),
              massPsf->object->getMasses().begin());
    weakMassPsf = massPsf->object;
    restraint = MakeRestraint(massPsf.get());
  }
  CHECK(weakMassPsf.expired());

  std::vector<std::weak_ptr<AtomSelection>> weakSelections;
  for (const CharmmTerm &term : mixed.phases[0].terms) {
    AtomSelectionHandle selection = MakeSelection(CHARMM_NUM_ATOMS, term.atoms);
    weakSelections.push_back(selection->object);

    if (term.geometry == Geometry::PLANE) {
      REQUIRE(apo_geometric_restraint_force_add_plane_from_selection(
                  restraint.get(), selection.get(), term.origin.data(), 3,
                  term.direction.data(), 3, term.forceConstant, term.offset,
                  term.activation, term.selectionMode) == APO_STATUS_OK);
    } else if (term.geometry == Geometry::CYLINDER) {
      REQUIRE(apo_geometric_restraint_force_add_cylinder_from_selection(
                  restraint.get(), selection.get(), term.origin.data(), 3,
                  term.direction.data(), 3, term.forceConstant, term.offset,
                  term.activation, term.selectionMode) == APO_STATUS_OK);
    } else {
      REQUIRE(apo_geometric_restraint_force_add_sphere_from_selection(
                  restraint.get(), selection.get(), term.origin.data(), 3,
                  term.forceConstant, term.offset, term.activation,
                  term.selectionMode) == APO_STATUS_OK);
    }
  }

  for (const std::weak_ptr<AtomSelection> &weakSelection : weakSelections)
    CHECK(weakSelection.expired());

  REQUIRE(apo_force_manager_subscribe_geometric_restraint_force(
              system.forceManager.get(), restraint.get(), "mmfp") ==
          APO_STATUS_OK);
  CheckManagerOutput(mixed, mixed.phases[0], baseline,
                     EvaluateManager(system.forceManager.get(), coordinates));

  REQUIRE(apo_force_manager_unsubscribe_geometric_restraint_force(
              system.forceManager.get(), restraint.get()) == APO_STATUS_OK);
  REQUIRE(apo_geometric_restraint_force_reset(restraint.get()) ==
          APO_STATUS_OK);

  const std::array<std::array<double, 3>, CHARMM_NUM_ATOMS> planeCoordinates =
      EvaluatedCoordinates(plane);
  const ManagerOutput planeBaseline =
      EvaluateManager(system.forceManager.get(), planeCoordinates);

  std::weak_ptr<CharmmPSF> weakReferencePsf;
  {
    CharmmPsfHandle referencePsf = MakePsf(CHARMM_NUM_ATOMS);
    weakReferencePsf = referencePsf->object;

    AddCharmmTerms(restraint.get(), referencePsf.get(), plane, plane.phases[0],
                   AtomRoute::ATOM_REFERENCE);
  }
  CHECK(weakReferencePsf.expired());

  REQUIRE(apo_force_manager_subscribe_geometric_restraint_force(
              system.forceManager.get(), restraint.get(), "mmfp") ==
          APO_STATUS_OK);
  CheckManagerOutput(
      plane, plane.phases[0], planeBaseline,
      EvaluateManager(system.forceManager.get(), planeCoordinates));
}

TEST_CASE("CapiGeometricRestraintForceManagerMatchesCharmm") {
  ZeroInteractionSystem system = MakeSystem();
  InitializeSystem(system);

  const std::vector<CharmmCase> cases = MakeCharmmCases();
  REQUIRE(cases.size() == 23);

  for (const CharmmCase &test : cases) {
    RunCharmmCase(system, test, AtomRoute::SELECTION);

    bool singleAtomTerms = true;
    for (const CharmmPhase &phase : test.phases) {
      for (const CharmmTerm &term : phase.terms)
        singleAtomTerms = singleAtomTerms && (term.atoms.size() == 1);
    }

    if (singleAtomTerms)
      RunCharmmCase(system, test, AtomRoute::ATOM_REFERENCE);
  }
}

TEST_CASE("CapiGeometricRestraintForceManagerPreservesStateAfterRejections") {
  const CharmmCase plane =
      FindCharmmCase("Positive harmonic constant above the plane");
  const std::array<std::array<double, 3>, CHARMM_NUM_ATOMS> coordinates =
      EvaluatedCoordinates(plane);
  const std::array<double, 3> noShift = {0.0, 0.0, 0.0};

  ZeroInteractionSystem system = MakeSystem();
  InitializeSystem(system);
  GeometricRestraintForceHandle restraint = MakeRestraint(system.psf.get());

  REQUIRE(apo_force_manager_subscribe_geometric_restraint_force(
              system.forceManager.get(), restraint.get(), "mmfp") ==
          APO_STATUS_OK);

  const ManagerOutput baseline =
      EvaluateManager(system.forceManager.get(), coordinates);
  AddCharmmTerms(restraint.get(), system.psf.get(), plane, plane.phases[0],
                 AtomRoute::SELECTION);
  CheckManagerOutput(plane, plane.phases[0], baseline,
                     EvaluateManager(system.forceManager.get(), coordinates));

  SECTION("ZeroPlaneDirection") {
    CharmmTerm zeroDirection = plane.phases[0].terms[0];
    zeroDirection.direction = {0.0, 0.0, 0.0};

    CheckMmfpNativeError(
        AddCharmmTerm(restraint.get(), system.psf.get(), zeroDirection, noShift,
                      AtomRoute::SELECTION),
        "apo_geometric_restraint_force_add_plane_from_selection",
        "Direction must have a positive finite norm", "appendTerm");
    CheckMmfpNativeError(
        AddCharmmTerm(restraint.get(), system.psf.get(), zeroDirection, noShift,
                      AtomRoute::ATOM_REFERENCE),
        "apo_geometric_restraint_force_add_plane_from_atom_reference",
        "Direction must have a positive finite norm", "appendTerm");
  }

  SECTION("ZeroRcmTotalMass") {
    const CharmmTerm zeroTotalMass =
        FindCharmmCase("RCM with unequal positive masses").phases[0].terms[0];

    REQUIRE(apo_geometric_restraint_force_set_masses(
                restraint.get(), ZERO_TOTAL_MASSES.data(),
                ZERO_TOTAL_MASSES.size()) == APO_STATUS_OK);
    CheckMmfpNativeError(
        AddCharmmTerm(restraint.get(), system.psf.get(), zeroTotalMass, noShift,
                      AtomRoute::SELECTION),
        "apo_geometric_restraint_force_add_plane_from_selection",
        "RCM total mass must be finite and nonzero; observed " +
            std::to_string(0.0),
        "validateRcmMasses");
  }

  CHECK(restraint->object->getNumTerms() == 1);
  CheckManagerOutput(plane, plane.phases[0], baseline,
                     EvaluateManager(system.forceManager.get(), coordinates));
}
