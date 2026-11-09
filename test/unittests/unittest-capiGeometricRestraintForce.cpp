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
#include "apo_test_helpers.h"
#include "apocharmm_c/AtomReference.h"
#include "apocharmm_c/AtomSelection.h"
#include "apocharmm_c/CharmmPsf.h"
#include "apocharmm_c/Error.h"
#include "apocharmm_c/GeometricRestraintForce.h"
#include "apocharmm_c/Status.h"
#include "apocharmm_c/detail/AtomReferenceHandle.h"
#include "apocharmm_c/detail/AtomSelectionHandle.h"
#include "apocharmm_c/detail/CharmmPsfHandle.h"
#include "apocharmm_c/detail/GeometricRestraintForceHandle.h"
#include "catch.hpp"

#include <cstddef>
#include <functional>
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

using GeometricRestraintForceHandle =
    std::unique_ptr<apo_geometric_restraint_force,
                    GeometricRestraintForceDeleter>;
using CharmmPsfHandle = std::unique_ptr<apo_charmm_psf, CharmmPsfDeleter>;
using AtomSelectionHandle =
    std::unique_ptr<apo_atom_selection, AtomSelectionDeleter>;
using AtomReferenceHandle =
    std::unique_ptr<apo_atom_reference, AtomReferenceDeleter>;

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

  SECTION("AcceptsCharmmCharacterizedUnusualMasses") {
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
