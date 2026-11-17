// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "AtomSelection.h"
#include "CharmmPSF.h"
#include "ForceManager.h"
#include "apo_test_helpers.h"
#include "apocharmm_c/Error.h"
#include "apocharmm_c/HarmonicCenterOfMassRestraintForce.h"
#include "apocharmm_c/Status.h"
#include "apocharmm_c/detail/AtomSelectionHandle.h"
#include "apocharmm_c/detail/CharmmPsfHandle.h"
#include "apocharmm_c/detail/ForceManagerHandle.h"
#include "apocharmm_c/detail/HarmonicCenterOfMassRestraintForceHandle.h"
#include "catch.hpp"

#include <limits>
#include <memory>
#include <string>
#include <string_view>

namespace {

constexpr int NUM_ATOMS = 2;
constexpr char HMCM_SOURCE_FILE[] = "src/HarmonicCenterOfMassRestraintForce.cu";
constexpr char FORCE_MANAGER_SOURCE_FILE[] = "include/ForceManager.h";

struct HarmonicCenterOfMassRestraintForceDeleter {
  void operator()(
      apo_harmonic_center_of_mass_restraint_force *restraint) const noexcept {
    apo_harmonic_center_of_mass_restraint_force_destroy(restraint);
    return;
  }
};

using HarmonicCenterOfMassRestraintForceHandle =
    std::unique_ptr<apo_harmonic_center_of_mass_restraint_force,
                    HarmonicCenterOfMassRestraintForceDeleter>;

apo_charmm_psf MakePsf(const int numAtoms) {
  apo_charmm_psf psf;
  psf.object = std::make_shared<CharmmPSF>();
  psf.object->setNumAtoms(numAtoms);
  return psf;
}

HarmonicCenterOfMassRestraintForceHandle MakeRestraint(void) {
  apo_charmm_psf psf = MakePsf(NUM_ATOMS);
  apo_harmonic_center_of_mass_restraint_force *restraint = nullptr;

  REQUIRE(apo_harmonic_center_of_mass_restraint_force_create(
              &restraint, &psf) == APO_STATUS_OK);
  REQUIRE(restraint != nullptr);

  return HarmonicCenterOfMassRestraintForceHandle(restraint);
}

apo_atom_selection
MakeSelection(const int numAtoms,
              const AtomSelection::InitialValue initialValue =
                  AtomSelection::InitialValue::ALL) {
  apo_atom_selection selection;
  selection.object = std::make_shared<AtomSelection>(numAtoms, initialValue);
  return selection;
}

apo_force_manager MakeForceManager(void) {
  apo_force_manager forceManager;
  forceManager.object = std::make_shared<ForceManager>();
  return forceManager;
}

template <typename Function>
void CheckRestraintHandleValidation(const std::string_view functionName,
                                    Function function) {
  apo_test::CheckStatusAndDiagnostic(
      function(nullptr), APO_STATUS_INVALID_ARGUMENT,
      std::string(functionName) +
          ": HarmonicCenterOfMassRestraintForce is NULL");

  apo_harmonic_center_of_mass_restraint_force emptyRestraint;
  apo_test::CheckStatusAndDiagnostic(
      function(&emptyRestraint), APO_STATUS_INVALID_ARGUMENT,
      std::string(functionName) +
          ": HarmonicCenterOfMassRestraintForce object is NULL");

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

void CheckHmcmNativeError(const apo_status status,
                          const std::string_view functionName,
                          const std::string_view expectedMessage,
                          const std::string_view sourceFunction) {
  apo_test::CheckNativeError(status, APO_STATUS_INVALID_ARGUMENT,
                             "InvalidArgument", functionName, expectedMessage,
                             HMCM_SOURCE_FILE, sourceFunction);
  return;
}

void CheckForceManagerNativeError(const apo_status status,
                                  const std::string_view functionName,
                                  const std::string_view expectedMessage,
                                  const std::string_view sourceFunction) {
  apo_test::CheckNativeError(status, APO_STATUS_INVALID_ARGUMENT,
                             "InvalidArgument", functionName, expectedMessage,
                             FORCE_MANAGER_SOURCE_FILE, sourceFunction);
  return;
}

} // namespace

TEST_CASE("CapiHarmonicCenterOfMassRestraintForceCreatesAndDestroys") {
  apo_charmm_psf psf = MakePsf(NUM_ATOMS);

  apo_test::CheckStatusAndDiagnostic(
      apo_harmonic_center_of_mass_restraint_force_create(nullptr, &psf),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_harmonic_center_of_mass_restraint_force_create: out pointer is "
      "NULL");

  apo_harmonic_center_of_mass_restraint_force staleRestraint;
  apo_harmonic_center_of_mass_restraint_force *restraint = &staleRestraint;
  apo_status status = APO_STATUS_OK;

  apo_test::CheckStatusAndDiagnostic(
      apo_harmonic_center_of_mass_restraint_force_create(&restraint, nullptr),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_harmonic_center_of_mass_restraint_force_create: CharmmPsf is "
      "NULL");
  CHECK(restraint == nullptr);

  apo_charmm_psf emptyPsf;
  restraint = &staleRestraint;
  apo_test::CheckStatusAndDiagnostic(
      apo_harmonic_center_of_mass_restraint_force_create(&restraint, &emptyPsf),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_harmonic_center_of_mass_restraint_force_create: CharmmPsf object "
      "is NULL");
  CHECK(restraint == nullptr);

  apo_charmm_psf uninitializedPsf;
  uninitializedPsf.object = std::make_shared<CharmmPSF>();
  restraint = &staleRestraint;
  CHECK_NOTHROW((status = apo_harmonic_center_of_mass_restraint_force_create(
                     &restraint, &uninitializedPsf)));

  apo_test::CheckNativeError(
      status, APO_STATUS_NOT_INITIALIZED, "NotInitialized",
      "apo_harmonic_center_of_mass_restraint_force_create",
      "CharmmPSF atom count is not initialized; observed -1", HMCM_SOURCE_FILE,
      "HarmonicCenterOfMassRestraintForce");
  CHECK(restraint == nullptr);

  apo_charmm_psf emptyAtomPsf = MakePsf(0);
  restraint = &staleRestraint;
  CHECK_NOTHROW((status = apo_harmonic_center_of_mass_restraint_force_create(
                     &restraint, &emptyAtomPsf)));

  CheckHmcmNativeError(status,
                       "apo_harmonic_center_of_mass_restraint_force_create",
                       "Atom count must be positive; observed 0",
                       "HarmonicCenterOfMassRestraintForce");
  CHECK(restraint == nullptr);

  apo_charmm_psf shortMassPsf = MakePsf(NUM_ATOMS);
  shortMassPsf.object->getMasses().pop_back();
  restraint = &staleRestraint;
  CHECK_NOTHROW((status = apo_harmonic_center_of_mass_restraint_force_create(
                     &restraint, &shortMassPsf)));

  apo_test::CheckNativeError(
      status, APO_STATUS_RUNTIME_ERROR, "Runtime",
      "apo_harmonic_center_of_mass_restraint_force_create",
      "PSF mass count mismatch; expected 2, observed 1", HMCM_SOURCE_FILE,
      "HarmonicCenterOfMassRestraintForce");
  CHECK(restraint == nullptr);

  const double infinity = std::numeric_limits<double>::infinity();
  apo_charmm_psf infiniteMassPsf = MakePsf(NUM_ATOMS);
  infiniteMassPsf.object->getMasses()[1] = infinity;
  restraint = &staleRestraint;
  CHECK_NOTHROW((status = apo_harmonic_center_of_mass_restraint_force_create(
                     &restraint, &infiniteMassPsf)));

  CheckHmcmNativeError(
      status, "apo_harmonic_center_of_mass_restraint_force_create",
      "Mass at index 1 must be finite; observed " + std::to_string(infinity),
      "HarmonicCenterOfMassRestraintForce");
  CHECK(restraint == nullptr);

  const std::weak_ptr<CharmmPSF> weakPsf = psf.object;
  CHECK(apo_harmonic_center_of_mass_restraint_force_create(&restraint, &psf) ==
        APO_STATUS_OK);
  REQUIRE(restraint != nullptr);
  CHECK(std::string(apo_last_error()).empty() == true);
  CHECK(restraint->object->getForce()->size() == NUM_ATOMS);
  psf.object.reset();
  CHECK(weakPsf.expired());

  CHECK_NOTHROW(apo_harmonic_center_of_mass_restraint_force_destroy(restraint));
  CHECK(std::string(apo_last_error()).empty() == true);

  REQUIRE(apo_harmonic_center_of_mass_restraint_force_set_force_constant(
              nullptr, 1.0) == APO_STATUS_INVALID_ARGUMENT);
  const std::string diagnostic = apo_last_error();
  REQUIRE(diagnostic.empty() == false);

  CHECK_NOTHROW(apo_harmonic_center_of_mass_restraint_force_destroy(nullptr));
  CHECK(std::string(apo_last_error()) == diagnostic);
}

TEST_CASE("CapiHarmonicCenterOfMassRestraintForceWeightsWithPsfMasses") {
  apo_charmm_psf psf = MakePsf(NUM_ATOMS);
  psf.object->getMasses() = {1.0, -1.0};

  apo_harmonic_center_of_mass_restraint_force *restraint = nullptr;
  REQUIRE(apo_harmonic_center_of_mass_restraint_force_create(
              &restraint, &psf) == APO_STATUS_OK);
  HarmonicCenterOfMassRestraintForceHandle ownedRestraint(restraint);
  psf.object->getMasses() = {1.0, 1.0};

  CheckHmcmNativeError(
      apo_harmonic_center_of_mass_restraint_force_set_mass_weighting(
          ownedRestraint.get(), true),
      "apo_harmonic_center_of_mass_restraint_force_set_mass_weighting",
      "Selected atoms must have a finite nonzero total weight; observed " +
          std::to_string(0.0),
      "updateSelectedAtoms");
}

TEST_CASE(
    "CapiHarmonicCenterOfMassRestraintForceValidatesEveryRestraintHandle") {
  const double referencePosition[3] = {0.0, 0.0, 0.0};
  const int referenceMask[3] = {1, 1, 1};
  const double masses[NUM_ATOMS] = {1.0, 2.0};
  apo_atom_selection selection = MakeSelection(NUM_ATOMS);
  apo_force_manager forceManager = MakeForceManager();

  CheckRestraintHandleValidation(
      "apo_harmonic_center_of_mass_restraint_force_set_selection",
      [&selection](apo_harmonic_center_of_mass_restraint_force *restraint)
          -> apo_status {
        return apo_harmonic_center_of_mass_restraint_force_set_selection(
            restraint, &selection);
      });

  CheckRestraintHandleValidation(
      "apo_harmonic_center_of_mass_restraint_force_set_force_constant",
      [](apo_harmonic_center_of_mass_restraint_force *restraint) -> apo_status {
        return apo_harmonic_center_of_mass_restraint_force_set_force_constant(
            restraint, 1.0);
      });

  CheckRestraintHandleValidation(
      "apo_harmonic_center_of_mass_restraint_force_set_reference_position",
      [&referencePosition,
       &referenceMask](apo_harmonic_center_of_mass_restraint_force *restraint)
          -> apo_status {
        return apo_harmonic_center_of_mass_restraint_force_set_reference_position(
            restraint, referencePosition, 3, referenceMask, 3);
      });

  CheckRestraintHandleValidation(
      "apo_harmonic_center_of_mass_restraint_force_set_reference_distance",
      [](apo_harmonic_center_of_mass_restraint_force *restraint) -> apo_status {
        return apo_harmonic_center_of_mass_restraint_force_set_reference_distance(
            restraint, 0.0);
      });

  CheckRestraintHandleValidation(
      "apo_harmonic_center_of_mass_restraint_force_set_masses",
      [&masses](apo_harmonic_center_of_mass_restraint_force *restraint)
          -> apo_status {
        return apo_harmonic_center_of_mass_restraint_force_set_masses(
            restraint, masses, NUM_ATOMS);
      });

  CheckRestraintHandleValidation(
      "apo_harmonic_center_of_mass_restraint_force_set_mass_weighting",
      [](apo_harmonic_center_of_mass_restraint_force *restraint) -> apo_status {
        return apo_harmonic_center_of_mass_restraint_force_set_mass_weighting(
            restraint, true);
      });

  CheckRestraintHandleValidation(
      "apo_force_manager_subscribe_harmonic_center_of_mass_restraint_force",
      [&forceManager](apo_harmonic_center_of_mass_restraint_force *restraint)
          -> apo_status {
        return apo_force_manager_subscribe_harmonic_center_of_mass_restraint_force(
            &forceManager, restraint, "hmcm");
      });

  CheckRestraintHandleValidation(
      "apo_force_manager_unsubscribe_harmonic_center_of_mass_restraint_force",
      [&forceManager](apo_harmonic_center_of_mass_restraint_force *restraint)
          -> apo_status {
        return apo_force_manager_unsubscribe_harmonic_center_of_mass_restraint_force(
            &forceManager, restraint);
      });
}

TEST_CASE("CapiHarmonicCenterOfMassRestraintForceValidatesSelectionAndPointerAr"
          "guments") {
  HarmonicCenterOfMassRestraintForceHandle restraint = MakeRestraint();
  const double referencePosition[3] = {0.0, 0.0, 0.0};
  const int referenceMask[3] = {1, 1, 1};

  apo_test::CheckStatusAndDiagnostic(
      apo_harmonic_center_of_mass_restraint_force_set_selection(restraint.get(),
                                                                nullptr),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_harmonic_center_of_mass_restraint_force_set_selection: "
      "AtomSelection is NULL");

  apo_atom_selection emptySelection;
  apo_test::CheckStatusAndDiagnostic(
      apo_harmonic_center_of_mass_restraint_force_set_selection(
          restraint.get(), &emptySelection),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_harmonic_center_of_mass_restraint_force_set_selection: "
      "AtomSelection object is NULL");

  apo_test::CheckStatusAndDiagnostic(
      apo_harmonic_center_of_mass_restraint_force_set_reference_position(
          restraint.get(), nullptr, 3, referenceMask, 3),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_harmonic_center_of_mass_restraint_force_set_reference_position: "
      "reference_position is NULL");

  apo_test::CheckStatusAndDiagnostic(
      apo_harmonic_center_of_mass_restraint_force_set_reference_position(
          restraint.get(), referencePosition, 3, nullptr, 3),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_harmonic_center_of_mass_restraint_force_set_reference_position: "
      "reference_mask is NULL");

  apo_test::CheckStatusAndDiagnostic(
      apo_harmonic_center_of_mass_restraint_force_set_masses(
          restraint.get(), nullptr, NUM_ATOMS),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_harmonic_center_of_mass_restraint_force_set_masses: masses is "
      "NULL");
}

TEST_CASE("CapiHarmonicCenterOfMassRestraintForceMapsNativeValidation") {
  HarmonicCenterOfMassRestraintForceHandle restraint = MakeRestraint();
  const double infinity = std::numeric_limits<double>::infinity();

  SECTION("SelectionAtomCount") {
    apo_atom_selection wrongSelection = MakeSelection(NUM_ATOMS + 1);
    const apo_status status =
        apo_harmonic_center_of_mass_restraint_force_set_selection(
            restraint.get(), &wrongSelection);

    CheckHmcmNativeError(
        status, "apo_harmonic_center_of_mass_restraint_force_set_selection",
        "Selection atom count mismatch; expected 2, observed 3",
        "setSelection");
  }

  SECTION("EmptySelection") {
    apo_atom_selection emptySelection =
        MakeSelection(NUM_ATOMS, AtomSelection::InitialValue::NONE);
    const apo_status status =
        apo_harmonic_center_of_mass_restraint_force_set_selection(
            restraint.get(), &emptySelection);

    CheckHmcmNativeError(
        status, "apo_harmonic_center_of_mass_restraint_force_set_selection",
        "Selection must contain at least one atom; observed 0", "setSelection");
  }

  SECTION("ScalarForceConstantFinite") {
    const apo_status status =
        apo_harmonic_center_of_mass_restraint_force_set_force_constant(
            restraint.get(), infinity);

    CheckHmcmNativeError(
        status,
        "apo_harmonic_center_of_mass_restraint_force_set_force_constant",
        "Force constant must be finite; observed " + std::to_string(infinity),
        "setForceConstant");
  }

  SECTION("ScalarForceConstantAcceptsEitherSign") {
    CHECK(apo_harmonic_center_of_mass_restraint_force_set_force_constant(
              restraint.get(), -1.0) == APO_STATUS_OK);
    CHECK(std::string(apo_last_error()).empty() == true);
  }

  SECTION("ReferencePositionArraySize") {
    const double referencePosition[2] = {0.0, 0.0};
    const int referenceMask[3] = {1, 1, 1};
    const apo_status status =
        apo_harmonic_center_of_mass_restraint_force_set_reference_position(
            restraint.get(), referencePosition, 2, referenceMask, 3);

    CheckHmcmNativeError(
        status,
        "apo_harmonic_center_of_mass_restraint_force_set_reference_position",
        "Reference-position array size mismatch; expected 3, observed 2",
        "setReferencePosition");
  }

  SECTION("ReferenceMaskArraySize") {
    const double referencePosition[3] = {0.0, 0.0, 0.0};
    const int referenceMask[2] = {1, 1};
    const apo_status status =
        apo_harmonic_center_of_mass_restraint_force_set_reference_position(
            restraint.get(), referencePosition, 3, referenceMask, 2);

    CheckHmcmNativeError(
        status,
        "apo_harmonic_center_of_mass_restraint_force_set_reference_position",
        "Reference-mask array size mismatch; expected 3, observed 2",
        "setReferencePosition");
  }

  SECTION("ReferencePositionFinite") {
    const double referencePosition[3] = {0.0, infinity, 0.0};
    const int referenceMask[3] = {1, 1, 1};
    const apo_status status =
        apo_harmonic_center_of_mass_restraint_force_set_reference_position(
            restraint.get(), referencePosition, 3, referenceMask, 3);

    CheckHmcmNativeError(
        status,
        "apo_harmonic_center_of_mass_restraint_force_set_reference_position",
        "Reference position at index 1 must be finite; observed " +
            std::to_string(infinity),
        "setReferencePosition");
  }

  SECTION("ReferenceMaskValue") {
    const double referencePosition[3] = {0.0, 0.0, 0.0};
    const int referenceMask[3] = {1, 2, 1};
    const apo_status status =
        apo_harmonic_center_of_mass_restraint_force_set_reference_position(
            restraint.get(), referencePosition, 3, referenceMask, 3);

    CheckHmcmNativeError(
        status,
        "apo_harmonic_center_of_mass_restraint_force_set_reference_position",
        "Reference mask at index 1 must be 0 or 1; observed 2",
        "setReferencePosition");
  }

  SECTION("ReferenceMaskActive") {
    const double referencePosition[3] = {0.0, 0.0, 0.0};
    const int referenceMask[3] = {0, 0, 0};
    const apo_status status =
        apo_harmonic_center_of_mass_restraint_force_set_reference_position(
            restraint.get(), referencePosition, 3, referenceMask, 3);

    CheckHmcmNativeError(
        status,
        "apo_harmonic_center_of_mass_restraint_force_set_reference_position",
        "Reference mask must activate at least one coordinate",
        "setReferencePosition");
  }

  SECTION("ReferenceDistanceFinite") {
    const apo_status status =
        apo_harmonic_center_of_mass_restraint_force_set_reference_distance(
            restraint.get(), infinity);

    CheckHmcmNativeError(
        status,
        "apo_harmonic_center_of_mass_restraint_force_set_reference_distance",
        "Reference distance must be finite; observed " +
            std::to_string(infinity),
        "setReferenceDistance");
  }

  SECTION("ReferenceDistanceAcceptsEitherSign") {
    CHECK(apo_harmonic_center_of_mass_restraint_force_set_reference_distance(
              restraint.get(), -1.0) == APO_STATUS_OK);
    CHECK(std::string(apo_last_error()).empty() == true);
  }

  SECTION("MassArraySize") {
    const double masses[1] = {1.0};
    const apo_status status =
        apo_harmonic_center_of_mass_restraint_force_set_masses(restraint.get(),
                                                               masses, 1);

    CheckHmcmNativeError(
        status, "apo_harmonic_center_of_mass_restraint_force_set_masses",
        "Mass array size mismatch; expected 2, observed 1", "setMasses");
  }

  SECTION("MassFinite") {
    const double masses[NUM_ATOMS] = {1.0, infinity};
    const apo_status status =
        apo_harmonic_center_of_mass_restraint_force_set_masses(
            restraint.get(), masses, NUM_ATOMS);

    CheckHmcmNativeError(
        status, "apo_harmonic_center_of_mass_restraint_force_set_masses",
        "Mass at index 1 must be finite; observed " + std::to_string(infinity),
        "setMasses");
  }

  SECTION("MassAcceptsEitherSign") {
    const double masses[NUM_ATOMS] = {1.0, -3.0};

    CHECK(apo_harmonic_center_of_mass_restraint_force_set_masses(
              restraint.get(), masses, NUM_ATOMS) == APO_STATUS_OK);
    CHECK(std::string(apo_last_error()).empty() == true);
  }

  SECTION("SelectedTotalWeightFromMasses") {
    const double masses[NUM_ATOMS] = {1.0, -1.0};
    const apo_status status =
        apo_harmonic_center_of_mass_restraint_force_set_masses(
            restraint.get(), masses, NUM_ATOMS);

    CheckHmcmNativeError(
        status, "apo_harmonic_center_of_mass_restraint_force_set_masses",
        "Selected atoms must have a finite nonzero total weight; observed " +
            std::to_string(0.0),
        "updateSelectedAtoms");
  }

  SECTION("SelectedTotalWeightFromMassWeighting") {
    const double masses[NUM_ATOMS] = {0.0, 0.0};
    apo_status status = apo_harmonic_center_of_mass_restraint_force_set_masses(
        restraint.get(), masses, NUM_ATOMS);

    CheckHmcmNativeError(
        status, "apo_harmonic_center_of_mass_restraint_force_set_masses",
        "Selected atoms must have a finite nonzero total weight; observed " +
            std::to_string(0.0),
        "updateSelectedAtoms");

    REQUIRE(apo_harmonic_center_of_mass_restraint_force_set_mass_weighting(
                restraint.get(), false) == APO_STATUS_OK);

    status = apo_harmonic_center_of_mass_restraint_force_set_mass_weighting(
        restraint.get(), true);

    CheckHmcmNativeError(
        status,
        "apo_harmonic_center_of_mass_restraint_force_set_mass_weighting",
        "Selected atoms must have a finite nonzero total weight; observed " +
            std::to_string(0.0),
        "updateSelectedAtoms");
  }
}

TEST_CASE(
    "CapiHarmonicCenterOfMassRestraintForceValidatesForceManagerSubscription") {
  HarmonicCenterOfMassRestraintForceHandle restraint = MakeRestraint();
  apo_force_manager forceManager = MakeForceManager();

  CheckForceManagerHandleValidation(
      "apo_force_manager_subscribe_harmonic_center_of_mass_restraint_force",
      [&restraint](apo_force_manager *candidate) -> apo_status {
        return apo_force_manager_subscribe_harmonic_center_of_mass_restraint_force(
            candidate, restraint.get(), "hmcm");
      });

  CheckForceManagerHandleValidation(
      "apo_force_manager_unsubscribe_harmonic_center_of_mass_restraint_force",
      [&restraint](apo_force_manager *candidate) -> apo_status {
        return apo_force_manager_unsubscribe_harmonic_center_of_mass_restraint_force(
            candidate, restraint.get());
      });

  apo_test::CheckStatusAndDiagnostic(
      apo_force_manager_subscribe_harmonic_center_of_mass_restraint_force(
          &forceManager, restraint.get(), nullptr),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_force_manager_subscribe_harmonic_center_of_mass_restraint_force: "
      "force_tag is NULL");

  apo_status status =
      apo_force_manager_subscribe_harmonic_center_of_mass_restraint_force(
          &forceManager, restraint.get(), "");
  CheckForceManagerNativeError(
      status,
      "apo_force_manager_subscribe_harmonic_center_of_mass_restraint_force",
      "Force tag must not be empty", "subscribe");

  CHECK(apo_force_manager_subscribe_harmonic_center_of_mass_restraint_force(
            &forceManager, restraint.get(), "hmcm") == APO_STATUS_OK);
  CHECK(std::string(apo_last_error()).empty() == true);

  status = apo_force_manager_subscribe_harmonic_center_of_mass_restraint_force(
      &forceManager, restraint.get(), "hmcm");
  CheckForceManagerNativeError(
      status,
      "apo_force_manager_subscribe_harmonic_center_of_mass_restraint_force",
      "Force is already subscribed to this ForceManager", "subscribe");

  CHECK(apo_force_manager_unsubscribe_harmonic_center_of_mass_restraint_force(
            &forceManager, restraint.get()) == APO_STATUS_OK);
  CHECK(std::string(apo_last_error()).empty() == true);

  status =
      apo_force_manager_unsubscribe_harmonic_center_of_mass_restraint_force(
          &forceManager, restraint.get());
  CheckForceManagerNativeError(
      status,
      "apo_force_manager_unsubscribe_harmonic_center_of_mass_restraint_force",
      "Force is not subscribed to this ForceManager", "unsubscribe");
}

TEST_CASE(
    "CapiHarmonicCenterOfMassRestraintForceSuccessClearsStaleDiagnostic") {
  HarmonicCenterOfMassRestraintForceHandle restraint = MakeRestraint();

  REQUIRE(apo_harmonic_center_of_mass_restraint_force_set_force_constant(
              nullptr, 1.0) == APO_STATUS_INVALID_ARGUMENT);
  REQUIRE(std::string(apo_last_error()).empty() == false);

  CHECK(apo_harmonic_center_of_mass_restraint_force_set_force_constant(
            restraint.get(), 1.0) == APO_STATUS_OK);
  CHECK(std::string(apo_last_error()).empty() == true);
}
