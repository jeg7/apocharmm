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
#include "apocharmm_c/HarmonicRestraintForce.h"
#include "apocharmm_c/Status.h"
#include "apocharmm_c/detail/AtomSelectionHandle.h"
#include "apocharmm_c/detail/CharmmPsfHandle.h"
#include "apocharmm_c/detail/ForceManagerHandle.h"
#include "apocharmm_c/detail/HarmonicRestraintForceHandle.h"
#include "catch.hpp"

#include <limits>
#include <memory>
#include <string>
#include <string_view>

namespace {

constexpr int NUM_ATOMS = 2;
constexpr char HARMONIC_SOURCE_FILE[] = "src/HarmonicRestraintForce.cu";
constexpr char FORCE_MANAGER_SOURCE_FILE[] = "include/ForceManager.h";

struct HarmonicRestraintForceDeleter {
  void operator()(apo_harmonic_restraint_force *restraint) const noexcept {
    apo_harmonic_restraint_force_destroy(restraint);
    return;
  }
};

using HarmonicRestraintForceHandle =
    std::unique_ptr<apo_harmonic_restraint_force,
                    HarmonicRestraintForceDeleter>;

apo_charmm_psf MakePsf(const int numAtoms) {
  apo_charmm_psf psf;
  psf.object = std::make_shared<CharmmPSF>();
  psf.object->setNumAtoms(numAtoms);
  return psf;
}

HarmonicRestraintForceHandle MakeRestraint(void) {
  apo_charmm_psf psf = MakePsf(NUM_ATOMS);
  apo_harmonic_restraint_force *restraint = nullptr;

  REQUIRE(apo_harmonic_restraint_force_create(&restraint, &psf) ==
          APO_STATUS_OK);
  REQUIRE(restraint != nullptr);

  return HarmonicRestraintForceHandle(restraint);
}

apo_atom_selection MakeSelection(const int numAtoms) {
  apo_atom_selection selection;
  selection.object = std::make_shared<AtomSelection>(numAtoms);
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
      std::string(functionName) + ": HarmonicRestraintForce is NULL");

  apo_harmonic_restraint_force emptyRestraint;
  apo_test::CheckStatusAndDiagnostic(
      function(&emptyRestraint), APO_STATUS_INVALID_ARGUMENT,
      std::string(functionName) + ": HarmonicRestraintForce object is NULL");

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

void CheckHarmonicNativeError(const apo_status status,
                              const std::string_view functionName,
                              const std::string_view expectedMessage,
                              const std::string_view sourceFunction) {
  apo_test::CheckNativeError(status, APO_STATUS_INVALID_ARGUMENT,
                             "InvalidArgument", functionName, expectedMessage,
                             HARMONIC_SOURCE_FILE, sourceFunction);
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

TEST_CASE("CapiHarmonicRestraintForceCreatesAndDestroys") {
  apo_charmm_psf psf = MakePsf(NUM_ATOMS);

  apo_test::CheckStatusAndDiagnostic(
      apo_harmonic_restraint_force_create(nullptr, &psf),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_harmonic_restraint_force_create: out pointer is NULL");

  apo_harmonic_restraint_force staleRestraint;
  apo_harmonic_restraint_force *restraint = &staleRestraint;
  apo_status status = APO_STATUS_OK;

  apo_test::CheckStatusAndDiagnostic(
      apo_harmonic_restraint_force_create(&restraint, nullptr),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_harmonic_restraint_force_create: CharmmPsf is NULL");
  CHECK(restraint == nullptr);

  apo_charmm_psf emptyPsf;
  restraint = &staleRestraint;
  apo_test::CheckStatusAndDiagnostic(
      apo_harmonic_restraint_force_create(&restraint, &emptyPsf),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_harmonic_restraint_force_create: CharmmPsf object is NULL");
  CHECK(restraint == nullptr);

  apo_charmm_psf uninitializedPsf;
  uninitializedPsf.object = std::make_shared<CharmmPSF>();
  restraint = &staleRestraint;
  CHECK_NOTHROW((status = apo_harmonic_restraint_force_create(
                     &restraint, &uninitializedPsf)));

  apo_test::CheckNativeError(
      status, APO_STATUS_NOT_INITIALIZED, "NotInitialized",
      "apo_harmonic_restraint_force_create",
      "CharmmPSF atom count is not initialized; observed -1",
      HARMONIC_SOURCE_FILE, "HarmonicRestraintForce");
  CHECK(restraint == nullptr);

  apo_charmm_psf emptyAtomPsf = MakePsf(0);
  restraint = &staleRestraint;
  CHECK_NOTHROW((
      status = apo_harmonic_restraint_force_create(&restraint, &emptyAtomPsf)));

  CheckHarmonicNativeError(status, "apo_harmonic_restraint_force_create",
                           "Atom count must be positive; observed 0",
                           "HarmonicRestraintForce");
  CHECK(restraint == nullptr);

  const std::weak_ptr<CharmmPSF> weakPsf = psf.object;
  CHECK(apo_harmonic_restraint_force_create(&restraint, &psf) == APO_STATUS_OK);
  REQUIRE(restraint != nullptr);
  CHECK(std::string(apo_last_error()).empty() == true);
  CHECK(restraint->object->getForce()->size() == NUM_ATOMS);
  psf.object.reset();
  CHECK(weakPsf.expired());

  CHECK_NOTHROW(apo_harmonic_restraint_force_destroy(restraint));
  CHECK(std::string(apo_last_error()).empty() == true);

  REQUIRE(apo_harmonic_restraint_force_set_force_constant(nullptr, 1.0) ==
          APO_STATUS_INVALID_ARGUMENT);
  const std::string diagnostic = apo_last_error();
  REQUIRE(diagnostic.empty() == false);

  CHECK_NOTHROW(apo_harmonic_restraint_force_destroy(nullptr));
  CHECK(std::string(apo_last_error()) == diagnostic);
}

TEST_CASE("CapiHarmonicRestraintForceValidatesEveryRestraintHandle") {
  const double forceConstants[NUM_ATOMS] = {1.0, 2.0};
  const double referenceCoordinates[3 * NUM_ATOMS] = {0.0, 0.0, 0.0,
                                                      1.0, 1.0, 1.0};
  const double masses[NUM_ATOMS] = {1.0, 2.0};
  const double scaleFactors[3] = {1.0, 1.0, 1.0};
  const double boxDimensions[3] = {50.0, 50.0, 50.0};
  apo_atom_selection selection = MakeSelection(NUM_ATOMS);
  apo_force_manager forceManager = MakeForceManager();

  CheckRestraintHandleValidation(
      "apo_harmonic_restraint_force_set_selection",
      [&selection](apo_harmonic_restraint_force *restraint) -> apo_status {
        return apo_harmonic_restraint_force_set_selection(restraint,
                                                          &selection);
      });

  CheckRestraintHandleValidation(
      "apo_harmonic_restraint_force_set_force_constant",
      [](apo_harmonic_restraint_force *restraint) -> apo_status {
        return apo_harmonic_restraint_force_set_force_constant(restraint, 1.0);
      });

  CheckRestraintHandleValidation(
      "apo_harmonic_restraint_force_set_force_constants",
      [&forceConstants](apo_harmonic_restraint_force *restraint) -> apo_status {
        return apo_harmonic_restraint_force_set_force_constants(
            restraint, forceConstants, NUM_ATOMS);
      });

  CheckRestraintHandleValidation(
      "apo_harmonic_restraint_force_set_reference_coordinates",
      [&referenceCoordinates](
          apo_harmonic_restraint_force *restraint) -> apo_status {
        return apo_harmonic_restraint_force_set_reference_coordinates(
            restraint, referenceCoordinates, 3 * NUM_ATOMS);
      });

  CheckRestraintHandleValidation(
      "apo_harmonic_restraint_force_set_masses",
      [&masses](apo_harmonic_restraint_force *restraint) -> apo_status {
        return apo_harmonic_restraint_force_set_masses(restraint, masses,
                                                       NUM_ATOMS);
      });

  CheckRestraintHandleValidation(
      "apo_harmonic_restraint_force_set_exponent",
      [](apo_harmonic_restraint_force *restraint) -> apo_status {
        return apo_harmonic_restraint_force_set_exponent(restraint, 2);
      });

  CheckRestraintHandleValidation(
      "apo_harmonic_restraint_force_set_scale_factors",
      [&scaleFactors](apo_harmonic_restraint_force *restraint) -> apo_status {
        return apo_harmonic_restraint_force_set_scale_factors(restraint,
                                                              scaleFactors, 3);
      });

  CheckRestraintHandleValidation(
      "apo_harmonic_restraint_force_set_box_dimensions",
      [&boxDimensions](apo_harmonic_restraint_force *restraint) -> apo_status {
        return apo_harmonic_restraint_force_set_box_dimensions(
            restraint, boxDimensions, 3);
      });

  CheckRestraintHandleValidation(
      "apo_force_manager_subscribe_harmonic_restraint_force",
      [&forceManager](apo_harmonic_restraint_force *restraint) -> apo_status {
        return apo_force_manager_subscribe_harmonic_restraint_force(
            &forceManager, restraint, "harm");
      });

  CheckRestraintHandleValidation(
      "apo_force_manager_unsubscribe_harmonic_restraint_force",
      [&forceManager](apo_harmonic_restraint_force *restraint) -> apo_status {
        return apo_force_manager_unsubscribe_harmonic_restraint_force(
            &forceManager, restraint);
      });
}

TEST_CASE("CapiHarmonicRestraintForceValidatesSelectionAndArrayArguments") {
  HarmonicRestraintForceHandle restraint = MakeRestraint();
  const double scalar = 0.0;

  apo_test::CheckStatusAndDiagnostic(
      apo_harmonic_restraint_force_set_selection(restraint.get(), nullptr),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_harmonic_restraint_force_set_selection: AtomSelection is NULL");

  apo_atom_selection emptySelection;
  apo_test::CheckStatusAndDiagnostic(
      apo_harmonic_restraint_force_set_selection(restraint.get(),
                                                 &emptySelection),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_harmonic_restraint_force_set_selection: AtomSelection object is "
      "NULL");

  apo_test::CheckStatusAndDiagnostic(
      apo_harmonic_restraint_force_set_force_constants(restraint.get(), nullptr,
                                                       NUM_ATOMS),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_harmonic_restraint_force_set_force_constants: force_constants is "
      "NULL");

  apo_test::CheckStatusAndDiagnostic(
      apo_harmonic_restraint_force_set_reference_coordinates(
          restraint.get(), nullptr, 3 * NUM_ATOMS),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_harmonic_restraint_force_set_reference_coordinates: "
      "reference_coordinates is NULL");

  apo_test::CheckStatusAndDiagnostic(
      apo_harmonic_restraint_force_set_reference_coordinates(restraint.get(),
                                                             &scalar, 0),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_harmonic_restraint_force_set_reference_coordinates: "
      "reference_coordinates must not be empty");

  apo_test::CheckStatusAndDiagnostic(
      apo_harmonic_restraint_force_set_reference_coordinates(restraint.get(),
                                                             &scalar, 4),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_harmonic_restraint_force_set_reference_coordinates: "
      "reference_coordinates length must be a positive multiple of 3");

  apo_test::CheckStatusAndDiagnostic(
      apo_harmonic_restraint_force_set_masses(restraint.get(), nullptr,
                                              NUM_ATOMS),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_harmonic_restraint_force_set_masses: masses is NULL");

  apo_test::CheckStatusAndDiagnostic(
      apo_harmonic_restraint_force_set_scale_factors(restraint.get(), nullptr,
                                                     3),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_harmonic_restraint_force_set_scale_factors: scale_factors is "
      "NULL");

  apo_test::CheckStatusAndDiagnostic(
      apo_harmonic_restraint_force_set_box_dimensions(restraint.get(), nullptr,
                                                      3),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_harmonic_restraint_force_set_box_dimensions: box_dimensions is "
      "NULL");
}

TEST_CASE("CapiHarmonicRestraintForceMapsNativeValidation") {
  HarmonicRestraintForceHandle restraint = MakeRestraint();
  const double infinity = std::numeric_limits<double>::infinity();

  SECTION("SelectionAtomCount") {
    apo_atom_selection wrongSelection = MakeSelection(NUM_ATOMS + 1);
    const apo_status status = apo_harmonic_restraint_force_set_selection(
        restraint.get(), &wrongSelection);

    CheckHarmonicNativeError(
        status, "apo_harmonic_restraint_force_set_selection",
        "Selection atom count mismatch; expected 2, observed 3",
        "setSelection");
  }

  SECTION("ScalarForceConstantFinite") {
    const apo_status status = apo_harmonic_restraint_force_set_force_constant(
        restraint.get(), infinity);

    CheckHarmonicNativeError(
        status, "apo_harmonic_restraint_force_set_force_constant",
        "Force constant must be finite; observed " + std::to_string(infinity),
        "setForceConstant");
  }

  SECTION("ScalarForceConstantNonnegative") {
    const apo_status status =
        apo_harmonic_restraint_force_set_force_constant(restraint.get(), -1.0);

    CheckHarmonicNativeError(
        status, "apo_harmonic_restraint_force_set_force_constant",
        "Force constant must be non-negative; observed " + std::to_string(-1.0),
        "setForceConstant");
  }

  SECTION("ForceConstantArraySize") {
    const double forceConstants[1] = {1.0};
    const apo_status status = apo_harmonic_restraint_force_set_force_constants(
        restraint.get(), forceConstants, 1);

    CheckHarmonicNativeError(
        status, "apo_harmonic_restraint_force_set_force_constants",
        "Force-constant array size mismatch; expected 2, observed 1",
        "setForceConstants");
  }

  SECTION("ForceConstantArrayFinite") {
    const double forceConstants[NUM_ATOMS] = {1.0, infinity};
    const apo_status status = apo_harmonic_restraint_force_set_force_constants(
        restraint.get(), forceConstants, NUM_ATOMS);

    CheckHarmonicNativeError(
        status, "apo_harmonic_restraint_force_set_force_constants",
        "Force constant at index 1 must be finite; observed " +
            std::to_string(infinity),
        "setForceConstants");
  }

  SECTION("ForceConstantArrayAcceptsEitherSign") {
    const double forceConstants[NUM_ATOMS] = {1.0, -1.0};

    CHECK(apo_harmonic_restraint_force_set_force_constants(
              restraint.get(), forceConstants, NUM_ATOMS) == APO_STATUS_OK);
    CHECK(std::string(apo_last_error()).empty() == true);
  }

  SECTION("ReferenceCoordinateAtomCount") {
    const double referenceCoordinates[3] = {0.0, 0.0, 0.0};
    const apo_status status =
        apo_harmonic_restraint_force_set_reference_coordinates(
            restraint.get(), referenceCoordinates, 3);

    CheckHarmonicNativeError(
        status, "apo_harmonic_restraint_force_set_reference_coordinates",
        "Reference-coordinate array size mismatch; expected 2, observed 1",
        "setReferenceCoordinates");
  }

  SECTION("ReferenceCoordinateXFinite") {
    const double referenceCoordinates[3 * NUM_ATOMS] = {infinity, 0.0, 0.0,
                                                        0.0,      0.0, 0.0};
    const apo_status status =
        apo_harmonic_restraint_force_set_reference_coordinates(
            restraint.get(), referenceCoordinates, 3 * NUM_ATOMS);

    CheckHarmonicNativeError(
        status, "apo_harmonic_restraint_force_set_reference_coordinates",
        "Reference coordinate at atom index 0, X component must be finite; "
        "observed " +
            std::to_string(infinity),
        "setReferenceCoordinates");
  }

  SECTION("ReferenceCoordinateYFinite") {
    const double referenceCoordinates[3 * NUM_ATOMS] = {0.0, 0.0,      0.0,
                                                        0.0, infinity, 0.0};
    const apo_status status =
        apo_harmonic_restraint_force_set_reference_coordinates(
            restraint.get(), referenceCoordinates, 3 * NUM_ATOMS);

    CheckHarmonicNativeError(
        status, "apo_harmonic_restraint_force_set_reference_coordinates",
        "Reference coordinate at atom index 1, Y component must be finite; "
        "observed " +
            std::to_string(infinity),
        "setReferenceCoordinates");
  }

  SECTION("ReferenceCoordinateZFinite") {
    const double referenceCoordinates[3 * NUM_ATOMS] = {0.0, 0.0, 0.0,
                                                        0.0, 0.0, infinity};
    const apo_status status =
        apo_harmonic_restraint_force_set_reference_coordinates(
            restraint.get(), referenceCoordinates, 3 * NUM_ATOMS);

    CheckHarmonicNativeError(
        status, "apo_harmonic_restraint_force_set_reference_coordinates",
        "Reference coordinate at atom index 1, Z component must be finite; "
        "observed " +
            std::to_string(infinity),
        "setReferenceCoordinates");
  }

  SECTION("MassArraySize") {
    const double masses[1] = {1.0};
    const apo_status status =
        apo_harmonic_restraint_force_set_masses(restraint.get(), masses, 1);

    CheckHarmonicNativeError(status, "apo_harmonic_restraint_force_set_masses",
                             "Mass array size mismatch; expected 2, observed 1",
                             "setMasses");
  }

  SECTION("MassFinite") {
    const double masses[NUM_ATOMS] = {1.0, infinity};
    const apo_status status = apo_harmonic_restraint_force_set_masses(
        restraint.get(), masses, NUM_ATOMS);

    CheckHarmonicNativeError(status, "apo_harmonic_restraint_force_set_masses",
                             "Mass at index 1 must be finite; observed " +
                                 std::to_string(infinity),
                             "setMasses");
  }

  SECTION("MassAcceptsEitherSign") {
    const double masses[NUM_ATOMS] = {1.0, -1.0};

    CHECK(apo_harmonic_restraint_force_set_masses(restraint.get(), masses,
                                                  NUM_ATOMS) == APO_STATUS_OK);
    CHECK(std::string(apo_last_error()).empty() == true);
  }

  SECTION("ExponentAcceptsEveryInteger") {
    CHECK(apo_harmonic_restraint_force_set_exponent(restraint.get(), -1) ==
          APO_STATUS_OK);
    CHECK(apo_harmonic_restraint_force_set_exponent(restraint.get(), 7) ==
          APO_STATUS_OK);
    CHECK(std::string(apo_last_error()).empty() == true);
  }

  SECTION("ScaleFactorArraySize") {
    const double scaleFactors[2] = {1.0, 1.0};
    const apo_status status = apo_harmonic_restraint_force_set_scale_factors(
        restraint.get(), scaleFactors, 2);

    CheckHarmonicNativeError(
        status, "apo_harmonic_restraint_force_set_scale_factors",
        "Scale-factor array size mismatch; expected 3, observed 2",
        "setScaleFactors");
  }

  SECTION("ScaleFactorFinite") {
    const double scaleFactors[3] = {1.0, infinity, 1.0};
    const apo_status status = apo_harmonic_restraint_force_set_scale_factors(
        restraint.get(), scaleFactors, 3);

    CheckHarmonicNativeError(
        status, "apo_harmonic_restraint_force_set_scale_factors",
        "Scale factor at index 1 must be finite; observed " +
            std::to_string(infinity),
        "setScaleFactors");
  }

  SECTION("ScaleFactorAcceptsEitherSign") {
    const double scaleFactors[3] = {2.0, 0.0, -1.0};

    CHECK(apo_harmonic_restraint_force_set_scale_factors(
              restraint.get(), scaleFactors, 3) == APO_STATUS_OK);
    CHECK(std::string(apo_last_error()).empty() == true);
  }

  SECTION("BoxDimensionArraySize") {
    const double boxDimensions[2] = {50.0, 50.0};
    const apo_status status = apo_harmonic_restraint_force_set_box_dimensions(
        restraint.get(), boxDimensions, 2);

    CheckHarmonicNativeError(
        status, "apo_harmonic_restraint_force_set_box_dimensions",
        "Box-dimension array size mismatch; expected 3, observed 2",
        "setBoxDimensions");
  }

  SECTION("BoxDimensionFinite") {
    const double boxDimensions[3] = {50.0, infinity, 50.0};
    const apo_status status = apo_harmonic_restraint_force_set_box_dimensions(
        restraint.get(), boxDimensions, 3);

    CheckHarmonicNativeError(
        status, "apo_harmonic_restraint_force_set_box_dimensions",
        "Box dimension at index 1 must be finite; observed " +
            std::to_string(infinity),
        "setBoxDimensions");
  }

  SECTION("BoxDimensionPositive") {
    const double boxDimensions[3] = {50.0, 0.0, 50.0};
    const apo_status status = apo_harmonic_restraint_force_set_box_dimensions(
        restraint.get(), boxDimensions, 3);

    CheckHarmonicNativeError(
        status, "apo_harmonic_restraint_force_set_box_dimensions",
        "Box dimension at index 1 must be positive; observed " +
            std::to_string(0.0),
        "setBoxDimensions");
  }
}

TEST_CASE("CapiHarmonicRestraintForceValidatesForceManagerSubscription") {
  HarmonicRestraintForceHandle restraint = MakeRestraint();
  apo_force_manager forceManager = MakeForceManager();

  CheckForceManagerHandleValidation(
      "apo_force_manager_subscribe_harmonic_restraint_force",
      [&restraint](apo_force_manager *candidate) -> apo_status {
        return apo_force_manager_subscribe_harmonic_restraint_force(
            candidate, restraint.get(), "harm");
      });

  CheckForceManagerHandleValidation(
      "apo_force_manager_unsubscribe_harmonic_restraint_force",
      [&restraint](apo_force_manager *candidate) -> apo_status {
        return apo_force_manager_unsubscribe_harmonic_restraint_force(
            candidate, restraint.get());
      });

  apo_test::CheckStatusAndDiagnostic(
      apo_force_manager_subscribe_harmonic_restraint_force(
          &forceManager, restraint.get(), nullptr),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_force_manager_subscribe_harmonic_restraint_force: force_tag is "
      "NULL");

  apo_status status = apo_force_manager_subscribe_harmonic_restraint_force(
      &forceManager, restraint.get(), "");
  CheckForceManagerNativeError(
      status, "apo_force_manager_subscribe_harmonic_restraint_force",
      "Force tag must not be empty", "subscribe");

  CHECK(apo_force_manager_subscribe_harmonic_restraint_force(
            &forceManager, restraint.get(), "harm") == APO_STATUS_OK);
  CHECK(std::string(apo_last_error()).empty() == true);

  status = apo_force_manager_subscribe_harmonic_restraint_force(
      &forceManager, restraint.get(), "harm");
  CheckForceManagerNativeError(
      status, "apo_force_manager_subscribe_harmonic_restraint_force",
      "Force is already subscribed to this ForceManager", "subscribe");

  CHECK(apo_force_manager_unsubscribe_harmonic_restraint_force(
            &forceManager, restraint.get()) == APO_STATUS_OK);
  CHECK(std::string(apo_last_error()).empty() == true);

  status = apo_force_manager_unsubscribe_harmonic_restraint_force(
      &forceManager, restraint.get());
  CheckForceManagerNativeError(
      status, "apo_force_manager_unsubscribe_harmonic_restraint_force",
      "Force is not subscribed to this ForceManager", "unsubscribe");
}

TEST_CASE("CapiHarmonicRestraintForceSuccessClearsStaleDiagnostic") {
  HarmonicRestraintForceHandle restraint = MakeRestraint();

  REQUIRE(apo_harmonic_restraint_force_set_force_constant(nullptr, 1.0) ==
          APO_STATUS_INVALID_ARGUMENT);
  REQUIRE(std::string(apo_last_error()).empty() == false);

  CHECK(apo_harmonic_restraint_force_set_force_constant(restraint.get(), 1.0) ==
        APO_STATUS_OK);
  CHECK(std::string(apo_last_error()).empty() == true);
}
