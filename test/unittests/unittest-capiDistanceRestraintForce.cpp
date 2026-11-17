// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "AtomReference.h"
#include "CharmmPSF.h"
#include "apo_test_helpers.h"
#include "apocharmm_c/DistanceRestraintForce.h"
#include "apocharmm_c/Error.h"
#include "apocharmm_c/Status.h"
#include "apocharmm_c/detail/AtomReferenceHandle.h"
#include "apocharmm_c/detail/CharmmPsfHandle.h"
#include "apocharmm_c/detail/DistanceRestraintForceHandle.h"
#include "catch.hpp"

#include <memory>
#include <string>

namespace {

constexpr int NUM_ATOMS = 3;
constexpr char CREATE_FUNCTION_NAME[] = "apo_distance_restraint_force_create";
constexpr char ADD_FUNCTION_NAME[] =
    "apo_distance_restraint_force_add_restraint";

struct DistanceRestraintForceDeleter {
  void operator()(apo_distance_restraint_force *restraint) const noexcept {
    apo_distance_restraint_force_destroy(restraint);
    return;
  }
};

struct AtomReferenceDeleter {
  void operator()(apo_atom_reference *reference) const noexcept {
    apo_atom_reference_destroy(reference);
    return;
  }
};

using DistanceRestraintForceHandle =
    std::unique_ptr<apo_distance_restraint_force,
                    DistanceRestraintForceDeleter>;

using AtomReferenceHandle =
    std::unique_ptr<apo_atom_reference, AtomReferenceDeleter>;

std::shared_ptr<CharmmPSF> MakePsf(const int numAtoms) {
  auto psf = std::make_shared<CharmmPSF>();
  psf->setNumAtoms(numAtoms);
  return psf;
}

AtomReferenceHandle MakeReference(const std::shared_ptr<const CharmmPSF> &psf,
                                  const int atomIndex) {
  AtomReferenceHandle reference(new apo_atom_reference());
  reference->object = std::make_unique<AtomReference>(psf, atomIndex);
  return reference;
}

DistanceRestraintForceHandle MakeRestraint(void) {
  apo_charmm_psf psf;
  psf.object = MakePsf(NUM_ATOMS);
  apo_distance_restraint_force *restraint = nullptr;

  REQUIRE(apo_distance_restraint_force_create(&restraint, &psf) ==
          APO_STATUS_OK);
  REQUIRE(restraint != nullptr);

  return DistanceRestraintForceHandle(restraint);
}

apo_status AddOnePair(apo_distance_restraint_force *restraint,
                      const apo_atom_reference *first,
                      const apo_atom_reference *second) {
  const apo_atom_reference *firstReferences[1] = {first};
  const apo_atom_reference *secondReferences[1] = {second};
  const double coefficients[1] = {1.0};

  return apo_distance_restraint_force_add_restraint(
      restraint, firstReferences, 1, secondReferences, 1, coefficients, 1, 2.0,
      1.0, 1, 2, APO_DISTANCE_RESTRAINT_CONDITION_NONE);
}

} // namespace

TEST_CASE("CapiDistanceRestraintForceCreatesAndDestroys") {
  apo_charmm_psf psf;
  psf.object = MakePsf(NUM_ATOMS);

  apo_test::CheckStatusAndDiagnostic(
      apo_distance_restraint_force_create(nullptr, &psf),
      APO_STATUS_INVALID_ARGUMENT,
      std::string(CREATE_FUNCTION_NAME) + ": out pointer is NULL");

  apo_distance_restraint_force staleRestraint;
  apo_distance_restraint_force *restraint = &staleRestraint;

  apo_test::CheckStatusAndDiagnostic(
      apo_distance_restraint_force_create(&restraint, nullptr),
      APO_STATUS_INVALID_ARGUMENT,
      std::string(CREATE_FUNCTION_NAME) + ": CharmmPsf is NULL");
  CHECK(restraint == nullptr);

  apo_charmm_psf emptyPsf;
  restraint = &staleRestraint;
  apo_test::CheckStatusAndDiagnostic(
      apo_distance_restraint_force_create(&restraint, &emptyPsf),
      APO_STATUS_INVALID_ARGUMENT,
      std::string(CREATE_FUNCTION_NAME) + ": CharmmPsf object is NULL");
  CHECK(restraint == nullptr);

  apo_charmm_psf uninitializedPsf;
  uninitializedPsf.object = std::make_shared<CharmmPSF>();
  restraint = &staleRestraint;
  apo_test::CheckNativeError(
      apo_distance_restraint_force_create(&restraint, &uninitializedPsf),
      APO_STATUS_NOT_INITIALIZED, "NotInitialized", CREATE_FUNCTION_NAME,
      "CharmmPSF atom count is not initialized; observed -1",
      "src/DistanceRestraintForce.cu", "DistanceRestraintForce");
  CHECK(restraint == nullptr);

  apo_charmm_psf emptyAtomPsf;
  emptyAtomPsf.object = MakePsf(0);
  restraint = &staleRestraint;
  apo_test::CheckNativeError(
      apo_distance_restraint_force_create(&restraint, &emptyAtomPsf),
      APO_STATUS_INVALID_ARGUMENT, "InvalidArgument", CREATE_FUNCTION_NAME,
      "Atom count must be positive; observed 0",
      "src/DistanceRestraintForce.cu", "DistanceRestraintForce");
  CHECK(restraint == nullptr);

  const std::weak_ptr<CharmmPSF> weakPsf = psf.object;
  REQUIRE(apo_distance_restraint_force_create(&restraint, &psf) ==
          APO_STATUS_OK);
  REQUIRE(restraint != nullptr);
  CHECK(std::string(apo_last_error()).empty());
  CHECK(restraint->object->getForce()->size() == NUM_ATOMS);
  psf.object.reset();
  CHECK(weakPsf.expired());

  CHECK_NOTHROW(apo_distance_restraint_force_destroy(restraint));
  CHECK(std::string(apo_last_error()).empty());

  REQUIRE(apo_distance_restraint_force_set_scale(nullptr, 1.0) ==
          APO_STATUS_INVALID_ARGUMENT);
  const std::string diagnostic = apo_last_error();
  REQUIRE(diagnostic.empty() == false);

  CHECK_NOTHROW(apo_distance_restraint_force_destroy(nullptr));
  CHECK(std::string(apo_last_error()) == diagnostic);
}

TEST_CASE("CapiDistanceRestraintForceAcceptsAtomReferencesAndRepeatedPairs") {
  DistanceRestraintForceHandle restraint = MakeRestraint();

  const std::shared_ptr<CharmmPSF> firstPsf = MakePsf(NUM_ATOMS);
  const std::shared_ptr<CharmmPSF> secondPsf = MakePsf(NUM_ATOMS);

  AtomReferenceHandle first = MakeReference(firstPsf, 0);
  AtomReferenceHandle second = MakeReference(secondPsf, 1);

  const apo_atom_reference *firstReferences[2] = {first.get(), first.get()};
  const apo_atom_reference *secondReferences[2] = {second.get(), second.get()};
  const double coefficients[2] = {1.0, 0.5};

  CHECK(apo_distance_restraint_force_add_restraint(
            restraint.get(), firstReferences, 2, secondReferences, 2,
            coefficients, 2, 2.0, 1.0, 1, 2,
            APO_DISTANCE_RESTRAINT_CONDITION_NONE) == APO_STATUS_OK);

  CHECK(std::string(apo_last_error()).empty());
}

TEST_CASE("CapiDistanceRestraintForceValidatesReferenceArraysAndElements") {
  DistanceRestraintForceHandle restraint = MakeRestraint();

  const std::shared_ptr<CharmmPSF> psf = MakePsf(NUM_ATOMS);
  AtomReferenceHandle first = MakeReference(psf, 0);
  AtomReferenceHandle second = MakeReference(psf, 1);

  const apo_atom_reference *firstReferences[1] = {first.get()};
  const apo_atom_reference *secondReferences[1] = {second.get()};
  const double coefficients[1] = {1.0};

  SECTION("NullFirstArray") {
    apo_test::CheckStatusAndDiagnostic(
        apo_distance_restraint_force_add_restraint(
            restraint.get(), nullptr, 1, secondReferences, 1, coefficients, 1,
            2.0, 1.0, 1, 2, APO_DISTANCE_RESTRAINT_CONDITION_NONE),
        APO_STATUS_INVALID_ARGUMENT,
        std::string(ADD_FUNCTION_NAME) + ": first_atom_references is NULL");
  }

  SECTION("NullSecondArray") {
    apo_test::CheckStatusAndDiagnostic(
        apo_distance_restraint_force_add_restraint(
            restraint.get(), firstReferences, 1, nullptr, 1, coefficients, 1,
            2.0, 1.0, 1, 2, APO_DISTANCE_RESTRAINT_CONDITION_NONE),
        APO_STATUS_INVALID_ARGUMENT,
        std::string(ADD_FUNCTION_NAME) + ": second_atom_references is NULL");
  }

  SECTION("NullCoefficientArray") {
    apo_test::CheckStatusAndDiagnostic(
        apo_distance_restraint_force_add_restraint(
            restraint.get(), firstReferences, 1, secondReferences, 1, nullptr,
            1, 2.0, 1.0, 1, 2, APO_DISTANCE_RESTRAINT_CONDITION_NONE),
        APO_STATUS_INVALID_ARGUMENT,
        std::string(ADD_FUNCTION_NAME) + ": coefficients is NULL");
  }

  SECTION("EndpointArrayLengthMismatch") {
    apo_test::CheckStatusAndDiagnostic(
        apo_distance_restraint_force_add_restraint(
            restraint.get(), firstReferences, 1, secondReferences, 0,
            coefficients, 1, 2.0, 1.0, 1, 2,
            APO_DISTANCE_RESTRAINT_CONDITION_NONE),
        APO_STATUS_INVALID_ARGUMENT,
        std::string(ADD_FUNCTION_NAME) +
            ": first_atom_references and second_atom_references must have "
            "matching lengths");
  }

  SECTION("NullFirstElement") {
    const apo_atom_reference *invalidFirstReferences[1] = {nullptr};

    apo_test::CheckStatusAndDiagnostic(
        apo_distance_restraint_force_add_restraint(
            restraint.get(), invalidFirstReferences, 1, secondReferences, 1,
            coefficients, 1, 2.0, 1.0, 1, 2,
            APO_DISTANCE_RESTRAINT_CONDITION_NONE),
        APO_STATUS_INVALID_ARGUMENT,
        std::string(ADD_FUNCTION_NAME) + ": first_atom_references[0] is NULL");
  }

  SECTION("NullFirstNativeObject") {
    apo_atom_reference emptyReference;
    const apo_atom_reference *invalidFirstReferences[1] = {&emptyReference};

    apo_test::CheckStatusAndDiagnostic(
        apo_distance_restraint_force_add_restraint(
            restraint.get(), invalidFirstReferences, 1, secondReferences, 1,
            coefficients, 1, 2.0, 1.0, 1, 2,
            APO_DISTANCE_RESTRAINT_CONDITION_NONE),
        APO_STATUS_INVALID_ARGUMENT,
        std::string(ADD_FUNCTION_NAME) +
            ": first_atom_references[0] object is NULL");
  }

  SECTION("NullSecondElement") {
    const apo_atom_reference *invalidSecondReferences[1] = {nullptr};

    apo_test::CheckStatusAndDiagnostic(
        apo_distance_restraint_force_add_restraint(
            restraint.get(), firstReferences, 1, invalidSecondReferences, 1,
            coefficients, 1, 2.0, 1.0, 1, 2,
            APO_DISTANCE_RESTRAINT_CONDITION_NONE),
        APO_STATUS_INVALID_ARGUMENT,
        std::string(ADD_FUNCTION_NAME) + ": second_atom_references[0] is NULL");
  }
}

TEST_CASE("CapiDistanceRestraintForceDelegatesExtractedIndexValidation") {
  DistanceRestraintForceHandle restraint = MakeRestraint();

  SECTION("SameExtractedIndexFromDifferentTopologies") {
    const std::shared_ptr<CharmmPSF> firstPsf = MakePsf(NUM_ATOMS);
    const std::shared_ptr<CharmmPSF> secondPsf = MakePsf(NUM_ATOMS);

    AtomReferenceHandle first = MakeReference(firstPsf, 1);
    AtomReferenceHandle second = MakeReference(secondPsf, 1);

    const apo_status status =
        AddOnePair(restraint.get(), first.get(), second.get());

    apo_test::CheckNativeError(
        status, APO_STATUS_INVALID_ARGUMENT, "InvalidArgument",
        ADD_FUNCTION_NAME,
        "Atom indices at pair index 0 must be distinct; observed (1, 1)",
        "src/DistanceRestraintForce.cu", "addRestraint");
  }

  SECTION("SourceValidIndexOutsideForceAtomCount") {
    const std::shared_ptr<CharmmPSF> psf = MakePsf(NUM_ATOMS + 1);

    AtomReferenceHandle first = MakeReference(psf, NUM_ATOMS);
    AtomReferenceHandle second = MakeReference(psf, 1);

    const apo_status status =
        AddOnePair(restraint.get(), first.get(), second.get());

    apo_test::CheckNativeError(
        status, APO_STATUS_INVALID_ARGUMENT, "InvalidArgument",
        ADD_FUNCTION_NAME,
        "First atom index at pair index 0 is out of range; expected [0, 3), "
        "observed 3",
        "src/DistanceRestraintForce.cu", "addRestraint");
  }

  SECTION("PairCoefficientCountMismatch") {
    const std::shared_ptr<CharmmPSF> psf = MakePsf(NUM_ATOMS);

    AtomReferenceHandle first = MakeReference(psf, 0);
    AtomReferenceHandle second = MakeReference(psf, 1);

    const apo_atom_reference *firstReferences[1] = {first.get()};
    const apo_atom_reference *secondReferences[1] = {second.get()};
    const double coefficientStorage = 1.0;

    const apo_status status = apo_distance_restraint_force_add_restraint(
        restraint.get(), firstReferences, 1, secondReferences, 1,
        &coefficientStorage, 0, 2.0, 1.0, 1, 2,
        APO_DISTANCE_RESTRAINT_CONDITION_NONE);

    apo_test::CheckNativeError(
        status, APO_STATUS_INVALID_ARGUMENT, "InvalidArgument",
        ADD_FUNCTION_NAME,
        "Pair and coefficient counts must match; observed 1 pairs and 0 "
        "coefficients",
        "src/DistanceRestraintForce.cu", "addRestraint");
  }
}

TEST_CASE("CapiDistanceRestraintForceRetainsNeitherReferencesNorPsfs") {
  DistanceRestraintForceHandle restraint = MakeRestraint();

  std::weak_ptr<const CharmmPSF> weakFirstPsf;
  std::weak_ptr<const CharmmPSF> weakSecondPsf;

  {
    const std::shared_ptr<CharmmPSF> firstPsf = MakePsf(NUM_ATOMS);
    const std::shared_ptr<CharmmPSF> secondPsf = MakePsf(NUM_ATOMS);

    weakFirstPsf = firstPsf;
    weakSecondPsf = secondPsf;

    AtomReferenceHandle first = MakeReference(firstPsf, 0);
    AtomReferenceHandle second = MakeReference(secondPsf, 1);

    REQUIRE(AddOnePair(restraint.get(), first.get(), second.get()) ==
            APO_STATUS_OK);
  }

  CHECK(weakFirstPsf.expired());
  CHECK(weakSecondPsf.expired());
  CHECK(restraint != nullptr);
}
