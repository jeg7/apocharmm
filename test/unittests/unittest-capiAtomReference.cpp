// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "CharmmPSF.h"
#include "apo_test_helpers.h"
#include "apocharmm_c/AtomReference.h"
#include "apocharmm_c/AtomSelector.h"
#include "apocharmm_c/Error.h"
#include "apocharmm_c/detail/AtomReferenceHandle.h"
#include "apocharmm_c/detail/AtomSelectorHandle.h"
#include "apocharmm_c/detail/CharmmPsfHandle.h"
#include "catch.hpp"

#include <memory>
#include <string>

namespace {

struct AtomReferenceDeleter {
  void operator()(apo_atom_reference *reference) const noexcept {
    apo_atom_reference_destroy(reference);
    return;
  }
};

struct AtomSelectorDeleter {
  void operator()(apo_atom_selector *selector) const noexcept {
    apo_atom_selector_destroy(selector);
    return;
  }
};

struct CharmmPsfDeleter {
  void operator()(apo_charmm_psf *psf) const noexcept {
    apo_charmm_psf_destroy(psf);
    return;
  }
};

using AtomReferenceHandle =
    std::unique_ptr<apo_atom_reference, AtomReferenceDeleter>;
using AtomSelectorHandle =
    std::unique_ptr<apo_atom_selector, AtomSelectorDeleter>;
using CharmmPsfHandle = std::unique_ptr<apo_charmm_psf, CharmmPsfDeleter>;

CharmmPsfHandle MakePsf(const int numAtoms) {
  CharmmPsfHandle psf(new apo_charmm_psf());
  psf->object = std::make_shared<CharmmPSF>();
  psf->object->setNumAtoms(numAtoms);
  return psf;
}

AtomReferenceHandle MakeReference(const apo_charmm_psf *psf,
                                  const int atomIndex) {
  apo_atom_reference *reference = nullptr;
  REQUIRE(apo_atom_reference_create(&reference, psf, atomIndex) ==
          APO_STATUS_OK);
  REQUIRE(reference != nullptr);
  return AtomReferenceHandle(reference);
}

AtomSelectorHandle MakeSelector(const apo_charmm_psf *psf) {
  apo_atom_selector *selector = nullptr;
  REQUIRE(apo_atom_selector_create(&selector, psf) == APO_STATUS_OK);
  REQUIRE(selector != nullptr);
  return AtomSelectorHandle(selector);
}

int GetAtomIndex(const apo_atom_reference *reference) {
  int atomIndex = -1;
  REQUIRE(apo_atom_reference_get_atom_index(&atomIndex, reference) ==
          APO_STATUS_OK);
  return atomIndex;
}

} // namespace

TEST_CASE("CapiAtomReferenceConstruction") {
  SECTION("FirstAndLastValidIndices") {
    CharmmPsfHandle psf = MakePsf(4);

    AtomReferenceHandle first = MakeReference(psf.get(), 0);
    AtomReferenceHandle last = MakeReference(psf.get(), 3);

    CHECK(GetAtomIndex(first.get()) == 0);
    CHECK(GetAtomIndex(last.get()) == 3);
    CHECK(std::string(apo_last_error()).empty());
  }

  SECTION("NullOutputPointer") {
    CharmmPsfHandle psf = MakePsf(4);

    const apo_status status = apo_atom_reference_create(nullptr, psf.get(), 0);

    apo_test::CheckStatusAndDiagnostic(
        status, APO_STATUS_INVALID_ARGUMENT,
        "apo_atom_reference_create: out pointer is NULL");
  }

  SECTION("NullPsfHandleClearsOutput") {
    apo_atom_reference staleReference;
    apo_atom_reference *reference = &staleReference;

    const apo_status status = apo_atom_reference_create(&reference, nullptr, 0);

    apo_test::CheckStatusAndDiagnostic(
        status, APO_STATUS_INVALID_ARGUMENT,
        "apo_atom_reference_create: CharmmPsf is NULL");
    CHECK(reference == nullptr);
  }

  SECTION("NullPsfObjectClearsOutput") {
    CharmmPsfHandle psf(new apo_charmm_psf());
    apo_atom_reference staleReference;
    apo_atom_reference *reference = &staleReference;

    const apo_status status =
        apo_atom_reference_create(&reference, psf.get(), 0);

    apo_test::CheckStatusAndDiagnostic(
        status, APO_STATUS_INVALID_ARGUMENT,
        "apo_atom_reference_create: CharmmPsf object is NULL");
    CHECK(reference == nullptr);
  }

  SECTION("UninitializedPsfClearsOutput") {
    CharmmPsfHandle psf(new apo_charmm_psf());
    psf->object = std::make_shared<CharmmPSF>();
    apo_atom_reference staleReference;
    apo_atom_reference *reference = &staleReference;

    const apo_status status =
        apo_atom_reference_create(&reference, psf.get(), 0);

    apo_test::CheckNativeError(
        status, APO_STATUS_NOT_INITIALIZED, "NotInitialized",
        "apo_atom_reference_create",
        "CharmmPSF atom count is not initialized; observed -1",
        "src/AtomReference.cpp", "AtomReference");
    CHECK(reference == nullptr);
  }

  SECTION("NegativeIndexClearsOutput") {
    CharmmPsfHandle psf = MakePsf(4);
    apo_atom_reference staleReference;
    apo_atom_reference *reference = &staleReference;

    const apo_status status =
        apo_atom_reference_create(&reference, psf.get(), -1);

    apo_test::CheckNativeError(
        status, APO_STATUS_INVALID_ARGUMENT, "InvalidArgument",
        "apo_atom_reference_create",
        "Atom index is out of range; expected [0, 4), observed -1",
        "src/AtomReference.cpp", "AtomReference");
    CHECK(reference == nullptr);
  }

  SECTION("IndexEqualToAtomCountClearsOutput") {
    CharmmPsfHandle psf = MakePsf(4);
    apo_atom_reference staleReference;
    apo_atom_reference *reference = &staleReference;

    const apo_status status =
        apo_atom_reference_create(&reference, psf.get(), 4);

    apo_test::CheckNativeError(
        status, APO_STATUS_INVALID_ARGUMENT, "InvalidArgument",
        "apo_atom_reference_create",
        "Atom index is out of range; expected [0, 4), observed 4",
        "src/AtomReference.cpp", "AtomReference");
    CHECK(reference == nullptr);
  }

  SECTION("SuccessClearsStaleDiagnostic") {
    CharmmPsfHandle psf = MakePsf(4);
    apo_atom_reference *failedReference = nullptr;

    REQUIRE(apo_atom_reference_create(&failedReference, psf.get(), -1) ==
            APO_STATUS_INVALID_ARGUMENT);
    REQUIRE(std::string(apo_last_error()).empty() == false);

    apo_atom_reference *reference = nullptr;
    REQUIRE(apo_atom_reference_create(&reference, psf.get(), 2) ==
            APO_STATUS_OK);
    AtomReferenceHandle ownedReference(reference);

    CHECK(ownedReference != nullptr);
    CHECK(std::string(apo_last_error()).empty());
  }
}

TEST_CASE("CapiAtomReferenceQueries") {
  CharmmPsfHandle firstPsf = MakePsf(4);
  CharmmPsfHandle secondPsf = MakePsf(4);

  AtomReferenceHandle firstZero = MakeReference(firstPsf.get(), 0);
  AtomReferenceHandle firstZeroCopy = MakeReference(firstPsf.get(), 0);
  AtomReferenceHandle firstOne = MakeReference(firstPsf.get(), 1);
  AtomReferenceHandle secondZero = MakeReference(secondPsf.get(), 0);
  AtomReferenceHandle secondOne = MakeReference(secondPsf.get(), 1);

  SECTION("GetAtomIndex") {
    int atomIndex = -1;

    REQUIRE(apo_atom_reference_get_atom_index(&atomIndex, firstOne.get()) ==
            APO_STATUS_OK);

    CHECK(atomIndex == 1);
    CHECK(std::string(apo_last_error()).empty());
  }

  SECTION("GetAtomIndexRejectsNullOutput") {
    const apo_status status =
        apo_atom_reference_get_atom_index(nullptr, firstZero.get());

    apo_test::CheckStatusAndDiagnostic(
        status, APO_STATUS_INVALID_ARGUMENT,
        "apo_atom_reference_get_atom_index: atom_index is NULL");
  }

  SECTION("GetAtomIndexInitializesOutputForNullHandle") {
    int atomIndex = 17;

    const apo_status status =
        apo_atom_reference_get_atom_index(&atomIndex, nullptr);

    apo_test::CheckStatusAndDiagnostic(
        status, APO_STATUS_INVALID_ARGUMENT,
        "apo_atom_reference_get_atom_index: AtomReference is NULL");
    CHECK(atomIndex == -1);
  }

  SECTION("GetAtomIndexInitializesOutputForNullObject") {
    apo_atom_reference reference;
    int atomIndex = 17;

    const apo_status status =
        apo_atom_reference_get_atom_index(&atomIndex, &reference);

    apo_test::CheckStatusAndDiagnostic(
        status, APO_STATUS_INVALID_ARGUMENT,
        "apo_atom_reference_get_atom_index: AtomReference object is NULL");
    CHECK(atomIndex == -1);
  }

  SECTION("SameTopologyAndDifferentTopology") {
    bool result = false;

    REQUIRE(apo_atom_reference_has_same_topology(
                &result, firstZero.get(), firstOne.get()) == APO_STATUS_OK);
    CHECK(result == true);

    REQUIRE(apo_atom_reference_has_same_topology(
                &result, firstZero.get(), secondZero.get()) == APO_STATUS_OK);
    CHECK(result == false);
  }

  SECTION("EqualityCombinations") {
    bool result = false;

    REQUIRE(apo_atom_reference_equals(&result, firstZero.get(),
                                      firstZeroCopy.get()) == APO_STATUS_OK);
    CHECK(result == true);

    REQUIRE(apo_atom_reference_equals(&result, firstZero.get(),
                                      firstOne.get()) == APO_STATUS_OK);
    CHECK(result == false);

    REQUIRE(apo_atom_reference_equals(&result, firstZero.get(),
                                      secondZero.get()) == APO_STATUS_OK);
    CHECK(result == false);

    REQUIRE(apo_atom_reference_equals(&result, firstZero.get(),
                                      secondOne.get()) == APO_STATUS_OK);
    CHECK(result == false);
  }

  SECTION("SameTopologyInitializesOutputOnFailures") {
    bool result = true;

    apo_status status =
        apo_atom_reference_has_same_topology(&result, nullptr, firstZero.get());
    apo_test::CheckStatusAndDiagnostic(
        status, APO_STATUS_INVALID_ARGUMENT,
        "apo_atom_reference_has_same_topology: AtomReference is NULL");
    CHECK(result == false);

    apo_atom_reference emptyFirst;
    result = true;
    status = apo_atom_reference_has_same_topology(&result, &emptyFirst,
                                                  firstZero.get());
    apo_test::CheckStatusAndDiagnostic(
        status, APO_STATUS_INVALID_ARGUMENT,
        "apo_atom_reference_has_same_topology: AtomReference object is NULL");
    CHECK(result == false);

    result = true;
    status =
        apo_atom_reference_has_same_topology(&result, firstZero.get(), nullptr);
    apo_test::CheckStatusAndDiagnostic(
        status, APO_STATUS_INVALID_ARGUMENT,
        "apo_atom_reference_has_same_topology: Other AtomReference is NULL");
    CHECK(result == false);

    apo_atom_reference emptyOther;
    result = true;
    status = apo_atom_reference_has_same_topology(&result, firstZero.get(),
                                                  &emptyOther);
    apo_test::CheckStatusAndDiagnostic(
        status, APO_STATUS_INVALID_ARGUMENT,
        "apo_atom_reference_has_same_topology: Other AtomReference object is "
        "NULL");
    CHECK(result == false);
  }

  SECTION("SameTopologyRejectsNullOutput") {
    const apo_status status = apo_atom_reference_has_same_topology(
        nullptr, firstZero.get(), firstOne.get());

    apo_test::CheckStatusAndDiagnostic(
        status, APO_STATUS_INVALID_ARGUMENT,
        "apo_atom_reference_has_same_topology: has_same_topology is NULL");
  }

  SECTION("EqualsInitializesOutputOnFailures") {
    bool result = true;

    apo_status status =
        apo_atom_reference_equals(&result, nullptr, firstZero.get());
    apo_test::CheckStatusAndDiagnostic(
        status, APO_STATUS_INVALID_ARGUMENT,
        "apo_atom_reference_equals: AtomReference is NULL");
    CHECK(result == false);

    apo_atom_reference emptyFirst;
    result = true;
    status = apo_atom_reference_equals(&result, &emptyFirst, firstZero.get());
    apo_test::CheckStatusAndDiagnostic(
        status, APO_STATUS_INVALID_ARGUMENT,
        "apo_atom_reference_equals: AtomReference object is NULL");
    CHECK(result == false);

    result = true;
    status = apo_atom_reference_equals(&result, firstZero.get(), nullptr);
    apo_test::CheckStatusAndDiagnostic(
        status, APO_STATUS_INVALID_ARGUMENT,
        "apo_atom_reference_equals: Other AtomReference is NULL");
    CHECK(result == false);

    apo_atom_reference emptyOther;
    result = true;
    status = apo_atom_reference_equals(&result, firstZero.get(), &emptyOther);
    apo_test::CheckStatusAndDiagnostic(
        status, APO_STATUS_INVALID_ARGUMENT,
        "apo_atom_reference_equals: Other AtomReference object is NULL");
    CHECK(result == false);
  }

  SECTION("EqualsRejectsNullOutput") {
    const apo_status status = apo_atom_reference_equals(
        nullptr, firstZero.get(), firstZeroCopy.get());

    apo_test::CheckStatusAndDiagnostic(
        status, APO_STATUS_INVALID_ARGUMENT,
        "apo_atom_reference_equals: equals is NULL");
  }

  SECTION("SuccessfulQueryClearsStaleDiagnostic") {
    int atomIndex = 17;
    REQUIRE(apo_atom_reference_get_atom_index(&atomIndex, nullptr) ==
            APO_STATUS_INVALID_ARGUMENT);
    REQUIRE(std::string(apo_last_error()).empty() == false);

    REQUIRE(apo_atom_reference_get_atom_index(&atomIndex, firstOne.get()) ==
            APO_STATUS_OK);

    CHECK(atomIndex == 1);
    CHECK(std::string(apo_last_error()).empty());
  }
}

TEST_CASE("CapiAtomReferenceLifetimeAndDestruction") {
  SECTION("ReferenceRetainsNativePsfAfterPublicHandleDestruction") {
    CharmmPsfHandle psf = MakePsf(4);
    std::weak_ptr<CharmmPSF> weakPsf = psf->object;
    AtomReferenceHandle reference = MakeReference(psf.get(), 2);

    psf.reset();

    CHECK(weakPsf.expired() == false);
    CHECK(GetAtomIndex(reference.get()) == 2);

    reference.reset();
    CHECK(weakPsf.expired() == true);
  }

  SECTION("NullSafeNoThrowDestructionPreservesDiagnostic") {
    int atomIndex = 17;
    REQUIRE(apo_atom_reference_get_atom_index(&atomIndex, nullptr) ==
            APO_STATUS_INVALID_ARGUMENT);
    const std::string diagnostic = apo_last_error();

    CHECK_NOTHROW(apo_atom_reference_destroy(nullptr));
    CHECK(std::string(apo_last_error()) == diagnostic);
  }

  SECTION("NullObjectDestructionDoesNotThrow") {
    auto *reference = new apo_atom_reference();

    CHECK_NOTHROW(apo_atom_reference_destroy(reference));
  }

  SECTION("OwnedDestructionDoesNotThrow") {
    CharmmPsfHandle psf = MakePsf(4);
    apo_atom_reference *reference = nullptr;
    REQUIRE(apo_atom_reference_create(&reference, psf.get(), 1) ==
            APO_STATUS_OK);
    REQUIRE(reference != nullptr);

    CHECK_NOTHROW(apo_atom_reference_destroy(reference));
  }
}

TEST_CASE("CapiAtomSelectorSelectAtom") {
  SECTION("Success") {
    CharmmPsfHandle psf = MakePsf(4);
    AtomSelectorHandle selector = MakeSelector(psf.get());
    apo_atom_reference *reference = nullptr;

    REQUIRE(apo_atom_selector_select_atom(&reference, selector.get(),
                                          "bynu 2") == APO_STATUS_OK);
    AtomReferenceHandle ownedReference(reference);

    CHECK(ownedReference != nullptr);
    CHECK(GetAtomIndex(ownedReference.get()) == 1);
    CHECK(std::string(apo_last_error()).empty());
  }

  SECTION("ValidationClearsOutput") {
    CharmmPsfHandle psf = MakePsf(4);
    AtomSelectorHandle selector = MakeSelector(psf.get());

    const apo_status nullOutStatus =
        apo_atom_selector_select_atom(nullptr, selector.get(), "bynu 2");
    apo_test::CheckStatusAndDiagnostic(
        nullOutStatus, APO_STATUS_INVALID_ARGUMENT,
        "apo_atom_selector_select_atom: out pointer is NULL");

    apo_atom_reference staleReference;
    apo_atom_reference *reference = &staleReference;
    apo_status status =
        apo_atom_selector_select_atom(&reference, nullptr, "bynu 2");
    apo_test::CheckStatusAndDiagnostic(
        status, APO_STATUS_INVALID_ARGUMENT,
        "apo_atom_selector_select_atom: AtomSelector is NULL");
    CHECK(reference == nullptr);

    apo_atom_selector emptySelector;
    reference = &staleReference;
    status =
        apo_atom_selector_select_atom(&reference, &emptySelector, "bynu 2");
    apo_test::CheckStatusAndDiagnostic(
        status, APO_STATUS_INVALID_ARGUMENT,
        "apo_atom_selector_select_atom: AtomSelector object is NULL");
    CHECK(reference == nullptr);

    reference = &staleReference;
    status = apo_atom_selector_select_atom(&reference, selector.get(), nullptr);
    apo_test::CheckStatusAndDiagnostic(
        status, APO_STATUS_INVALID_ARGUMENT,
        "apo_atom_selector_select_atom: selection_string is NULL or empty");
    CHECK(reference == nullptr);

    reference = &staleReference;
    status = apo_atom_selector_select_atom(&reference, selector.get(), "");
    apo_test::CheckStatusAndDiagnostic(
        status, APO_STATUS_INVALID_ARGUMENT,
        "apo_atom_selector_select_atom: selection_string is NULL or empty");
    CHECK(reference == nullptr);
  }

  SECTION("ZeroMatchesClearOutput") {
    CharmmPsfHandle psf = MakePsf(4);
    AtomSelectorHandle selector = MakeSelector(psf.get());
    apo_atom_reference staleReference;
    apo_atom_reference *reference = &staleReference;

    const apo_status status =
        apo_atom_selector_select_atom(&reference, selector.get(), "none");

    apo_test::CheckNativeError(
        status, APO_STATUS_INVALID_ARGUMENT, "InvalidArgument",
        "apo_atom_selector_select_atom",
        "Atom selection must match exactly one atom; observed 0",
        "src/AtomSelector.cpp", "selectAtom");
    CHECK(reference == nullptr);
  }

  SECTION("MultipleMatchesClearOutput") {
    CharmmPsfHandle psf = MakePsf(4);
    AtomSelectorHandle selector = MakeSelector(psf.get());
    apo_atom_reference staleReference;
    apo_atom_reference *reference = &staleReference;

    const apo_status status =
        apo_atom_selector_select_atom(&reference, selector.get(), "all");

    apo_test::CheckNativeError(
        status, APO_STATUS_INVALID_ARGUMENT, "InvalidArgument",
        "apo_atom_selector_select_atom",
        "Atom selection must match exactly one atom; observed 4",
        "src/AtomSelector.cpp", "selectAtom");
    CHECK(reference == nullptr);
  }

  SECTION("ParserErrorPropagatesAndClearsOutput") {
    CharmmPsfHandle psf = MakePsf(4);
    AtomSelectorHandle selector = MakeSelector(psf.get());
    apo_atom_reference staleReference;
    apo_atom_reference *reference = &staleReference;

    const apo_status status = apo_atom_selector_select_atom(
        &reference, selector.get(), ".around. bynu 2");

    apo_test::CheckNativeError(
        status, APO_STATUS_INVALID_ARGUMENT, "InvalidArgument",
        "apo_atom_selector_select_atom",
        "Unknown dotted atom selection operator \".around.\"",
        "src/SelectionTokenizer.cpp", "getDottedTokenType");
    CHECK(reference == nullptr);
  }

  SECTION("SuccessClearsStaleDiagnostic") {
    CharmmPsfHandle psf = MakePsf(4);
    AtomSelectorHandle selector = MakeSelector(psf.get());
    apo_atom_reference *reference = nullptr;

    REQUIRE(apo_atom_selector_select_atom(&reference, selector.get(), "none") ==
            APO_STATUS_INVALID_ARGUMENT);
    REQUIRE(reference == nullptr);
    REQUIRE(std::string(apo_last_error()).empty() == false);

    REQUIRE(apo_atom_selector_select_atom(&reference, selector.get(),
                                          "bynu 3") == APO_STATUS_OK);
    AtomReferenceHandle ownedReference(reference);

    CHECK(GetAtomIndex(ownedReference.get()) == 2);
    CHECK(std::string(apo_last_error()).empty());
  }

  SECTION("ReferenceOutlivesSelectorAndPublicPsfHandle") {
    CharmmPsfHandle psf = MakePsf(4);
    std::weak_ptr<CharmmPSF> weakPsf = psf->object;
    AtomSelectorHandle selector = MakeSelector(psf.get());
    apo_atom_reference *reference = nullptr;

    REQUIRE(apo_atom_selector_select_atom(&reference, selector.get(),
                                          "bynu 4") == APO_STATUS_OK);
    AtomReferenceHandle ownedReference(reference);

    selector.reset();
    psf.reset();

    CHECK(weakPsf.expired() == false);
    CHECK(GetAtomIndex(ownedReference.get()) == 3);

    ownedReference.reset();
    CHECK(weakPsf.expired() == true);
  }
}
