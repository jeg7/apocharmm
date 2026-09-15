// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "ApoCharmmError.h"
#include "AtomReference.h"
#include "CharmmPSF.h"
#include "apo_test_helpers.h"
#include "catch.hpp"

#include <memory>
#include <type_traits>
#include <utility>

namespace {

std::shared_ptr<CharmmPSF> MakePsfWithAtomCount(const int numAtoms) {
  auto psf = std::make_shared<CharmmPSF>();
  psf->setNumAtoms(numAtoms);
  return psf;
}

static_assert(std::is_default_constructible<AtomReference>::value == false,
              "AtomReference must not be default constructible");
static_assert(std::is_nothrow_copy_constructible<AtomReference>::value,
              "AtomReference copy construction must be noexcept");
static_assert(std::is_nothrow_copy_assignable<AtomReference>::value,
              "AtomReference copy assignment must be noexcept");
static_assert(std::is_nothrow_move_constructible<AtomReference>::value,
              "AtomReference move construction must be noexcept");
static_assert(std::is_nothrow_move_assignable<AtomReference>::value,
              "AtomReference move assignment must be noexcept");
static_assert(std::is_polymorphic<AtomReference>::value == false,
              "AtomReference must not be polymorphic");

} // namespace

TEST_CASE("AtomReferenceConstructionAndValueSemantics") {
  SECTION("ConstructsAtBoundaryIndicesAndReturnsQueries") {
    auto psf = MakePsfWithAtomCount(3);

    const AtomReference first(psf, 0);
    const AtomReference last(psf, 2);

    CHECK(first.getAtomIndex() == 0);
    CHECK(last.getAtomIndex() == 2);
    CHECK(first.getPsf().get() == psf.get());
    CHECK(last.getPsf().get() == psf.get());
  }

  SECTION("CopyConstruction") {
    auto psf = MakePsfWithAtomCount(3);
    const AtomReference source(psf, 1);

    const AtomReference copy(source);

    CHECK(copy.getAtomIndex() == 1);
    CHECK(copy.getPsf().get() == psf.get());
    CHECK(copy == source);
  }

  SECTION("CopyAssignment") {
    auto sourcePsf = MakePsfWithAtomCount(3);
    auto destinationPsf = MakePsfWithAtomCount(3);
    const AtomReference source(sourcePsf, 2);
    AtomReference destination(destinationPsf, 0);

    destination = source;

    CHECK(destination.getAtomIndex() == 2);
    CHECK(destination.getPsf().get() == sourcePsf.get());
    CHECK(destination == source);
  }

  SECTION("MoveConstruction") {
    auto psf = MakePsfWithAtomCount(3);
    AtomReference source(psf, 1);

    const AtomReference moved(std::move(source));

    CHECK(moved.getAtomIndex() == 1);
    CHECK(moved.getPsf().get() == psf.get());
  }

  SECTION("MoveAssignment") {
    auto sourcePsf = MakePsfWithAtomCount(3);
    auto destinationPsf = MakePsfWithAtomCount(3);
    AtomReference source(sourcePsf, 2);
    AtomReference destination(destinationPsf, 0);

    destination = std::move(source);

    CHECK(destination.getAtomIndex() == 2);
    CHECK(destination.getPsf().get() == sourcePsf.get());
  }
}

TEST_CASE("AtomReferenceRejectsInvalidInputs") {
  SECTION("NullPsfPrecedesIndexValidation") {
    const std::shared_ptr<const CharmmPSF> psf = nullptr;

    apo_test::CheckApoCharmmError(
        [psf](void) -> void {
          static_cast<void>(AtomReference(psf, -1));
          return;
        },
        ApoCharmmErrorCode::InvalidArgument,
        "AtomReference requires a non-null PSF");
  }

  SECTION("UninitializedPsfPrecedesIndexValidation") {
    auto psf = std::make_shared<CharmmPSF>();

    apo_test::CheckApoCharmmError(
        [psf](void) -> void {
          static_cast<void>(AtomReference(psf, -1));
          return;
        },
        ApoCharmmErrorCode::NotInitialized,
        "CharmmPSF atom count is not initialized; observed -1");
  }

  SECTION("NegativeIndex") {
    auto psf = MakePsfWithAtomCount(3);

    apo_test::CheckApoCharmmError(
        [psf](void) -> void {
          static_cast<void>(AtomReference(psf, -1));
          return;
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Atom index is out of range; expected [0, 3), observed -1");
  }

  SECTION("IndexEqualToAtomCount") {
    auto psf = MakePsfWithAtomCount(3);

    apo_test::CheckApoCharmmError(
        [psf](void) -> void {
          static_cast<void>(AtomReference(psf, 3));
          return;
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Atom index is out of range; expected [0, 3), observed 3");
  }

  SECTION("IndexGreaterThanAtomCount") {
    auto psf = MakePsfWithAtomCount(3);

    apo_test::CheckApoCharmmError(
        [psf](void) -> void {
          static_cast<void>(AtomReference(psf, 4));
          return;
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Atom index is out of range; expected [0, 3), observed 4");
  }

  SECTION("ZeroAtomPsf") {
    auto psf = MakePsfWithAtomCount(0);

    apo_test::CheckApoCharmmError(
        [psf](void) -> void {
          static_cast<void>(AtomReference(psf, 0));
          return;
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Atom index is out of range; expected [0, 0), observed 0");
  }
}

TEST_CASE("AtomReferenceTopologyIdentityAndEquality") {
  auto firstPsf = MakePsfWithAtomCount(3);
  auto secondPsf = MakePsfWithAtomCount(3);

  const AtomReference first(firstPsf, 1);
  const AtomReference sameTopologySameIndex(firstPsf, 1);
  const AtomReference sameTopologyDifferentIndex(firstPsf, 2);
  const AtomReference differentTopologySameIndex(secondPsf, 1);
  const AtomReference differentTopologyDifferentIndex(secondPsf, 2);

  CHECK(first.hasSameTopology(sameTopologySameIndex) == true);
  CHECK(first.hasSameTopology(sameTopologyDifferentIndex) == true);
  CHECK(first.hasSameTopology(differentTopologySameIndex) == false);
  CHECK(first.hasSameTopology(differentTopologyDifferentIndex) == false);

  CHECK((first == sameTopologySameIndex) == true);
  CHECK((first != sameTopologySameIndex) == false);

  CHECK((first == sameTopologyDifferentIndex) == false);
  CHECK((first != sameTopologyDifferentIndex) == true);

  CHECK((first == differentTopologySameIndex) == false);
  CHECK((first != differentTopologySameIndex) == true);

  CHECK((first == differentTopologyDifferentIndex) == false);
  CHECK((first != differentTopologyDifferentIndex) == true);
}

TEST_CASE("AtomReferencePsfOwnership") {
  SECTION("ReferenceRetainsPsfAfterOriginalOwnerIsDestroyed") {
    std::weak_ptr<const CharmmPSF> weakPsf;

    const AtomReference reference = [&weakPsf](void) -> AtomReference {
      auto psf = MakePsfWithAtomCount(2);
      weakPsf = psf;
      return AtomReference(psf, 1);
    }();

    CHECK(weakPsf.expired() == false);
    CHECK(reference.getAtomIndex() == 1);
    CHECK(reference.getPsf().get() == weakPsf.lock().get());
  }

  SECTION("GetPsfReturnsSharedOwnership") {
    std::weak_ptr<const CharmmPSF> weakPsf;
    std::shared_ptr<const CharmmPSF> returnedPsf;

    {
      auto psf = MakePsfWithAtomCount(2);
      weakPsf = psf;
      const AtomReference reference(psf, 0);

      returnedPsf = reference.getPsf();
      CHECK(returnedPsf.get() == psf.get());
    }

    CHECK(weakPsf.expired() == false);
    returnedPsf.reset();
    CHECK(weakPsf.expired() == true);
  }

  SECTION("PsfIsReleasedAfterFinalReferenceIsDestroyed") {
    std::weak_ptr<const CharmmPSF> weakPsf;

    {
      auto psf = MakePsfWithAtomCount(2);
      weakPsf = psf;
      const AtomReference reference(psf, 0);

      psf.reset();
      CHECK(weakPsf.expired() == false);
    }

    CHECK(weakPsf.expired() == true);
  }
}
