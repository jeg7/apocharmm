// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II, Félix Aviat, Samarjeet Prasad
//
// ENDLICENSE

#include "ApoCharmmError.h"
#include "CudaContainer.h"
#include "apo_test_helpers.h"
#include "catch.hpp"

#include <type_traits>
#include <utility>
#include <vector>

static_assert(std::is_nothrow_move_constructible_v<CudaContainer<int>>);
static_assert(std::is_nothrow_move_assignable_v<CudaContainer<int>>);

TEST_CASE("CudaContainerConstruction") {
  SECTION("DefaultConstructor") {
    CudaContainer<int> c;

    CHECK(c.size() == 0);
    CHECK(c.getHostArray().empty() == true);
    CHECK(c.getDeviceArray().empty() == true);
    CHECK(c.getDeviceArray().size() == 0);
    CHECK(c.getDeviceArray().capacity() == 0);
    CHECK(c.getDeviceArray().data() == nullptr);
  }

  SECTION("SizeConstructor") {
    constexpr std::size_t n = 5;
    CudaContainer<int> c(n);

    CHECK(c.size() == n);
    CHECK(c.getHostArray().size() == n);
    CHECK(c.getDeviceArray().size() == n);
    CHECK(c.getDeviceArray().capacity() == n);
    CHECK(c.getDeviceArray().data() != nullptr);

    c.set(7);

    const std::vector<int> expected(n, 7);
    apo_test::CheckHostAndDeviceEqual(c, expected);
  }

  SECTION("HostVectorConstructor") {
    const std::vector<int> expected = {1, 2, 3};
    CudaContainer<int> c(expected);

    apo_test::CheckHostAndDeviceEqual(c, expected);
  }

  SECTION("HostRvalueConstructorTransfersHostStorage") {
    const std::vector<int> expected = {1, 2, 3};
    std::vector<int> source = expected;
    int *const sourceData = source.data();

    CudaContainer<int> c(std::move(source));

    CHECK(c.getHostArray().data() == sourceData);
    CHECK(source.empty() == true);
    apo_test::CheckHostAndDeviceEqual(c, expected);
  }

  SECTION("DeviceVectorConstructor") {
    const std::vector<int> expected = {1, 2, 3};
    DeviceVector<int> deviceVector(expected);
    CudaContainer<int> c(deviceVector);

    apo_test::CheckHostAndDeviceEqual(c, expected);
  }

  SECTION("DeviceVectorRvalueConstructorTransfersDeviceStorage") {
    const std::vector<int> expected = {1, 2, 3};
    DeviceVector<int> source(expected);
    int *const sourceData = source.data();

    CudaContainer<int> c(std::move(source));

    CHECK(c.getDeviceArray().data() == sourceData);
    CHECK(source.data() == nullptr);
    CHECK(source.size() == 0);
    CHECK(source.capacity() == 0);
    apo_test::CheckHostAndDeviceEqual(c, expected);
  }

  SECTION("CopyConstructorDeepCopy") {
    const std::vector<int> expected = {1, 2, 3};
    const std::vector<int> changed = {4, 5, 6};

    CudaContainer<int> c1(expected);
    CudaContainer<int> c2(c1);

    CHECK(c1.size() == c2.size());
    CHECK(c1.getHostArray().data() != c2.getHostArray().data());
    CHECK(c1.getDeviceArray().data() != c2.getDeviceArray().data());

    c1.set(changed);

    apo_test::CheckHostAndDeviceEqual(c1, changed);
    apo_test::CheckHostAndDeviceEqual(c2, expected);
  }

  SECTION("MoveConstructorTransfersBothMirrors") {
    const std::vector<int> deviceValues = {1, 2, 3};
    const std::vector<int> hostValues = {9, 2, 3};

    CudaContainer<int> source(deviceValues);
    source.getHostArray()[0] = hostValues[0];

    int *const sourceHostData = source.getHostArray().data();
    int *const sourceDeviceData = source.getDeviceArray().data();

    CudaContainer<int> moved(std::move(source));

    CHECK(moved.getHostArray().data() == sourceHostData);
    CHECK(moved.getDeviceArray().data() == sourceDeviceData);
    CHECK(moved.getHostArray() == hostValues);
    CHECK(apo_test::CopyToHost(moved.getDeviceArray()) == deviceValues);

    CHECK(source.getHostArray().empty() == true);
    CHECK(source.getDeviceArray().size() == 0);
    CHECK(source.getDeviceArray().capacity() == 0);
    CHECK(source.getDeviceArray().data() == nullptr);
  }

  SECTION("DoubleContainer") {
    const std::vector<double> expected = {1.25, 2.5, 3.75};
    CudaContainer<double> c(expected);

    apo_test::CheckHostAndDeviceEqual(c, expected);
  }
}

TEST_CASE("CudaContainerElementAccess") {
  SECTION("AtReadsAndWritesHostArray") {
    CudaContainer<int> c(std::vector<int>{1, 2, 3});

    CHECK(c.at(0) == 1);
    CHECK(c.at(1) == 2);
    CHECK(c.at(2) == 3);

    c.at(1) = 7;

    const std::vector<int> expected = {1, 7, 3};
    CHECK(c.getHostArray() == expected);

    c.transferToDevice();

    CHECK(apo_test::CopyToHost(c.getDeviceArray()) == expected);
  }

  SECTION("AtThrowsOutOfRange") {
    CudaContainer<int> c(std::vector<int>{1, 2, 3});
    const CudaContainer<int> &constContainer = c;

    apo_test::CheckApoCharmmError(
        [&c](void) -> void { (void)c.at(3); },
        ApoCharmmErrorCode::InvalidArgument,
        "CudaContainer index is out of range; expected [0, 3), observed 3");
    apo_test::CheckApoCharmmError(
        [&constContainer](void) -> void { (void)constContainer.at(3); },
        ApoCharmmErrorCode::InvalidArgument,
        "CudaContainer index is out of range; expected [0, 3), observed 3");
  }

  SECTION("OperatorIndexReadsAndWritesHostArray") {
    CudaContainer<int> c(std::vector<int>{1, 2, 3});

    CHECK(c[0] == 1);
    CHECK(c[1] == 2);
    CHECK(c[2] == 3);

    c[2] = 9;

    const std::vector<int> expected = {1, 2, 9};
    CHECK(c.getHostArray() == expected);

    c.transferToDevice();

    CHECK(apo_test::CopyToHost(c.getDeviceArray()) == expected);
  }

  SECTION("ConstElementAccess") {
    CudaContainer<int> c(std::vector<int>{1, 2, 3});
    const CudaContainer<int> &constContainer = c;

    CHECK(constContainer.at(0) == 1);
    CHECK(constContainer[2] == 3);
  }
}

TEST_CASE("CudaContainerAssignment") {
  SECTION("AssignFromHostVector") {
    const std::vector<int> expected = {1, 2, 3};
    CudaContainer<int> c;

    c = expected;

    apo_test::CheckHostAndDeviceEqual(c, expected);
  }

  SECTION("AssignFromHostRvalueVectorTransfersHostStorage") {
    const std::vector<int> expected = {1, 2, 3};
    std::vector<int> source = expected;
    int *const sourceData = source.data();

    CudaContainer<int> c(std::vector<int>{9});
    c = std::move(source);

    CHECK(c.getHostArray().data() == sourceData);
    CHECK(source.empty() == true);
    apo_test::CheckHostAndDeviceEqual(c, expected);
  }

  SECTION("AssignFromHostVectorShrinksExistingContainer") {
    const std::vector<int> expected = {1, 2, 3};
    CudaContainer<int> c(8);

    c = expected;

    apo_test::CheckHostAndDeviceEqual(c, expected);
  }

  SECTION("AssignFromDeviceVector") {
    const std::vector<int> expected = {1, 2, 3};
    DeviceVector<int> deviceVector(expected);
    CudaContainer<int> c;

    c = deviceVector;

    apo_test::CheckHostAndDeviceEqual(c, expected);
  }

  SECTION("AssignFromDeviceRvalueVectorTransfersDeviceStorage") {
    const std::vector<int> expected = {1, 2, 3};
    DeviceVector<int> source(expected);
    int *const sourceData = source.data();

    CudaContainer<int> c(std::vector<int>{9});
    c = std::move(source);

    CHECK(c.getDeviceArray().data() == sourceData);
    CHECK(source.data() == nullptr);
    CHECK(source.size() == 0);
    CHECK(source.capacity() == 0);
    apo_test::CheckHostAndDeviceEqual(c, expected);
  }

  SECTION("CopyAssignmentDeepCopy") {
    const std::vector<int> expected = {1, 2, 3};
    const std::vector<int> changed = {4, 5, 6};

    CudaContainer<int> c1(expected);
    CudaContainer<int> c2(8);

    c2 = c1;

    CHECK(c1.getHostArray().data() != c2.getHostArray().data());
    CHECK(c1.getDeviceArray().data() != c2.getDeviceArray().data());

    c1.set(changed);

    apo_test::CheckHostAndDeviceEqual(c1, changed);
    apo_test::CheckHostAndDeviceEqual(c2, expected);
  }

  SECTION("MoveAssignmentTransfersBothMirrors") {
    const std::vector<int> deviceValues = {1, 2, 3};
    const std::vector<int> hostValues = {9, 2, 3};

    CudaContainer<int> source(deviceValues);
    source.getHostArray()[0] = hostValues[0];

    int *const sourceHostData = source.getHostArray().data();
    int *const sourceDeviceData = source.getDeviceArray().data();

    CudaContainer<int> assigned(std::vector<int>{8});
    assigned = std::move(source);

    CHECK(assigned.getHostArray().data() == sourceHostData);
    CHECK(assigned.getDeviceArray().data() == sourceDeviceData);
    CHECK(assigned.getHostArray() == hostValues);
    CHECK(apo_test::CopyToHost(assigned.getDeviceArray()) == deviceValues);

    CHECK(source.getHostArray().empty() == true);
    CHECK(source.getDeviceArray().size() == 0);
    CHECK(source.getDeviceArray().capacity() == 0);
    CHECK(source.getDeviceArray().data() == nullptr);
  }

  SECTION("SelfAssignment") {
    const std::vector<int> expected = {1, 2, 3};
    CudaContainer<int> c(expected);

    CHECK_NOTHROW(c = c);
    apo_test::CheckHostAndDeviceEqual(c, expected);
  }

  SECTION("SelfMoveAssignment") {
    const std::vector<int> expected = {1, 2, 3};
    CudaContainer<int> c(expected);

    int *const hostData = c.getHostArray().data();
    int *const deviceData = c.getDeviceArray().data();

    CHECK_NOTHROW(c = std::move(c));

    CHECK(c.getHostArray().data() == hostData);
    CHECK(c.getDeviceArray().data() == deviceData);
    apo_test::CheckHostAndDeviceEqual(c, expected);
  }
}

TEST_CASE("CudaContainerModifiers") {
  SECTION("Clear") {
    CudaContainer<int> c(std::vector<int>{1, 2, 3});

    c.clear();

    CHECK(c.size() == 0);
    CHECK(c.getHostArray().empty() == true);
    CHECK(c.getDeviceArray().empty() == true);
    CHECK(c.getDeviceArray().size() == 0);
    CHECK(c.getDeviceArray().capacity() == 0);
    CHECK(c.getDeviceArray().data() == nullptr);

    CHECK_NOTHROW(c.clear());
    CHECK(c.size() == 0);
    CHECK(c.getHostArray().empty() == true);
    CHECK(c.getDeviceArray().empty() == true);
    CHECK(c.getDeviceArray().data() == nullptr);
  }

  SECTION("ResizeFromEmpty") {
    constexpr std::size_t n = 5;
    CudaContainer<int> c;

    c.resize(n);

    CHECK(c.size() == n);
    CHECK(c.getHostArray() == std::vector<int>(n, 0));
    CHECK(c.getDeviceArray().size() == n);
    CHECK(c.getDeviceArray().capacity() == n);
    CHECK(c.getDeviceArray().data() != nullptr);
  }

  SECTION("ResizeGrowPreserve") {
    CudaContainer<int> c(std::vector<int>{1, 2, 3});

    c.resize(5);

    CHECK(c.size() == 5);
    CHECK(c.getHostArray() == std::vector<int>{1, 2, 3, 0, 0});
    CHECK(c.getDeviceArray().size() == 5);
    CHECK(apo_test::CopyToHost(c.getDeviceArray(), 3) ==
          std::vector<int>{1, 2, 3});
  }

  SECTION("ResizeShrinkPreserve") {
    const std::vector<int> expected = {1, 2, 3};
    CudaContainer<int> c(std::vector<int>{1, 2, 3, 4, 5});

    c.resize(expected.size());

    apo_test::CheckHostAndDeviceEqual(c, expected);
  }

  SECTION("ResizeToZero") {
    CudaContainer<int> c(std::vector<int>{1, 2, 3});

    c.resize(0);

    CHECK(c.size() == 0);
    CHECK(c.getHostArray().empty() == true);
    CHECK(c.getDeviceArray().empty() == true);
    CHECK(c.getDeviceArray().size() == 0);
  }

  SECTION("ShrinkToFitNonempty") {
    const std::vector<int> expected = {1, 2, 3};
    CudaContainer<int> c(std::vector<int>{1, 2, 3, 4, 5});

    c.resize(expected.size());
    c.shrink_to_fit();

    apo_test::CheckHostAndDeviceEqual(c, expected);
    CHECK(c.getDeviceArray().capacity() == expected.size());
  }

  SECTION("ShrinkToFitEmpty") {
    CudaContainer<int> c;

    CHECK_NOTHROW(c.shrink_to_fit());

    CHECK(c.size() == 0);
    CHECK(c.getHostArray().empty() == true);
    CHECK(c.getDeviceArray().empty() == true);
    CHECK(c.getDeviceArray().capacity() == 0);
    CHECK(c.getDeviceArray().data() == nullptr);
  }

  SECTION("PushBackFromEmpty") {
    const std::vector<int> expected = {4};
    CudaContainer<int> c;

    c.push_back(4);

    apo_test::CheckHostAndDeviceEqual(c, expected);
  }

  SECTION("PushBackMany") {
    std::vector<int> expected(100);
    for (std::size_t i = 0; i < expected.size(); i++)
      expected[i] = static_cast<int>(i + 1);

    CudaContainer<int> c;
    for (std::size_t i = 0; i < expected.size(); i++)
      c.push_back(static_cast<int>(i + 1));

    apo_test::CheckHostAndDeviceEqual(c, expected);
  }

  SECTION("PushBackAfterClear") {
    const std::vector<int> expected = {64};
    CudaContainer<int> c(std::vector<int>{1, 2, 3});

    c.clear();
    c.push_back(64);

    apo_test::CheckHostAndDeviceEqual(c, expected);
  }
}

TEST_CASE("CudaContainerSetAndTransfer") {
  SECTION("SetFromHostVectorResizeDevice") {
    const std::vector<int> expected = {1, 2, 3};
    CudaContainer<int> c(5);

    c.set(expected);

    apo_test::CheckHostAndDeviceEqual(c, expected);
  }

  SECTION("SetFromDeviceVectorResizeHost") {
    const std::vector<int> expected = {1, 2, 3};
    CudaContainer<int> c(std::vector<int>{9, 9, 9, 9, 9});
    DeviceVector<int> deviceVector(expected);

    c.set(deviceVector);

    apo_test::CheckHostAndDeviceEqual(c, expected);
  }

  SECTION("SetScalar") {
    CudaContainer<int> c(4);

    c.set(11);

    const std::vector<int> expected(4, 11);
    apo_test::CheckHostAndDeviceEqual(c, expected);
  }

  SECTION("SetToValueAlias") {
    CudaContainer<int> c(4);

    c.setToValue(-2);

    const std::vector<int> expected(4, -2);
    apo_test::CheckHostAndDeviceEqual(c, expected);
  }

  SECTION("TransferToDevice") {
    CudaContainer<int> c(std::vector<int>{1, 2, 3});

    c.getHostArray()[1] = 20;
    c.transferToDevice();

    const std::vector<int> expected = {1, 20, 3};
    CHECK(c.getHostArray() == expected);
    CHECK(apo_test::CopyToHost(c.getDeviceArray()) == expected);
  }

  SECTION("TransferToHost") {
    CudaContainer<int> c(std::vector<int>{1, 2, 3});
    const std::vector<int> expected = {4, 5, 6};

    apo_test::CopyToDevice(c.getDeviceArray(), expected);
    c.transferToHost();

    CHECK(c.getHostArray() == expected);
    CHECK(apo_test::CopyToHost(c.getDeviceArray()) == expected);
  }

  SECTION("TransferAliases") {
    CudaContainer<int> c(std::vector<int>{1, 2, 3});

    c.getHostArray()[0] = 7;
    c.transferFromHost();

    const std::vector<int> hostChange = {7, 2, 3};
    CHECK(c.getHostArray() == hostChange);
    CHECK(apo_test::CopyToHost(c.getDeviceArray()) == hostChange);

    const std::vector<int> deviceChange = {4, 5, 6};
    apo_test::CopyToDevice(c.getDeviceArray(), deviceChange);
    c.transferFromDevice();

    CHECK(c.getHostArray() == deviceChange);
    CHECK(apo_test::CopyToHost(c.getDeviceArray()) == deviceChange);
  }
}
