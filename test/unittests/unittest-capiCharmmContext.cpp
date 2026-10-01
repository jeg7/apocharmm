// BEGINLICENSE
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "ApoCharmmError.h"
#include "CharmmContext.h"
#include "CharmmPSF.h"
#include "CharmmParameters.h"
#include "Coordinates.h"
#include "ForceManager.h"
#include "apo_test_helpers.h"
#include "apocharmm_c/CharmmContext.h"
#include "apocharmm_c/Error.h"
#include "apocharmm_c/Status.h"
#include "apocharmm_c/detail/CharmmContextHandle.h"
#include "apocharmm_c/detail/CharmmCrdHandle.h"
#include "apocharmm_c/detail/CharmmParametersHandle.h"
#include "apocharmm_c/detail/CharmmPsfHandle.h"
#include "apocharmm_c/detail/ForceManagerHandle.h"
#include "catch.hpp"

#include <cstddef>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

static_assert(std::is_nothrow_destructible_v<CharmmContext>);

namespace {

struct CharmmContextDeleter {
  void operator()(apo_charmm_context *context) const noexcept {
    apo_charmm_context_destroy(context);
    return;
  }
};

using CharmmContextHandle =
    std::unique_ptr<apo_charmm_context, CharmmContextDeleter>;

struct ContextInputs {
  std::shared_ptr<CharmmPSF> psf;
  std::shared_ptr<CharmmParameters> parameters;
  std::shared_ptr<ForceManager> forceManager;
  apo_charmm_psf psfHandle;
  apo_charmm_parameters parametersHandle;
  apo_force_manager forceManagerHandle;

  ContextInputs()
      : psf(std::make_shared<CharmmPSF>(apo_test::GetDataDir() /
                                        "nacl_pair.psf")),
        parameters(std::make_shared<CharmmParameters>(apo_test::GetTopparDir() /
                                                      "toppar_water_ions.str")),
        forceManager(std::make_shared<ForceManager>(psf, parameters)),
        psfHandle(), parametersHandle(), forceManagerHandle() {
    psfHandle.object = psf;
    parametersHandle.object = parameters;
    forceManagerHandle.psf = psf;
    forceManagerHandle.parameters = parameters;
    forceManagerHandle.object = forceManager;
  }
};

std::shared_ptr<CharmmParameters> MakeParameters(void) {
  return std::make_shared<CharmmParameters>(apo_test::GetTopparDir() /
                                            "toppar_water_ions.str");
}

std::shared_ptr<CharmmPSF> MakePsf(void) {
  return std::make_shared<CharmmPSF>(apo_test::GetDataDir() / "nacl_pair.psf");
}

class LogicallyInitializedForceManager final : public ForceManager {
public:
  bool isInitialized(void) const override { return true; }
};

std::shared_ptr<CharmmContext> MakeLogicallyInitializedContext(void) {
  std::shared_ptr<LogicallyInitializedForceManager> forceManager =
      std::make_shared<LogicallyInitializedForceManager>();
  forceManager->setPsf(MakePsf());

  return std::make_shared<CharmmContext>(forceManager);
}

CharmmContextHandle MakeDefaultContext(void) {
  CharmmContextHandle handle(new apo_charmm_context());
  handle->object = std::make_shared<CharmmContext>();
  return handle;
}

CharmmContextHandle MakeContext(ContextInputs &inputs) {
  apo_charmm_context *rawContext = nullptr;
  REQUIRE(apo_charmm_context_create(&rawContext, &inputs.forceManagerHandle) ==
          APO_STATUS_OK);
  REQUIRE(rawContext != nullptr);
  return CharmmContextHandle(rawContext);
}

void CheckVelocityFileError(CharmmContext &context, const std::string &contents,
                            const ApoCharmmErrorCode expectedCode,
                            const std::string &expectedMessage) {
  const std::string fileName = "unittest-charmmContextErrors.vel";
  apo_test::RemoveIfExists(fileName);
  apo_test::WriteTextFile(fileName, contents);

  apo_test::CheckApoCharmmError(
      [&context, &fileName](void) {
        context.setVelocitiesFromCHARMMVelocityFile(fileName);
      },
      expectedCode, expectedMessage);

  apo_test::RemoveIfExists(fileName);
  return;
}

template <typename Function>
void CheckCharmmContextHandleValidation(const std::string_view functionName,
                                        Function function) {
  apo_test::CheckStatusAndDiagnostic(
      function(nullptr), APO_STATUS_INVALID_ARGUMENT,
      std::string(functionName) + ": CharmmContext is NULL");

  apo_charmm_context emptyContext;
  apo_test::CheckStatusAndDiagnostic(
      function(&emptyContext), APO_STATUS_INVALID_ARGUMENT,
      std::string(functionName) + ": CharmmContext object is NULL");

  return;
}

} // namespace

TEST_CASE("CharmmContextConstructorsAndObjectSettersUseApoCharmmError") {
  std::shared_ptr<CharmmPSF> psf = MakePsf();
  std::shared_ptr<CharmmParameters> parameters = MakeParameters();

  SECTION("NullPsfConstructor") {
    apo_test::CheckApoCharmmError(
        [&parameters](void) {
          CharmmContext context(nullptr, parameters);
          static_cast<void>(context);
        },
        ApoCharmmErrorCode::InvalidArgument, "CharmmPSF must not be null");
  }

  SECTION("NullParametersConstructor") {
    apo_test::CheckApoCharmmError(
        [&psf](void) {
          CharmmContext context(psf, nullptr);
          static_cast<void>(context);
        },
        ApoCharmmErrorCode::InvalidArgument,
        "CharmmParameters must not be null");
  }

  SECTION("NullForceManagerConstructor") {
    apo_test::CheckApoCharmmError(
        [](void) {
          CharmmContext context{std::shared_ptr<ForceManager>()};
          static_cast<void>(context);
        },
        ApoCharmmErrorCode::InvalidArgument, "ForceManager must not be null");
  }

  SECTION("NullParametersSetter") {
    CharmmContext context;
    apo_test::CheckApoCharmmError([&context](void) { context.setPrm(nullptr); },
                                  ApoCharmmErrorCode::InvalidArgument,
                                  "CharmmParameters must not be null");
  }

  SECTION("NullPsfSetter") {
    CharmmContext context;
    apo_test::CheckApoCharmmError([&context](void) { context.setPsf(nullptr); },
                                  ApoCharmmErrorCode::InvalidArgument,
                                  "CharmmPSF must not be null");
  }

  SECTION("PsfAtomCountMismatch") {
    CharmmContext context;
    context.setNumAtoms(1);
    psf->setNumAtoms(2);

    apo_test::CheckApoCharmmError(
        [&context, &psf](void) { context.setPsf(psf); },
        ApoCharmmErrorCode::InvalidArgument,
        "Atom count mismatch between CharmmContext and CharmmPSF; expected 1, "
        "observed 2");
  }

  SECTION("NullForceManagerSetter") {
    CharmmContext context;
    apo_test::CheckApoCharmmError(
        [&context](void) { context.setForceManager(nullptr); },
        ApoCharmmErrorCode::InvalidArgument, "ForceManager must not be null");
  }

  SECTION("InvalidAtomCount") {
    CharmmContext context;
    apo_test::CheckApoCharmmError([&context](void) { context.setNumAtoms(0); },
                                  ApoCharmmErrorCode::InvalidArgument,
                                  "Atom count must be positive; observed 0");
  }
}

TEST_CASE("CharmmContextCoordinateAndChargeValidation") {
  SECTION("CoordinateChargeState") {
    CharmmContext context;
    apo_test::CheckApoCharmmError(
        [&context](void) {
          context.setCoordinatesCharges(std::vector<double4>{});
        },
        ApoCharmmErrorCode::NotInitialized,
        "Atom count must be set before assigning coordinates and charges");
  }

  SECTION("CoordinateChargeCount") {
    CharmmContext context;
    context.setNumAtoms(2);
    apo_test::CheckApoCharmmError(
        [&context](void) {
          context.setCoordinatesCharges(
              std::vector<double4>{make_double4(0.0, 0.0, 0.0, 0.0)});
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Coordinate and charge count mismatch; expected 2, observed 1");
  }

  SECTION("CoordinateChargeForceManagerCount") {
    std::shared_ptr<CharmmContext> context = MakeLogicallyInitializedContext();
    context->setNumAtoms(3);

    apo_test::CheckApoCharmmError(
        [&context](void) {
          context->setCoordinatesCharges(
              std::vector<double4>{make_double4(0.0, 0.0, 0.0, 0.0),
                                   make_double4(1.0, 0.0, 0.0, 0.0),
                                   make_double4(2.0, 0.0, 0.0, 0.0)});
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Coordinate and charge count mismatch with ForceManager; expected 2, "
        "observed 3");
  }

  SECTION("CoordinateChargeRowWidth") {
    CharmmContext context;
    context.setNumAtoms(1);
    apo_test::CheckApoCharmmError(
        [&context](void) {
          context.setCoordinatesCharges(
              std::vector<std::vector<double>>{{0.0, 0.0, 0.0}});
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Each coordinate and charge entry must contain exactly 4 values");
  }

  SECTION("CoordinateChargeFlatWidth") {
    CharmmContext context;
    context.setNumAtoms(1);
    apo_test::CheckApoCharmmError(
        [&context](void) {
          context.setCoordinatesCharges(std::vector<double>{0.0, 0.0, 0.0});
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Coordinate and charge array length must be a multiple of 4; observed "
        "3");
  }

  SECTION("CoordinateState") {
    CharmmContext context;
    apo_test::CheckApoCharmmError(
        [&context](void) { context.setCoordinates(std::vector<double3>{}); },
        ApoCharmmErrorCode::NotInitialized,
        "Atom count must be set before assigning coordinates");
  }

  SECTION("CoordinateCount") {
    CharmmContext context;
    context.setNumAtoms(2);
    apo_test::CheckApoCharmmError(
        [&context](void) {
          context.setCoordinates(
              std::vector<double3>{make_double3(0.0, 0.0, 0.0)});
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Coordinate count mismatch; expected 2, observed 1");
  }

  SECTION("CoordinateRowWidth") {
    CharmmContext context;
    context.setNumAtoms(1);
    apo_test::CheckApoCharmmError(
        [&context](void) {
          context.setCoordinates(std::vector<std::vector<double>>{{0.0, 0.0}});
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Each coordinate entry must contain exactly 3 values");
  }

  SECTION("CoordinateFlatWidth") {
    CharmmContext context;
    context.setNumAtoms(1);
    apo_test::CheckApoCharmmError(
        [&context](void) {
          context.setCoordinates(std::vector<double>{0.0, 0.0});
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Coordinate array length must be a multiple of 3; observed 2");
  }

  SECTION("NullCoordinatesObject") {
    CharmmContext context;
    apo_test::CheckApoCharmmError(
        [&context](void) {
          context.setCoordinates(std::shared_ptr<Coordinates>());
        },
        ApoCharmmErrorCode::InvalidArgument, "Coordinates must not be null");
  }

  SECTION("ChargeState") {
    CharmmContext context;
    apo_test::CheckApoCharmmError(
        [&context](void) { context.setCharges({0.0}); },
        ApoCharmmErrorCode::NotInitialized,
        "Atom count must be set before assigning charges");
  }

  SECTION("ChargeCount") {
    CharmmContext context;
    context.setNumAtoms(2);
    apo_test::CheckApoCharmmError(
        [&context](void) { context.setCharges({0.0}); },
        ApoCharmmErrorCode::InvalidArgument,
        "Charge count mismatch; expected 2, observed 1");
  }
}

TEST_CASE("CharmmContextVelocityAndMassValidation") {
  SECTION("VelocityInverseMassState") {
    CharmmContext context;
    apo_test::CheckApoCharmmError(
        [&context](void) {
          context.setVelocitiesInverseMasses(std::vector<double4>{});
        },
        ApoCharmmErrorCode::NotInitialized,
        "Atom count must be set before assigning velocities and inverse "
        "masses");
  }

  SECTION("VelocityInverseMassCount") {
    CharmmContext context;
    context.setNumAtoms(2);
    apo_test::CheckApoCharmmError(
        [&context](void) {
          context.setVelocitiesInverseMasses(
              std::vector<double4>{make_double4(0.0, 0.0, 0.0, 1.0)});
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Velocity and inverse-mass count mismatch; expected 2, observed 1");
  }

  SECTION("VelocityInverseMassRowWidth") {
    CharmmContext context;
    context.setNumAtoms(1);
    apo_test::CheckApoCharmmError(
        [&context](void) {
          context.setVelocitiesInverseMasses(
              std::vector<std::vector<double>>{{0.0, 0.0, 0.0}});
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Each velocity and inverse-mass entry must contain exactly 4 values");
  }

  SECTION("VelocityInverseMassFlatWidth") {
    CharmmContext context;
    context.setNumAtoms(1);
    apo_test::CheckApoCharmmError(
        [&context](void) {
          context.setVelocitiesInverseMasses(
              std::vector<double>{0.0, 0.0, 0.0});
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Velocity and inverse-mass array length must be a multiple of 4; "
        "observed 3");
  }

  SECTION("VelocityState") {
    CharmmContext context;
    apo_test::CheckApoCharmmError(
        [&context](void) { context.setVelocities(std::vector<double3>{}); },
        ApoCharmmErrorCode::NotInitialized,
        "Atom count must be set before assigning velocities");
  }

  SECTION("VelocityCount") {
    CharmmContext context;
    context.setNumAtoms(2);
    apo_test::CheckApoCharmmError(
        [&context](void) {
          context.setVelocities(
              std::vector<double3>{make_double3(0.0, 0.0, 0.0)});
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Velocity count mismatch; expected 2, observed 1");
  }

  SECTION("VelocityRowWidth") {
    CharmmContext context;
    context.setNumAtoms(1);
    apo_test::CheckApoCharmmError(
        [&context](void) {
          context.setVelocities(std::vector<std::vector<double>>{{0.0, 0.0}});
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Each velocity entry must contain exactly 3 values");
  }

  SECTION("VelocityFlatWidth") {
    CharmmContext context;
    context.setNumAtoms(1);
    apo_test::CheckApoCharmmError(
        [&context](void) {
          context.setVelocities(std::vector<double>{0.0, 0.0});
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Velocity array length must be a multiple of 3; observed 2");
  }

  SECTION("MassState") {
    CharmmContext context;
    apo_test::CheckApoCharmmError(
        [&context](void) { context.setMasses({1.0}); },
        ApoCharmmErrorCode::NotInitialized,
        "Atom count must be set before assigning masses");
  }

  SECTION("MassCount") {
    CharmmContext context;
    context.setNumAtoms(2);
    apo_test::CheckApoCharmmError(
        [&context](void) { context.setMasses({1.0}); },
        ApoCharmmErrorCode::InvalidArgument,
        "Mass count mismatch; expected 2, observed 1");
  }
}

TEST_CASE("CharmmContextVelocityFileValidation") {
  SECTION("MissingAtomCountState") {
    CharmmContext context;
    apo_test::CheckApoCharmmError(
        [&context](void) {
          context.setVelocitiesFromCHARMMVelocityFile("unused.vel");
        },
        ApoCharmmErrorCode::NotInitialized,
        "Atom count must be set before loading velocities from a CHARMM "
        "velocity file");
  }

  CharmmContext context;
  context.setNumAtoms(2);

  SECTION("EmptyFileName") {
    apo_test::CheckApoCharmmError(
        [&context](void) { context.setVelocitiesFromCHARMMVelocityFile(""); },
        ApoCharmmErrorCode::InvalidArgument,
        "CHARMM velocity file name must not be empty");
  }

  SECTION("OpenFailure") {
    const std::string fileName = "unittest-charmmContextErrors-missing.vel";
    apo_test::RemoveIfExists(fileName);
    apo_test::CheckApoCharmmError(
        [&context, &fileName](void) {
          context.setVelocitiesFromCHARMMVelocityFile(fileName);
        },
        ApoCharmmErrorCode::Runtime,
        "Could not open CHARMM velocity file \"" + fileName + "\"");
  }

  SECTION("MissingAtomCountRecord") {
    CheckVelocityFileError(
        context, "* title\n* another title\n", ApoCharmmErrorCode::Runtime,
        "Could not read atom count from CHARMM velocity file "
        "\"unittest-charmmContextErrors.vel\"");
  }

  SECTION("InvalidAtomCountRecord") {
    CheckVelocityFileError(
        context, "not-an-integer\n", ApoCharmmErrorCode::Runtime,
        "Could not parse atom count from CHARMM velocity file "
        "\"unittest-charmmContextErrors.vel\"");
  }

  SECTION("AtomCountMismatch") {
    CheckVelocityFileError(
        context, "3\n", ApoCharmmErrorCode::InvalidArgument,
        "CHARMM velocity atom count mismatch; expected 2, observed 3");
  }

  SECTION("MissingVelocityRecord") {
    CheckVelocityFileError(
        context, "2\n1 1 RES A 0.0 0.0 0.0 SEG 1 0.0\n",
        ApoCharmmErrorCode::Runtime,
        "Could not read velocity record 2 from CHARMM velocity file "
        "\"unittest-charmmContextErrors.vel\"");
  }

  SECTION("InvalidVelocityRecord") {
    CheckVelocityFileError(
        context, "2\ninvalid record\n2 1 RES B 0.0 0.0 0.0 SEG 1 0.0\n",
        ApoCharmmErrorCode::Runtime,
        "Could not parse velocity record 1 from CHARMM velocity file "
        "\"unittest-charmmContextErrors.vel\"");
  }

  SECTION("ValidVelocityFile") {
    const std::string fileName = "unittest-charmmContextErrors.vel";
    apo_test::RemoveIfExists(fileName);
    apo_test::WriteTextFile(
        fileName, "* title\n2\n1 1 RES A 1.0 2.0 3.0 SEG 1 0.0\n2 1 RES B 4.0 "
                  "5.0 6.0 SEG 1 0.0\n");

    CHECK_NOTHROW(context.setVelocitiesFromCHARMMVelocityFile(fileName));
    const CudaContainer<double4> &velocityMass =
        context.getVelocitiesInverseMasses();
    CHECK(velocityMass[0].x == Approx(1.0));
    CHECK(velocityMass[0].y == Approx(2.0));
    CHECK(velocityMass[0].z == Approx(3.0));
    CHECK(velocityMass[1].x == Approx(4.0));
    CHECK(velocityMass[1].y == Approx(5.0));
    CHECK(velocityMass[1].z == Approx(6.0));

    apo_test::RemoveIfExists(fileName);
  }
}

TEST_CASE("CharmmContextScalarAndBoxValidation") {
  SECTION("NegativeTemperature") {
    CharmmContext context;
    apo_test::CheckApoCharmmError(
        [&context](void) { context.setTemperature(-1.0); },
        ApoCharmmErrorCode::InvalidArgument,
        "Temperature must be finite and non-negative");
  }

  SECTION("NonfiniteTemperature") {
    CharmmContext context;
    apo_test::CheckApoCharmmError(
        [&context](void) {
          context.setTemperature(std::numeric_limits<double>::infinity());
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Temperature must be finite and non-negative");
  }

  SECTION("InvalidBoxLength") {
    CharmmContext context;
    apo_test::CheckApoCharmmError(
        [&context](void) { context.setBoxDimensions({20.0, 20.0}); },
        ApoCharmmErrorCode::InvalidArgument,
        "Box dimensions must contain exactly 3 positive values");
  }

  SECTION("InvalidBoxValue") {
    CharmmContext context;
    apo_test::CheckApoCharmmError(
        [&context](void) { context.setBoxDimensions({20.0, 0.0, 20.0}); },
        ApoCharmmErrorCode::InvalidArgument,
        "Box dimensions must contain exactly 3 positive values");
  }
}

TEST_CASE("CharmmContextStateHelpersUseNotInitialized") {
  SECTION("MissingPsf") {
    CharmmContext context;
    apo_test::CheckApoCharmmError(
        [&context](void) { static_cast<void>(context.getBonds()); },
        ApoCharmmErrorCode::NotInitialized, "CharmmPSF is not set");
  }

  SECTION("MissingForceManager") {
    CharmmContext context;
    apo_test::CheckApoCharmmError(
        [&context](void) { static_cast<void>(context.getKappa()); },
        ApoCharmmErrorCode::NotInitialized, "ForceManager is not set");
  }

  SECTION("UninitializedForceManager") {
    CharmmContext context;
    context.setForceManager(std::make_shared<ForceManager>());
    apo_test::CheckApoCharmmError(
        [&context](void) { static_cast<void>(context.getShakeAtoms()); },
        ApoCharmmErrorCode::NotInitialized, "ForceManager is not initialized");
  }

  SECTION("MissingBoxDimensions") {
    CharmmContext context;
    apo_test::CheckApoCharmmError(
        [&context](void) { static_cast<void>(context.getVolume()); },
        ApoCharmmErrorCode::NotInitialized, "Box dimensions are not set");
  }

  SECTION("MissingPsfForConstraints") {
    CharmmContext context;
    apo_test::CheckApoCharmmError(
        [&context](void) { context.useHolonomicConstraints(true); },
        ApoCharmmErrorCode::NotInitialized, "CharmmPSF is not set");
  }

  SECTION("MissingForceManagerForConstraints") {
    CharmmContext context;
    std::shared_ptr<CharmmPSF> psf = MakePsf();
    context.setPsf(psf);
    apo_test::CheckApoCharmmError(
        [&context](void) { context.useHolonomicConstraints(true); },
        ApoCharmmErrorCode::NotInitialized, "ForceManager is not set");
  }

  SECTION("MissingImageBoxDimensions") {
    std::shared_ptr<CharmmContext> context = MakeLogicallyInitializedContext();

    apo_test::CheckApoCharmmError(
        [&context](void) { context->imageCentering(); },
        ApoCharmmErrorCode::NotInitialized, "Box dimensions are not set");
  }
}

TEST_CASE("CharmmContextSpecializedOperationValidation") {
  SECTION("AssignVelocityState") {
    CharmmContext context;
    apo_test::CheckApoCharmmError(
        [&context](void) { context.assignVelocitiesAtTemperature(300.0); },
        ApoCharmmErrorCode::NotInitialized,
        "Atom count must be set before assigning velocities at a temperature");
  }

  SECTION("ComputeTemperatureStorageState") {
    CharmmContext context;
    apo_test::CheckApoCharmmError(
        [&context](void) { static_cast<void>(context.computeTemperature()); },
        ApoCharmmErrorCode::NotInitialized,
        "Atom count and velocity storage must be initialized before computing "
        "temperature");
  }

  SECTION("ComputeTemperatureDegreesOfFreedomState") {
    CharmmContext context;
    context.setNumAtoms(1);
    apo_test::CheckApoCharmmError(
        [&context](void) { static_cast<void>(context.computeTemperature()); },
        ApoCharmmErrorCode::NotInitialized,
        "Degrees of freedom must be initialized before computing temperature");
  }

  SECTION("CalculateKineticEnergyState") {
    CharmmContext context;
    apo_test::CheckApoCharmmError(
        [&context](void) { context.calculateKineticEnergy(); },
        ApoCharmmErrorCode::NotInitialized,
        "Atom count and velocity storage must be initialized before "
        "calculating kinetic energy");
  }
}

TEST_CASE("CharmmContextCheckedCudaPathsExecute") {
  SECTION("KineticEnergyKernel") {
    CharmmContext context;
    context.setNumAtoms(2);
    context.setVelocitiesInverseMasses(std::vector<double4>{
        make_double4(1.0, 2.0, 3.0, 1.0), make_double4(0.0, 0.0, 0.0, 1.0)});

    CHECK(context.getKineticEnergy() == Approx(7.0));
  }
}

TEST_CASE("CapiCharmmContextConstructorsAndDestroy") {
  ContextInputs inputs;

  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_create(nullptr, &inputs.forceManagerHandle),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_create: out pointer is NULL");

  apo_charmm_context staleContext;
  apo_charmm_context *context = &staleContext;

  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_create(&context, nullptr), APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_create: ForceManager is NULL");
  CHECK(context == nullptr);

  apo_force_manager emptyForceManager;
  context = &staleContext;
  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_create(&context, &emptyForceManager),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_create: ForceManager object is NULL");
  CHECK(context == nullptr);

  context = nullptr;
  REQUIRE(apo_charmm_context_create(&context, &inputs.forceManagerHandle) ==
          APO_STATUS_OK);
  REQUIRE(context != nullptr);
  CHECK_NOTHROW(apo_charmm_context_destroy(context));
  CHECK(std::string(apo_last_error()).empty() == true);

  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_create_from_psf_parameters(nullptr, &inputs.psfHandle,
                                                    &inputs.parametersHandle),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_create_from_psf_parameters: out pointer is NULL");

  context = &staleContext;
  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_create_from_psf_parameters(&context, nullptr,
                                                    &inputs.parametersHandle),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_create_from_psf_parameters: CharmmPsf is NULL");
  CHECK(context == nullptr);

  apo_charmm_psf emptyPsf;
  context = &staleContext;
  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_create_from_psf_parameters(&context, &emptyPsf,
                                                    &inputs.parametersHandle),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_create_from_psf_parameters: CharmmPsf object is "
      "NULL");
  CHECK(context == nullptr);

  context = &staleContext;
  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_create_from_psf_parameters(&context, &inputs.psfHandle,
                                                    nullptr),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_create_from_psf_parameters: CharmmParameters is "
      "NULL");
  CHECK(context == nullptr);

  apo_charmm_parameters emptyParameters;
  context = &staleContext;
  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_create_from_psf_parameters(&context, &inputs.psfHandle,
                                                    &emptyParameters),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_create_from_psf_parameters: CharmmParameters object "
      "is NULL");
  CHECK(context == nullptr);

  context = nullptr;
  REQUIRE(apo_charmm_context_create_from_psf_parameters(
              &context, &inputs.psfHandle, &inputs.parametersHandle) ==
          APO_STATUS_OK);
  REQUIRE(context != nullptr);
  CHECK_NOTHROW(apo_charmm_context_destroy(context));
  CHECK(std::string(apo_last_error()).empty() == true);

  CHECK_NOTHROW(apo_charmm_context_destroy(nullptr));
  CHECK(std::string(apo_last_error()).empty() == true);
}

TEST_CASE("CapiCharmmContextValidatesEveryFunctionHandle") {
  ContextInputs inputs;
  const double xyzq[8] = {0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 1.0};
  const double xyz[6] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
  const double charges[2] = {-1.0, 1.0};
  const double xyzm[8] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  const double boxDimensions[3] = {20.0, 20.0, 20.0};
  const int grid[3] = {32, 32, 32};
  int numAtoms = 0;
  double outputDouble[8] = {};
  double scalar = 0.0;
  int outputGrid[3] = {};
  int integer = 0;
  apo_pbc pbc = APO_PBC_UNSET;
  apo_force_manager *forceManager = nullptr;

  CheckCharmmContextHandleValidation(
      "apo_charmm_context_set_prm",
      [&inputs](apo_charmm_context *context) -> apo_status {
        return apo_charmm_context_set_prm(context, &inputs.parametersHandle);
      });
  CheckCharmmContextHandleValidation(
      "apo_charmm_context_set_psf",
      [&inputs](apo_charmm_context *context) -> apo_status {
        return apo_charmm_context_set_psf(context, &inputs.psfHandle);
      });
  CheckCharmmContextHandleValidation(
      "apo_charmm_context_set_force_manager",
      [&inputs](apo_charmm_context *context) -> apo_status {
        return apo_charmm_context_set_force_manager(context,
                                                    &inputs.forceManagerHandle);
      });
  CheckCharmmContextHandleValidation(
      "apo_charmm_context_set_coordinates_charges",
      [&xyzq](apo_charmm_context *context) -> apo_status {
        return apo_charmm_context_set_coordinates_charges(context, xyzq, 8);
      });
  CheckCharmmContextHandleValidation(
      "apo_charmm_context_set_coordinates_from_array",
      [&xyz](apo_charmm_context *context) -> apo_status {
        return apo_charmm_context_set_coordinates_from_array(context, xyz, 6);
      });
  CheckCharmmContextHandleValidation(
      "apo_charmm_context_set_coordinates",
      [](apo_charmm_context *context) -> apo_status {
        return apo_charmm_context_set_coordinates(context, nullptr);
      });
  CheckCharmmContextHandleValidation(
      "apo_charmm_context_set_charges",
      [&charges](apo_charmm_context *context) -> apo_status {
        return apo_charmm_context_set_charges(context, charges, 2);
      });
  CheckCharmmContextHandleValidation(
      "apo_charmm_context_set_velocities_inverse_masses",
      [&xyzm](apo_charmm_context *context) -> apo_status {
        return apo_charmm_context_set_velocities_inverse_masses(context, xyzm,
                                                                8);
      });
  CheckCharmmContextHandleValidation(
      "apo_charmm_context_set_velocities",
      [&xyz](apo_charmm_context *context) -> apo_status {
        return apo_charmm_context_set_velocities(context, xyz, 6);
      });
  CheckCharmmContextHandleValidation(
      "apo_charmm_context_set_box_dimensions",
      [&boxDimensions](apo_charmm_context *context) -> apo_status {
        return apo_charmm_context_set_box_dimensions(context, boxDimensions, 3);
      });
  CheckCharmmContextHandleValidation(
      "apo_charmm_context_set_kappa",
      [](apo_charmm_context *context) -> apo_status {
        return apo_charmm_context_set_kappa(context, 0.34);
      });
  CheckCharmmContextHandleValidation(
      "apo_charmm_context_set_cutoff",
      [](apo_charmm_context *context) -> apo_status {
        return apo_charmm_context_set_cutoff(context, 14.0);
      });
  CheckCharmmContextHandleValidation(
      "apo_charmm_context_set_ctonnb",
      [](apo_charmm_context *context) -> apo_status {
        return apo_charmm_context_set_ctonnb(context, 12.0);
      });
  CheckCharmmContextHandleValidation(
      "apo_charmm_context_set_ctofnb",
      [](apo_charmm_context *context) -> apo_status {
        return apo_charmm_context_set_ctofnb(context, 10.0);
      });
  CheckCharmmContextHandleValidation(
      "apo_charmm_context_set_fft_grid",
      [&grid](apo_charmm_context *context) -> apo_status {
        return apo_charmm_context_set_fft_grid(context, grid, 3);
      });
  CheckCharmmContextHandleValidation(
      "apo_charmm_context_set_pme_spline_order",
      [](apo_charmm_context *context) -> apo_status {
        return apo_charmm_context_set_pme_spline_order(context, 4);
      });
  CheckCharmmContextHandleValidation(
      "apo_charmm_context_set_periodic_boundary_condition",
      [](apo_charmm_context *context) -> apo_status {
        return apo_charmm_context_set_periodic_boundary_condition(context,
                                                                  APO_PBC_P1);
      });
  CheckCharmmContextHandleValidation(
      "apo_charmm_context_set_vdw_type",
      [](apo_charmm_context *context) -> apo_status {
        return apo_charmm_context_set_vdw_type(context, 3);
      });
  CheckCharmmContextHandleValidation(
      "apo_charmm_context_set_random_seed",
      [](apo_charmm_context *context) -> apo_status {
        return apo_charmm_context_set_random_seed(context, 7);
      });
  CheckCharmmContextHandleValidation(
      "apo_charmm_context_use_holonomic_constraints",
      [](apo_charmm_context *context) -> apo_status {
        return apo_charmm_context_use_holonomic_constraints(context, true);
      });
  CheckCharmmContextHandleValidation(
      "apo_charmm_context_get_num_atoms",
      [&numAtoms](apo_charmm_context *context) -> apo_status {
        return apo_charmm_context_get_num_atoms(&numAtoms, context);
      });
  CheckCharmmContextHandleValidation(
      "apo_charmm_context_get_coordinates_charges",
      [&outputDouble](apo_charmm_context *context) -> apo_status {
        return apo_charmm_context_get_coordinates_charges(outputDouble, 8,
                                                          context);
      });
  CheckCharmmContextHandleValidation(
      "apo_charmm_context_get_velocity_mass",
      [&outputDouble](apo_charmm_context *context) -> apo_status {
        return apo_charmm_context_get_velocity_mass(outputDouble, 8, context);
      });
  CheckCharmmContextHandleValidation(
      "apo_charmm_context_get_box_dimensions",
      [&outputDouble](apo_charmm_context *context) -> apo_status {
        return apo_charmm_context_get_box_dimensions(outputDouble, 3, context);
      });
  CheckCharmmContextHandleValidation(
      "apo_charmm_context_get_kappa",
      [&scalar](apo_charmm_context *context) -> apo_status {
        return apo_charmm_context_get_kappa(&scalar, context);
      });
  CheckCharmmContextHandleValidation(
      "apo_charmm_context_get_cutoff",
      [&scalar](apo_charmm_context *context) -> apo_status {
        return apo_charmm_context_get_cutoff(&scalar, context);
      });
  CheckCharmmContextHandleValidation(
      "apo_charmm_context_get_ctonnb",
      [&scalar](apo_charmm_context *context) -> apo_status {
        return apo_charmm_context_get_ctonnb(&scalar, context);
      });
  CheckCharmmContextHandleValidation(
      "apo_charmm_context_get_ctofnb",
      [&scalar](apo_charmm_context *context) -> apo_status {
        return apo_charmm_context_get_ctofnb(&scalar, context);
      });
  CheckCharmmContextHandleValidation(
      "apo_charmm_context_get_fft_grid",
      [&outputGrid](apo_charmm_context *context) -> apo_status {
        return apo_charmm_context_get_fft_grid(outputGrid, 3, context);
      });
  CheckCharmmContextHandleValidation(
      "apo_charmm_context_get_pme_spline_order",
      [&integer](apo_charmm_context *context) -> apo_status {
        return apo_charmm_context_get_pme_spline_order(&integer, context);
      });
  CheckCharmmContextHandleValidation(
      "apo_charmm_context_get_periodic_boundary_condition",
      [&pbc](apo_charmm_context *context) -> apo_status {
        return apo_charmm_context_get_periodic_boundary_condition(&pbc,
                                                                  context);
      });
  CheckCharmmContextHandleValidation(
      "apo_charmm_context_get_vdw_type",
      [&integer](apo_charmm_context *context) -> apo_status {
        return apo_charmm_context_get_vdw_type(&integer, context);
      });
  CheckCharmmContextHandleValidation(
      "apo_charmm_context_get_force_manager",
      [&forceManager](apo_charmm_context *context) -> apo_status {
        return apo_charmm_context_get_force_manager(&forceManager, context);
      });
  CheckCharmmContextHandleValidation(
      "apo_charmm_context_assign_velocities_at_temperature",
      [](apo_charmm_context *context) -> apo_status {
        return apo_charmm_context_assign_velocities_at_temperature(context,
                                                                   300.0);
      });
  CheckCharmmContextHandleValidation(
      "apo_charmm_context_compute_temperature",
      [&scalar](apo_charmm_context *context) -> apo_status {
        return apo_charmm_context_compute_temperature(&scalar, context);
      });
}

TEST_CASE("CapiCharmmContextValidatesPointersAndArrayShapes") {
  ContextInputs inputs;
  CharmmContextHandle context = MakeContext(inputs);
  const double values[8] = {};
  const double boxDimensions[3] = {20.0, 20.0, 20.0};
  const int grid[3] = {32, 32, 32};

  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_set_prm(context.get(), nullptr),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_set_prm: CharmmParameters is NULL");
  apo_charmm_parameters emptyParameters;
  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_set_prm(context.get(), &emptyParameters),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_set_prm: CharmmParameters object is NULL");

  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_set_psf(context.get(), nullptr),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_set_psf: CharmmPsf is NULL");
  apo_charmm_psf emptyPsf;
  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_set_psf(context.get(), &emptyPsf),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_set_psf: CharmmPsf object is NULL");

  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_set_force_manager(context.get(), nullptr),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_set_force_manager: ForceManager is NULL");
  apo_force_manager emptyForceManager;
  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_set_force_manager(context.get(), &emptyForceManager),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_set_force_manager: ForceManager object is NULL");

  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_set_coordinates_charges(context.get(), nullptr, 8),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_set_coordinates_charges: xyzq is NULL");
  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_set_coordinates_charges(context.get(), values, 0),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_set_coordinates_charges: xyzq must not be empty");
  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_set_coordinates_charges(context.get(), values, 3),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_set_coordinates_charges: xyzq length must be a "
      "positive multiple of 4");

  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_set_coordinates_from_array(context.get(), nullptr, 6),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_set_coordinates_from_array: xyz is NULL");
  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_set_coordinates_from_array(context.get(), values, 0),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_set_coordinates_from_array: xyz must not be empty");
  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_set_coordinates_from_array(context.get(), values, 2),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_set_coordinates_from_array: xyz length must be a "
      "positive multiple of 3");

  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_set_coordinates(context.get(), nullptr),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_set_coordinates: CharmmCrd is NULL");
  apo_charmm_crd emptyCrd;
  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_set_coordinates(context.get(), &emptyCrd),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_set_coordinates: CharmmCrd object is NULL");

  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_set_charges(context.get(), nullptr, 2),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_set_charges: charges is NULL");

  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_set_velocities_inverse_masses(context.get(), nullptr,
                                                       8),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_set_velocities_inverse_masses: xyzm is NULL");
  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_set_velocities_inverse_masses(context.get(), values,
                                                       0),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_set_velocities_inverse_masses: xyzm must not be "
      "empty");
  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_set_velocities_inverse_masses(context.get(), values,
                                                       3),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_set_velocities_inverse_masses: xyzm length must be a "
      "positive multiple of 4");

  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_set_velocities(context.get(), nullptr, 6),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_set_velocities: xyz is NULL");
  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_set_velocities(context.get(), values, 0),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_set_velocities: xyz must not be empty");
  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_set_velocities(context.get(), values, 2),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_set_velocities: xyz length must be a positive "
      "multiple of 3");

  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_set_box_dimensions(context.get(), nullptr, 3),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_set_box_dimensions: box_dimensions is NULL");
  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_set_box_dimensions(context.get(), boxDimensions, 2),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_set_box_dimensions: box_dimensions must contain "
      "exactly 3 elements");

  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_set_fft_grid(context.get(), nullptr, 3),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_set_fft_grid: grid is NULL");
  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_set_fft_grid(context.get(), grid, 2),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_set_fft_grid: grid must contain exactly 3 elements");

  const apo_status unsetPbcStatus =
      apo_charmm_context_set_periodic_boundary_condition(context.get(),
                                                         APO_PBC_UNSET);
  CHECK(unsetPbcStatus == APO_STATUS_INVALID_ARGUMENT);

  const std::string unsetPbcDiagnostic(apo_last_error());
  CHECK(unsetPbcDiagnostic.find(
            "apoCHARMM error [InvalidArgument]: Periodic boundary condition "
            "must be PBC::P1 or PBC::P21; observed 0") != std::string::npos);
  CHECK(unsetPbcDiagnostic.find("function: setPeriodicBoundaryCondition") !=
        std::string::npos);

  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_set_periodic_boundary_condition(
          context.get(), static_cast<apo_pbc>(-1)),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_set_periodic_boundary_condition: invalid periodic "
      "boundary condition");
}

TEST_CASE("CapiCharmmContextValidatesOutputPointersAndBuffers") {
  ContextInputs inputs;
  CharmmContextHandle context = MakeContext(inputs);

  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_get_num_atoms(nullptr, context.get()),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_get_num_atoms: num_atoms is NULL");

  double output[8] = {};
  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_get_coordinates_charges(nullptr, 8, context.get()),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_get_coordinates_charges: Coordinate and charge "
      "output buffer is NULL");
  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_get_coordinates_charges(output, 7, context.get()),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_get_coordinates_charges: Coordinate and charge "
      "output buffer is too small");

  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_get_velocity_mass(nullptr, 8, context.get()),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_get_velocity_mass: Velocity and mass output buffer "
      "is NULL");
  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_get_velocity_mass(output, 7, context.get()),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_get_velocity_mass: Velocity and mass output buffer "
      "is too small");

  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_get_box_dimensions(nullptr, 3, context.get()),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_get_box_dimensions: Box dimension buffer is NULL");
  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_get_box_dimensions(output, 2, context.get()),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_get_box_dimensions: Box dimension buffer is too "
      "small");

  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_get_kappa(nullptr, context.get()),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_get_kappa: kappa is NULL");
  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_get_cutoff(nullptr, context.get()),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_get_cutoff: cutoff is NULL");
  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_get_ctonnb(nullptr, context.get()),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_get_ctonnb: ctonnb is NULL");
  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_get_ctofnb(nullptr, context.get()),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_get_ctofnb: ctofnb is NULL");

  int grid[3] = {};
  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_get_fft_grid(nullptr, 3, context.get()),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_get_fft_grid: FFT grid buffer is NULL");
  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_get_fft_grid(grid, 2, context.get()),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_get_fft_grid: FFT grid buffer is too small");

  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_get_pme_spline_order(nullptr, context.get()),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_get_pme_spline_order: order is NULL");
  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_get_periodic_boundary_condition(nullptr,
                                                         context.get()),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_get_periodic_boundary_condition: pbc is NULL");
  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_get_vdw_type(nullptr, context.get()),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_get_vdw_type: vdw_type is NULL");
  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_get_force_manager(nullptr, context.get()),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_get_force_manager: out pointer is NULL");
  apo_test::CheckStatusAndDiagnostic(
      apo_charmm_context_compute_temperature(nullptr, context.get()),
      APO_STATUS_INVALID_ARGUMENT,
      "apo_charmm_context_compute_temperature: temperature is NULL");
}

TEST_CASE("CapiCharmmContextSetVelocitiesIsImplemented") {
  CharmmContextHandle context = MakeDefaultContext();
  context->object->setNumAtoms(2);

  const double velocities[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  CHECK(apo_charmm_context_set_velocities(context.get(), velocities, 6) ==
        APO_STATUS_OK);

  double velocityMass[8] = {};
  REQUIRE(apo_charmm_context_get_velocity_mass(velocityMass, 8,
                                               context.get()) == APO_STATUS_OK);
  CHECK(velocityMass[0] == Approx(1.0));
  CHECK(velocityMass[1] == Approx(2.0));
  CHECK(velocityMass[2] == Approx(3.0));
  CHECK(velocityMass[4] == Approx(4.0));
  CHECK(velocityMass[5] == Approx(5.0));
  CHECK(velocityMass[6] == Approx(6.0));
  CHECK(std::string(apo_last_error()).empty() == true);
}

TEST_CASE("CapiCharmmContextMapsStateAndNativeValidation") {
  SECTION("UnsetAtomCountIsForwarded") {
    CharmmContextHandle context = MakeDefaultContext();
    int numAtoms = 17;

    CHECK(apo_charmm_context_get_num_atoms(&numAtoms, context.get()) ==
          APO_STATUS_OK);
    CHECK(numAtoms == -1);
    CHECK(std::string(apo_last_error()).empty() == true);
  }

  SECTION("MissingForceManager") {
    CharmmContextHandle context = MakeDefaultContext();
    apo_force_manager staleForceManager;
    apo_force_manager *forceManager = &staleForceManager;

    apo_test::CheckStatusAndDiagnostic(
        apo_charmm_context_get_force_manager(&forceManager, context.get()),
        APO_STATUS_NOT_INITIALIZED,
        "apo_charmm_context_get_force_manager: ForceManager is not set");
    CHECK(forceManager == nullptr);
  }

  SECTION("MissingVelocityState") {
    CharmmContextHandle context = MakeDefaultContext();
    const double velocities[3] = {0.0, 0.0, 0.0};
    apo_status status = APO_STATUS_OK;

    CHECK_NOTHROW((status = apo_charmm_context_set_velocities(context.get(),
                                                              velocities, 3)));
    apo_test::CheckNativeError(
        status, APO_STATUS_NOT_INITIALIZED, "NotInitialized",
        "apo_charmm_context_set_velocities",
        "Atom count must be set before assigning velocities",
        "src/CharmmContext.cu", "setVelocities");
  }

  SECTION("VelocityCountMismatch") {
    CharmmContextHandle context = MakeDefaultContext();
    context->object->setNumAtoms(2);
    const double velocities[3] = {0.0, 0.0, 0.0};
    apo_status status = APO_STATUS_OK;

    CHECK_NOTHROW((status = apo_charmm_context_set_velocities(context.get(),
                                                              velocities, 3)));
    apo_test::CheckNativeError(
        status, APO_STATUS_INVALID_ARGUMENT, "InvalidArgument",
        "apo_charmm_context_set_velocities",
        "Velocity count mismatch; expected 2, observed 1",
        "src/CharmmContext.cu", "setVelocities");
  }

  SECTION("ScalarValidationIsOwnedByForceManager") {
    CharmmContextHandle context = MakeDefaultContext();
    context->force_manager = std::make_shared<ForceManager>();
    context->object->setForceManager(context->force_manager);
    apo_status status = APO_STATUS_OK;

    CHECK_NOTHROW((status = apo_charmm_context_set_kappa(context.get(), -1.0)));
    apo_test::CheckNativeError(status, APO_STATUS_INVALID_ARGUMENT,
                               "InvalidArgument",
                               "apo_charmm_context_set_kappa",
                               "Kappa must be non-negative; observed -1.000000",
                               "src/ForceManager.cu", "setKappa");

    CHECK_NOTHROW(
        (status = apo_charmm_context_set_cutoff(context.get(), -1.0)));
    apo_test::CheckNativeError(status, APO_STATUS_INVALID_ARGUMENT,
                               "InvalidArgument",
                               "apo_charmm_context_set_cutoff",
                               "Cutoff must be positive; observed -1.000000",
                               "src/ForceManager.cu", "setCutoff");

    CHECK_NOTHROW(
        (status = apo_charmm_context_set_ctonnb(context.get(), -1.0)));
    apo_test::CheckNativeError(status, APO_STATUS_INVALID_ARGUMENT,
                               "InvalidArgument",
                               "apo_charmm_context_set_ctonnb",
                               "Ctonnb must be positive; observed -1.000000",
                               "src/ForceManager.cu", "setCtonnb");

    CHECK_NOTHROW(
        (status = apo_charmm_context_set_ctofnb(context.get(), -1.0)));
    apo_test::CheckNativeError(status, APO_STATUS_INVALID_ARGUMENT,
                               "InvalidArgument",
                               "apo_charmm_context_set_ctofnb",
                               "Ctofnb must be positive; observed -1.000000",
                               "src/ForceManager.cu", "setCtofnb");

    CHECK_NOTHROW(
        (status = apo_charmm_context_set_pme_spline_order(context.get(), 0)));
    apo_test::CheckNativeError(status, APO_STATUS_INVALID_ARGUMENT,
                               "InvalidArgument",
                               "apo_charmm_context_set_pme_spline_order",
                               "PME spline order must be positive; observed 0",
                               "src/ForceManager.cu", "setPmeSplineOrder");

    CHECK_NOTHROW((status = apo_charmm_context_set_vdw_type(context.get(), 0)));
    apo_test::CheckNativeError(
        status, APO_STATUS_INVALID_ARGUMENT, "InvalidArgument",
        "apo_charmm_context_set_vdw_type",
        "Van der Waals type must be in [1, 6]; observed 0",
        "src/ForceManager.cu", "setVdwType");
  }
}

TEST_CASE("CapiCharmmContextPreservesCachedStateOnNativeFailure") {
  CharmmContextHandle context = MakeDefaultContext();
  context->object->setNumAtoms(1);

  std::shared_ptr<CharmmPSF> oldPsf = std::make_shared<CharmmPSF>();
  oldPsf->setNumAtoms(1);
  context->psf = oldPsf;

  apo_charmm_psf replacementPsf;
  replacementPsf.object = std::make_shared<CharmmPSF>();
  replacementPsf.object->setNumAtoms(2);

  apo_status status = APO_STATUS_OK;
  CHECK_NOTHROW(
      (status = apo_charmm_context_set_psf(context.get(), &replacementPsf)));
  apo_test::CheckNativeError(
      status, APO_STATUS_INVALID_ARGUMENT, "InvalidArgument",
      "apo_charmm_context_set_psf",
      "Atom count mismatch between CharmmContext and CharmmPSF; expected 1, "
      "observed 2",
      "src/CharmmContext.cu", "setPsf");
  CHECK(context->psf == oldPsf);
}

TEST_CASE("CapiCharmmContextSuccessClearsStaleDiagnostic") {
  CharmmContextHandle context = MakeDefaultContext();

  REQUIRE(apo_charmm_context_get_num_atoms(nullptr, context.get()) ==
          APO_STATUS_INVALID_ARGUMENT);
  REQUIRE(std::string(apo_last_error()) ==
          "apo_charmm_context_get_num_atoms: num_atoms is NULL");

  int numAtoms = 17;
  CHECK(apo_charmm_context_get_num_atoms(&numAtoms, context.get()) ==
        APO_STATUS_OK);
  CHECK(numAtoms == -1);
  CHECK(std::string(apo_last_error()).empty() == true);
}
