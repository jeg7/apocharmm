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
#include "CharmmContext.h"
#include "CharmmCrd.h"
#include "CharmmPSF.h"
#include "CharmmParameters.h"
#include "CudaIntegrator.h"
#include "DcdSubscriber.h"
#include "ForceManager.h"
#include "apo_test_helpers.h"
#include "catch.hpp"

#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <limits>
#include <memory>
#include <streambuf>
#include <string>
#include <vector>

namespace {

constexpr double TIME_STEP = 0.001;
constexpr int REPORT_FREQUENCY = 2;
constexpr double FLOAT_TOLERANCE = 1.0e-6;

const std::vector<double> BOX_DIMENSIONS = {50.0, 50.0, 50.0};

struct DcdFrame {
  std::array<double, 6> xtal = {};
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> z;
};

struct DcdFile {
  int cord = 0;
  int numFrames = 0;
  int start = 0;
  int reportFrequency = 0;
  int numSteps = 0;
  int numDegreesOfFreedom = 0;
  int hasXtalData = 0;
  int version = 0;
  int numAtoms = 0;
  float timeStep = 0.0f;
  std::vector<DcdFrame> frames;
};

class FailingStreamBuffer : public std::streambuf {
protected:
  std::streamsize xsputn(const char *, const std::streamsize) override {
    return 0;
  }

  int_type overflow(const int_type) override { return traits_type::eof(); }

  int sync(void) override { return -1; }
};

class TestDcdIntegrator : public CudaIntegrator {
public:
  using CudaIntegrator::CudaIntegrator;

  void setCompletedSteps(const unsigned long long int step) {
    m_TotNumSteps = step;
  }
};

class TestDcdSubscriber : public DcdSubscriber {
public:
  using DcdSubscriber::DcdSubscriber;

  std::fstream &getOutputFile(void) { return m_FileStream; }
};

template <typename T> T ReadValue(std::ifstream &input) {
  T value{};
  input.read(reinterpret_cast<char *>(&value), sizeof(T));
  REQUIRE(input.good());

  return value;
}

template <typename T>
T ReadBufferValue(const std::vector<char> &buffer, const std::size_t offset) {
  REQUIRE(offset + sizeof(T) <= buffer.size());

  T value{};
  std::memcpy(static_cast<void *>(&value),
              static_cast<const void *>(buffer.data() + offset), sizeof(T));

  return value;
}

std::vector<char> ReadBytes(std::ifstream &input, const std::size_t count) {
  std::vector<char> buffer(count);
  input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
  REQUIRE(input.good());

  return buffer;
}

std::vector<float> ReadFloatBlock(std::ifstream &input, const int numAtoms) {
  const int blockSizeBytes = ReadValue<int>(input);
  REQUIRE(blockSizeBytes == numAtoms * static_cast<int>(sizeof(float)));

  std::vector<float> values(static_cast<std::size_t>(numAtoms));
  input.read(reinterpret_cast<char *>(values.data()), blockSizeBytes);
  REQUIRE(input.good());

  const int endBlockSizeBytes = ReadValue<int>(input);
  REQUIRE(endBlockSizeBytes == blockSizeBytes);

  return values;
}

DcdFile ReadDcdFile(const std::filesystem::path &filePath) {
  std::ifstream input(filePath, std::ios::binary);
  REQUIRE(input.good());

  DcdFile dcd;

  const int controlBlockSize = ReadValue<int>(input);
  REQUIRE(controlBlockSize == 84);

  const std::vector<char> controlBlock =
      ReadBytes(input, static_cast<std::size_t>(controlBlockSize));

  const int controlBlockEndSize = ReadValue<int>(input);
  REQUIRE(controlBlockEndSize == controlBlockSize);

  dcd.cord = ReadBufferValue<int>(controlBlock, 0);
  dcd.numFrames = ReadBufferValue<int>(controlBlock, 4);
  dcd.start = ReadBufferValue<int>(controlBlock, 8);
  dcd.reportFrequency = ReadBufferValue<int>(controlBlock, 12);
  dcd.numSteps = ReadBufferValue<int>(controlBlock, 16);
  dcd.numDegreesOfFreedom = ReadBufferValue<int>(controlBlock, 32);
  dcd.timeStep = ReadBufferValue<float>(controlBlock, 40);
  dcd.hasXtalData = ReadBufferValue<int>(controlBlock, 44);
  dcd.version = ReadBufferValue<int>(controlBlock, 80);

  const int titleBlockSize = ReadValue<int>(input);
  REQUIRE(titleBlockSize == 164);
  ReadBytes(input, static_cast<std::size_t>(titleBlockSize));
  const int titleEndBlocksize = ReadValue<int>(input);
  REQUIRE(titleEndBlocksize == titleBlockSize);

  const int atomBlockSize = ReadValue<int>(input);
  REQUIRE(atomBlockSize == 4);
  dcd.numAtoms = ReadValue<int>(input);
  const int atomEndBlockSize = ReadValue<int>(input);
  REQUIRE(atomEndBlockSize == atomBlockSize);

  while (input.peek() != std::ifstream::traits_type::eof()) {
    DcdFrame frame;

    const int xtalBlockSize = ReadValue<int>(input);
    REQUIRE(xtalBlockSize == 6 * static_cast<int>(sizeof(double)));
    for (std::size_t i = 0; i < frame.xtal.size(); i++)
      frame.xtal[i] = ReadValue<double>(input);
    const int xtalEndBlockSize = ReadValue<int>(input);
    REQUIRE(xtalEndBlockSize == xtalBlockSize);

    frame.x = ReadFloatBlock(input, dcd.numAtoms);
    frame.y = ReadFloatBlock(input, dcd.numAtoms);
    frame.z = ReadFloatBlock(input, dcd.numAtoms);

    dcd.frames.push_back(frame);
  }

  return dcd;
}

std::shared_ptr<CharmmContext> CreateContext(void) {
  auto prm = std::make_shared<CharmmParameters>(apo_test::GetTopparDir() /
                                                "toppar_water_ions.str");
  auto psf =
      std::make_shared<CharmmPSF>(apo_test::GetDataDir() / "nacl_pair.psf");
  auto crd =
      std::make_shared<CharmmCrd>(apo_test::GetDataDir() / "nacl_pair.cor");

  auto ctx = std::make_shared<CharmmContext>(psf, prm);
  ctx->setBoxDimensions(BOX_DIMENSIONS);
  ctx->setCoordinates(crd);
  ctx->useHolonomicConstraints(false);

  return ctx;
}

void CheckFrameMatchesContext(const DcdFrame &frame,
                              std::shared_ptr<CharmmContext> ctx) {
  ctx->getCoordinatesChargesSP().transferToHost();

  const std::vector<float4> &xyzq =
      ctx->getCoordinatesChargesSP().getHostArray();

  REQUIRE(frame.x.size() == xyzq.size());
  REQUIRE(frame.y.size() == xyzq.size());
  REQUIRE(frame.z.size() == xyzq.size());

  for (std::size_t i = 0; i < xyzq.size(); i++) {
    CHECK(frame.x[i] == Approx(xyzq[i].x).margin(FLOAT_TOLERANCE));
    CHECK(frame.y[i] == Approx(xyzq[i].y).margin(FLOAT_TOLERANCE));
    CHECK(frame.z[i] == Approx(xyzq[i].z).margin(FLOAT_TOLERANCE));
  }

  return;
}

} // namespace

TEST_CASE("DcdSubscriberConstructorsAndReportFrequency") {
  SECTION("DefaultReportFrequencyConstructor") {
    const std::filesystem::path filePath = "tmp_dcd_subscriber_default.dcd";
    apo_test::RemoveIfExists(filePath);

    DcdSubscriber subscriber(filePath);

    CHECK(subscriber.getFilePath() == filePath);
    CHECK(subscriber.getReportFrequency() == 1000);

    subscriber.setReportFrequency(REPORT_FREQUENCY);
    CHECK(subscriber.getReportFrequency() == REPORT_FREQUENCY);

    std::ifstream file(filePath, std::ios::binary);
    CHECK(file.good());

    apo_test::RemoveIfExists(filePath);
  }

  SECTION("ExplicitReportFrequencyConstructor") {
    const std::string fileName = "tmp_dcd_subscriber_frequency.dcd";
    apo_test::RemoveIfExists(fileName);

    DcdSubscriber subscriber(fileName, REPORT_FREQUENCY);

    CHECK(subscriber.getFilePath() == fileName);
    CHECK(subscriber.getReportFrequency() == REPORT_FREQUENCY);

    apo_test::RemoveIfExists(fileName);
  }

  SECTION("RejectsEmptyOutputFileName") {
    apo_test::CheckApoCharmmError([]() { (void)DcdSubscriber(""); },
                                  ApoCharmmErrorCode::InvalidArgument,
                                  "Output file name must not be empty");
  }

  SECTION("RejectsZeroReportFrequency") {
    apo_test::CheckApoCharmmError(
        []() { (void)DcdSubscriber("tmp_dcd_zero_frequency.dcd", 0); },
        ApoCharmmErrorCode::InvalidArgument,
        "Subscriber report frequency must be positive; observed 0");
  }

  SECTION("RejectsNegativeReportFrequency") {
    apo_test::CheckApoCharmmError(
        []() { (void)DcdSubscriber("tmp_dcd_negative_frequency.dcd", -1); },
        ApoCharmmErrorCode::InvalidArgument,
        "Subscriber report frequency must be positive; observed -1");
  }

  SECTION("RejectsMissingOutputDirectory") {
    apo_test::CheckApoCharmmError(
        []() { (void)DcdSubscriber("missing_dcd_subscriber_dir/output.dcd"); },
        ApoCharmmErrorCode::InvalidArgument,
        "Output directory does not exist: missing_dcd_subscriber_dir");
  }

  SECTION("RejectsMissingOutputDirectoryWithReportFrequency") {
    apo_test::CheckApoCharmmError(
        []() {
          (void)DcdSubscriber("missing_dcd_subscriber_dir/output.dcd",
                              REPORT_FREQUENCY);
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Output directory does not exist: missing_dcd_subscriber_dir");
  }

  SECTION("RejectsOutputOpenFailure") {
    apo_test::CheckApoCharmmError([]() { (void)DcdSubscriber("."); },
                                  ApoCharmmErrorCode::Runtime,
                                  "Failed to open DCD file for writing: .");
  }

  SECTION("RejectsOutputOpenFailureWithReportFrequency") {
    apo_test::CheckApoCharmmError(
        []() { (void)DcdSubscriber(".", REPORT_FREQUENCY); },
        ApoCharmmErrorCode::Runtime, "Failed to open DCD file for writing: .");
  }

  SECTION("OpenFileRequiresOutputFileName") {
    const std::string fileName = "tmp_dcd_open_requires_name.dcd";
    apo_test::RemoveIfExists(fileName);

    DcdSubscriber subscriber(fileName);
    subscriber.getFilePath().clear();

    apo_test::CheckApoCharmmError([&]() { subscriber.openFile(); },
                                  ApoCharmmErrorCode::NotInitialized,
                                  "DcdSubscriber output file name is not set");

    apo_test::RemoveIfExists(fileName);
  }
}

TEST_CASE("DcdSubscriberUpdateValidatesRequiredState") {
  SECTION("RequiresOutputFileName") {
    const std::string fileName = "tmp_dcd_update_requires_name.dcd";
    apo_test::RemoveIfExists(fileName);

    DcdSubscriber subscriber(fileName);
    subscriber.getFilePath().clear();

    apo_test::CheckApoCharmmError(
        [&]() { subscriber.update(); }, ApoCharmmErrorCode::NotInitialized,
        "DcdSubscriber requires an output file before update");

    apo_test::RemoveIfExists(fileName);
  }

  SECTION("RequiresCharmmContext") {
    const std::string fileName = "tmp_dcd_update_requires_context.dcd";
    apo_test::RemoveIfExists(fileName);

    auto integrator = std::make_shared<TestDcdIntegrator>(TIME_STEP);

    DcdSubscriber subscriber(fileName, REPORT_FREQUENCY);
    subscriber.setIntegrator(integrator);

    apo_test::CheckApoCharmmError(
        [&]() { subscriber.update(); }, ApoCharmmErrorCode::NotInitialized,
        "DcdSubscriber requires a CharmmContext before update");

    apo_test::RemoveIfExists(fileName);
  }

  SECTION("RequiresIntegrator") {
    const std::string fileName = "tmp_dcd_update_requires_integrator.dcd";
    apo_test::RemoveIfExists(fileName);

    auto ctx = CreateContext();

    DcdSubscriber subscriber(fileName, REPORT_FREQUENCY);
    subscriber.setCharmmContext(ctx);

    apo_test::CheckApoCharmmError(
        [&]() { subscriber.update(); }, ApoCharmmErrorCode::NotInitialized,
        "DcdSubscriber requires an integrator before update");

    apo_test::RemoveIfExists(fileName);
  }

  SECTION("RequiresOpenOutputFile") {
    const std::string fileName = "tmp_dcd_update_requires_open_file.dcd";
    apo_test::RemoveIfExists(fileName);

    auto ctx = CreateContext();
    auto integrator = std::make_shared<TestDcdIntegrator>(TIME_STEP);

    TestDcdSubscriber subscriber(fileName, REPORT_FREQUENCY);
    subscriber.setCharmmContext(ctx);
    subscriber.setIntegrator(integrator);
    subscriber.getOutputFile().close();

    apo_test::CheckApoCharmmError(
        [&]() { subscriber.update(); }, ApoCharmmErrorCode::NotInitialized,
        "DcdSubscriber output file is not open for writing: " + fileName);

    apo_test::RemoveIfExists(fileName);
  }

  SECTION("RequiresPositiveBoxDimensions") {
    const std::string fileName = "tmp_dcd_update_requires_box.dcd";
    apo_test::RemoveIfExists(fileName);

    auto ctx = CreateContext();
    ctx->getBoxDimensions()[0] = 0.0;
    auto integrator = std::make_shared<TestDcdIntegrator>(TIME_STEP);

    DcdSubscriber subscriber(fileName, REPORT_FREQUENCY);
    subscriber.setCharmmContext(ctx);
    subscriber.setIntegrator(integrator);

    apo_test::CheckApoCharmmError(
        [&]() { subscriber.update(); }, ApoCharmmErrorCode::NotInitialized,
        "DcdSubscriber requires three positive box dimensions before update");

    apo_test::RemoveIfExists(fileName);
  }

  SECTION("RequiresAtLeastOneAtom") {
    const std::string fileName = "tmp_dcd_update_requires_atoms.dcd";
    apo_test::RemoveIfExists(fileName);

    auto ctx = std::make_shared<CharmmContext>();
    ctx->setBoxDimensions(BOX_DIMENSIONS);
    auto integrator = std::make_shared<TestDcdIntegrator>(TIME_STEP);

    DcdSubscriber subscriber(fileName, REPORT_FREQUENCY);
    subscriber.setCharmmContext(ctx);
    subscriber.setIntegrator(integrator);

    apo_test::CheckApoCharmmError(
        [&]() { subscriber.update(); }, ApoCharmmErrorCode::NotInitialized,
        "DcdSubscriber requires at least one atom before update");

    apo_test::RemoveIfExists(fileName);
  }
}

TEST_CASE("DcdSubscriberReportsMetadataAndOutputFailures") {
  SECTION("RejectsFrameMetadataOverflow") {
    const std::string fileName = "tmp_dcd_metadata_overflow.dcd";
    apo_test::RemoveIfExists(fileName);

    auto ctx = CreateContext();
    auto integrator = std::make_shared<TestDcdIntegrator>(TIME_STEP);

    DcdSubscriber subscriber(fileName, std::numeric_limits<int>::max());
    subscriber.setCharmmContext(ctx);
    subscriber.setIntegrator(integrator);

    subscriber.update();
    const auto maxStep =
        static_cast<unsigned long long int>(std::numeric_limits<int>::max());
    integrator->setCompletedSteps(maxStep);
    subscriber.update();
    integrator->setCompletedSteps(2ULL * maxStep);

    apo_test::CheckApoCharmmError(
        [&]() { subscriber.update(); }, ApoCharmmErrorCode::Runtime,
        "DcdSubscriber frame metadata exceeds DCD integer range");

    apo_test::RemoveIfExists(fileName);
  }

  SECTION("ReportsHeaderWriteFailure") {
    const std::string fileName = "tmp_dcd_header_write_failure.dcd";
    apo_test::RemoveIfExists(fileName);

    auto ctx = CreateContext();
    auto integrator = std::make_shared<TestDcdIntegrator>(TIME_STEP);
    FailingStreamBuffer failingBuffer;

    TestDcdSubscriber subscriber(fileName, REPORT_FREQUENCY);
    subscriber.setCharmmContext(ctx);
    subscriber.setIntegrator(integrator);
    subscriber.getOutputFile().std::basic_ios<char>::rdbuf(&failingBuffer);

    apo_test::CheckApoCharmmError([&]() { subscriber.update(); },
                                  ApoCharmmErrorCode::Runtime,
                                  "Failed to write DCD header: " + fileName);

    apo_test::RemoveIfExists(fileName);
  }

  SECTION("ReportsFrameWriteFailure") {
    const std::string fileName = "tmp_dcd_frame_write_failure.dcd";
    apo_test::RemoveIfExists(fileName);

    auto ctx = CreateContext();
    auto integrator = std::make_shared<TestDcdIntegrator>(TIME_STEP);
    FailingStreamBuffer failingBuffer;

    TestDcdSubscriber subscriber(fileName, REPORT_FREQUENCY);
    subscriber.setCharmmContext(ctx);
    subscriber.setIntegrator(integrator);

    subscriber.update();
    integrator->setCompletedSteps(REPORT_FREQUENCY);
    subscriber.getOutputFile().std::basic_ios<char>::rdbuf(&failingBuffer);

    apo_test::CheckApoCharmmError([&]() { subscriber.update(); },
                                  ApoCharmmErrorCode::Runtime,
                                  "Failed to write DCD frame: " + fileName);

    apo_test::RemoveIfExists(fileName);
  }
}

TEST_CASE("DcdSubscriberOpenFileResetsTrajectoryState") {
  const std::string fileName = "tmp_dcd_reopen.dcd";
  apo_test::RemoveIfExists(fileName);

  auto ctx = CreateContext();
  auto integrator = std::make_shared<TestDcdIntegrator>(TIME_STEP);

  DcdSubscriber subscriber(fileName, REPORT_FREQUENCY);
  subscriber.setCharmmContext(ctx);
  subscriber.setIntegrator(integrator);

  subscriber.update();
  subscriber.openFile();
  subscriber.update();

  const DcdFile dcd = ReadDcdFile(fileName);
  CHECK(dcd.numFrames == 1);
  CHECK(dcd.numSteps == 0);
  REQUIRE(dcd.frames.size() == 1);

  apo_test::RemoveIfExists(fileName);
}

TEST_CASE("DcdSubscriberUpdateWritesOneFrame") {
  const std::string fileName = "tmp_dcd_subscriber_one_frame.dcd";
  apo_test::RemoveIfExists(fileName);

  auto ctx = CreateContext();
  auto integrator = std::make_shared<TestDcdIntegrator>(TIME_STEP);

  DcdSubscriber subscriber(fileName, REPORT_FREQUENCY);
  subscriber.setCharmmContext(ctx);
  subscriber.setIntegrator(integrator);

  apo_test::CheckApoCharmmError([&]() { subscriber.setCharmmContext(ctx); },
                                ApoCharmmErrorCode::InvalidArgument,
                                "Subscriber already has a CharmmContext");
  apo_test::CheckApoCharmmError([&]() { subscriber.setIntegrator(integrator); },
                                ApoCharmmErrorCode::InvalidArgument,
                                "Subscriber already has an Integrator");

  subscriber.update();

  const DcdFile dcd = ReadDcdFile(fileName);

  CHECK(dcd.cord == 1146244931);
  CHECK(dcd.numFrames == 1);
  CHECK(dcd.start == 0);
  CHECK(dcd.reportFrequency == REPORT_FREQUENCY);
  CHECK(dcd.numSteps == 0);
  CHECK(dcd.numDegreesOfFreedom == ctx->getNumDegreesOfFreedom());
  CHECK(dcd.timeStep == Approx(static_cast<float>(TIME_STEP / 0.0488882129))
                            .margin(FLOAT_TOLERANCE));
  CHECK(dcd.hasXtalData == 1);
  CHECK(dcd.version == 52);
  CHECK(dcd.numAtoms == ctx->getNumAtoms());

  REQUIRE(dcd.frames.size() == 1);
  CHECK(dcd.frames[0].xtal[0] == Approx(BOX_DIMENSIONS[0]));
  CHECK(dcd.frames[0].xtal[2] == Approx(BOX_DIMENSIONS[1]));
  CHECK(dcd.frames[0].xtal[5] == Approx(BOX_DIMENSIONS[2]));
  CheckFrameMatchesContext(dcd.frames[0], ctx);

  apo_test::RemoveIfExists(fileName);
}

TEST_CASE("DcdSubscriberUpdatePatchesHeaderForMultipleFrames") {
  const std::string fileName = "tmp_dcd_subscriber_two_frames.dcd";
  apo_test::RemoveIfExists(fileName);

  auto ctx = CreateContext();
  auto integrator = std::make_shared<TestDcdIntegrator>(TIME_STEP);

  DcdSubscriber subscriber(fileName, REPORT_FREQUENCY);
  subscriber.setCharmmContext(ctx);
  subscriber.setIntegrator(integrator);

  subscriber.update();
  integrator->setCompletedSteps(REPORT_FREQUENCY);
  subscriber.update();

  const DcdFile dcd = ReadDcdFile(fileName);

  CHECK(dcd.numFrames == 2);
  CHECK(dcd.reportFrequency == REPORT_FREQUENCY);
  CHECK(dcd.numSteps == REPORT_FREQUENCY);
  CHECK(dcd.numAtoms == ctx->getNumAtoms());

  REQUIRE(dcd.frames.size() == 2);
  CheckFrameMatchesContext(dcd.frames[0], ctx);
  CheckFrameMatchesContext(dcd.frames[1], ctx);

  apo_test::RemoveIfExists(fileName);
}

TEST_CASE("DcdSubscriberPreservesFirstStepAndRejectsMetadataChanges") {
  const std::string fileName = "tmp_dcd_metadata_consistency.dcd";
  apo_test::RemoveIfExists(fileName);
  auto ctx = CreateContext();
  auto integrator = std::make_shared<TestDcdIntegrator>(TIME_STEP);
  DcdSubscriber subscriber(fileName, REPORT_FREQUENCY);
  subscriber.setCharmmContext(ctx);
  subscriber.setIntegrator(integrator);
  subscriber.onPropagationStart(98, 5);
  integrator->setCompletedSteps(100);
  subscriber.update();
  integrator->setCompletedSteps(100 + REPORT_FREQUENCY);
  subscriber.update();

  const DcdFile dcd = ReadDcdFile(fileName);
  CHECK(dcd.start == 100);
  CHECK(dcd.numFrames == 2);
  CHECK(dcd.numSteps == 5);

  SECTION("DuplicateFrameStep") {
    apo_test::CheckApoCharmmError(
        [&]() { subscriber.update(); }, ApoCharmmErrorCode::InvalidArgument,
        "DcdSubscriber frames must be separated by the reporting frequency");
  }
  SECTION("IrregularFrameStep") {
    subscriber.onPropagationStart(102, 8);
    integrator->setCompletedSteps(105);
    apo_test::CheckApoCharmmError(
        [&]() { subscriber.update(); }, ApoCharmmErrorCode::InvalidArgument,
        "DcdSubscriber frames must be separated by the reporting frequency");
  }
  SECTION("ChangedTimeStep") {
    subscriber.onPropagationStart(102, 8);
    integrator->setCompletedSteps(104);
    integrator->setTimeStep(TIME_STEP * 2.0);
    apo_test::CheckApoCharmmError(
        [&]() { subscriber.update(); }, ApoCharmmErrorCode::InvalidArgument,
        "DcdSubscriber trajectory metadata changed; open a new DCD file");
  }
  SECTION("ChangedReportingFrequency") {
    subscriber.onPropagationStart(102, 8);
    integrator->setCompletedSteps(104);
    subscriber.setReportFrequency(REPORT_FREQUENCY * 2);
    apo_test::CheckApoCharmmError(
        [&]() { subscriber.update(); }, ApoCharmmErrorCode::InvalidArgument,
        "DcdSubscriber trajectory metadata changed; open a new DCD file");
  }
  apo_test::RemoveIfExists(fileName);
}

TEST_CASE("DcdSubscriberPreservesPlannedUnsavedTailAndShortRequests") {
  const std::string fileName = "tmp_dcd_planned_tail.dcd";
  apo_test::RemoveIfExists(fileName);
  auto ctx = CreateContext();
  auto integrator = std::make_shared<TestDcdIntegrator>(TIME_STEP);
  DcdSubscriber subscriber(fileName, 10);
  subscriber.setCharmmContext(ctx);
  subscriber.setIntegrator(integrator);
  subscriber.onPropagationStart(0, 25);
  integrator->setCompletedSteps(10);
  subscriber.update();
  integrator->setCompletedSteps(20);
  subscriber.update();
  const DcdFile dcd = ReadDcdFile(fileName);
  CHECK(dcd.start == 10);
  CHECK(dcd.numFrames == 2);
  CHECK(dcd.numSteps == 25);

  // No frame is written by this shorter-than-frequency request. Its plan is
  // visible immediately, even if another subscriber or a later step fails.
  subscriber.onPropagationStart(25, 3);
  const DcdFile shortRequest = ReadDcdFile(fileName);
  CHECK(shortRequest.numFrames == 2);
  CHECK(shortRequest.numSteps == 28);

  // Like CHARMM's MOD(ISTEP,NSAVC) with ISTEP counted from the run origin, the
  // integrator schedules the next frame at step 30 regardless of how the
  // remaining steps are split across calls. The plan now extends to step 50.
  subscriber.onPropagationStart(28, 22);
  const DcdFile extendedRequest = ReadDcdFile(fileName);
  CHECK(extendedRequest.numFrames == 2);
  CHECK(extendedRequest.numSteps == 50);
  integrator->setCompletedSteps(30);
  subscriber.update();
  const DcdFile thirdFrame = ReadDcdFile(fileName);
  CHECK(thirdFrame.start == 10);
  CHECK(thirdFrame.numFrames == 3);
  CHECK(thirdFrame.numSteps == 50);
  apo_test::RemoveIfExists(fileName);
}

TEST_CASE("DcdSubscriberExtendsAlignedRunsAndClearsPlanOnReopen") {
  const std::string fileName = "tmp_dcd_multiple_runs.dcd";
  apo_test::RemoveIfExists(fileName);
  auto ctx = CreateContext();
  auto integrator = std::make_shared<TestDcdIntegrator>(TIME_STEP);
  DcdSubscriber subscriber(fileName, 10);
  subscriber.setCharmmContext(ctx);
  subscriber.setIntegrator(integrator);
  subscriber.onPropagationStart(0, 20);
  integrator->setCompletedSteps(10);
  subscriber.update();
  integrator->setCompletedSteps(20);
  subscriber.update();
  subscriber.onPropagationStart(20, 15);
  integrator->setCompletedSteps(30);
  subscriber.update();
  const DcdFile dcd = ReadDcdFile(fileName);
  CHECK(dcd.start == 10);
  CHECK(dcd.numFrames == 3);
  CHECK(dcd.numSteps == 35);

  subscriber.openFile();
  integrator->setCompletedSteps(35);
  subscriber.update();
  const DcdFile manualFirst = ReadDcdFile(fileName);
  CHECK(manualFirst.start == 35);
  CHECK(manualFirst.numSteps == 0);
  integrator->setCompletedSteps(45);
  subscriber.update();
  const DcdFile manualSecond = ReadDcdFile(fileName);
  CHECK(manualSecond.numFrames == 2);
  CHECK(manualSecond.numSteps == 10);
  apo_test::RemoveIfExists(fileName);
}

TEST_CASE("DcdSubscriberValidatesPropagationAnnouncements") {
  const std::string fileName = "tmp_dcd_invalid_plan.dcd";
  apo_test::RemoveIfExists(fileName);
  auto ctx = CreateContext();
  auto integrator = std::make_shared<TestDcdIntegrator>(TIME_STEP);
  DcdSubscriber subscriber(fileName, REPORT_FREQUENCY);
  subscriber.setCharmmContext(ctx);
  subscriber.setIntegrator(integrator);

  apo_test::CheckApoCharmmError(
      [&]() { subscriber.onPropagationStart(0, 0); },
      ApoCharmmErrorCode::InvalidArgument,
      "DcdSubscriber propagation request must be positive");
  apo_test::CheckApoCharmmError(
      [&]() {
        subscriber.onPropagationStart(
            std::numeric_limits<unsigned long long int>::max(), 1);
      },
      ApoCharmmErrorCode::InvalidArgument,
      "DcdSubscriber propagation request exceeds the step counter range");
  subscriber.onPropagationStart(10, 10);
  integrator->setCompletedSteps(12);
  subscriber.update();
  apo_test::CheckApoCharmmError(
      [&]() { subscriber.onPropagationStart(11, 10); },
      ApoCharmmErrorCode::InvalidArgument,
      "DcdSubscriber integration step metadata is inconsistent");
  apo_test::CheckApoCharmmError(
      [&]() { subscriber.onPropagationStart(9, 1); },
      ApoCharmmErrorCode::InvalidArgument,
      "DcdSubscriber integration step metadata is inconsistent");

  // Clearing the origin after frames exist (what a restart load does) is a
  // new run; CHARMM would open a new trajectory unit.
  subscriber.setStartStep(std::nullopt);
  apo_test::CheckApoCharmmError(
      [&]() { subscriber.onPropagationStart(12, 10); },
      ApoCharmmErrorCode::InvalidArgument,
      "DcdSubscriber run origin was reset; open a new DCD file");
  apo_test::CheckApoCharmmError(
      [&]() { subscriber.update(); }, ApoCharmmErrorCode::InvalidArgument,
      "DcdSubscriber run origin was reset; open a new DCD file");

  subscriber.openFile();
  subscriber.onPropagationStart(0, std::numeric_limits<int>::max());
  apo_test::CheckApoCharmmError(
      [&]() {
        subscriber.onPropagationStart(static_cast<unsigned long long int>(
                                          std::numeric_limits<int>::max()),
                                      1);
      },
      ApoCharmmErrorCode::Runtime,
      "DcdSubscriber frame metadata exceeds DCD integer range");
  apo_test::RemoveIfExists(fileName);
}

TEST_CASE("DcdSubscriberUsesFirstAnnouncementAsRunOrigin") {
  const std::string fileName = "tmp_dcd_initial_short_run.dcd";
  apo_test::RemoveIfExists(fileName);
  auto ctx = CreateContext();
  auto integrator = std::make_shared<TestDcdIntegrator>(TIME_STEP);
  DcdSubscriber subscriber(fileName, 10);
  subscriber.setCharmmContext(ctx);
  subscriber.setIntegrator(integrator);
  // A 5-step request followed by a 25-step request is one 30-step CHARMM run:
  // frames at steps 10, 20, and 30 with START=10 and NSTEP=30.
  subscriber.onPropagationStart(0, 5);
  CHECK(subscriber.getStartStep().value() == 0ULL);
  subscriber.onPropagationStart(5, 25);
  CHECK(subscriber.getStartStep().value() == 0ULL);
  integrator->setCompletedSteps(10);
  subscriber.update();
  const DcdFile dcd = ReadDcdFile(fileName);
  CHECK(dcd.start == 10);
  CHECK(dcd.numFrames == 1);
  CHECK(dcd.numSteps == 30);
  integrator->setCompletedSteps(20);
  subscriber.update();
  const DcdFile second = ReadDcdFile(fileName);
  CHECK(second.numFrames == 2);
  CHECK(second.numSteps == 30);

  // Reopening starts a new run at the step count first observed afterwards,
  // like a CHARMM run restarted at NPRIV=1005 saving at 1015, 1025, ...
  subscriber.openFile();
  CHECK_FALSE(subscriber.getStartStep().has_value());
  subscriber.onPropagationStart(1005, 20);
  CHECK(subscriber.getStartStep().value() == 1005ULL);
  integrator->setCompletedSteps(1015);
  subscriber.update();
  const DcdFile restarted = ReadDcdFile(fileName);
  CHECK(restarted.start == 1015);
  CHECK(restarted.numFrames == 1);
  CHECK(restarted.numSteps == 20);
  apo_test::RemoveIfExists(fileName);
}
