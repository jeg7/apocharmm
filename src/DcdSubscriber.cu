// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#include "DcdSubscriber.h"

#include "ApoCharmmError.h"
#include "CharmmContext.h"
#include "Constants.h"
#include "CudaIntegrator.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <vector>

static_assert(CHAR_BIT == 8 && sizeof(std::int32_t) == 4,
              "DCD output requires 8-bit bytes and 32-bit integers");
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "DCD output requires IEEE 754 binary32 coordinates");
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
              "DCD output requires IEEE 754 binary64 cell values");

// Every count in a DCD control record is a signed 32-bit integer
static constexpr unsigned long long int MAX_DCD_INTEGER =
    static_cast<unsigned long long int>(
        std::numeric_limits<std::int32_t>::max());

DcdSubscriber::DcdSubscriber(const std::filesystem::path &filePath)
    : Subscriber() {
  m_NumFramesWritten = 0;
  m_IsHeaderWritten = false;
  m_HeaderReportFrequency = 0;
  m_HeaderNumAtoms = 0;
  m_HeaderNumDegreesOfFreedom = 0;
  m_HeaderTimeStep = 0.0;
  m_LastStepWritten = 0;
  m_PlannedEndStep = 0;
  this->setFilePath(filePath);
  this->openFile();
}

DcdSubscriber::DcdSubscriber(const std::filesystem::path &filePath,
                             const int reportFrequency)
    : Subscriber() {
  m_NumFramesWritten = 0;
  m_IsHeaderWritten = false;
  m_HeaderReportFrequency = 0;
  m_HeaderNumAtoms = 0;
  m_HeaderNumDegreesOfFreedom = 0;
  m_HeaderTimeStep = 0.0;
  m_LastStepWritten = 0;
  m_PlannedEndStep = 0;
  this->setReportFrequency(reportFrequency);
  this->setFilePath(filePath);
  this->openFile();
}

void DcdSubscriber::onPropagationStart(const unsigned long long int startStep,
                                       const int numSteps) {
  APOCHARMM_REQUIRE(numSteps > 0, ApoCharmmErrorCode::InvalidArgument,
                    "DcdSubscriber propagation request must be positive");

  const unsigned long long int requestedSteps =
      static_cast<unsigned long long int>(numSteps);
  APOCHARMM_REQUIRE(
      requestedSteps <=
          std::numeric_limits<unsigned long long int>::max() - startStep,
      ApoCharmmErrorCode::InvalidArgument,
      "DcdSubscriber propagation request exceeds the step counter range");

  // An empty origin means a new run: the first announcement after subscription
  // or after reopening. Once frames exist, a new run (for example after a
  // restart load) needs a new file, as CHARMM opens a new trajectory unit.
  APOCHARMM_REQUIRE(m_StartStep.has_value() || !m_IsHeaderWritten,
                    ApoCharmmErrorCode::InvalidArgument,
                    "DcdSubscriber run origin was reset; open a new DCD file");

  if (!m_StartStep.has_value())
    m_StartStep = startStep;

  const unsigned long long int runStart = *m_StartStep;
  APOCHARMM_REQUIRE(startStep >= runStart && startStep >= m_LastStepWritten,
                    ApoCharmmErrorCode::InvalidArgument,
                    "DcdSubscriber integration step metadata is inconsistent");

  const unsigned long long int plannedEnd = startStep + requestedSteps;
  const unsigned long long int plannedSpan = plannedEnd - runStart;
  APOCHARMM_REQUIRE(plannedSpan <= MAX_DCD_INTEGER, ApoCharmmErrorCode::Runtime,
                    "DcdSubscriber frame metadata exceeds DCD integer range");

  if (m_IsHeaderWritten) {
    APOCHARMM_REQUIRE(
        m_ReportFrequency == m_HeaderReportFrequency,
        ApoCharmmErrorCode::InvalidArgument,
        "DcdSubscriber trajectory metadata changed; open a new DCD file");
    APOCHARMM_REQUIRE(m_FileStream.is_open(),
                      ApoCharmmErrorCode::NotInitialized,
                      "DcdSubscriber output file is not open for writing: " +
                          m_FilePath.string());

    // CHARMM records NSTEP as the planned run length; extend it to the newly
    // announced endpoint even when this request produces no frame.
    const std::int32_t numStepsWritten = static_cast<std::int32_t>(plannedSpan);

    m_FileStream.seekp(20);
    m_FileStream.write(reinterpret_cast<const char *>(&numStepsWritten),
                       sizeof(std::int32_t));
    m_FileStream.seekp(0, std::ofstream::end);
    m_FileStream.flush();
    APOCHARMM_REQUIRE(m_FileStream.good(), ApoCharmmErrorCode::Runtime,
                      "Failed to update DCD run metadata: " +
                          m_FilePath.string());
  }

  m_PlannedEndStep = plannedEnd;

  return;
}

void DcdSubscriber::update(void) {
  APOCHARMM_REQUIRE(!m_FilePath.empty(), ApoCharmmErrorCode::NotInitialized,
                    "DcdSubscriber requires an output file before update");

  APOCHARMM_REQUIRE(m_CharmmContext != nullptr,
                    ApoCharmmErrorCode::NotInitialized,
                    "DcdSubscriber requires a CharmmContext before update");

  APOCHARMM_REQUIRE(m_Integrator != nullptr, ApoCharmmErrorCode::NotInitialized,
                    "DcdSubscriber requires an integrator before update");

  APOCHARMM_REQUIRE(m_FileStream.is_open(), ApoCharmmErrorCode::NotInitialized,
                    "DcdSubscriber output file is not open for writing: " +
                        m_FilePath.string());

  const std::vector<double> &boxDimensions =
      m_CharmmContext->getBoxDimensions();

  APOCHARMM_REQUIRE(
      boxDimensions.size() == 3 && std::isfinite(boxDimensions[0]) &&
          std::isfinite(boxDimensions[1]) && std::isfinite(boxDimensions[2]) &&
          boxDimensions[0] > 0.0 && boxDimensions[1] > 0.0 &&
          boxDimensions[2] > 0.0,
      ApoCharmmErrorCode::NotInitialized,
      "DcdSubscriber requires three positive box dimensions before update");

  const int numAtoms = m_CharmmContext->getNumAtoms();

  APOCHARMM_REQUIRE(numAtoms > 0, ApoCharmmErrorCode::NotInitialized,
                    "DcdSubscriber requires at least one atom before update");

  APOCHARMM_REQUIRE(
      numAtoms <= std::numeric_limits<std::int32_t>::max() / 4,
      ApoCharmmErrorCode::Runtime,
      "DcdSubscriber coordinate record exceeds DCD integer range");

  const unsigned long long int currentStep = m_Integrator->getTotNumSteps();

  // A manual snapshot without an announced run starts the run at this step
  APOCHARMM_REQUIRE(m_StartStep.has_value() || !m_IsHeaderWritten,
                    ApoCharmmErrorCode::InvalidArgument,
                    "DcdSubscriber run origin was reset; open a new DCD file");

  if (!m_StartStep.has_value())
    m_StartStep = currentStep;

  const unsigned long long int fileStart = *m_StartStep;
  const unsigned long long int plannedEnd =
      std::max(m_PlannedEndStep, currentStep);
  APOCHARMM_REQUIRE(fileStart <= currentStep,
                    ApoCharmmErrorCode::InvalidArgument,
                    "DcdSubscriber integration step metadata is inconsistent");

  if (m_IsHeaderWritten) {
    APOCHARMM_REQUIRE(
        m_ReportFrequency == m_HeaderReportFrequency &&
            m_Integrator->getTimeStep() == m_HeaderTimeStep &&
            numAtoms == m_HeaderNumAtoms &&
            m_CharmmContext->getNumDegreesOfFreedom() ==
                m_HeaderNumDegreesOfFreedom,
        ApoCharmmErrorCode::InvalidArgument,
        "DcdSubscriber trajectory metadata changed; open a new DCD file");
    APOCHARMM_REQUIRE(
        currentStep >= m_LastStepWritten &&
            currentStep - m_LastStepWritten ==
                static_cast<unsigned long long int>(m_HeaderReportFrequency),
        ApoCharmmErrorCode::InvalidArgument,
        "DcdSubscriber frames must be separated by the reporting frequency");
  }

  const unsigned long long int nextNumFramesWritten =
      static_cast<unsigned long long int>(m_NumFramesWritten) + 1ULL;
  const unsigned long long int nextNumStepsWritten = plannedEnd - fileStart;

  APOCHARMM_REQUIRE(nextNumFramesWritten <= MAX_DCD_INTEGER &&
                        nextNumStepsWritten <= MAX_DCD_INTEGER,
                    ApoCharmmErrorCode::Runtime,
                    "DcdSubscriber frame metadata exceeds DCD integer range");

  // Write header if needed
  if (!m_IsHeaderWritten)
    this->writeHeader();

  this->writeXtalData();

  // Write coordinate data
  this->writeCoordData();

  // Update header to account for new data that was written
  const std::int32_t numFramesWritten =
      static_cast<std::int32_t>(nextNumFramesWritten);
  const std::int32_t numStepsWritten =
      static_cast<std::int32_t>(nextNumStepsWritten);

  m_FileStream.seekp(8);
  m_FileStream.write(reinterpret_cast<const char *>(&numFramesWritten),
                     sizeof(std::int32_t));

  m_FileStream.seekp(20);
  m_FileStream.write(reinterpret_cast<const char *>(&numStepsWritten),
                     sizeof(std::int32_t));

  m_FileStream.seekp(0, std::ofstream::end);
  m_FileStream.flush();

  APOCHARMM_REQUIRE(m_FileStream.good(), ApoCharmmErrorCode::Runtime,
                    "Failed to write DCD frame: " + m_FilePath.string());

  m_NumFramesWritten = numFramesWritten;
  m_LastStepWritten = currentStep;

  return;
}

void DcdSubscriber::openFile(void) {
  APOCHARMM_REQUIRE(!m_FilePath.empty(), ApoCharmmErrorCode::NotInitialized,
                    "DcdSubscriber output file name is not set");

  this->checkPath(m_FilePath);

  if (m_FileStream.is_open())
    m_FileStream.close();

  m_FileStream.clear();
  m_FileStream.open(m_FilePath, std::ios::out | std::ios::binary);

  APOCHARMM_REQUIRE(m_FileStream.is_open() && m_FileStream.good(),
                    ApoCharmmErrorCode::Runtime,
                    "Failed to open DCD file for writing: " +
                        m_FilePath.string());

  m_NumFramesWritten = 0;
  m_IsHeaderWritten = false;
  m_HeaderReportFrequency = 0;
  m_HeaderNumAtoms = 0;
  m_HeaderNumDegreesOfFreedom = 0;
  m_HeaderTimeStep = 0.0;
  m_LastStepWritten = 0;
  m_PlannedEndStep = 0;
  // A new file is a new CHARMM-style run; the next propagation sets its origin.
  m_StartStep.reset();

  return;
}

void DcdSubscriber::writeHeader(void) {
  // START is the completed step of this first frame; DELTA is the time step in
  // AKMA units stored as the bits of an IEEE binary32 value.
  const unsigned long long int startStep = m_Integrator->getTotNumSteps();
  APOCHARMM_REQUIRE(startStep <= MAX_DCD_INTEGER, ApoCharmmErrorCode::Runtime,
                    "DcdSubscriber frame metadata exceeds DCD integer range");

  const float delta = static_cast<float>(m_Integrator->getTimeStep() /
                                         charmm::constants::timfac);
  APOCHARMM_REQUIRE(
      std::isfinite(delta) && delta > 0.0f, ApoCharmmErrorCode::InvalidArgument,
      "DcdSubscriber time step is not representable in DCD format");

  std::array<std::int32_t, 20> control{};
  control[1] = static_cast<std::int32_t>(startStep);
  control[2] = static_cast<std::int32_t>(m_ReportFrequency);
  control[7] =
      static_cast<std::int32_t>(m_CharmmContext->getNumDegreesOfFreedom());
  std::memcpy(&control[9], &delta, sizeof(float));
  control[10] = 1; // One symmetric unit-cell matrix record per frame
  control[19] = charmm::constants::vernum;

  const std::int32_t controlSize = 84;
  m_FileStream.write(reinterpret_cast<const char *>(&controlSize),
                     sizeof(std::int32_t));
  m_FileStream.write("CORD", 4);
  m_FileStream.write(
      reinterpret_cast<const char *>(control.data()),
      static_cast<std::streamsize>(control.size() * sizeof(std::int32_t)));
  m_FileStream.write(reinterpret_cast<const char *>(&controlSize),
                     sizeof(std::int32_t));

  std::array<char, 160> titles;
  titles.fill(' ');

  const std::string title = "* GENERATED BY APOCHARMM";
  std::memcpy(titles.data(), title.data(), title.size());
  titles[80] = '*';

  const std::int32_t titleSize = 164;
  const std::int32_t numTitles = 2;
  m_FileStream.write(reinterpret_cast<const char *>(&titleSize),
                     sizeof(std::int32_t));
  m_FileStream.write(reinterpret_cast<const char *>(&numTitles),
                     sizeof(std::int32_t));
  m_FileStream.write(titles.data(),
                     static_cast<std::streamsize>(titles.size()));
  m_FileStream.write(reinterpret_cast<const char *>(&titleSize),
                     sizeof(std::int32_t));

  const std::int32_t atomSize = 4;
  const std::int32_t numAtoms =
      static_cast<std::int32_t>(m_CharmmContext->getNumAtoms());
  m_FileStream.write(reinterpret_cast<const char *>(&atomSize),
                     sizeof(std::int32_t));
  m_FileStream.write(reinterpret_cast<const char *>(&numAtoms),
                     sizeof(std::int32_t));
  m_FileStream.write(reinterpret_cast<const char *>(&atomSize),
                     sizeof(std::int32_t));

  m_FileStream.flush();

  APOCHARMM_REQUIRE(m_FileStream.good(), ApoCharmmErrorCode::Runtime,
                    "Failed to write DCD header: " + m_FilePath.string());

  m_HeaderReportFrequency = m_ReportFrequency;
  m_HeaderNumAtoms = m_CharmmContext->getNumAtoms();
  m_HeaderNumDegreesOfFreedom = m_CharmmContext->getNumDegreesOfFreedom();
  m_HeaderTimeStep = m_Integrator->getTimeStep();
  m_IsHeaderWritten = true;

  return;
}

void DcdSubscriber::writeXtalData(void) {
  const std::vector<double> &boxDims = m_CharmmContext->getBoxDimensions();
  const std::array<double, 6> cell = {boxDims[0], 0.0, boxDims[1],
                                      0.0,        0.0, boxDims[2]};
  const std::int32_t blockSizeBytes = 48;

  m_FileStream.write(reinterpret_cast<const char *>(&blockSizeBytes),
                     sizeof(std::int32_t));
  m_FileStream.write(reinterpret_cast<const char *>(cell.data()),
                     blockSizeBytes);
  m_FileStream.write(reinterpret_cast<const char *>(&blockSizeBytes),
                     sizeof(std::int32_t));

  return;
}

void DcdSubscriber::writeCoordData(void) {
  m_CharmmContext->getCoordinatesChargesSP().transferToHost();
  const float4 *xyzq =
      m_CharmmContext->getCoordinatesChargesSP().getHostArray().data();
  const int numAtoms = m_CharmmContext->getNumAtoms();
  const std::int32_t blockSizeBytes = static_cast<std::int32_t>(numAtoms) *
                                      static_cast<std::int32_t>(sizeof(float));

  // Write X coordinate block
  m_FileStream.write(reinterpret_cast<const char *>(&blockSizeBytes),
                     sizeof(std::int32_t));
  for (int i = 0; i < numAtoms; i++)
    m_FileStream.write(reinterpret_cast<const char *>(&xyzq[i].x),
                       sizeof(float));
  m_FileStream.write(reinterpret_cast<const char *>(&blockSizeBytes),
                     sizeof(std::int32_t));

  // Write Y coordinate block
  m_FileStream.write(reinterpret_cast<const char *>(&blockSizeBytes),
                     sizeof(std::int32_t));
  for (int i = 0; i < numAtoms; i++)
    m_FileStream.write(reinterpret_cast<const char *>(&xyzq[i].y),
                       sizeof(float));
  m_FileStream.write(reinterpret_cast<const char *>(&blockSizeBytes),
                     sizeof(std::int32_t));

  // Write Z coordinate block
  m_FileStream.write(reinterpret_cast<const char *>(&blockSizeBytes),
                     sizeof(std::int32_t));
  for (int i = 0; i < numAtoms; i++)
    m_FileStream.write(reinterpret_cast<const char *>(&xyzq[i].z),
                       sizeof(float));
  m_FileStream.write(reinterpret_cast<const char *>(&blockSizeBytes),
                     sizeof(std::int32_t));

  return;
}
