// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#include "Subscriber.h"

#include "ApoCharmmError.h"
#include "CudaIntegrator.h"

#include <filesystem>
#include <fstream>
#include <system_error>

Subscriber::Subscriber(void)
    : m_ReportFrequency(1000), m_StartStep(), m_FilePath(), m_FileStream(),
      m_CharmmContext(nullptr), m_Integrator(nullptr) {}

Subscriber::Subscriber(const std::filesystem::path &filePath) : Subscriber() {
  this->setFilePath(filePath);
  this->openFile();
}

Subscriber::Subscriber(const std::filesystem::path &filePath,
                       const int reportFrequency)
    : Subscriber() {
  this->setReportFrequency(reportFrequency);
  this->setFilePath(filePath);
  this->openFile();
}

void Subscriber::setReportFrequency(const int reportFrequency) {
  APOCHARMM_REQUIRE(reportFrequency > 0, ApoCharmmErrorCode::InvalidArgument,
                    "Subscriber report frequency must be positive; observed " +
                        std::to_string(reportFrequency));

  m_ReportFrequency = reportFrequency;

  return;
}

void Subscriber::setFilePath(const std::filesystem::path &filePath) {
  APOCHARMM_REQUIRE(!filePath.empty(), ApoCharmmErrorCode::InvalidArgument,
                    "Output file name must not be empty");

  this->checkPath(filePath);

  m_FilePath = filePath;

  return;
}

void Subscriber::setCharmmContext(std::shared_ptr<CharmmContext> ctx) {
  APOCHARMM_REQUIRE(m_CharmmContext == nullptr,
                    ApoCharmmErrorCode::InvalidArgument,
                    "Subscriber already has a CharmmContext");

  m_CharmmContext = ctx;

  return;
}

void Subscriber::setIntegrator(std::shared_ptr<CudaIntegrator> integrator) {
  APOCHARMM_REQUIRE(m_Integrator == nullptr,
                    ApoCharmmErrorCode::InvalidArgument,
                    "Subscriber already has an Integrator");

  m_Integrator = integrator;

  return;
}

int Subscriber::getReportFrequency(void) const { return m_ReportFrequency; }

const std::filesystem::path &Subscriber::getFilePath(void) const {
  return m_FilePath;
}

std::filesystem::path &Subscriber::getFilePath(void) { return m_FilePath; }

void Subscriber::onPropagationStart(const unsigned long long int startStep,
                                    const int numSteps) {
  static_cast<void>(startStep);
  static_cast<void>(numSteps);
  return;
}

const std::optional<unsigned long long int> &
Subscriber::getStartStep(void) const {
  return m_StartStep;
}

void Subscriber::setStartStep(
    const std::optional<unsigned long long int> startStep) {
  m_StartStep = startStep;
  return;
}

void Subscriber::checkPath(const std::filesystem::path &filePath) {
  const std::filesystem::path parentPath = filePath.parent_path();

  if (parentPath.empty())
    return;

  std::error_code errorCode;
  const bool parentExists = std::filesystem::exists(parentPath, errorCode);

  APOCHARMM_REQUIRE(!errorCode && parentExists,
                    ApoCharmmErrorCode::InvalidArgument,
                    "Output directory does not exist: " + parentPath.string());

  return;
}

void Subscriber::openFile(void) {
  APOCHARMM_REQUIRE(!m_FilePath.empty(), ApoCharmmErrorCode::NotInitialized,
                    "Subscriber output file name is not set");

  this->checkPath(m_FilePath);

  if (m_FileStream.is_open())
    m_FileStream.close();

  m_FileStream.clear();
  m_FileStream.open(m_FilePath, std::ios::out);

  APOCHARMM_REQUIRE(m_FileStream.is_open() && m_FileStream.good(),
                    ApoCharmmErrorCode::Runtime,
                    "Failed to open subscriber output file for writing: " +
                        m_FilePath.string());

  return;
}

void Subscriber::addCommentSection(const std::string &commentLines) {
  APOCHARMM_REQUIRE(!commentLines.empty(), ApoCharmmErrorCode::InvalidArgument,
                    "Comment section must not be empty");

  APOCHARMM_REQUIRE(m_FileStream.is_open(), ApoCharmmErrorCode::NotInitialized,
                    "Subscriber output file is not open");

  // Make sure that last char is a line break
  std::string commentSection = commentLines;
  if (commentSection.back() != '\n')
    commentSection += "\n";

  m_FileStream << commentSection;
  m_FileStream.flush();

  APOCHARMM_REQUIRE(
      m_FileStream.good(), ApoCharmmErrorCode::Runtime,
      "Failed to write comment section to subscriber output file: " +
          m_FilePath.string());

  return;
}
