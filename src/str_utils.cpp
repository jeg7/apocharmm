// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "str_utils.h"

#include "ApoCharmmError.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <istream>
#include <limits>
#include <locale>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace {

template <typename T>
bool try_parse_number(T &value, const std::string &token) {
  static_assert(std::is_same_v<T, int> ||
                    std::is_same_v<T, unsigned long long int> ||
                    std::is_same_v<T, double>,
                "Unsupported numeric value type");

  std::size_t parsed_characters = 0;
  T parsed_value{};

  try {
    if constexpr (std::is_same_v<T, int>)
      parsed_value = std::stoi(token, &parsed_characters);
    else if constexpr (std::is_same_v<T, unsigned long long int>)
      parsed_value = std::stoull(token, &parsed_characters);
    else
      parsed_value = std::stod(token, &parsed_characters);
  } catch (const std::invalid_argument &) {
    return false;
  } catch (const std::out_of_range &) {
    return false;
  }

  if (parsed_characters != token.size())
    return false;

  value = parsed_value;

  return true;
}

std::string make_invalid_value_message(const std::string_view token,
                                       const std::string_view field_name,
                                       const std::string_view context) {
  return "Invalid " + std::string(field_name) + " value \"" +
         std::string(token) + "\" in " + std::string(context);
}

} // namespace

void apo::ltrim_ip(std::string &str) {
  str.erase(str.begin(),
            std::find_if(str.begin(), str.end(),
                         [](unsigned char c) { return !std::isspace(c); }));
  return;
}

void apo::rtrim_ip(std::string &str) {
  str.erase(std::find_if(str.rbegin(), str.rend(),
                         [](unsigned char c) { return !std::isspace(c); })
                .base(),
            str.end());
  return;
}

void apo::trim_ip(std::string &str) {
  apo::rtrim_ip(str);
  apo::ltrim_ip(str);
  return;
}

std::string apo::ltrim(const std::string_view str) {
  std::string s{str};
  apo::ltrim_ip(s);
  return s;
}

std::string apo::rtrim(const std::string_view str) {
  std::string s{str};
  apo::rtrim_ip(s);
  return s;
}

std::string apo::trim(const std::string_view str) {
  std::string s{str};
  apo::trim_ip(s);
  return s;
}

void apo::to_lower_ip(std::string &str) {
  std::transform(
      str.begin(), str.end(), str.begin(),
      [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return;
}

void apo::to_upper_ip(std::string &str) {
  std::transform(
      str.begin(), str.end(), str.begin(),
      [](const unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return;
}

std::string apo::to_lower(const std::string_view str) {
  std::string s{str};
  apo::to_lower_ip(s);
  return s;
}

std::string apo::to_upper(const std::string_view str) {
  std::string s{str};
  apo::to_upper_ip(s);
  return s;
}

std::vector<std::string> apo::split(const std::string_view str,
                                    const std::string_view delimiter) {
  APOCHARMM_REQUIRE(!delimiter.empty(), ApoCharmmErrorCode::InvalidArgument,
                    "Split delimiter must not be empty");

  const std::string trimmed = apo::trim(str);
  std::string_view remaining(trimmed);
  std::vector<std::string> tokens;

  std::size_t delimiter_position = 0;

  while ((delimiter_position = remaining.find(delimiter)) !=
         std::string_view::npos) {
    tokens.emplace_back(remaining.substr(0, delimiter_position));
    remaining.remove_prefix(delimiter_position + delimiter.size());

    while (!remaining.empty() &&
           std::isspace(static_cast<unsigned char>(remaining.front())) != 0) {
      remaining.remove_prefix(1);
    }
  }

  tokens.emplace_back(remaining);

  return tokens;
}

bool apo::try_get_line(std::string &line, std::size_t &pos,
                       const std::string_view file_data) {
  if (pos >= file_data.size()) {
    line.clear();
    pos = file_data.size();
    return false;
  }

  const std::size_t line_end = file_data.find('\n', pos);

  if (line_end == std::string_view::npos) {
    line.assign(file_data.substr(pos));
    pos = file_data.size();
  } else {
    line.assign(file_data.substr(pos, line_end - pos));
    pos = line_end + 1;
  }

  if (!line.empty() && (line.back() == '\r'))
    line.pop_back();

  return true;
}

bool apo::try_get_line(std::string &line, std::size_t &pos,
                       std::size_t &line_number,
                       const std::string_view file_data) {
  if (!apo::try_get_line(line, pos, file_data))
    return false;

  line_number++;

  return true;
}

bool apo::try_get_line(std::string &line, std::size_t &line_number,
                       std::istream &input) {
  line.clear();

  if (!std::getline(input, line))
    return false;

  if (!line.empty() && (line.back() == '\r'))
    line.pop_back();

  line_number++;

  return true;
}

void apo::get_line(std::string &line, std::size_t &pos,
                   const std::string_view file_data) {
  static_cast<void>(apo::try_get_line(line, pos, file_data));
  return;
}

void apo::get_line(std::string &line, std::size_t &pos,
                   std::size_t &line_number, const std::string_view file_data,
                   const std::string_view record_name,
                   const std::string_view source_name) {
  APOCHARMM_REQUIRE(apo::try_get_line(line, pos, line_number, file_data),
                    ApoCharmmErrorCode::Runtime,
                    "Unexpected end of file while reading " +
                        std::string(record_name) + " in " +
                        std::string(source_name));
  return;
}

void apo::get_line(std::string &line, std::size_t &line_number,
                   std::istream &input, const std::string_view record_name,
                   const std::string_view source_name) {
  if (apo::try_get_line(line, line_number, input))
    return;

  APOCHARMM_REQUIRE(!input.bad(), ApoCharmmErrorCode::Runtime,
                    "Failed while reading " + std::string(record_name) +
                        " in " + std::string(source_name));

  APOCHARMM_THROW(ApoCharmmErrorCode::Runtime,
                  "Unexpected end of file while reading " +
                      std::string(record_name) + " in " +
                      std::string(source_name));
}

void apo::find_required_line(std::istream &input, std::size_t &line_number,
                             const std::string_view target,
                             const std::string_view record_name,
                             const std::string_view source_name) {
  std::string line = "";

  while (apo::try_get_line(line, line_number, input)) {
    if (line == target)
      return;
  }

  APOCHARMM_REQUIRE(!input.bad(), ApoCharmmErrorCode::Runtime,
                    "Failed while searching for " + std::string(record_name) +
                        " in " + std::string(source_name));

  APOCHARMM_THROW(ApoCharmmErrorCode::Runtime,
                  "Could not find " + std::string(record_name) + " in " +
                      std::string(source_name));
}

bool apo::try_get_fixed_width_field(std::string_view &field,
                                    const std::string_view line,
                                    const std::size_t offset,
                                    const std::size_t width) noexcept {
  field = std::string_view{};

  if ((offset > line.size()) || (width > line.size() - offset))
    return false;

  field = line.substr(offset, width);

  return true;
}

void apo::read_file_into_string(std::string &file_data,
                                const std::filesystem::path &file_path) {
  const std::string file_name = file_path.string();
  std::ifstream ifs(file_path, std::ios::in | std::ios::binary);

  APOCHARMM_REQUIRE(ifs.is_open(), ApoCharmmErrorCode::Runtime,
                    "Failed to open file \"" + file_name + "\"");

  ifs.seekg(0, std::ios::end);
  const std::streampos end_position = ifs.tellg();

  APOCHARMM_REQUIRE(end_position != std::streampos(-1),
                    ApoCharmmErrorCode::Runtime,
                    "Failed to determine size of file \"" + file_name + "\"");

  // Store the size of the file
  const std::streamoff file_size = end_position - std::streampos(0);

  APOCHARMM_REQUIRE(file_size >= 0, ApoCharmmErrorCode::Runtime,
                    "File size is invalid for \"" + file_name + "\"");

  const std::uintmax_t unsigned_file_size =
      static_cast<std::uintmax_t>(file_size);

  APOCHARMM_REQUIRE(
      (unsigned_file_size <=
       static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max())) &&
          (unsigned_file_size <=
           static_cast<std::uintmax_t>(
               std::numeric_limits<std::streamsize>::max())),
      ApoCharmmErrorCode::Runtime,
      "File is too large to read into memory: \"" + file_name + "\"");

  // Initialize a std::string with length fileSize, filled with null characters
  std::string contents(static_cast<std::size_t>(unsigned_file_size), '\0');

  // Go back to the beginning of the file and store the contents in the string
  ifs.seekg(0, std::ios::beg);

  APOCHARMM_REQUIRE(ifs.good(), ApoCharmmErrorCode::Runtime,
                    "Failed to seek in file \"" + file_name + "\"");

  if (!contents.empty()) {
    const std::streamsize expected_size =
        static_cast<std::streamsize>(contents.size());

    ifs.read(contents.data(), expected_size);

    APOCHARMM_REQUIRE(!ifs.bad() && (ifs.gcount() == expected_size),
                      ApoCharmmErrorCode::Runtime,
                      "Failed while reading file \"" + file_name + "\"");
  }

  file_data.swap(contents);

  return;
}

bool apo::try_parse_int(int &value, const std::string_view token) {
  const std::string normalized_token = apo::trim(token);
  int parsed_value = 0;

  if (!try_parse_number<int>(parsed_value, normalized_token))
    return false;

  value = parsed_value;

  return true;
}

bool apo::try_parse_ull(unsigned long long int &value,
                        const std::string_view token) {
  const std::string normalized_token = apo::trim(token);

  if (normalized_token.empty() || (normalized_token.front() == '-'))
    return false;

  unsigned long long int parsed_value = 0;

  if (!try_parse_number<unsigned long long int>(parsed_value, normalized_token))
    return false;

  value = parsed_value;

  return true;
}

bool apo::try_parse_double(double &value, const std::string_view token) {
  std::string normalized_token = apo::trim(token);

  std::replace(normalized_token.begin(), normalized_token.end(), 'D', 'E');
  std::replace(normalized_token.begin(), normalized_token.end(), 'd', 'E');

  double parsed_value = 0.0;

  if (!try_parse_number<double>(parsed_value, normalized_token))
    return false;

  value = parsed_value;

  return true;
}

bool apo::try_parse_finite_double(double &value, const std::string_view token) {
  double parsed_value = 0.0;

  if (!apo::try_parse_double(parsed_value, token) ||
      !std::isfinite(parsed_value)) {
    return false;
  }

  value = parsed_value;

  return true;
}

int apo::parse_int(const std::string_view token,
                   const std::string_view field_name,
                   const std::string_view context) {
  int value = 0;

  APOCHARMM_REQUIRE(
      apo::try_parse_int(value, token), ApoCharmmErrorCode::Runtime,
      make_invalid_value_message(apo::trim(token), field_name, context));

  return value;
}

unsigned long long int apo::parse_ull(const std::string_view token,
                                      const std::string_view field_name,
                                      const std::string_view context) {
  unsigned long long int value = 0;

  APOCHARMM_REQUIRE(
      apo::try_parse_ull(value, token), ApoCharmmErrorCode::Runtime,
      make_invalid_value_message(apo::trim(token), field_name, context));

  return value;
}

double apo::parse_double(const std::string_view token,
                         const std::string_view field_name,
                         const std::string_view context) {
  double value = 0.0;

  APOCHARMM_REQUIRE(
      apo::try_parse_finite_double(value, token), ApoCharmmErrorCode::Runtime,
      make_invalid_value_message(apo::trim(token), field_name, context));

  return value;
}

bool apo::contains_wildcard(const std::string_view str) {
  return str.find_first_of("*%#+") != std::string_view::npos;
}

std::string apo::cDoubleToFortSciStr(const double val, const int prec) {
  APOCHARMM_REQUIRE(prec >= 0, ApoCharmmErrorCode::InvalidArgument,
                    "Precision must be non-negative; observed " +
                        std::to_string(prec));

  std::ostringstream oss;
  oss.imbue(std::locale::classic());
  oss << std::scientific << std::setprecision(prec) << val;
  std::string str = oss.str();
  std::replace(str.begin(), str.end(), 'e', 'D');
  return str;
}

double apo::fortSciStrToCDouble(const std::string_view str) {
  return apo::parse_double(str, "floating-point", "Fortran scientific value");
}

bool apo::try_parse_fortran_double(double &value,
                                   const std::string_view token) {
  std::string normalized_token = apo::trim(token);
  std::size_t pos = 0;
  const auto is_digit = [](const char c) -> bool {
    return (c >= '0' && c <= '9');
  };

  if (!normalized_token.empty() &&
      ((normalized_token.front() == '+') || (normalized_token.front() == '-')))
    pos++;

  bool has_mantissa_digit = false;
  bool has_nonzero_mantissa = false;
  const auto consume_mantissa_digits = [&](void) -> void {
    while ((pos < normalized_token.size()) && is_digit(normalized_token[pos])) {
      has_mantissa_digit = true;
      has_nonzero_mantissa =
          (has_nonzero_mantissa || (normalized_token[pos] != '0'));
      pos++;
    }
    return;
  };

  consume_mantissa_digits();
  if ((pos < normalized_token.size()) && (normalized_token[pos] == '.')) {
    pos++;
    consume_mantissa_digits();
  }
  if (!has_mantissa_digit)
    return false;

  if (pos < normalized_token.size()) {
    if ((normalized_token[pos] == 'D') || (normalized_token[pos] == 'd') ||
        (normalized_token[pos] == 'E') || (normalized_token[pos] == 'e')) {
      normalized_token[pos++] = 'E';
    } else if ((normalized_token[pos] == '+') || (normalized_token[pos] == '-'))
      normalized_token.insert(pos++, "E");
    else
      return false;

    if ((pos < normalized_token.size()) &&
        ((normalized_token[pos] == '+') || (normalized_token[pos] == '-'))) {
      pos++;
    }

    const std::size_t first_exponent_digit = pos;
    while ((pos < normalized_token.size()) && is_digit(normalized_token[pos]))
      pos++;
    if ((pos == first_exponent_digit) || (pos != normalized_token.size()))
      return false;
  }

  // JEG261029: std::stod may report ERANGE for a representable subnormal.
  // Classic-locale stream extraction accepts that value without consulting the
  // process locale.
  double parsed_value = 0.0;
  std::istringstream input(normalized_token);
  input.imbue(std::locale::classic());
  input >> parsed_value;
  if (input.fail() || !std::isfinite(parsed_value) ||
      ((parsed_value == 0.0) && has_nonzero_mantissa)) {
    return false;
  }
  input >> std::ws;
  if (!input.eof())
    return false;

  value = parsed_value;

  return true;
}

double apo::parse_fortran_double(const std::string_view token,
                                 const std::string_view field_name,
                                 const std::string_view context) {
  double value = 0.0;
  APOCHARMM_REQUIRE(
      apo::try_parse_fortran_double(value, token), ApoCharmmErrorCode::Runtime,
      make_invalid_value_message(apo::trim(token), field_name, context));
  return value;
}

std::string apo::format_fortran_double(const double value,
                                       const std::size_t width,
                                       const int precision) {
  APOCHARMM_REQUIRE(std::isfinite(value), ApoCharmmErrorCode::InvalidArgument,
                    "Fortran numeric fields must be finite");
  APOCHARMM_REQUIRE(
      (precision >= 1) && (width >= 6) &&
          (static_cast<std::size_t>(precision) <= width - 6),
      ApoCharmmErrorCode::InvalidArgument,
      "Fortran field width must accommodate a positive precision");

  const std::string scientific =
      apo::cDoubleToFortSciStr(std::abs(value), precision - 1);
  double roundtrip_value = 0.0;
  APOCHARMM_REQUIRE(
      apo::try_parse_fortran_double(roundtrip_value, scientific),
      ApoCharmmErrorCode::InvalidArgument,
      "Value is outside the finite Fortran field round-trip range");

  const std::size_t exponent_position = scientific.find('D');
  APOCHARMM_REQUIRE(exponent_position != std::string::npos,
                    ApoCharmmErrorCode::Runtime,
                    "Failed to format Fortran numeric field");

  std::string digits = scientific.substr(0, exponent_position);
  digits.erase(std::remove(digits.begin(), digits.end(), '.'), digits.end());

  const int exponent =
      (value == 0.0) ? 0
                     : apo::parse_int(scientific.substr(exponent_position + 1),
                                      "exponent", "Fortran numeric field") +
                           1;

  std::ostringstream output;
  output.imbue(std::locale::classic());

  if (std::signbit(value))
    output << '-';

  output << "0." << digits;

  if (std::abs(exponent) < 100)
    output << 'D';

  output << ((exponent < 0) ? '-' : '+') << std::setfill('0')
         << std::setw((std::abs(exponent) < 100) ? 2 : 3) << std::abs(exponent);

  std::string field = output.str();
  APOCHARMM_REQUIRE(field.size() <= width, ApoCharmmErrorCode::InvalidArgument,
                    "Fortran numeric field exceeds its requested width");
  field.insert(0, width - field.size(), ' ');

  return field;
}

void apo::get_nonblank_line(std::string &line, std::size_t &line_number,
                            std::istream &input,
                            const std::string_view record_name,
                            const std::string_view source_name) {
  do {
    apo::get_line(line, line_number, input, record_name, source_name);
  } while (apo::trim(line).empty());
  return;
}

void apo::get_expected_line(std::size_t &line_number, std::istream &input,
                            const std::string_view expected,
                            const std::string_view source_name) {
  std::string line;
  apo::get_nonblank_line(line, line_number, input,
                         "line \"" + std::string(expected) + "\"", source_name);
  APOCHARMM_REQUIRE(line == expected, ApoCharmmErrorCode::Runtime,
                    "Expected line \"" + std::string(expected) + "\" in " +
                        std::string(source_name) + " but found \"" + line +
                        "\"");
  return;
}

void apo::parse_fixed_width_fortran_doubles(
    double *values, const std::size_t count, const std::string_view line,
    const std::size_t offset, const std::size_t width,
    const std::string_view field_name, const std::string_view context) {
  APOCHARMM_REQUIRE((values != nullptr) || (count == 0),
                    ApoCharmmErrorCode::InvalidArgument,
                    "Fortran field destination must not be null");

  for (std::size_t i = 0; i < count; i++) {
    values[i] = apo::parse_fixed_width_fortran_double(
        line, offset + i * width, width, field_name, context);
  }

  const std::size_t end = offset + count * width;
  APOCHARMM_REQUIRE((end >= line.size()) || apo::trim(line.substr(end)).empty(),
                    ApoCharmmErrorCode::Runtime,
                    "Unexpected data after \"" + std::string(field_name) +
                        "\" in " + std::string(context));

  return;
}

void apo::read_fixed_width_fortran_values(std::size_t &line_number,
                                          std::istream &input, double *values,
                                          const std::size_t count,
                                          const std::size_t values_per_record,
                                          const std::size_t width,
                                          const std::string_view record_name,
                                          const std::string_view source_name) {
  APOCHARMM_REQUIRE(values_per_record > 0, ApoCharmmErrorCode::InvalidArgument,
                    "Fortran record must hold at least one value");

  for (std::size_t first = 0; first < count; first += values_per_record) {
    std::string line;
    apo::get_line(line, line_number, input, record_name, source_name);
    apo::parse_fixed_width_fortran_doubles(
        values + first, std::min(values_per_record, count - first), line, 0,
        width, record_name, source_name);
  }

  return;
}

void apo::write_fixed_width_fortran_values(std::ostream &output,
                                           const double *values,
                                           const std::size_t count,
                                           const std::size_t values_per_record,
                                           const std::size_t width,
                                           const int precision) {
  APOCHARMM_REQUIRE(values_per_record > 0, ApoCharmmErrorCode::InvalidArgument,
                    "Fortran record must hold at least one value");
  APOCHARMM_REQUIRE(values != nullptr || count == 0,
                    ApoCharmmErrorCode::InvalidArgument,
                    "Fortran field source must not be null");

  for (std::size_t i = 0; i < count; i++) {
    output << apo::format_fortran_double(values[i], width, precision);
    if ((i + 1) % values_per_record == 0 || i + 1 == count)
      output << '\n';
  }

  return;
}

std::string apo::get_fixed_width_field(const std::string_view line,
                                       const std::size_t offset,
                                       const std::size_t width,
                                       const std::string_view field_name,
                                       const std::string_view context) {
  std::string_view field;

  APOCHARMM_REQUIRE(apo::try_get_fixed_width_field(field, line, offset, width),
                    ApoCharmmErrorCode::Runtime,
                    "Field \"" + std::string(field_name) +
                        "\" is truncated in " + std::string(context));

  return std::string(field);
}

double apo::parse_fixed_width_double(const std::string_view line,
                                     const std::size_t offset,
                                     const std::size_t width,
                                     const std::string_view field_name,
                                     const std::string_view context) {
  const std::string field =
      apo::get_fixed_width_field(line, offset, width, field_name, context);

  return apo::parse_double(field, field_name, context);
}

double apo::parse_fixed_width_fortran_double(const std::string_view line,
                                             const std::size_t offset,
                                             const std::size_t width,
                                             const std::string_view field_name,
                                             const std::string_view context) {
  const std::string field =
      apo::get_fixed_width_field(line, offset, width, field_name, context);
  return apo::parse_fortran_double(field, field_name, context);
}

int apo::parse_fixed_width_int(const std::string_view line,
                               const std::size_t offset,
                               const std::size_t width,
                               const std::string_view field_name,
                               const std::string_view context) {
  const std::string field =
      apo::get_fixed_width_field(line, offset, width, field_name, context);

  return apo::parse_int(field, field_name, context);
}

unsigned long long int
apo::parse_fixed_width_ull(const std::string_view line,
                           const std::size_t offset, const std::size_t width,
                           const std::string_view field_name,
                           const std::string_view context) {
  const std::string field =
      apo::get_fixed_width_field(line, offset, width, field_name, context);

  return apo::parse_ull(field, field_name, context);
}
