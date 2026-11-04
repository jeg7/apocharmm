// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "apo_test_helpers.h"
#include "catch.hpp"
#include "str_utils.h"

#include <array>
#include <cmath>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

TEST_CASE("StrUtilsTryGetLineHandlesFinalAndEmptyLines") {
  const std::string fileData = "first\n\nthird";
  std::size_t position = 0;
  std::size_t lineNumber = 0;
  std::string line = "stale";

  CHECK(apo::try_get_line(line, position, lineNumber, fileData));
  CHECK(line == "first");
  CHECK(lineNumber == 1);

  CHECK(apo::try_get_line(line, position, lineNumber, fileData));
  CHECK(line.empty());
  CHECK(lineNumber == 2);

  CHECK(apo::try_get_line(line, position, lineNumber, fileData));
  CHECK(line == "third");
  CHECK(lineNumber == 3);
  CHECK(position == fileData.size());

  CHECK_FALSE(apo::try_get_line(line, position, lineNumber, fileData));
  CHECK(line.empty());
  CHECK(position == fileData.size());
  CHECK(lineNumber == 3);
}

TEST_CASE("StrUtilsGetLineReportsUnexpectedEndOfFile") {
  std::size_t position = 0;
  std::size_t lineNumber = 0;
  std::string line;

  apo_test::CheckApoCharmmError(
      [&line, &position, &lineNumber](void) {
        apo::get_line(line, position, lineNumber, "", "ATOM records",
                      "PSF \"test.psf\"");
      },
      ApoCharmmErrorCode::Runtime,
      "Unexpected end of file while reading ATOM records in PSF \"test.psf\"");
}

TEST_CASE("StrUtilsExtractsFixedWidthFieldsSafely") {
  const std::string line = "0123456789";
  std::string_view field = "stale";

  CHECK(apo::try_get_fixed_width_field(field, line, 2, 4));
  CHECK(field == "2345");

  CHECK(apo::try_get_fixed_width_field(field, line, line.size(), 0));
  CHECK(field.empty());

  CHECK_FALSE(apo::try_get_fixed_width_field(field, line, 8, 3));
  CHECK(field.empty());

  CHECK_FALSE(apo::try_get_fixed_width_field(
      field, line, std::numeric_limits<std::size_t>::max(), 1));
  CHECK(field.empty());
}

TEST_CASE("StrUtilsParsesIntegersStrictly") {
  int value = 99;

  CHECK(apo::try_parse_int(value, " -17 "));
  CHECK(value == -17);

  value = 99;
  CHECK_FALSE(apo::try_parse_int(value, ""));
  CHECK(value == 99);

  CHECK_FALSE(apo::try_parse_int(value, "12BAD"));
  CHECK(value == 99);

  CHECK_FALSE(apo::try_parse_int(value, "999999999999999999999"));
  CHECK(value == 99);

  apo_test::CheckApoCharmmError(
      [](void) {
        static_cast<void>(apo::parse_int(
            "BAD", "atom index", "BOND section of PSF \"test.psf\" at line 7"));
      },
      ApoCharmmErrorCode::Runtime,
      "Invalid atom index value \"BAD\" in BOND section of PSF \"test.psf\" "
      "at line 7");
}

TEST_CASE("StrUtilsParsesUnsignedIntegersStrictly") {
  unsigned long long int value = 99;

  CHECK(apo::try_parse_ull(value, " 17 "));
  CHECK(value == 17);

  value = 99;
  CHECK_FALSE(apo::try_parse_ull(value, "-1"));
  CHECK(value == 99);

  CHECK_FALSE(apo::try_parse_ull(value, "12BAD"));
  CHECK(value == 99);

  const std::string outOfRange =
      std::to_string(std::numeric_limits<unsigned long long int>::max()) + "0";
  CHECK_FALSE(apo::try_parse_ull(value, outOfRange));
  CHECK(value == 99);

  apo_test::CheckApoCharmmError(
      [](void) {
        static_cast<void>(apo::parse_ull(
            "-1", "count", "ATOM section of PSF \"test.psf\" at line 4"));
      },
      ApoCharmmErrorCode::Runtime,
      "Invalid count value \"-1\" in ATOM section of PSF \"test.psf\" at "
      "line 4");
}

TEST_CASE("StrUtilsParsesFloatingPointValuesStrictly") {
  double value = 99.0;

  CHECK(apo::try_parse_finite_double(value, "1.25"));
  CHECK(value == Approx(1.25));

  CHECK(apo::try_parse_finite_double(value, "-2.5E+01"));
  CHECK(value == Approx(-25.0));

  CHECK(apo::try_parse_finite_double(value, "1.25D+02"));
  CHECK(value == Approx(125.0));

  CHECK(apo::try_parse_finite_double(value, "2.5d-01"));
  CHECK(value == Approx(0.25));

  value = 99.0;
  CHECK_FALSE(apo::try_parse_finite_double(value, "1.0BAD"));
  CHECK(value == Approx(99.0));

  CHECK_FALSE(apo::try_parse_finite_double(value, "1E9999"));
  CHECK(value == Approx(99.0));

  CHECK_FALSE(apo::try_parse_finite_double(value, "NAN"));
  CHECK(value == Approx(99.0));

  CHECK_FALSE(apo::try_parse_finite_double(value, "INF"));
  CHECK(value == Approx(99.0));

  double nonfiniteValue = 0.0;
  CHECK(apo::try_parse_double(nonfiniteValue, "NAN"));
  CHECK(std::isnan(nonfiniteValue));

  apo_test::CheckApoCharmmError(
      [](void) {
        static_cast<void>(apo::parse_double(
            "BAD", "charge", "ATOM section of PSF \"test.psf\" at line 5"));
      },
      ApoCharmmErrorCode::Runtime,
      "Invalid charge value \"BAD\" in ATOM section of PSF \"test.psf\" at "
      "line 5");
}

TEST_CASE("StrUtilsReadsStreamsAndNormalizesLineEndings") {
  std::istringstream input("first\r\nsecond\nthird");
  std::size_t lineNumber = 0;
  std::string line = "stale";

  CHECK(apo::try_get_line(line, lineNumber, input));
  CHECK(line == "first");
  CHECK(lineNumber == 1);

  apo::get_line(line, lineNumber, input, "second record", "test stream");
  CHECK(line == "second");
  CHECK(lineNumber == 2);

  apo::get_line(line, lineNumber, input, "third record", "test stream");
  CHECK(line == "third");
  CHECK(lineNumber == 3);

  CHECK_FALSE(apo::try_get_line(line, lineNumber, input));
  CHECK(line.empty());
  CHECK(lineNumber == 3);
}

TEST_CASE("StrUtilsFindsRequiredStreamLines") {
  std::istringstream input("header\n"
                           "ignored\n"
                           " !TARGET\r\n"
                           "payload\n");

  std::size_t lineNumber = 0;
  std::string line;

  apo::find_required_line(input, lineNumber, " !TARGET", "target section",
                          "test stream");
  CHECK(lineNumber == 3);

  apo::get_line(line, lineNumber, input, "payload record", "test stream");
  CHECK(line == "payload");
  CHECK(lineNumber == 4);
}

TEST_CASE("StrUtilsReportsStreamReadFailures") {
  SECTION("UnexpectedEndOfFile") {
    std::istringstream input("");
    std::size_t lineNumber = 0;
    std::string line;

    apo_test::CheckApoCharmmError(
        [&]() {
          apo::get_line(line, lineNumber, input, "ATOM record", "test stream");
        },
        ApoCharmmErrorCode::Runtime,
        "Unexpected end of file while reading ATOM record in test stream");
  }

  SECTION("MissingRequiredLine") {
    std::istringstream input("header\npayload\n");
    std::size_t lineNumber = 0;

    apo_test::CheckApoCharmmError(
        [&]() {
          apo::find_required_line(input, lineNumber, " !TARGET",
                                  "target section", "test stream");
        },
        ApoCharmmErrorCode::Runtime,
        "Could not find target section in test stream");
  }
}

TEST_CASE("StrUtilsParsesFixedWidthFields") {
  CHECK(apo::get_fixed_width_field("0123456789", 2, 4, "digits",
                                   "test record") == "2345");

  CHECK(apo::parse_fixed_width_int("  -17  ", 0, 7, "count", "test record") ==
        -17);

  CHECK(apo::parse_fixed_width_ull("  42  ", 0, 6, "count", "test record") ==
        42ULL);

  CHECK(apo::parse_fixed_width_double(" 1.25D+02 ", 0, 10, "value",
                                      "test record") == Approx(125.0));

  apo_test::CheckApoCharmmError(
      [](void) {
        static_cast<void>(
            apo::get_fixed_width_field("123", 2, 2, "count", "test record"));
      },
      ApoCharmmErrorCode::Runtime,
      "Field \"count\" is truncated in test record");

  apo_test::CheckApoCharmmError(
      [](void) {
        static_cast<void>(
            apo::parse_fixed_width_int(" BAD ", 0, 5, "count", "test record"));
      },
      ApoCharmmErrorCode::Runtime,
      "Invalid count value \"BAD\" in test record");

  apo_test::CheckApoCharmmError(
      [](void) {
        static_cast<void>(
            apo::parse_fixed_width_ull(" -1 ", 0, 4, "count", "test record"));
      },
      ApoCharmmErrorCode::Runtime, "Invalid count value \"-1\" in test record");

  apo_test::CheckApoCharmmError(
      [](void) {
        static_cast<void>(apo::parse_fixed_width_double(
            " INF ", 0, 5, "floating-point", "test record"));
      },
      ApoCharmmErrorCode::Runtime,
      "Invalid floating-point value \"INF\" in test record");
}

TEST_CASE("StrUtilsSplitRejectsEmptyDelimiter") {
  apo_test::CheckApoCharmmError(
      [](void) { static_cast<void>(apo::split("ABC", "")); },
      ApoCharmmErrorCode::InvalidArgument, "Split delimiter must not be empty");
}

TEST_CASE("StrUtilsSplitPreservesCurrentWhitespaceBehavior") {
  CHECK(apo::split("  A   B C  ") == std::vector<std::string>{"A", "B", "C"});

  CHECK(apo::split("A,,B", ",") == std::vector<std::string>{"A", "", "B"});

  CHECK(apo::split("A,   B", ",") == std::vector<std::string>{"A", "B"});
}

TEST_CASE("StrUtilsParsesFortranDecimalValuesStrictly") {
  double value = 99.0;
  for (const std::string token :
       {"1.25E+02", "1.25D+02", "1.25d+02", "1.25+02", "  +125.0  "}) {
    INFO("token=" << token);
    CHECK(apo::try_parse_fortran_double(value, token));
    CHECK(value == 125.0);
  }
  CHECK(apo::parse_fortran_double("0.100000000000000+101", "value", "test") ==
        1.0e100);
  CHECK(apo::parse_fortran_double("-0.100000000000000-099", "value", "test") ==
        -1.0e-100);
  CHECK(apo::parse_fortran_double(".5", "value", "test") == 0.5);
  CHECK(apo::parse_fortran_double("12", "value", "test") == 12.0);
  CHECK(apo::parse_fortran_double("1.", "value", "test") == 1.0);
  CHECK(std::signbit(apo::parse_fortran_double("-0.0D+00", "value", "test")));

  for (const std::string token :
       {"", "+", ".", "NAN", "INF", "-INF", "0x1p+0", "0.123D+999", "0.1D-999",
        "1.2BAD", "1.2D", "1.2+", "1.2+-03", "1.2D+01tail", "1 .2", "1,2"}) {
    INFO("invalid token=" << token);
    value = 99.0;
    CHECK_FALSE(apo::try_parse_fortran_double(value, token));
    CHECK(value == 99.0);
  }
  apo_test::CheckApoCharmmError(
      []() {
        static_cast<void>(apo::parse_fortran_double("BAD", "value", "test"));
      },
      ApoCharmmErrorCode::Runtime, "Invalid value value \"BAD\" in test");

  // The existing parser deliberately retains its distinct nonfinite contract.
  CHECK(apo::try_parse_double(value, "NAN"));
  CHECK(std::isnan(value));
}

TEST_CASE("StrUtilsFormatsFortranFieldsAndPreservesRepresentableSubnormals") {
  const std::array<double, 10> values = {
      0.0,
      -0.0,
      1.0,
      -1.23456789012345,
      1.0e100,
      -1.0e100,
      1.0e-100,
      -1.0e-100,
      std::numeric_limits<double>::min(),
      std::numeric_limits<double>::denorm_min()};
  for (const double value : values) {
    INFO("value=" << value);
    const std::string field = apo::format_fortran_double(value, 22, 15);
    REQUIRE(field.size() == 22);
    const double restored =
        apo::parse_fixed_width_fortran_double(field, 0, 22, "value", "test");
    CHECK(std::signbit(restored) == std::signbit(value));
    if (value != 0.0)
      CHECK(restored != 0.0);
    CHECK(std::abs(restored - value) <=
          std::abs(value) * 5.0e-15 +
              std::numeric_limits<double>::denorm_min());
  }
  CHECK(apo::format_fortran_double(1.0, 22, 15) == " 0.100000000000000D+01");
  CHECK(apo::format_fortran_double(-1.0, 22, 15) == "-0.100000000000000D+01");
  CHECK(apo::format_fortran_double(1.0e100, 22, 15) ==
        " 0.100000000000000+101");
  CHECK(apo::format_fortran_double(1.25, 12, 5) == " 0.12500D+01");
  const auto max_value = std::numeric_limits<double>::max();
  CHECK(apo::parse_fortran_double(apo::format_fortran_double(max_value, 24, 17),
                                  "value", "test") == max_value);

  // Rounding DBL_MAX to fifteen significant digits overflows binary64.
  for (const double value : {std::numeric_limits<double>::max(),
                             -std::numeric_limits<double>::max()}) {
    apo_test::CheckApoCharmmError(
        [value]() {
          static_cast<void>(apo::format_fortran_double(value, 22, 15));
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Value is outside the finite Fortran field round-trip range");
  }
  for (const double value : {std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN()}) {
    apo_test::CheckApoCharmmError(
        [value]() {
          static_cast<void>(apo::format_fortran_double(value, 22, 15));
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Fortran numeric fields must be finite");
  }
  for (const std::pair<std::size_t, int> &field :
       {std::pair<std::size_t, int>{22, 0}, std::pair<std::size_t, int>{22, -1},
        std::pair<std::size_t, int>{20, 15}}) {
    apo_test::CheckApoCharmmError(
        [field]() {
          static_cast<void>(
              apo::format_fortran_double(1.0, field.first, field.second));
        },
        ApoCharmmErrorCode::InvalidArgument,
        "Fortran field width must accommodate a positive precision");
  }
  // The sign of -1.0 needs the twenty-second column.
  apo_test::CheckApoCharmmError(
      []() { static_cast<void>(apo::format_fortran_double(-1.0, 21, 15)); },
      ApoCharmmErrorCode::InvalidArgument,
      "Fortran numeric field exceeds its requested width");
}

TEST_CASE("StrUtilsKeepsAdjacentSignedFortranFieldsSeparate") {
  const std::string line = apo::format_fortran_double(-1.25, 22, 15) +
                           apo::format_fortran_double(-2.5e100, 22, 15);
  REQUIRE(line.size() == 44);
  REQUIRE(line[22] == '-');
  CHECK(apo::parse_fixed_width_fortran_double(line, 0, 22, "first", "test") ==
        -1.25);
  CHECK(apo::parse_fixed_width_fortran_double(line, 22, 22, "second", "test") ==
        -2.5e100);
  apo_test::CheckApoCharmmError(
      [&]() {
        static_cast<void>(apo::parse_fixed_width_fortran_double(
            std::string_view(line).substr(0, 43), 22, 22, "second", "test"));
      },
      ApoCharmmErrorCode::Runtime, "Field \"second\" is truncated in test");
  const auto max_seed = std::numeric_limits<unsigned long long int>::max();
  const std::string seed_token = std::to_string(max_seed);
  const std::string seed_field =
      std::string(22 - seed_token.size(), ' ') + seed_token;
  CHECK(apo::parse_fixed_width_ull(seed_field, 0, 22, "SEED", "test") ==
        max_seed);
}

TEST_CASE("StrUtilsFortranConversionsIgnoreTheGlobalNumericLocale") {
  class CommaDecimalPoint : public std::numpunct<char> {
    char do_decimal_point() const override { return ','; }
  };
  struct RestoreLocale {
    std::locale previous = std::locale();
    ~RestoreLocale() { std::locale::global(previous); }
  } restore;
  std::locale::global(std::locale(restore.previous, new CommaDecimalPoint));

  CHECK(apo::format_fortran_double(1.25, 12, 5) == " 0.12500D+01");
  CHECK(apo::parse_fortran_double("1.25D+00", "value", "test") == 1.25);
  double value = 99.0;
  CHECK_FALSE(apo::try_parse_fortran_double(value, "1,25"));
  CHECK(value == 99.0);
}

TEST_CASE("StrUtilsReadsFortranRecordsWithFormatReversion") {
  const auto field = [](const double value) {
    return apo::format_fortran_double(value, 22, 15);
  };
  // A (/A) label, four values written with (3D22.15) (three plus one record),
  // two (2D22.15) rows, and a record with trailing junk.
  std::istringstream input("\n !LABEL\n" + field(1.0) + field(2.0) +
                           field(3.0) + "\n" + field(4.0) + "\n" + field(5.0) +
                           field(6.0) + "\n" + field(7.0) + field(8.0) + "\n" +
                           field(9.0) + "  junk\n");
  std::size_t lineNumber = 0;

  apo::get_expected_line(lineNumber, input, " !LABEL", "test stream");
  CHECK(lineNumber == 2);

  std::array<double, 4> values{};
  apo::read_fixed_width_fortran_values(lineNumber, input, values.data(),
                                       values.size(), 3, 22, "values",
                                       "test stream");
  CHECK((values == std::array<double, 4>{1.0, 2.0, 3.0, 4.0}));
  CHECK(lineNumber == 4);

  // The writer reproduces the (3D22.15) records the reader consumed.
  std::ostringstream written;
  apo::write_fixed_width_fortran_values(written, values.data(), values.size(),
                                        3, 22, 15);
  CHECK(written.str() ==
        field(1.0) + field(2.0) + field(3.0) + "\n" + field(4.0) + "\n");
  apo_test::CheckApoCharmmError(
      [&]() {
        apo::write_fixed_width_fortran_values(written, values.data(),
                                              values.size(), 0, 22, 15);
      },
      ApoCharmmErrorCode::InvalidArgument,
      "Fortran record must hold at least one value");

  std::vector<std::array<double, 2>> rows;
  apo::read_fixed_width_fortran_rows(lineNumber, input, rows, 2, 22, "rows",
                                     "test stream");
  REQUIRE(rows.size() == 2);
  CHECK((rows[0] == std::array<double, 2>{5.0, 6.0}));
  CHECK((rows[1] == std::array<double, 2>{7.0, 8.0}));
  CHECK(lineNumber == 6);

  apo_test::CheckApoCharmmError(
      [&]() {
        std::array<double, 1> tail{};
        apo::read_fixed_width_fortran_values(lineNumber, input, tail.data(),
                                             tail.size(), 3, 22, "tail",
                                             "test stream");
      },
      ApoCharmmErrorCode::Runtime,
      "Unexpected data after \"tail\" in test stream");

  std::array<double, 3> offsets{};
  apo::parse_fixed_width_fortran_doubles(offsets.data(), offsets.size(),
                                         "     IGN" + field(1.5) + field(2.5) +
                                             field(3.5) + "   ",
                                         8, 22, "offsets", "test stream");
  CHECK((offsets == std::array<double, 3>{1.5, 2.5, 3.5}));

  std::istringstream wrongLabel("\n !OTHER\n");
  std::size_t wrongLineNumber = 0;
  apo_test::CheckApoCharmmError(
      [&]() {
        apo::get_expected_line(wrongLineNumber, wrongLabel, " !LABEL",
                               "test stream");
      },
      ApoCharmmErrorCode::Runtime,
      "Expected line \" !LABEL\" in test stream but found \" !OTHER\"");

  std::istringstream blanks("\n   \n");
  std::size_t blankLineNumber = 0;
  std::string line;
  apo_test::CheckApoCharmmError(
      [&]() {
        apo::get_nonblank_line(line, blankLineNumber, blanks, "payload",
                               "test stream");
      },
      ApoCharmmErrorCode::Runtime,
      "Unexpected end of file while reading payload in test stream");
  CHECK(blankLineNumber == 2);
}
