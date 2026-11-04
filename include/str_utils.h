// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

namespace apo {

void ltrim_ip(std::string &str);
void rtrim_ip(std::string &str);
void trim_ip(std::string &str);

std::string ltrim(const std::string_view str);
std::string rtrim(const std::string_view str);
std::string trim(const std::string_view str);

void to_lower_ip(std::string &str);
void to_upper_ip(std::string &str);

std::string to_lower(const std::string_view str);
std::string to_upper(const std::string_view str);

std::vector<std::string> split(const std::string_view str,
                               const std::string_view delimiter = " ");

bool try_get_line(std::string &line, std::size_t &pos,
                  const std::string_view file_data);

bool try_get_line(std::string &line, std::size_t &pos, std::size_t &line_number,
                  const std::string_view file_data);

bool try_get_line(std::string &line, std::size_t &line_number,
                  std::istream &input);

void get_line(std::string &line, std::size_t &pos,
              const std::string_view file_data);

void get_line(std::string &line, std::size_t &pos, std::size_t &line_number,
              const std::string_view file_data,
              const std::string_view record_name,
              const std::string_view source_name);

void get_line(std::string &line, std::size_t &line_number, std::istream &input,
              const std::string_view record_name,
              const std::string_view source_name);

void find_required_line(std::istream &input, std::size_t &line_number,
                        const std::string_view target,
                        const std::string_view record_name,
                        const std::string_view source_name);

bool try_get_fixed_width_field(std::string_view &field,
                               const std::string_view line,
                               const std::size_t offset,
                               const std::size_t width) noexcept;

void read_file_into_string(std::string &file_data,
                           const std::filesystem::path &file_path);

bool try_parse_int(int &value, const std::string_view token);

bool try_parse_ull(unsigned long long int &value, const std::string_view token);

bool try_parse_double(double &value, const std::string_view token);

bool try_parse_finite_double(double &value, const std::string_view token);

int parse_int(const std::string_view token, const std::string_view field_name,
              const std::string_view context);

unsigned long long int parse_ull(const std::string_view token,
                                 const std::string_view field_name,
                                 const std::string_view context);

double parse_double(const std::string_view token,
                    const std::string_view field_name,
                    const std::string_view context);

bool contains_wildcard(const std::string_view str);

std::string cDoubleToFortSciStr(const double val, const int prec);

double fortSciStrToCDouble(const std::string_view str);

// Parse a finite decimal value using E/e, D/d, or a Fortran exponent with its
// letter omitted (for example, "0.123456789012345+103"). Parsing is
// locale-independent; nonzero values that underflow to zero are rejected. These
// functions do not apply a Fortran edit descriptor's implied scale.
bool try_parse_fortran_double(double &value, const std::string_view token);

double parse_fortran_double(const std::string_view token,
                            const std::string_view field_name,
                            const std::string_view context);

// Format a finite value as a zero-scale Fortran Dw.d field. Precision is the
// number of digits after the decimal point in the normalized 0.ddd mantissa.
// Three-digit exponents omit D. Reject fields that do not fit, and values whose
// requested decimal rounding would overflow or underflow a double.
std::string format_fortran_double(const double value, const std::size_t width,
                                  const int precision);

// Read lines until a nonblank one is found. Fortran (/A) output, which CHARMM
// uses for section labels, writes an empty record before the labelled one.
void get_nonblank_line(std::string &line, std::size_t &line_number,
                       std::istream &input, const std::string_view record_name,
                       const std::string_view source_name);

// Read the next nonblank line and require it to equal `expected` exactly.
// Unlike find_required_line, no lines are skipped while looking for it.
void get_expected_line(std::size_t &line_number, std::istream &input,
                       const std::string_view expected,
                       const std::string_view source_name);

// Parse `count` consecutive Fortran Dw.d fields of `width` columns starting at
// `offset` into `values`. Only blanks may follow the last field.
void parse_fixed_width_fortran_doubles(double *values, const std::size_t count,
                                       const std::string_view line,
                                       const std::size_t offset,
                                       const std::size_t width,
                                       const std::string_view field_name,
                                       const std::string_view context);

// Read `count` values written with a (kDw.d) edit descriptor, where k is
// `values_per_record`: Fortran format reversion places k values on each record
// and the remainder on the last one. Every record must hold exactly its share.
void read_fixed_width_fortran_values(std::size_t &line_number,
                                     std::istream &input, double *values,
                                     const std::size_t count,
                                     const std::size_t values_per_record,
                                     const std::size_t width,
                                     const std::string_view record_name,
                                     const std::string_view source_name);

// Read `count` records that each hold exactly N Dw.d fields of `width` columns,
// such as one (3D22.15) record per atom.
template <std::size_t N>
void read_fixed_width_fortran_rows(std::size_t &line_number,
                                   std::istream &input,
                                   std::vector<std::array<double, N>> &rows,
                                   const std::size_t count,
                                   const std::size_t width,
                                   const std::string_view record_name,
                                   const std::string_view source_name) {
  rows.resize(count);

  for (std::array<double, N> &row : rows) {
    apo::read_fixed_width_fortran_values(line_number, input, row.data(), N, N,
                                         width, record_name, source_name);
  }

  return;
}

// Write `count` values as a (kDw.d) edit descriptor would, where k is
// `values_per_record`: k fields per record and the remainder on the last one.
// This is the inverse of read_fixed_width_fortran_values.
void write_fixed_width_fortran_values(std::ostream &output,
                                      const double *values,
                                      const std::size_t count,
                                      const std::size_t values_per_record,
                                      const std::size_t width,
                                      const int precision);

std::string get_fixed_width_field(const std::string_view line,
                                  const std::size_t offset,
                                  const std::size_t width,
                                  const std::string_view field_name,
                                  const std::string_view context);

double parse_fixed_width_double(const std::string_view line,
                                const std::size_t offset,
                                const std::size_t width,
                                const std::string_view field_name,
                                const std::string_view context);

double parse_fixed_width_fortran_double(const std::string_view line,
                                        const std::size_t offset,
                                        const std::size_t width,
                                        const std::string_view field_name,
                                        const std::string_view context);

int parse_fixed_width_int(const std::string_view line, const std::size_t offset,
                          const std::size_t width,
                          const std::string_view field_name,
                          const std::string_view context);

unsigned long long int parse_fixed_width_ull(const std::string_view line,
                                             const std::size_t offset,
                                             const std::size_t width,
                                             const std::string_view field_name,
                                             const std::string_view context);

} // namespace apo
