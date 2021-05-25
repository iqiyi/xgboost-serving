/*
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
02110-1301, USA.
*/
// Copyright 2010
// Author: yiwang@tencent.com (Yi Wang)
//
#include "tensorflow_serving/util/strings/split.h"

using std::set;
using std::string;
using std::vector;

// In most cases, delim contains only one character.  In this case, we
// use CalculateReserveForVector to count the number of elements
// should be reserved in result vector, and thus optimize SplitStringUsing.
template <typename StringType>
static int CalculateReserveForVector(const StringType &full,
                                     const char *delim) {
  int count = 0;
  if (delim[0] != '\0' && delim[1] == '\0') {
    // Optimize the common case where delim is a single character.
    char c = delim[0];
    const char *p = full.data();
    const char *end = p + full.size();
    while (p != end) {
      if (*p == c) { // This could be optimized with hasless(v,1) trick.
        ++p;
      } else {
        while (++p != end && *p != c) {
          // Skip to the next occurence of the delimiter.
        }
        ++count;
      }
    }
  }
  return count;
}

void SplitStringUsing(const string &full, const char *delim,
                      vector<string> *result) {
  result->reserve(CalculateReserveForVector(full, delim));
  back_insert_iterator<vector<string>> it(*result);
  SplitStringToIteratorUsing(full, delim, it);
}

void SplitStringAllowEmpty(const string &full, const char *delim,
                           vector<string> *result) {
  back_insert_iterator<vector<string>> it(*result);
  SplitStringToIteratorAllowEmpty(full, delim, it);
}

void SplitStringToSetUsing(const string &full, const char *delim,
                           set<string> *result) {
  simple_insert_iterator<set<string>> it(result);
  SplitStringToIteratorUsing(full, delim, it);
}

void SplitStringToSetAllowEmpty(const string &full, const char *delim,
                                set<string> *result) {
  simple_insert_iterator<set<string>> it(result);
  SplitStringToIteratorAllowEmpty(full, delim, it);
}

void SplitStringPieceUsing(const StringPiece &full, const char *delim,
                           vector<StringPiece> *result) {
  result->reserve(CalculateReserveForVector(full, delim));
  back_insert_iterator<vector<StringPiece>> it(*result);
  SplitStringToIteratorUsing(full, delim, it);
}

void SplitStringPieceAllowEmpty(const StringPiece &full, const char *delim,
                                vector<StringPiece> *result) {
  back_insert_iterator<vector<StringPiece>> it(*result);
  SplitStringToIteratorAllowEmpty(full, delim, it);
}

void SplitStringPieceToSetUsing(const StringPiece &full, const char *delim,
                                set<StringPiece> *result) {
  simple_insert_iterator<set<StringPiece>> it(result);
  SplitStringToIteratorUsing(full, delim, it);
}

void SplitStringPieceToSetAllowEmpty(const StringPiece &full, const char *delim,
                                     set<StringPiece> *result) {
  simple_insert_iterator<set<StringPiece>> it(result);
  SplitStringToIteratorAllowEmpty(full, delim, it);
}
