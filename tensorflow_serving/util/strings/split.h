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
// Copyright 2010 Tencent Inc.
// Author: yiwang@tencent.com (Yi Wang)
//
// This file declares string splitting utilities.
//
#ifndef STRINGS_SPLIT_H_
#define STRINGS_SPLIT_H_

#include <set>
#include <string>
#include <vector>

#include "tensorflow_serving/util/strings/string_piece.h"

// Subdivide string |full| into substrings according to delimitors
// given in |delim|.  |delim| should pointing to a string including
// one or more characters.  Each character is considerred a possible
// delimitor.  For example,
//   vector<string> substrings;
//   SplitStringUsing("apple orange\tbanana", "\t ", &substrings);
// results in three substrings:
//   substrings.size() == 3
//   substrings[0] == "apple"
//   substrings[1] == "orange"
//   substrings[2] == "banana"
void SplitStringUsing(const std::string &full, const char *delim,
                      std::vector<std::string> *result);

// This function differs from SplitStringUsing as it parses adjacent delimitors
// into empty strings while SplitStringUsing takes them as a single delimitor,
// and it does not ignore leading and trailing delimitors while SplitStringUsing
// does.
void SplitStringAllowEmpty(const std::string &full, const char *delim,
                           std::vector<std::string> *result);

// This function has the same semnatic as SplitStringUsing.  Results
// are saved in an STL set container.
void SplitStringToSetUsing(const std::string &full, const char *delim,
                           std::set<std::string> *result);

// Like SplitStringAllowEmpty, but results are saved in an STL set container.
void SplitStringToSetAllowEmpty(const std::string &full, const char *delim,
                                std::set<std::string> *result);

// Similar to above, split for StringPiece.
void SplitStringPieceUsing(const StringPiece &full, const char *delim,
                           std::vector<StringPiece> *result);
void SplitStringPieceAllowEmpty(const StringPiece &full, const char *delim,
                                std::vector<StringPiece> *result);
void SplitStringPieceToSetUsing(const StringPiece &full, const char *delim,
                                std::set<StringPiece> *result);
void SplitStringPieceToSetAllowEmpty(const StringPiece &full, const char *delim,
                                     std::set<StringPiece> *result);

template <typename T> struct simple_insert_iterator {
  explicit simple_insert_iterator(T *t) : t_(t) {}

  simple_insert_iterator<T> &operator=(const typename T::value_type &value) {
    t_->insert(value);
    return *this;
  }

  simple_insert_iterator<T> &operator*() { return *this; }
  simple_insert_iterator<T> &operator++() { return *this; }
  simple_insert_iterator<T> &operator++(int placeholder) { return *this; }

  T *t_;
};

template <typename T> struct back_insert_iterator {
  explicit back_insert_iterator(T &t) : t_(t) {}

  back_insert_iterator<T> &operator=(const typename T::value_type &value) {
    t_.push_back(value);
    return *this;
  }

  back_insert_iterator<T> &operator*() { return *this; }
  back_insert_iterator<T> &operator++() { return *this; }
  back_insert_iterator<T> operator++(int placeholder) { return *this; }

  T &t_;
};

template <typename StringType, typename ITR>
static inline void SplitStringToIteratorUsing(const StringType &full,
                                              const char *delim, ITR &result) {
  // Optimize the common case where delim is a single character.
  if (delim[0] != '\0' && delim[1] == '\0') {
    char c = delim[0];
    const char *p = full.data();
    const char *end = p + full.size();
    while (p != end) {
      if (*p == c) {
        ++p;
      } else {
        const char *start = p;
        while (++p != end && *p != c) {
          // Skip to the next occurence of the delimiter.
        }
        *result++ = StringType(start, p - start);
      }
    }
    return;
  }

  std::string::size_type begin_index, end_index;
  begin_index = full.find_first_not_of(delim);
  while (begin_index != std::string::npos) {
    end_index = full.find_first_of(delim, begin_index);
    if (end_index == std::string::npos) {
      *result++ = full.substr(begin_index);
      return;
    }
    *result++ = full.substr(begin_index, (end_index - begin_index));
    begin_index = full.find_first_not_of(delim, end_index);
  }
}

template <typename StringType, typename ITR>
static inline void SplitStringToIteratorAllowEmpty(const StringType &full,
                                                   const char *delim,
                                                   ITR &result) {
  // Optimize the common case where delim is a single character.
  if (delim[0] != '\0' && delim[1] == '\0') {
    char c = delim[0];
    const char *start = full.data();
    const char *end = start + full.size();
    for (const char *p = full.data(); p != end; ++p) {
      if (*p == c) {
        *result++ = StringType(start, p - start);
        start = p + 1;
      }
    }
    *result++ = StringType(start, end - start);
    return;
  }

  std::string::size_type begin_index, end_index;
  for (begin_index = 0; begin_index != std::string::npos;) {
    end_index = full.find_first_of(delim, begin_index);
    if (end_index == std::string::npos) {
      *result++ = full.substr(begin_index);
      return;
    }
    *result++ = full.substr(begin_index, (end_index - begin_index));
    begin_index = end_index + 1;
  }
}

// This function is used to subdivide full_string
// into key-value map pairs.
// Normally, full_string is a GET query string.
// The format is as follow: "a=1&b=5&c=9",
// in this case, group_separator is "&" and
// key_value_separator is "=".
// So, the result_pair is {a:1, b:5, c:9}.
template <typename MapType>
void SplitStringToKeyValuePair(const StringPiece &full_string,
                               const char *group_separator,
                               const char *key_value_separator,
                               MapType *result_pairs) {
  std::vector<StringPiece> params;
  SplitStringPieceUsing(full_string, group_separator, &params);

  for (size_t i = 0; i < params.size(); ++i) {
    std::vector<StringPiece> pair;
    pair.reserve(2);
    SplitStringPieceUsing(params[i], key_value_separator, &pair);
    if (pair.size() == 2) {
      result_pairs->insert(make_pair(pair[0].ToString(), pair[1].ToString()));
    }
  }
}

#endif
