/*

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Author: Gaidong Mou  (mougaidong@qiyi.com)
// Author: Hao Ziyu <haoziyu@qiyi.com>
// This class contains utility functions for processing numeric strings.

#ifndef STRINGS_NUMERIC_H_
#define STRINGS_NUMERIC_H_

#include <string>

// Checks whether 'str' is a decimal digit string, return true if so.
bool IsDigitalString(const std::string &str);

// Convert numeric string 'str' to integer,  result stored in 'out'.
// Return true if converted successfully, otherwise false.
bool safe_strtol(const char *str, int32_t *out);
bool safe_strtol(const std::string &str, int32_t *out);

bool safe_strtoul(const char *str, uint32_t *out);
bool safe_strtoul(const std::string &str, uint32_t *out);

bool safe_strtoll(const char *str, int64_t *out);
bool safe_strtoll(const std::string &str, int64_t *out);

bool safe_strtoull(const char *str, uint64_t *out);
bool safe_strtoull(const std::string &str, uint64_t *out);

// For floating point number.
bool safe_strtof(const char *str, float *out);
bool safe_strtof(const std::string &str, float *out);
bool safe_strtod(const char *str, double *out);
bool safe_strtod(const std::string &str, double *out);

#endif
