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

#include "tensorflow_serving/util/strings/numeric.h"

#include <cctype>
#include <cstdlib>
#include <cstring>

using namespace std;

bool IsDigitalString(const string &str) {
  for (size_t i = 0; i < str.length(); ++i) {
    if (!isdigit(str[i])) {
      return false;
    }
  }

  return true;
}

bool safe_strtol(const char *str, int32_t *out) {
  errno = 0;
  *out = 0;
  char *endptr;
  long l = strtol(str, &endptr, 10);
  if (errno == ERANGE) {
    return false;
  }

  if (isspace(*endptr) || (*endptr == '\0' && endptr != str)) {
    *out = l;
    return true;
  }
  return false;
}

bool safe_strtol(const string &str, int32_t *out) {
  return safe_strtol(str.c_str(), out);
}

bool safe_strtoul(const char *str, uint32_t *out) {
  char *endptr = NULL;
  unsigned long l = 0;
  *out = 0;
  errno = 0;

  l = strtoul(str, &endptr, 10);
  if (errno == ERANGE) {
    return false;
  }

  if (isspace(*endptr) || (*endptr == '\0' && endptr != str)) {
    if ((long)l < 0) {
      /* only check for negative signs in the uncommon case when
       * the unsigned number is so big that it's negative as a
       * signed number. */
      if (strchr(str, '-') != NULL) {
        return false;
      }
    }
    *out = l;
    return true;
  }

  return false;
}

bool safe_strtoul(const string &str, uint32_t *out) {
  return safe_strtoul(str.c_str(), out);
}

bool safe_strtoll(const char *str, int64_t *out) {
  errno = 0;
  *out = 0;
  char *endptr;
  long long ll = strtoll(str, &endptr, 10);
  if (errno == ERANGE) {
    return false;
  }

  if (isspace(*endptr) || (*endptr == '\0' && endptr != str)) {
    *out = ll;
    return true;
  }

  return false;
}

bool safe_strtoll(const string &str, int64_t *out) {
  return safe_strtoll(str.c_str(), out);
}

bool safe_strtoull(const char *str, uint64_t *out) {
  errno = 0;
  *out = 0;
  char *endptr;
  unsigned long long ull = strtoull(str, &endptr, 10);
  if (errno == ERANGE) {
    return false;
  }

  if (isspace(*endptr) || (*endptr == '\0' && endptr != str)) {
    if ((long long)ull < 0) {
      /* only check for negative signs in the uncommon case when
       * the unsigned number is so big that it's negative as a
       * signed number. */
      if (strchr(str, '-') != NULL) {
        return false;
      }
    }
    *out = ull;
    return true;
  }

  return false;
}

bool safe_strtoull(const string &str, uint64_t *out) {
  return safe_strtoull(str.c_str(), out);
}

bool safe_strtof(const char *str, float *out) {
  errno = 0;
  *out = 0;
  char *endptr;

  const float f = strtof(str, &endptr);
  if (errno == ERANGE) {
    return false;
  }

  if (isspace(*endptr) || (*endptr == '\0' && endptr != str)) {
    *out = f;
    return true;
  }

  return false;
}

bool safe_strtof(const std::string &str, float *out) {
  return safe_strtof(str.c_str(), out);
}

bool safe_strtod(const char *str, double *out) {
  errno = 0;
  *out = 0;
  char *endptr;

  const double d = strtod(str, &endptr);
  if (errno == ERANGE) {
    return false;
  }

  if (isspace(*endptr) || (*endptr == '\0' && endptr != str)) {
    *out = d;
    return true;
  }

  return false;
}

bool safe_strtod(const string &str, double *out) {
  return safe_strtod(str.c_str(), out);
}
