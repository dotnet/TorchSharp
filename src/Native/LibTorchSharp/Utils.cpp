#include "Utils.h"

#include <cstring>
#include <fstream>

thread_local char * torch_last_err = NULL;

const char * make_sharable_string(const std::string str)
{
    size_t size = sizeof(str);
    char* result = new char[size];
    strncpy(result, str.c_str(), size);
    result[size - 1] = '\0';
    return result;
}
