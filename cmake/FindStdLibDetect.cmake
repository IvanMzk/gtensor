# GTensor - computation library
# Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
#
# Distributed under the Boost Software License, Version 1.0.
# The full license is in the file LICENSE.txt, distributed with this software.

#FindStdLibDetect module determines which standart libray is used: libstdc++ or libc++
#This module defines variables StdLibDetect_LIBSTDC++, StdLibDetect_LIBC++ or StdLibDetect_MSVCLIBC++ depending on test result

include(CheckCXXSourceCompiles)

set (LIBSTDC++_DETECT_SOURCE [[
#include <iostream>
int a =
#ifdef __GLIBCXX__
1;
#endif
int main(){
    return 0;
}
]])

set (LIBC++_DETECT_SOURCE [[
#include <iostream>
int a =
#ifdef _LIBCPP_VERSION
1;
#endif
int main(){
    return 0;
}
]])

set (MSVCLIBC++_DETECT_SOURCE [[
#include <iostream>
int a =
#ifdef _CPPLIB_VER
1;
#endif
int main(){
    return 0;
}
]])

check_cxx_source_compiles("${LIBSTDC++_DETECT_SOURCE}" StdLibDetect_LIBSTDC++)
check_cxx_source_compiles("${LIBC++_DETECT_SOURCE}" StdLibDetect_LIBC++)
check_cxx_source_compiles("${MSVCLIBC++_DETECT_SOURCE}" StdLibDetect_MSVCLIBC++)

