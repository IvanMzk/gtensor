# GTensor - computation library
# Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
#
# Distributed under the Boost Software License, Version 1.0.
# The full license is in the file LICENSE.txt, distributed with this software.

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)
include(${CMAKE_CURRENT_LIST_DIR}/FindAvxDetect.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/gtensorTargets.cmake)