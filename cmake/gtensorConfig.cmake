set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

include("${CMAKE_CURRENT_LIST_DIR}/gtensorTargets.cmake")