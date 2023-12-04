#ifndef AVX_HELPER_
#define AVX_HELPER_

#include <complex>
#include <immintrin.h>
#include "common.hpp"

#if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__)
    #if defined(__AVX__)
        #define HAS_AVX 1
    #else
        #define HAS_AVX 0
    #endif
    #if defined(__AVX2__)
        #define HAS_AVX2 1
    #else
        #define HAS_AVX2 0
    #endif
    #if defined(__FMA__)
        #define HAS_FMA 1
    #else
        #define HAS_FMA 0
    #endif
#elif defined(_MSC_VER)
    #if defined(__AVX__)
        #define HAS_AVX 1
    #else
        #define HAS_AVX 0
    #endif
    #if defined(__AVX2__)
        #define HAS_AVX2 1
        #define HAS_FMA 1
    #else
        #define HAS_AVX2 0
        #define HAS_FMA 0
    #endif
#else
    #define HAS_AVX 0
    #define HAS_AVX2 0
    #define HAS_FMA 0
#endif

namespace gtensor{
namespace detail{

//primary declarations
template<typename U>
ALWAYS_INLINE auto avx_mul(__m256d a, __m256d b);
template<typename U>
ALWAYS_INLINE auto avx_mul(__m256 a, __m256 b);
template<typename U>
ALWAYS_INLINE auto avx_madd(__m256d a, __m256d b, __m256d c);
template<typename U>
ALWAYS_INLINE auto avx_madd(__m256 a, __m256 b, __m256 c);

//floating point helpers
template<typename U>
ALWAYS_INLINE auto avx_zero(){
    if constexpr (std::is_same_v<U,double>){
        return _mm256_setzero_pd();
    }else if constexpr (std::is_same_v<U,float>){
        return _mm256_setzero_ps();
    }else{
        static_assert(detail::always_false<U>);
    }
}
ALWAYS_INLINE auto avx_broadcast(const double* const buf){
    return _mm256_broadcast_sd(buf);
}
ALWAYS_INLINE auto avx_broadcast(const float* const buf){
    return _mm256_broadcast_ss(buf);
}
ALWAYS_INLINE auto avx_load(const double* const buf){
    return _mm256_load_pd(buf);
}
ALWAYS_INLINE auto avx_load(const float* const buf){
    return _mm256_load_ps(buf);
}
ALWAYS_INLINE auto avx_loadu(const double* const buf){
    return _mm256_loadu_pd(buf);
}
ALWAYS_INLINE auto avx_loadu(const float* const buf){
    return _mm256_loadu_ps(buf);
}
ALWAYS_INLINE void avx_store(double* const buf, __m256d y){
    _mm256_store_pd(buf, y);
}
ALWAYS_INLINE void avx_store(float* const buf, __m256 y){
    _mm256_store_ps(buf, y);
}
template<>
ALWAYS_INLINE auto avx_mul<double>(__m256d a, __m256d b){
    return _mm256_mul_pd(a,b);
}
template<>
ALWAYS_INLINE auto avx_mul<float>(__m256 a, __m256 b){
    return _mm256_mul_ps(a,b);
}
template<>
ALWAYS_INLINE auto avx_madd<double>(__m256d a, __m256d b, __m256d c){
    return _mm256_fmadd_pd(a,b,c);
}
template<>
ALWAYS_INLINE auto avx_madd<float>(__m256 a, __m256 b, __m256 c){
    return _mm256_fmadd_ps(a,b,c);
}

//complex helpers
ALWAYS_INLINE auto avx_broadcast(const std::complex<double>* const buf){
    return _mm256_broadcast_pd(reinterpret_cast<const __m128d*>(buf));
}
ALWAYS_INLINE auto avx_broadcast(const std::complex<float>* const buf){
    return _mm256_castpd_ps(_mm256_broadcast_sd(reinterpret_cast<const double*>(reinterpret_cast<const float*>(buf))));
}
ALWAYS_INLINE auto avx_load(const std::complex<double>* const buf){
    return _mm256_load_pd(reinterpret_cast<const double*>(buf));
}
ALWAYS_INLINE auto avx_load(const std::complex<float>* const buf){
    return _mm256_load_ps(reinterpret_cast<const float*>(buf));
}
ALWAYS_INLINE auto avx_loadu(const std::complex<double>* const buf){
    return _mm256_loadu_pd(reinterpret_cast<const double*>(buf));
}
ALWAYS_INLINE auto avx_loadu(const std::complex<float>* const buf){
    return _mm256_loadu_ps(reinterpret_cast<const float*>(buf));
}
ALWAYS_INLINE void avx_store(std::complex<double>* const buf, __m256d y){
    _mm256_store_pd(reinterpret_cast<double*>(buf),y);
}
ALWAYS_INLINE void avx_store(std::complex<float>* const buf, __m256 y){
    _mm256_store_ps(reinterpret_cast<float*>(buf),y);
}
template<>
ALWAYS_INLINE auto avx_mul<std::complex<double>>(__m256d a, __m256d b){
    auto y0 = _mm256_movedup_pd(a);
    auto y1 = _mm256_shuffle_pd(a,a,0b1111);
    y0 = _mm256_mul_pd(y0,b);
    y1 = _mm256_mul_pd(y1,b);
    y1 = _mm256_shuffle_pd(y1,y1,0b0101);
    return _mm256_addsub_pd(y0,y1);
}
template<>
ALWAYS_INLINE auto avx_mul<std::complex<float>>(__m256 a, __m256 b){
    auto y0 = _mm256_moveldup_ps(a);
    auto y1 = _mm256_shuffle_ps(a,a,0b11'11'01'01);
    y0 = _mm256_mul_ps(y0,b);
    y1 = _mm256_mul_ps(y1,b);
    y1 = _mm256_shuffle_ps(y1,y1,0b10'11'00'01);
    return _mm256_addsub_ps(y0,y1);
}
template<>
ALWAYS_INLINE auto avx_madd<std::complex<double>>(__m256d a, __m256d b, __m256d c){
    return _mm256_add_pd(c,avx_mul<std::complex<double>>(a,b));
}
template<>
ALWAYS_INLINE auto avx_madd<std::complex<float>>(__m256 a, __m256 b, __m256 c){
    return _mm256_add_ps(c,avx_mul<std::complex<float>>(a,b));
}
}   //end of namespace detail
}   //end of namespace gtensor
#endif