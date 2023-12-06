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

template<typename U>
ALWAYS_INLINE auto avx_zero(){
    if constexpr (std::is_same_v<U,double>){
        return _mm256_setzero_pd();
    }else if constexpr (std::is_same_v<U,float>){
        return _mm256_setzero_ps();
    }else if constexpr (std::is_same_v<U,std::complex<double>>){
        return _mm256_setzero_pd();
    }else if constexpr (std::is_same_v<U,std::complex<float>>){
        return _mm256_setzero_ps();
    }else{
        static_assert(detail::always_false<U>);
    }
}
template<typename U>
ALWAYS_INLINE auto avx_broadcast(const U* const buf){
    if constexpr (std::is_same_v<U,double>){
        return _mm256_broadcast_sd(buf);
    }else if constexpr (std::is_same_v<U,float>){
        return _mm256_broadcast_ss(buf);
    }else if constexpr (std::is_same_v<U,std::complex<double>>){
        return _mm256_broadcast_pd(reinterpret_cast<const __m128d*>(buf));
    }else if constexpr (std::is_same_v<U,std::complex<float>>){
        return _mm256_castpd_ps(_mm256_broadcast_sd(reinterpret_cast<const double*>(reinterpret_cast<const float*>(buf))));
    }else{
        static_assert(detail::always_false<U>);
    }
}
template<typename U>
ALWAYS_INLINE auto avx_load(const U* const buf){
    if constexpr (std::is_same_v<U,double>){
        return _mm256_load_pd(buf);
    }else if constexpr (std::is_same_v<U,float>){
        return _mm256_load_ps(buf);
    }else if constexpr (std::is_same_v<U,std::complex<double>>){
        return _mm256_load_pd(reinterpret_cast<const double*>(buf));
    }else if constexpr (std::is_same_v<U,std::complex<float>>){
        return _mm256_load_ps(reinterpret_cast<const float*>(buf));
    }else{
        static_assert(detail::always_false<U>);
    }
}
template<typename U>
ALWAYS_INLINE auto avx_loadu(const U* const buf){
    if constexpr (std::is_same_v<U,double>){
        return _mm256_loadu_pd(buf);
    }else if constexpr (std::is_same_v<U,float>){
        return _mm256_loadu_ps(buf);
    }else if constexpr (std::is_same_v<U,std::complex<double>>){
        return _mm256_loadu_pd(reinterpret_cast<const double*>(buf));
    }else if constexpr (std::is_same_v<U,std::complex<float>>){
        return _mm256_loadu_ps(reinterpret_cast<const float*>(buf));
    }else{
        static_assert(detail::always_false<U>);
    }
}
template<typename U, typename Y>
ALWAYS_INLINE auto avx_store(U* const buf, Y y){
    if constexpr (std::is_same_v<U,double>){
        return _mm256_store_pd(buf,y);
    }else if constexpr (std::is_same_v<U,float>){
        return _mm256_store_ps(buf,y);
    }else if constexpr (std::is_same_v<U,std::complex<double>>){
        return _mm256_store_pd(reinterpret_cast<double*>(buf),y);
    }else if constexpr (std::is_same_v<U,std::complex<float>>){
        return _mm256_store_ps(reinterpret_cast<float*>(buf),y);
    }else{
        static_assert(detail::always_false<U>);
    }
}
template<typename U, typename Y>
ALWAYS_INLINE auto avx_mul(Y a, Y b){
    if constexpr (std::is_same_v<U,double>){
        return _mm256_mul_pd(a,b);
    }else if constexpr (std::is_same_v<U,float>){
        return _mm256_mul_ps(a,b);
    }else if constexpr (std::is_same_v<U,std::complex<double>>){
        auto y0 = _mm256_movedup_pd(a);
        auto y1 = _mm256_shuffle_pd(a,a,0b1111);
        y0 = _mm256_mul_pd(y0,b);
        y1 = _mm256_mul_pd(y1,b);
        y1 = _mm256_shuffle_pd(y1,y1,0b0101);
        return _mm256_addsub_pd(y0,y1);
    }else if constexpr (std::is_same_v<U,std::complex<float>>){
        auto y0 = _mm256_moveldup_ps(a);
        auto y1 = _mm256_shuffle_ps(a,a,0b11'11'01'01);
        y0 = _mm256_mul_ps(y0,b);
        y1 = _mm256_mul_ps(y1,b);
        y1 = _mm256_shuffle_ps(y1,y1,0b10'11'00'01);
        return _mm256_addsub_ps(y0,y1);
    }else{
        static_assert(detail::always_false<U>);
    }
}
template<typename U, typename Y>
ALWAYS_INLINE auto avx_madd(Y a, Y b, Y c){
    if constexpr (std::is_same_v<U,double>){
        if constexpr (HAS_FMA){
            return _mm256_fmadd_pd(a,b,c);
        }else{
            return _mm256_add_pd(_mm256_mul_pd(a,b),c);
        }
    }else if constexpr (std::is_same_v<U,float>){
        if constexpr (HAS_FMA){
            return _mm256_fmadd_ps(a,b,c);
        }else{
            return _mm256_add_ps(_mm256_mul_ps(a,b),c);
        }
    }else if constexpr (std::is_same_v<U,std::complex<double>>){
        return _mm256_add_pd(c,avx_mul<std::complex<double>>(a,b));
    }else if constexpr (std::is_same_v<U,std::complex<float>>){
        return _mm256_add_ps(c,avx_mul<std::complex<float>>(a,b));
    }else{
        static_assert(detail::always_false<U>);
    }
}

}   //end of namespace detail
}   //end of namespace gtensor
#endif