#ifndef RANDOM_HPP_
#define RANDOM_HPP_

#include <random>
#include "tensor.hpp"

namespace gtensor{

//random module implementation

struct random
{

    template<typename It, typename BitGenerator, typename Distribution>
    static void generate_distribution(It first, It last, BitGenerator& bit_generator, Distribution distribution){
        std::generate(
            first,
            last,
            [&bit_generator,&distribution](){
                return distribution(bit_generator);
            }
        );
    }

    template<typename T, typename Order, typename Config, typename Size, typename BitGenerator, typename Distribution>
    static auto make_distribution(Size&& size, BitGenerator& bit_generator, Distribution distribution){
        using tensor_type = tensor<T,Order,config::extend_config_t<Config,T>>;
        using shape_type = typename tensor_type::shape_type;
        tensor_type res(detail::make_shape_of_type<shape_type>(std::forward<Size>(size)));
        generate_distribution(res.begin(), res.end(), bit_generator, distribution);
        return res;
    }

    template<typename Config, typename BitGenerator>
    class generator
    {
        using bit_generator_type = BitGenerator;

        bit_generator_type bit_generator_{};

        template<typename...Seeds>
        bit_generator_type make_bit_generator(const Seeds&...seeds){
            auto seq = std::seed_seq(std::initializer_list<int>{seeds...});
            return bit_generator_type(seq);
        }
        template<typename U>
        bit_generator_type make_bit_generator(std::initializer_list<U> seeds){
            auto seq = std::seed_seq(seeds);
            return bit_generator_type(seq);
        }
    public:
        generator() = default;

        template<typename...Seeds>
        explicit generator(const Seeds&...seeds):
            bit_generator_{make_bit_generator(seeds...)}
        {}
        template<typename U>
        generator(std::initializer_list<U> seeds):
            bit_generator_{make_bit_generator(seeds)}
        {}

        //make tensor of samples of integral type drawn from uniform distribution
        template<typename T=int, typename Order=config::c_order, typename U, typename Size>
        auto integers(const U& low_, const U& high_, Size&& size, bool end_point=false){
            static_assert(math::numeric_traits<T>::is_integral(),"T must be of integral type");
            static_assert(math::numeric_traits<U>::is_integral(),"low,high must be of integral type");
            const auto low = static_cast<T>(low_);
            const auto high = end_point ? static_cast<T>(high_) : static_cast<T>(high_-1);
            return make_distribution<T,Order,Config>(std::forward<Size>(size), bit_generator_, std::uniform_int_distribution<T>(low,high));
        }

        //make tensor of samples of floating point type drawn from uniform distribution
        template<typename T=double, typename Order=config::c_order, typename U, typename Size>
        auto random(const U& low_, const U& high_, Size&& size){
            static_assert(math::numeric_traits<T>::is_floating_point(),"T must be of floating point type");
            const auto low = static_cast<T>(low_);
            const auto high = static_cast<T>(high_);
            return make_distribution<T,Order,Config>(std::forward<Size>(size), bit_generator_, std::uniform_real_distribution<T>(low,high));
        }

        //make tensor of samples drawn from a binomial distribution
        //drawn sample represents number of successes in sequence of n experiments, each of which succeeds with probability p
        //n trials number, must be >=0
        //p probability of success, in range [0,1]
        template<typename T=int, typename Order=config::c_order, typename U, typename V, typename Size>
        auto binomial(const U& n_, const V& p_, Size&& size){
            static_assert(math::numeric_traits<T>::is_integral(),"T must be of integral type");
            const auto n = static_cast<math::make_integral_t<U>>(n_);
            const auto p = static_cast<math::make_floating_point_t<V>>(p_);
            return make_distribution<T,Order,Config>(std::forward<Size>(size), bit_generator_, std::binomial_distribution<T>(n,p));
        }

        //make tensor of samples drawn from a negative binomial distribution
        //drawn sample represents number of failures in a sequence of experiments, each succeeds with probability p, before exactly k successes occur
        //k successes number, must be >0
        //p probability of success, in range (0,1]
        template<typename T=int, typename Order=config::c_order, typename U, typename V, typename Size>
        auto negative_binomial(const U& k_, const V& p_, Size&& size){
            static_assert(math::numeric_traits<T>::is_integral(),"T must be of integral type");
            const auto k = static_cast<math::make_integral_t<U>>(k_);
            const auto p = static_cast<math::make_floating_point_t<V>>(p_);
            return make_distribution<T,Order,Config>(std::forward<Size>(size), bit_generator_, std::negative_binomial_distribution<T>(k,p));
        }

        //make tensor of samples drawn from a poisson distribution
        //drawn sample represents number of occurrences of random event, if the expected, number of its occurrence under the same conditions (on the same time/space interval) is mean.
        //mean expected number of events occurring in a fixed-time/space interval, must be > 0
        template<typename T=int, typename Order=config::c_order, typename V, typename Size>
        auto poisson(const V& mean_, Size&& size){
            static_assert(math::numeric_traits<T>::is_integral(),"T must be of integral type");
            const auto mean = static_cast<math::make_floating_point_t<V>>(mean_);
            return make_distribution<T,Order,Config>(std::forward<Size>(size), bit_generator_, std::poisson_distribution<T>(mean));
        }

        //make tensor of samples drawn from a exponential distribution
        //drawn sample represents the time/distance until the next random event if random events occur at constant rate lambda per unit of time/distance.
        //lambda - rate of event occurrence
        template<typename T=double, typename Order=config::c_order, typename V, typename Size>
        auto exponential(const V& lambda_, Size&& size){
            static_assert(math::numeric_traits<T>::is_floating_point(),"T must be of floating point type");
            const auto lambda = static_cast<math::make_floating_point_t<V>>(lambda_);
            return make_distribution<T,Order,Config>(std::forward<Size>(size), bit_generator_, std::exponential_distribution<T>(lambda));
        }

        //make tensor of samples drawn from a gamma distribution
        //shape - gamma distribution parameter, sometimes designated "k"
        //scale - gamma distribution parameter, sometimes designated "theta"
        template<typename T=double, typename Order=config::c_order, typename U, typename V, typename Size>
        auto gamma(const U& shape_, const V& scale_, Size&& size){
            static_assert(math::numeric_traits<T>::is_floating_point(),"T must be of floating point type");
            const auto shape = static_cast<math::make_floating_point_t<U>>(shape_);
            const auto scale = static_cast<math::make_floating_point_t<V>>(scale_);
            return make_distribution<T,Order,Config>(std::forward<Size>(size), bit_generator_, std::gamma_distribution<T>(shape,scale));
        }

        //make tensor of samples drawn from a weibull distribution
        //shape - weibull distribution parameter, sometimes designated "k"
        //scale - weibull distribution parameter, sometimes designated "lambda"
        template<typename T=double, typename Order=config::c_order, typename U, typename V, typename Size>
        auto weibull(const U& shape_, const V& scale_, Size&& size){
            static_assert(math::numeric_traits<T>::is_floating_point(),"T must be of floating point type");
            const auto shape = static_cast<math::make_floating_point_t<U>>(shape_);
            const auto scale = static_cast<math::make_floating_point_t<V>>(scale_);
            return make_distribution<T,Order,Config>(std::forward<Size>(size), bit_generator_, std::weibull_distribution<T>(shape,scale));
        }

    };

    template<typename BitGenerator, typename Config=config::default_config, typename...Seeds>
    static auto make_rng(const Seeds&...seeds){
        return generator<Config, BitGenerator>(seeds...);
    }

    template<typename Config=config::default_config, typename...Seeds>
    static auto default_rng(const Seeds&...seeds){
        return generator<Config, std::mt19937_64>(seeds...);
    }

};  //end of struct random

//random module frontend

template<typename BitGenerator, typename Config=config::default_config, typename...Seeds>
auto make_rng(const Seeds&...seeds){
    return random_selector_t<Config>::template make_rng<BitGenerator, Config>(seeds...);
}

template<typename Config=config::default_config, typename...Seeds>
auto default_rng(const Seeds&...seeds){
    return random_selector_t<Config>::template default_rng<Config>(seeds...);
}


}   //end of namespace gtensor
#endif