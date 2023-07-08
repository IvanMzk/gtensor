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

        template<typename T=int, typename Order=config::c_order, typename U, typename Size>
        auto integers(const U& low_, const U& high_, Size&& size, bool end_point=false){
            using value_type = math::make_integral_t<T>;
            using tensor_type = tensor<T,Order,config::extend_config_t<Config,T>>;
            using shape_type = typename tensor_type::shape_type;
            static_assert(math::numeric_traits<U>::is_integral(),"low,high must be of integral type");
            tensor_type res(detail::make_shape_of_type<shape_type>(std::forward<Size>(size)));
            const auto low = static_cast<value_type>(low_);
            const auto high = end_point ? static_cast<value_type>(high_) : static_cast<value_type>(high_-1);
            generate_distribution(res.begin(), res.end(), bit_generator_, std::uniform_int_distribution<value_type>{low,high});
            return res;
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