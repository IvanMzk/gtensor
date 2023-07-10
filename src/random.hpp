#ifndef RANDOM_HPP_
#define RANDOM_HPP_

#include <random>
#include <algorithm>
#include "tensor.hpp"
#include "reduce.hpp"

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

        template<typename Container>
        bit_generator_type make_bit_generator(const Container& seeds){
            auto seq = std::seed_seq(seeds.begin(),seeds.end());
            return bit_generator_type(seq);
        }

    public:
        generator() = default;

        template<typename Container>
        generator(const Container& seeds):
            bit_generator_{make_bit_generator(seeds)}
        {}

        template<typename U>
        generator(std::initializer_list<U> seeds):
            bit_generator_{make_bit_generator(seeds)}
        {}

        template<typename...Seeds>
        explicit generator(const Seeds&...seeds):
            generator(std::initializer_list<unsigned int>{static_cast<unsigned int>(seeds)...})
        {}

        //distribution generation methods
        //common description:
        //method return tensor of random samples
        //method has size parameter which can be scalar or container which specifies tensor's shape
        //method can be parameterized with value_type, and layout (template parameters)

        //make tensor of samples of integral type drawn from uniform distribution
        //samples drawn from ranga [low, high) if end_point false
        //samples drawn from ranga [low, high] if end_point true
        template<typename T=int, typename Order=config::c_order, typename U, typename Size>
        auto integers(const U& low_, const U& high_, Size&& size, bool end_point=false){
            static_assert(math::numeric_traits<T>::is_integral(),"T must be of integral type");
            static_assert(math::numeric_traits<U>::is_integral(),"low,high must be of integral type");
            const auto low = static_cast<T>(low_);
            const auto high = end_point ? static_cast<T>(high_) : static_cast<T>(high_-1);
            return make_distribution<T,Order,Config>(std::forward<Size>(size), bit_generator_, std::uniform_int_distribution<T>(low,high));
        }

        //make tensor of samples of floating point type drawn from uniform distribution
        //samples drawn from range [low,high)
        template<typename T=double, typename Order=config::c_order, typename U, typename Size>
        auto uniform(const U& low_, const U& high_, Size&& size){
            static_assert(math::numeric_traits<T>::is_floating_point(),"T must be of floating point type");
            const auto low = static_cast<T>(low_);
            const auto high = static_cast<T>(high_);
            return make_distribution<T,Order,Config>(std::forward<Size>(size), bit_generator_, std::uniform_real_distribution<T>(low,high));
        }

        //make tensor of samples of floating point type drawn from uniform distribution
        //samples drawn from range [0,1)
        template<typename T=double, typename Order=config::c_order, typename Size>
        auto random(Size&& size){
            static_assert(math::numeric_traits<T>::is_floating_point(),"T must be of floating point type");
            return uniform<T>(0.0,1.0,std::forward<Size>(size));
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

        //make tensor of samples drawn from a normal distribution
        //mean - normal distribution parameter
        //stdev - normal distribution parameter, must be >0
        template<typename T=double, typename Order=config::c_order, typename U, typename V, typename Size>
        auto normal(const U& mean_, const V& stdev_, Size&& size){
            static_assert(math::numeric_traits<T>::is_floating_point(),"T must be of floating point type");
            const auto mean = static_cast<math::make_floating_point_t<U>>(mean_);
            const auto stdev = static_cast<math::make_floating_point_t<V>>(stdev_);
            return make_distribution<T,Order,Config>(std::forward<Size>(size), bit_generator_, std::normal_distribution<T>(mean,stdev));
        }

        //make tensor of samples drawn from a lognormal distribution
        //mean - lognormal distribution parameter
        //stdev - lognormal distribution parameter, must be >0
        template<typename T=double, typename Order=config::c_order, typename U, typename V, typename Size>
        auto lognormal(const U& mean_, const V& stdev_, Size&& size){
            static_assert(math::numeric_traits<T>::is_floating_point(),"T must be of floating point type");
            const auto mean = static_cast<math::make_floating_point_t<U>>(mean_);
            const auto stdev = static_cast<math::make_floating_point_t<V>>(stdev_);
            return make_distribution<T,Order,Config>(std::forward<Size>(size), bit_generator_, std::lognormal_distribution<T>(mean,stdev));
        }

        //make tensor of samples drawn from a chisquare distribution
        //df - chisquare distribution parameter - degrees of freeedom, must be >0
        template<typename T=double, typename Order=config::c_order, typename U, typename Size>
        auto chisquare(const U& df_, Size&& size){
            static_assert(math::numeric_traits<T>::is_floating_point(),"T must be of floating point type");
            const auto df = static_cast<math::make_floating_point_t<U>>(df_);
            return make_distribution<T,Order,Config>(std::forward<Size>(size), bit_generator_, std::chi_squared_distribution<T>(df));
        }

        //make tensor of samples drawn from a cauchy distribution
        //location - cauchy distribution parameter
        //scale - cauchy distribution parameter, must be >0
        template<typename T=double, typename Order=config::c_order, typename U, typename V, typename Size>
        auto cauchy(const U& location_, const V& scale_, Size&& size){
            static_assert(math::numeric_traits<T>::is_floating_point(),"T must be of floating point type");
            const auto location = static_cast<math::make_floating_point_t<U>>(location_);
            const auto scale = static_cast<math::make_floating_point_t<V>>(scale_);
            return make_distribution<T,Order,Config>(std::forward<Size>(size), bit_generator_, std::cauchy_distribution<T>(location,scale));
        }

        //make tensor of samples drawn from a fisher distribution
        //dfnum - fisher distribution parameter - degrees of freedom in numerator, must be >0
        //dfden - fisher distribution parameter - Degrees of freedom in denominator, must be >0
        template<typename T=double, typename Order=config::c_order, typename U, typename V, typename Size>
        auto f(const U& dfnum_, const V& dfden_, Size&& size){
            static_assert(math::numeric_traits<T>::is_floating_point(),"T must be of floating point type");
            const auto dfnum = static_cast<math::make_floating_point_t<U>>(dfnum_);
            const auto dfden = static_cast<math::make_floating_point_t<V>>(dfden_);
            return make_distribution<T,Order,Config>(std::forward<Size>(size), bit_generator_, std::fisher_f_distribution<T>(dfnum,dfden));
        }

        //make tensor of samples drawn from a student distribution
        //df - student distribution parameter - degrees of freeedom, must be >0
        template<typename T=double, typename Order=config::c_order, typename U, typename Size>
        auto t(const U& df_, Size&& size){
            static_assert(math::numeric_traits<T>::is_floating_point(),"T must be of floating point type");
            const auto df = static_cast<math::make_floating_point_t<U>>(df_);
            return make_distribution<T,Order,Config>(std::forward<Size>(size), bit_generator_, std::student_t_distribution<T>(df));
        }

        //shuffling and permutation methods

        //do shuffle in-place, the order of sub-arrays is changed but their contents remains the same
        template<typename DimT=int, typename...Ts>
        void shuffle(basic_tensor<Ts...>& t, const DimT& axis_=0){
            using tensor_type = basic_tensor<Ts...>;
            using order = typename tensor_type::order;
            using config_type = typename tensor_type::config_type;
            using dim_type = typename tensor_type::dim_type;
            using index_type = typename tensor_type::index_type;
            using integral_type = long long int;
            using distribution_type = std::uniform_int_distribution<integral_type>;
            using distribution_param_type = distribution_type::param_type;
            const auto dim = t.dim();
            const auto size = t.size();
            const auto axis = detail::make_axis(dim,axis_);
            distribution_type distribution{};
            auto a = t.template traverse_order_adapter<order>();
            if (dim == 1){
                auto indexer = a.create_indexer();
                for (auto i=size-1; i>0; --i)
                {
                    using std::swap;
                    const auto j = distribution(bit_generator_, distribution_param_type(0, static_cast<const integral_type&>(i)));
                    swap(indexer[i], indexer[static_cast<const index_type&>(j)]);
                }
            }else{
                const auto& shape = t.shape();
                const auto axis_size = shape[axis];
                const auto chunk_size = size / axis_size;
                auto w1 = t.create_walker();
                auto w2 = t.create_walker();
                using walker_type = decltype(t.create_walker());
                using predicate_type = detail::reduce_traverse_predicate<config_type, dim_type>;
                using iterator_type = walker_iterator<config_type,walker_type,order,predicate_type>;
                predicate_type predicate{axis, true};
                for (auto i=axis_size-1; i>0; --i){
                    const auto j = distribution(bit_generator_, distribution_param_type(0, static_cast<const integral_type&>(i)));
                    w1.reset_back();
                    w2.reset_back();
                    w1.walk(axis,i);
                    w2.walk(axis,static_cast<const index_type&>(j));
                    std::swap_ranges(
                        iterator_type{w1,t.shape(),t.descriptor().strides_div(),index_type{0},predicate},
                        iterator_type{w1,t.shape(),t.descriptor().strides_div(),chunk_size,predicate},
                        iterator_type{w2,t.shape(),t.descriptor().strides_div(),index_type{0},predicate}
                    );
                }
            }
        }


    };  //end of class generator

    template<std::size_t...I>
    static auto random_seeds_helper(std::index_sequence<I...>){
        auto make_seed = [](auto){return std::random_device{}();};
        return std::array<unsigned int, sizeof...(I)>{make_seed(I)...};
    }

    //return std::array of N non-deterministic integer random numbers
    template<std::size_t N=5>
    static auto random_seeds(){
        return random_seeds_helper(std::make_index_sequence<N>{});
    }

    //return random number generator object
    //BitGenerator should satisfy UniformRandomBitGenerator and RandomNumberEngine named requirements
    //tensors of random samples produced by generator object use Config as their config_type, value_type and layout can be specified using template arguments of generator methods
    //seeds should be integral numbers to initialize BitGenerator state
    template<typename BitGenerator, typename Config=config::default_config, typename...Seeds>
    static auto rng(const Seeds&...seeds){
        return generator<Config, BitGenerator>(seeds...);
    }
    //return random number generator object initialized with random seeds, new for each call
    template<typename BitGenerator, typename Config=config::default_config>
    static auto rng(){
        return rng<BitGenerator,Config>(random_seeds());
    }

    //return random number generator object with default BitGenerator
    template<typename Config=config::default_config, typename...Seeds>
    static auto default_rng(const Seeds&...seeds){
        return rng<std::mt19937_64, Config>(seeds...);
    }
    //return random number generator object with default BitGenerator and random seeds, new for each call
    template<typename Config=config::default_config>
    static auto default_rng(){
        return rng<std::mt19937_64, Config>();
    }

};  //end of struct random

//random module frontend

//return random number generator object
//BitGenerator should satisfy UniformRandomBitGenerator and RandomNumberEngine named requirements
//tensors of random samples produced by generator object use Config as their config_type, value_type and layout can be specified using template arguments of generator methods
//seeds should be integral numbers to initialize BitGenerator state
template<typename BitGenerator, typename Config=config::default_config, typename...Seeds>
auto rng(const Seeds&...seeds){
    return random_selector_t<Config>::template rng<BitGenerator, Config>(seeds...);
}
//return random number generator object initialized with random seeds, new for each call
template<typename BitGenerator, typename Config=config::default_config>
auto rng(){
    return random_selector_t<Config>::template rng<BitGenerator, Config>();
}

//return random number generator object with default BitGenerator
template<typename Config=config::default_config, typename...Seeds>
auto default_rng(const Seeds&...seeds){
    return random_selector_t<Config>::template default_rng<Config>(seeds...);
}
//return random number generator object with default BitGenerator and random seeds, new for each call
template<typename Config=config::default_config>
auto default_rng(){
    return random_selector_t<Config>::template default_rng<Config>();
}


}   //end of namespace gtensor
#endif