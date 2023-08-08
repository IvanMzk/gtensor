#ifndef RANDOM_HPP_
#define RANDOM_HPP_

#include <random>
#include <algorithm>
#include <numeric>
#include "module_selector.hpp"
#include "math.hpp"
#include "indexing.hpp"
#include "builder.hpp"
#include "reduce.hpp"


namespace gtensor{

namespace detail{

template<typename ShT, typename DimT>
auto check_shuffle_args(const ShT& input_shape, const DimT& axis){
    const auto input_dim = detail::make_dim(input_shape);
    if (axis>=input_dim){
        throw axis_error("axis out of bounds");
    }
}

template<typename ShT, typename IdxT, typename Size, typename Probabilities, typename DimT>
auto check_choice_args(const ShT& input_shape, const IdxT& input_size, const Size& size, bool replace, const Probabilities& p, const DimT& axis){
    const auto input_dim = detail::make_dim(input_shape);
    const auto axis_size = input_shape[axis];
    const auto size_size = detail::make_size<IdxT>(size);
    if (axis>=input_dim){
        throw axis_error("axis out of bounds");
    }
    if (input_size==0 && size_size!=0){
        throw value_error("t cannot be empty unless no samples are taken");
    }
    if (!replace){
        if (size_size>axis_size){
            throw value_error("cannot take a larger sample than population without remplacement");
        }
    }
    if constexpr (detail::is_container_v<Probabilities>){
        const auto p_size = p.size();
        if (axis_size!=static_cast<const IdxT&>(p_size)){
            throw value_error("p must be the same size as size along axis");
        }
        if (!replace){
            const auto p_zeros = std::count(p.begin(),p.end(),0);
            const auto samplable_axis_size = p_size - p_zeros;
            if (size_size>samplable_axis_size){
                throw value_error("cannot take a larger sample than population with non-zero probabilities, without remplacement");
            }
        }
    }

}

}   //end of namespace detail

//random module implementation

struct random
{
private:

    template<typename It, typename BitGenerator, typename Distribution>
    static void generate_distribution(It first, It last, BitGenerator&& bit_generator, Distribution distribution){
        std::generate(
            first,
            last,
            [&bit_generator,&distribution](){
                return distribution(bit_generator);
            }
        );
    }

    template<typename T, typename Order, typename Config, typename Size, typename BitGenerator, typename Distribution>
    static auto make_distribution(Size&& size, BitGenerator&& bit_generator, Distribution distribution){
        using tensor_type = tensor<T,Order,config::extend_config_t<Config,T>>;
        using shape_type = typename tensor_type::shape_type;
        tensor_type res(detail::make_shape_of_type<shape_type>(std::forward<Size>(size)));
        auto a = res.traverse_order_adapter(Order{});
        generate_distribution(a.begin(), a.end(), bit_generator, distribution);
        return res;
    }

    template<typename It, typename BitGenerator>
    static auto shuffle_range(It first, It last, BitGenerator&& bit_generator){
        using integral_type = long long int;
        using difference_type = typename std::iterator_traits<It>::difference_type;
        using distribution_type = std::uniform_int_distribution<integral_type>;
        using distribution_param_type = distribution_type::param_type;
        const auto n = last-first;
        distribution_type distribution{};
        for (auto i=n-1; i>0; --i)
        {
            using std::swap;
            const auto j = distribution(bit_generator, distribution_param_type(0, static_cast<const integral_type&>(i)));
            swap(first[i], first[static_cast<const difference_type&>(j)]);
        }
    }

    template<typename It, typename PIt>
    static void generate_cdf(It cdf_first, const It cdf_last, PIt p_first){
        using value_type = typename std::iterator_traits<It>::value_type;
        static_assert(math::numeric_traits<value_type>::is_floating_point(),"cdf must be of floating point type");
        auto cdf_it = cdf_first;
        value_type cum = 0;
        for (;cdf_it!=cdf_last; ++cdf_it,++p_first){
            const auto& p = static_cast<const value_type&>(*p_first);
            if (p<0){
                throw value_error("probabilities can't be negative");
            }
            cum+=p;
            *cdf_it=cum;
        }
        if (cum == 0){
            throw value_error("probabilities sums to zero");
        }
        if (!math::isclose(cum,1.0)){  //normalize
            const auto normalizer = 1/cum;
            std::for_each(cdf_first,cdf_last,[normalizer](auto& e){e*=normalizer;});
        }
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

        template<typename BitGenerator_, std::enable_if_t<!std::is_same_v<BitGenerator_,generator>,int> =0>
        explicit generator(BitGenerator_&& bit_generator__):
            bit_generator_{std::forward<BitGenerator_>(bit_generator__)}
        {}

        template<typename Container>
        explicit generator(const Container& seeds):
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

        auto bit_generator()const{
            return bit_generator_;
        }

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
            using index_type = typename tensor_type::index_type;
            using integral_type = long long int;
            using distribution_type = std::uniform_int_distribution<integral_type>;
            using distribution_param_type = distribution_type::param_type;
            const auto dim = t.dim();
            const auto axis = detail::make_axis(dim,axis_);
            detail::check_shuffle_args(t.shape(),axis);
            distribution_type distribution{};
            auto a = t.traverse_order_adapter(order{});
            if (dim == 1){
                shuffle_range(a.begin(),a.end(),bit_generator_);
            }else if(t.size()>1){
                const auto& shape = t.shape();
                const auto axis_size = shape[axis];
                auto w1 = t.create_walker();
                auto w2 = t.create_walker();
                auto axes_iterator_maker = detail::make_axes_iterator_maker<config_type>(shape,axis,order{});
                for (auto i=axis_size-1; i>0; --i){
                    const auto j = distribution(bit_generator_, distribution_param_type(0, static_cast<const integral_type&>(i)));
                    if (i!=j){
                        w1.reset_back();
                        w2.reset_back();
                        w1.walk(axis,i);
                        w2.walk(axis,static_cast<const index_type&>(j));
                        std::swap_ranges(
                            axes_iterator_maker.begin(w1,std::true_type{}),
                            axes_iterator_maker.end(w1,std::true_type{}),
                            axes_iterator_maker.begin(w2,std::true_type{})
                        );
                    }
                }
            }
        }

        //if t is of tensor type return shuffled copy
        //if t is of integral type return shuffled arange(t)
        template<typename T, typename DimT=int>
        auto permutation(const T& t, const DimT& axis=0){
            static constexpr bool is_t_tensor = detail::is_tensor_v<T>;
            static_assert(is_t_tensor || math::numeric_traits<T>::is_integral(),"t must be of tensor or integral type");
            if constexpr (is_t_tensor){
                using order = typename T::order;
                auto res = t.copy(order{});
                shuffle(res,axis);
                return res;
            }else{  //t integral
                auto res = arange<T>(t);
                shuffle(res);
                return res;
            }
        }

        //return copy, permuted along given axis
        //if no axis specified flatten tensor is shuffled
        template<typename DimT=detail::no_value, typename...Ts>
        auto permuted(const basic_tensor<Ts...>& t, const DimT& axis_=DimT{}){
            using tensor_type = basic_tensor<Ts...>;
            using order = typename tensor_type::order;
            if constexpr (!std::is_same_v<DimT,detail::no_value>){
                const auto axis = detail::make_axis(t.dim(),axis_);
                auto res = t.copy(order{});
                transform(
                    res,
                    axis,
                    [this](auto first, auto last){
                        shuffle_range(first,last,bit_generator_);
                    }
                );
                return res;
            }else{  //shuffle flatten
                auto res = t.flatten();
                shuffle_range(res.begin(),res.end(),bit_generator_);
                return res;
            }
        }

        //if t is of tensor type return random sample taken from given tensor
        //if t is of integral type return random sample taken from arange(t)
        //size can be scalar or container
        template<typename T, typename Size, typename DimT, typename Probabilities=detail::no_value>
        auto choice(const T& t, Size&& size, bool replace=true, const Probabilities& p=Probabilities{}, const DimT& axis=0, bool shuffle=true){
            static constexpr bool is_t_tensor = detail::is_tensor_v<T>;
            static_assert(is_t_tensor || math::numeric_traits<T>::is_integral(),"t must be of tensor or integral type");
            if constexpr (is_t_tensor){
                return choice_(t,size,replace,p,axis,shuffle);
            }else{  //t integral
                return choice_(arange<T>(t),size,replace,p,0,shuffle);
            }
        }

    private:
        template<typename DimT, typename...Ts, typename Size, typename Probabilities=detail::no_value>
        auto choice_(const basic_tensor<Ts...>& t, Size&& size, bool replace=true, const Probabilities& p=Probabilities{}, const DimT& axis_=0, bool shuffle=true){
            using tensor_type = basic_tensor<Ts...>;
            using order = typename tensor_type::order;
            using config_type = typename tensor_type::config_type;
            using index_type = typename tensor_type::index_type;
            using shape_type = typename tensor_type::shape_type;
            using integral_type = long long int;
            using floating_point_type = double;
            using container_type = typename config_type::template container<floating_point_type>;
            using map_type = typename config_type::template container<bool>;
            using container_difference_type = typename container_type::difference_type;
            using map_difference_type = typename map_type::difference_type;
            using index_tensor_type = tensor<index_type,order,config_type>;
            static constexpr bool is_p = detail::is_container_v<Probabilities>;
            static_assert(is_p || std::is_same_v<Probabilities,detail::no_value>,"p must be container or no_value");

            const auto axis = detail::make_axis(t.dim(),axis_);
            detail::check_choice_args(t.shape(),t.size(),size,replace,p,axis);
            const auto axis_size = t.shape()[axis];
            index_tensor_type indexes(detail::make_shape_of_type<shape_type>(std::forward<Size>(size)));
            if (indexes.empty()){
                return take(t,indexes,axis);
            }
            const auto indexes_size = indexes.size();
            auto a_indexes = indexes.traverse_order_adapter(order{});
            if constexpr (is_p){
                container_type cdf(static_cast<const container_difference_type&>(p.size()));
                const auto cdf_first = cdf.begin();
                const auto cdf_last = cdf.end();
                std::uniform_real_distribution<floating_point_type> distribution{0,1};
                if (replace){
                    generate_cdf(cdf_first,cdf_last,p.begin());
                    std::generate(
                        a_indexes.begin(),
                        a_indexes.end(),
                        [this,cdf_first,cdf_last,&distribution](){
                            auto pos = std::lower_bound(cdf_first,cdf_last,distribution(bit_generator_));
                            return static_cast<const index_type&>(pos - cdf_first);
                        }
                    );
                }else{  //without replacement, probability
                    map_type map(static_cast<const map_difference_type&>(indexes_size),false);
                    container_type prob(p.begin(),p.end());
                    auto indexes_it = a_indexes.begin();
                    const auto indexes_last = a_indexes.end();
                    while (indexes_it!=indexes_last){
                        generate_cdf(cdf_first,cdf_last,prob.begin());
                        //heuristic, gives twice more trys to left indexes to be selected, tradeof with generate_cdf more times
                        for (auto n = 2*(indexes_last-indexes_it); n!=0; --n){
                            const auto j = std::lower_bound(cdf_first,cdf_last,distribution(bit_generator_)) - cdf_first;
                            if (!map[j]){
                                *indexes_it=static_cast<const index_type&>(j);
                                ++indexes_it;
                                map[j] = true;
                                prob[j] = 0;
                                if (indexes_it==indexes_last){
                                    break;
                                }
                            }
                        }
                    }
                }
            }else{
                if (replace){
                    generate_distribution(
                        a_indexes.begin(),
                        a_indexes.end(),
                        bit_generator_,
                        std::uniform_int_distribution{integral_type{0},static_cast<const integral_type&>(axis_size-1)}
                    );
                }else{  //without replacement, no probability
                    using distribution_type = std::uniform_int_distribution<integral_type>;
                    using distribution_param_type = distribution_type::param_type;
                    map_type map(indexes_size,false);
                    distribution_type distribution{};
                    auto indexes_it = a_indexes.begin();
                    //floyd's algorithm
                    for (auto i=axis_size-indexes_size; i!=axis_size; ++i,++indexes_it){
                        const auto j = distribution(bit_generator_, distribution_param_type(0, static_cast<const integral_type&>(i)));
                        const auto& j_ = static_cast<const map_difference_type&>(j);
                        if (map[j_]){
                            *indexes_it = i;
                            map[static_cast<const map_difference_type&>(i)] = true;
                        }else{
                            *indexes_it = j;
                            map[j_] = true;
                        }
                    }
                    if (shuffle){
                       shuffle_range(a_indexes.begin(),a_indexes.end(),bit_generator_);
                    }
                }
            }
            return take(t,indexes,axis);
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

public:
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