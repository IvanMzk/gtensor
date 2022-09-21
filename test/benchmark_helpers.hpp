#ifndef BENCHMARK_HELPERS_HPP_
#define BENCHMARK_HELPERS_HPP_

#include <type_traits>

namespace benchmark_helpers{

template<std::size_t Depth>
struct asymmetric_tree_maker{
    static constexpr std::size_t depth = Depth;
    static constexpr char* name = "asymmetric_tree_maker";
    template<std::size_t Dep = Depth, typename T1, typename T2, std::enable_if_t< (Dep>1) ,int> = 0 >
    auto operator()(const T1& t1, const T2& t2){
        return operator()<Dep-1>(t1,t2+t1);
    }
    template<std::size_t Dep = Depth, typename T1, typename T2, std::enable_if_t< Dep==1,int> = 0 >
    auto operator()(const T1& t1, const T2& t2){
        return t2+t1;
    }
};
template<std::size_t Depth>
struct asymmetric_tree_trivial_subtree_maker{
    static constexpr std::size_t depth = Depth;
    static constexpr char* name = "asymmetric_tree_trivial_subtree_maker";
    template<typename T1, typename T2>
    auto operator()(const T1& t1, const T2& t2){
        return helper(t1,t2,t1);
    }
    template<std::size_t Dep = Depth, typename T1, typename T2, typename T3, std::enable_if_t< (Dep>Depth/2) ,int> = 0 >
    auto helper(const T1& t1, const T2& t2, const T3& sum){
        return helper<Dep-1>(t1,t2,sum+t1);
    }
    template<std::size_t Dep = Depth, typename T1, typename T2, typename T3, std::enable_if_t< (Dep<=Depth/2) && (Dep>1) ,int> = 0 >
    auto helper(const T1& t1, const T2& t2, const T3& sum){
        return helper<Dep-1>(t1,t2,sum+t2);
    }
    template<std::size_t Dep = Depth, typename T1, typename T2, typename T3, std::enable_if_t< Dep==1,int> = 0 >
    auto helper(const T1& t1, const T2& t2, const T3& sum){
        return sum+t2;
    }
};
template<std::size_t Depth>
struct symmetric_tree_maker{
    static constexpr std::size_t depth = Depth;
    static constexpr char* name = "symmetric_tree_maker";
    template<std::size_t Dep = Depth, typename T1, typename T2, std::enable_if_t< (Dep>1) ,int> = 0 >
    auto operator()(const T1& t1, const T2& t2){
        return operator()<Dep-1>(t2+t1,t2+t1);
    }
    template<std::size_t Dep = Depth, typename T1, typename T2, std::enable_if_t< Dep==1,int> = 0 >
    auto operator()(const T1& t1, const T2& t2){
        return t2+t1;
    }
};


template<std::size_t Depth, typename T1, typename T2, std::enable_if_t< (Depth>1) ,int> = 0 >
auto make_asymmetric_tree(const T1& t1, const T2& t2){
    return make_asymmetric_tree<Depth-1>(t1,t2+t1);
}
template<std::size_t Depth, typename T1, typename T2, std::enable_if_t< Depth==1,int> = 0 >
auto make_asymmetric_tree(const T1& t1, const T2& t2){
    return t2+t1;
}

template<std::size_t Depth, typename T1, typename T2, std::enable_if_t< (Depth>1) ,int> = 0 >
auto make_symmetric_tree(const T1& t1, const T2& t2){
    return make_symmetric_tree<Depth-1>(t2+t1,t2+t1);
}
template<std::size_t Depth, typename T1, typename T2, std::enable_if_t< Depth==1,int> = 0 >
auto make_symmetric_tree(const T1& t1, const T2& t2){
    return t2+t1;
}

}   //end of namespace benchmark_helpers


#endif