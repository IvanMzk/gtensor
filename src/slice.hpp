#ifndef SLICE_HPP_
#define SLICE_HPP_
#include <array>
#include "config.hpp"

namespace gtensor{

class subscript_exception : public std::runtime_error{
    public: subscript_exception(const char* what):runtime_error(what){}
};

struct Nop{};

namespace detail{

enum class mask_type : char {
    nop=0b00000000, __k = 0b00000001, _j_ = 0b00000010, _jk = 0b00000011, i__ = 0b00000100, i_k = 0b00000101, ij_ = 0b00000110, ijk = 0b00000111
};
inline constexpr mask_type operator&(mask_type lhs, mask_type rhs){return static_cast<mask_type>(static_cast<char>(lhs)&static_cast<char>(rhs));}
inline constexpr mask_type operator|(mask_type lhs, mask_type rhs){return static_cast<mask_type>(static_cast<char>(lhs)|static_cast<char>(rhs));}
inline constexpr bool is_i(mask_type mask){return static_cast<bool>(mask&mask_type::i__);}
inline constexpr bool is_j(mask_type mask){return static_cast<bool>(mask&mask_type::_j_);}
inline constexpr bool is_k(mask_type mask){return static_cast<bool>(mask&mask_type::__k);}

template<typename IdxT, typename NopT = Nop>
struct slice_item{
    using index_type = IdxT;
    using nop_type = NopT;
    slice_item():nop{1}{}
    slice_item(const nop_type&):nop{1}{}
    template<typename U>
    slice_item(const U& i_):i{i_}{}
    const index_type i{};
    const char nop{0};
};

template<typename T, typename N>
inline mask_type make_mask(const slice_item<T,N>& i){
    return i.nop ?  mask_type::__k : mask_type::i_k;
}
template<typename T, typename N>
inline mask_type make_mask(const slice_item<T,N>& i, const slice_item<T,N>& j){
    return i.nop ?
                j.nop ? mask_type::__k : mask_type::_jk:    //i.nop
                j.nop ? mask_type::i_k : mask_type::ijk;    //!i.nop
}
template<typename T, typename N>
inline mask_type make_mask(std::initializer_list<slice_item<T,N>> l){
    auto it = l.begin();
    switch (l.size()){
        case 0:
            return mask_type::__k;
        case 1:
            return make_mask(it[0]);
        case 2:
            return make_mask(it[0],it[1]);
        default:
            return make_mask(it[0],it[1]);
    }
}

}   //end of namespace detail


template<typename IdxT, typename NopT = Nop>
struct slice{
    using index_type = IdxT;
    using nop_type = NopT;
    using mask_type = typename detail::mask_type;
    using slice_item_type = typename detail::slice_item<index_type, nop_type>;

    slice():
        mask{make_mask(nop_type{})},
        start{make_start_stop(nop_type{})},
        stop{make_start_stop(nop_type{})},
        step{make_step(nop_type{})}
    {}
    slice(std::initializer_list<slice_item_type> l):
        mask{detail::make_mask(l)},
        start{is_start() ? l.begin()[0].i : index_type{}},
        stop{is_stop() ? l.begin()[1].i : index_type{}},
        step{l.size()>2 ? l.begin()[2].nop ? index_type{1} : l.begin()[2].i : index_type{1} }
    {}
    template<typename Start, typename Stop, typename Step>
    slice(const Start& start__, const Stop& stop__, const Step& step__):
        mask{make_mask(start__,stop__)},
        start{make_start_stop(start__)},
        stop{make_start_stop(stop__)},
        step{make_step(step__)}
    {
        static_assert(std::is_convertible_v<Start,index_type>||std::is_same_v<Start,nop_type>);
        static_assert(std::is_convertible_v<Stop,index_type>||std::is_same_v<Stop,nop_type>);
        static_assert(std::is_convertible_v<Step,index_type>||std::is_same_v<Step,nop_type>);
    }
    template<typename Start, typename Stop>
    slice(const Start& start__, const Stop& stop__):
        mask{make_mask(start__,stop__)},
        start{make_start_stop(start__)},
        stop{make_start_stop(stop__)},
        step{make_step(nop_type{})}
    {
        static_assert(std::is_convertible_v<Start,index_type>||std::is_same_v<Start,nop_type>);
        static_assert(std::is_convertible_v<Stop,index_type>||std::is_same_v<Stop,nop_type>);
    }
    template<typename Start>
    explicit slice(const Start& start__):
        mask{make_mask(start__)},
        start{make_start_stop(start__)},
        stop{make_start_stop(nop_type{})},
        step{make_step(nop_type{})}
    {
        static_assert(std::is_convertible_v<Start,index_type>||std::is_same_v<Start,nop_type>);
    }

    bool is_start()const{return detail::is_i(mask);}
    bool is_stop()const{return detail::is_j(mask);}
    bool is_step()const{return detail::is_k(mask);}
    friend bool operator==(const slice& lhs, const slice& rhs){return lhs.mask==rhs.mask && lhs.start==rhs.start && lhs.stop==rhs.stop && lhs.step==rhs.step;}
    friend std::ostream& operator<<(std::ostream& os, const slice& lhs){
        os <<"("<<lhs.start<<" "<<lhs.stop<<" "<<lhs.step<<")";
        return os;
    }

    template<typename Start>
    constexpr mask_type make_mask(Start){
        if constexpr (std::is_convertible_v<Start,index_type>){
            return mask_type::i_k;
        }else{
            return mask_type::__k;
        }
    }
    template<typename Start, typename Stop>
    mask_type make_mask(Start start__,Stop){
        mask_type mask{make_mask(start__)};
        if constexpr (std::is_convertible_v<Stop,index_type>){
            mask = mask|mask_type::_j_;
        }
        return mask;
    }
    auto make_start_stop(const index_type& item_){return item_;}
    auto make_start_stop(nop_type){return index_type{};}
    auto make_step(const index_type& item_){return item_;}
    auto make_step(nop_type){return index_type{1};}

    const mask_type mask;
    const index_type start;
    const index_type stop;
    const index_type step;
};

template<typename CfgT>
struct slice_traits{
    using nop_type = Nop;
    using slice_type = slice<typename CfgT::index_type, nop_type>;
    using slice_item_type = detail::slice_item<typename CfgT::index_type, nop_type>;
    using slice_init_type = std::initializer_list<slice_item_type>;
    using slices_init_type = std::initializer_list<slice_init_type>;
    using slices_container_type = typename CfgT::template container<slice_type>;
};

namespace detail{

template<typename> inline constexpr bool is_slice = false;
template<typename T, typename N> inline constexpr bool is_slice<gtensor::slice<T,N>> = true;
template<typename...Ts> inline constexpr bool is_slices = (... && is_slice<Ts>);
template<typename Slices, typename = void> inline constexpr bool is_slices_container = false;
template<typename Slices> inline constexpr bool is_slices_container<Slices, std::void_t<decltype(std::begin(std::declval<Slices&>()))> > =
    is_slice<typename std::iterator_traits<decltype(std::begin(std::declval<Slices&>()))>::value_type>;

/*
* start,stop,step
* slice{i,j,k}
* if i is not given it defaults to 0 for k > 0 and n-1 for k < 0 .
* If j is not given it defaults to n for k > 0 and -n-1 for k < 0 .
* If k is not given it defaults to 1.
*
* Negative i and j are interpreted as n + i and n + j where n is the number of elements in the corresponding dimension.
* Negative k makes stepping go towards smaller indices.
*/
/*n is corresponding shape element, may be unsigned*/
template<typename T, typename N>
inline auto fill_slice(const slice<T,N>& slice_, const T& n){
    using slice_type = slice<T,N>;
    using index_type = T;
    const index_type zero_index(0);
    return slice_.step > index_type(0) ?
                slice_.is_start() ?
                    slice_.is_stop() ? slice_type{slice_.start < zero_index ? slice_.start + n:slice_.start, slice_.stop < zero_index  ? slice_.stop + n:slice_.stop, slice_.step} : slice_type{slice_.start < zero_index  ? slice_.start + n:slice_.start,n, slice_.step}
                :
                    slice_.is_stop() ? slice_type{zero_index,slice_.stop < zero_index  ? slice_.stop + n:slice_.stop, slice_.step} : slice_type{zero_index,n, slice_.step}
            :
                slice_.is_start() ?
                    slice_.is_stop() ? slice_type{slice_.start < zero_index ? slice_.start + n:slice_.start, slice_.stop < zero_index  ? slice_.stop + n:slice_.stop, slice_.step} : slice_type{slice_.start < zero_index  ? slice_.start + n:slice_.start,index_type(-1), slice_.step}
                :
                    slice_.is_stop() ? slice_type{n-index_type(1),slice_.stop < zero_index  ? slice_.stop + n:slice_.stop, slice_.step} : slice_type{n-index_type(1),index_type(-1), slice_.step};
}

/*
* check filled slice for valid values given shape element n
* slice{i,j,k}
* k !=0 - garantes by slice type
* if k>0:  i < j  and i,j in range [0,n)
* if k<0:  i > j  and i,j in range [0,n)
* if slice is ok do nothing, throw subscript_exception otherwise
*/
template<typename T, typename N>
inline void check_slice(const slice<T,N>& slice_, const T& n){
    using index_type = T;
    const index_type zero_index(0);
    if (
        slice_.step > zero_index ? slice_.start<slice_.stop && slice_.start>=zero_index && slice_.start<n && slice_.stop>zero_index && slice_.stop<=n ? true : false
        :
        slice_.start>slice_.stop && slice_.start>=zero_index && slice_.start<n && slice_.stop>=index_type(-1) && slice_.stop<n-index_type(1) ? true : false
        ){
        }else{
            throw subscript_exception("invalid slice subscript");
        }
}

//make collection of filled slices from slices args
template<std::size_t I, typename SlicesContainer, typename ShT>
inline void fill_slices_helper(SlicesContainer&, const ShT&){}
template<std::size_t I, typename SlicesContainer, typename ShT,typename Sub, typename...Subs>
inline void fill_slices_helper(SlicesContainer& res, const ShT& shape, const Sub& sub, const Subs&...subs){
    res.push_back(fill_slice(sub,shape[I]));
    fill_slices_helper<I+1>(res,shape,subs...);
}
template<typename SlicesContainer, typename ShT, typename...Subs, typename std::enable_if_t<is_slices<Subs...>,int> = 0 >
inline SlicesContainer fill_slices(const ShT& shape, const Subs&...subs){
    SlicesContainer res{};
    res.reserve(sizeof...(Subs));
    fill_slices_helper<0>(res,shape,subs...);
    return res;
}

//make collection of filled slices from init_list of intit_list of slice_items
template<typename SlicesContainer, typename ShT>
inline SlicesContainer fill_slices(const ShT& shape, std::initializer_list<std::initializer_list<slice_item<typename SlicesContainer::value_type::index_type, typename SlicesContainer::value_type::nop_type>>> subs){
    using slice_type = typename SlicesContainer::value_type;
    SlicesContainer res{};
    res.reserve(subs.size());
    auto sh_it = shape.begin();
    for_each(subs.begin(), subs.end(), [&sh_it, &res](const auto& sub){res.push_back(fill_slice(slice_type(sub),*sh_it));++sh_it;});
    return res;
}

//make collection of filled slices from slices container
template<typename SlicesContainer, typename ShT, typename Slices, std::enable_if_t<detail::is_slices_container<Slices>,int> =0>
inline SlicesContainer fill_slices(const ShT& shape, const Slices& subs){
    SlicesContainer res{};
    res.reserve(subs.size());
    auto sh_it = shape.begin();
    for_each(subs.begin(), subs.end(), [&sh_it, &res](const auto& sub){res.push_back(fill_slice(sub,*sh_it));++sh_it;});
    return res;
}

//check filled slices
template<typename ShT, typename SlicesContainer>
inline void check_slices(const ShT& shape, const SlicesContainer& slices){
    if (slices.size()>shape.size()){throw subscript_exception("subscripts number exceeds dim");}
    auto sh_it = shape.begin();
    std::for_each(slices.begin(), slices.end(), [&](const auto& slice){check_slice(slice,*sh_it);++sh_it;});
}

template<typename ShT, typename...T>
inline void check_slices_number(const ShT& shape, std::initializer_list<std::initializer_list<slice_item<T...>>> subs){
    if (subs.size()>shape.size()){throw subscript_exception("subscripts number exceeds dim");}
}
template<typename ShT, typename...Subs, std::enable_if_t<is_slices<Subs...>, int> =0 >
inline void check_slices_number(const ShT& shape, const Subs&...){
    if (sizeof...(Subs)>shape.size()){throw subscript_exception("subscripts number exceeds dim");}
}
template<typename ShT, typename SlicesContainer, std::enable_if_t<detail::is_slices_container<SlicesContainer>,int> =0>
inline void check_slices_number(const ShT& shape, const SlicesContainer& subs){
    if (subs.size()>shape.size()){throw subscript_exception("subscripts number exceeds dim");}
}
template<typename ShT, typename SizeT>
inline void check_slice_direction(const ShT& shape, const SizeT& direction){
    if (direction>=shape.size()){throw subscript_exception("subscripts number exceeds dim");}
}

}   //end of namespace detail
}   //end of namespace gtensor


#endif