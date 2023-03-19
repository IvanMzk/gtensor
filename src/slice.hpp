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
inline static char operator&(mask_type lhs, mask_type rhs){return static_cast<char>(lhs)&static_cast<char>(rhs);}

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
inline mask_type mask(const slice_item<T,N>& i){
    return i.nop ?  mask_type::__k : mask_type::i_k;
}
template<typename T, typename N>
inline mask_type mask(const slice_item<T,N>& i, const slice_item<T,N>& j){
    return i.nop ?
                j.nop ? mask_type::__k : mask_type::_jk:    //i.nop
                j.nop ? mask_type::i_k : mask_type::ijk;    //!i.nop
}
template<typename T, typename N>
inline mask_type mask(std::initializer_list<slice_item<T,N>> l){
    auto it = l.begin();
    switch (l.size()){
        case 0:
            return mask_type::__k;
        case 1:
            return detail::mask(it[0]);
        case 2:
            return detail::mask(it[0],it[1]);
        default:
            return detail::mask(it[0],it[1]);
    }
}

}   //end of namespace detail

/*
* k is always set
*/
template<typename IdxT, typename NopT = Nop>
struct slice{
    using index_type = IdxT;
    using nop_type = NopT;
    using mask_type = typename detail::mask_type;
    using slice_item_type = typename detail::slice_item<index_type, nop_type>;

    slice(std::initializer_list<slice_item_type> l):
        mask{detail::mask(l)},
        start{is_start() ? l.begin()[0].i : index_type{}},
        stop{is_stop() ? l.begin()[1].i : index_type{}},
        step{l.size()>2 ? l.begin()[2].nop ? index_type{1} : l.begin()[2].i : index_type{1} }
    {}

    slice() = default;
    slice(const nop_type&, const nop_type&, const nop_type&):
        slice()
    {}
    slice(const nop_type&, const nop_type&, const index_type& step_):
        mask{mask_type::__k},
        step{step_}
    {}
    slice(const nop_type&, const index_type& stop_, const nop_type&):
        mask{mask_type::_jk},
        stop{stop_}
    {}
    slice(const index_type& start_, const nop_type&, const nop_type&):
        mask{mask_type::i_k},
        start{start_}
    {}
    slice(const nop_type&, const index_type& stop_, const index_type& step_):
        mask{mask_type::_jk},
        stop{stop_},
        step{step_}
    {}
    slice(const index_type& start_, const nop_type&, const index_type& step_):
        mask{mask_type::i_k},
        start{start_},
        step{step_}
    {}
    slice(const index_type& start_, const index_type& stop_, const nop_type&):
        mask{mask_type::ijk},
        start{start_},
        stop{stop_}
    {}
    slice(const index_type& start_, const index_type& stop_, const index_type& step_):
        mask{mask_type::ijk},
        start{start_},
        stop{stop_},
        step{step_}
    {}
    slice(const nop_type&, const index_type& stop_):
        mask{mask_type::_jk},
        stop{stop_}
    {}
    slice(const index_type& start_, const nop_type&):
        mask{mask_type::i_k},
        start{start_}
    {}
    slice(const nop_type&, const nop_type&):
        slice()
    {}
    slice(const index_type& start_, const index_type& stop_):
        mask{mask_type::ijk},
        start{start_},
        stop{stop_}
    {}
    explicit slice(const index_type& start_):
        mask{mask_type::i_k},
        start{start_}
    {}
    explicit slice(const nop_type& ):
        slice()
    {}

    bool is_start()const{return mask & mask_type::i__;}
    bool is_stop()const{return mask & mask_type::_j_;}
    bool is_step()const{return mask& mask_type::__k;}
    friend bool operator==(const slice& lhs, const slice& rhs){return lhs.mask==rhs.mask && lhs.start==rhs.start && lhs.stop==rhs.stop && lhs.step==rhs.step;}
    friend std::ostream& operator<<(std::ostream& os, const slice& lhs){
        os <<"("<<lhs.start<<" "<<lhs.stop<<" "<<lhs.step<<")";
        return os;
    }
    const mask_type mask{mask_type::__k};
    const index_type start{};
    const index_type stop{};
    const index_type step{1};
};

template<typename CfgT>
struct slice_traits{
    using nop_type = Nop;
    using slice_type = slice<typename CfgT::index_type, nop_type>;
    using slice_item_type = detail::slice_item<typename CfgT::index_type, nop_type>;
    using slice_init_type = std::initializer_list<slice_item_type>;
    using slices_init_type = std::initializer_list<slice_init_type>;
    using slices_collection_type = std::vector<slice_type>;
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
* if slice is ok return true
*/
template<typename T, typename N>
inline void check_slice(const slice<T,N>& slice_, const T& n){
    using index_type = T;
    const index_type zero_index(0);
    if (
        slice_.step > zero_index ? slice_.start<slice_.stop && slice_.start>=zero_index && slice_.start<n && slice_.stop>zero_index && slice_.stop<=n ? true : false
        :
        slice_.start>slice_.stop && slice_.start>=zero_index && slice_.start<n && slice_.stop>=index_type(-1) && slice_.stop<n-index_type(1) ? true : false
        ){}
        else{throw subscript_exception("invalid slice subscript");}
}

/*make collection of filled slices from slices args*/
template<std::size_t I, typename R, typename ShT>
inline void fill_slices_helper(R&, const ShT&){}
template<std::size_t I, typename R, typename ShT,typename Sub, typename...Subs>
inline void fill_slices_helper(R& res, const ShT& shape, const Sub& sub, const Subs&...subs){
    res.push_back(fill_slice(sub,shape[I]));
    fill_slices_helper<I+1>(res,shape,subs...);
}
template<typename SlT, typename ShT, typename...Subs, typename std::enable_if_t<is_slices<Subs...>,int> = 0 >
inline auto fill_slices(const ShT& shape, const Subs&...subs){
    using slice_type = SlT;
    std::vector<slice_type> res{};
    res.reserve(sizeof...(Subs));
    fill_slices_helper<0>(res,shape,subs...);
    return res;
}

/*make collection of filled slices from init_list of intit_list of slice_items*/
template<typename SlT, typename ShT>
inline std::vector<SlT> fill_slices(const ShT& shape, std::initializer_list<std::initializer_list<slice_item<typename SlT::index_type, typename SlT::nop_type>>> subs){
    using slice_type = SlT;
    std::vector<slice_type> res{};
    res.reserve(subs.size());
    auto sh_begin{shape.begin()};
    for_each(subs.begin(), subs.end(), [&sh_begin, &res](const auto& sub){res.push_back(fill_slice(slice_type(sub),*sh_begin));++sh_begin;});
    return res;
}

/*check filled slices*/
template<typename ShT, typename SsT>
inline void check_slices(const ShT& shape, const SsT& slices){
    if (slices.size()>shape.size()){throw subscript_exception("subscripts number exceeds dim");}
    auto sh_begin = shape.begin();
    std::for_each(slices.begin(), slices.end(), [&](const auto& slice){check_slice(slice,*sh_begin);++sh_begin;});
}

template<typename ShT, typename...T>
inline void check_slices_number(const ShT& shape, std::initializer_list<std::initializer_list<slice_item<T...>>> subs){
    if (subs.size()>shape.size()){throw subscript_exception("subscripts number exceeds dim");}
}
template<typename ShT, typename...Subs, std::enable_if_t<is_slices<Subs...>, int> =0 >
inline void check_slices_number(const ShT& shape, const Subs&...){
    if (sizeof...(Subs)>shape.size()){throw subscript_exception("subscripts number exceeds dim");}
}
template<typename ShT, typename Slices, std::enable_if_t<detail::is_slices_container<Slices>,int> =0>
inline void check_slices_number(const ShT& shape, const Slices& slices){
    if (slices.size()>shape.size()){throw subscript_exception("subscripts number exceeds dim");}
}



}   //end of namespace detail
}   //end of namespace gtensor


#endif