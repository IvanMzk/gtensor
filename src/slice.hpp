#ifndef SLICE_HPP_
#define SLICE_HPP_
#include <array>
#include "config.hpp"

namespace gtensor{

class slice_exception : public std::runtime_error{
    public: slice_exception(const char* what):runtime_error(what){}
};

struct Nop{};
struct Rtag{};

namespace detail{

enum class mask_type : char {
    __k = 0b00000001, _j_ = 0b00000010, _jk = 0b00000011, i__ = 0b00000100, i_k = 0b00000101, ij_ = 0b00000110, ijk = 0b00000111, r___ = 0b00001000, ri_k = 0b00001101, nop = 0b00000000
};
inline constexpr mask_type operator&(mask_type lhs, mask_type rhs){return static_cast<mask_type>(static_cast<char>(lhs)&static_cast<char>(rhs));}
inline constexpr mask_type operator|(mask_type lhs, mask_type rhs){return static_cast<mask_type>(static_cast<char>(lhs)|static_cast<char>(rhs));}
inline constexpr bool is_i(mask_type mask){return static_cast<bool>(mask&mask_type::i__);}
inline constexpr bool is_j(mask_type mask){return static_cast<bool>(mask&mask_type::_j_);}
inline constexpr bool is_k(mask_type mask){return static_cast<bool>(mask&mask_type::__k);}
inline constexpr bool is_r(mask_type mask){return static_cast<bool>(mask&mask_type::r___);}

template<typename IdxT, typename NopT = Nop, typename RtagT = Rtag>
class slice_item{
public:
    using index_type = IdxT;
    using nop_type = NopT;
    using reduce_tag_type = Rtag;
    slice_item():
        mask_{mask_type::nop},
        item_{}
    {}
    slice_item(nop_type):
        slice_item()
    {}
    slice_item(reduce_tag_type):
        mask_{mask_type::r___},
        item_{}
    {}
    template<typename U>
    slice_item(const U& item__):
        mask_{mask_type::i__},
        item_{item__}
    {
        static_assert(std::is_convertible_v<U,index_type>);
    }
    index_type item()const{return item_;};
    bool is_nop()const{return mask_ == mask_type::nop;}
    bool is_reduce()const{return mask_ == mask_type::r___;}
    bool is_item()const{return mask_ == mask_type::i__;}

private:
    const mask_type mask_;
    const index_type item_;
};

template<typename T, typename N, typename R>
inline mask_type make_mask(const slice_item<T,N,R>& i){
    return i.is_nop() ?  mask_type::__k : mask_type::i_k;
}
template<typename T, typename N, typename R>
inline mask_type make_mask(const slice_item<T,N,R>& i, const slice_item<T,N,R>& j){
    if (j.is_reduce()){
        return mask_type::ri_k;
    }else{
        return i.is_nop() ?
            j.is_nop() ? mask_type::__k : mask_type::_jk:    //i.nop
            j.is_nop() ? mask_type::i_k : mask_type::ijk;    //!i.nop
    }
}
template<typename T, typename N, typename R>
inline mask_type make_mask(std::initializer_list<slice_item<T,N,R>> l){
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

template<typename T, typename N, typename R>
inline void check_slice_item_list(std::initializer_list<slice_item<T,N,R>> l){
    const auto size = l.size();
    if (size > 3){
        throw slice_exception("max slice_item list size is 3");
    }
    auto it = l.begin();
    auto end = l.end();
    for (;it!=end; ++it){
        if ((*it).is_reduce()){
            if (size != 2 || std::distance(it,end) != 1 || (*l.begin()).is_nop()){
                throw slice_exception("invalid slice_item list");
            }
            break;
        }
    }
}

}   //end of namespace detail

template<typename IdxT, typename NopT = Nop, typename RtagT = Rtag>
class slice{
    using mask_type = typename detail::mask_type;
public:
    using index_type = IdxT;
    using nop_type = NopT;
    using reduce_tag_type = RtagT;
    using slice_item_type = typename detail::slice_item<index_type, nop_type, reduce_tag_type>;

    slice():
        mask_{make_mask(nop_type{})},
        start_{make_start_stop(nop_type{})},
        stop_{make_start_stop(nop_type{})},
        step_{make_step(nop_type{})}
    {}
    slice(std::initializer_list<slice_item_type> l):
        mask_{detail::make_mask(l)},
        start_{is_start() ? l.begin()[0].item() : index_type{}},
        stop_{is_stop() ? l.begin()[1].item() : index_type{}},
        step_{l.size()>2 ? (l.begin()[2].is_nop() ? index_type{1} : l.begin()[2].item()) : index_type{1} }
    {
        detail::check_slice_item_list(l);
    }
    template<typename Start, typename Stop, typename Step>
    slice(const Start& start__, const Stop& stop__, const Step& step__):
        mask_{make_mask(start__,stop__)},
        start_{make_start_stop(start__)},
        stop_{make_start_stop(stop__)},
        step_{make_step(step__)}
    {
        static_assert(std::is_convertible_v<Start,index_type>||std::is_same_v<Start,nop_type>);
        static_assert(std::is_convertible_v<Stop,index_type>||std::is_same_v<Stop,nop_type>);
        static_assert(std::is_convertible_v<Step,index_type>||std::is_same_v<Step,nop_type>);
    }
    template<typename Start, typename Stop, std::enable_if_t<!std::is_same_v<Stop, reduce_tag_type> ,int> =0>
    slice(const Start& start__, const Stop& stop__):
        mask_{make_mask(start__,stop__)},
        start_{make_start_stop(start__)},
        stop_{make_start_stop(stop__)},
        step_{make_step(nop_type{})}
    {
        static_assert(std::is_convertible_v<Start,index_type>||std::is_same_v<Start,nop_type>);
        static_assert(std::is_convertible_v<Stop,index_type>||std::is_same_v<Stop,nop_type>);
    }
    template<typename Start>
    explicit slice(const Start& start__):
        mask_{make_mask(start__)},
        start_{make_start_stop(start__)},
        stop_{make_start_stop(nop_type{})},
        step_{make_step(nop_type{})}
    {
        static_assert(std::is_convertible_v<Start,index_type>||std::is_same_v<Start,nop_type>);
    }
    slice(const index_type& start__, reduce_tag_type):
        mask_{make_mask(reduce_tag_type{})},
        start_{make_start_stop(start__)},
        stop_{make_start_stop(nop_type{})},
        step_{make_step(nop_type{})}
    {}

    index_type start()const{return start_;}
    index_type stop()const{return stop_;}
    index_type step()const{return step_;}
    bool is_start()const{return detail::is_i(mask_);}
    bool is_stop()const{return detail::is_j(mask_);}
    bool is_step()const{return detail::is_k(mask_);}
    bool is_reduce()const{return detail::is_r(mask_);}
    friend bool operator==(const slice& lhs, const slice& rhs){return lhs.mask_==rhs.mask_ && lhs.start_==rhs.start_ && lhs.stop_==rhs.stop_ && lhs.step_==rhs.step_;}
    friend std::ostream& operator<<(std::ostream& os, const slice& lhs){
        os <<"("<<lhs.start_<<" "<<lhs.stop_<<" "<<lhs.step_<<")";
        return os;
    }

private:
    constexpr mask_type make_mask(reduce_tag_type){
        return mask_type::ri_k;
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

    mask_type mask_;
    index_type start_;
    index_type stop_;
    index_type step_;
};

}   //end of namespace gtensor
#endif