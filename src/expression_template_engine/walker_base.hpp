#ifndef WALKER_BASE_HPP_
#define WALKER_BASE_HPP_

namespace gtensor{

namespace detail{
}   //end of namespace detail

template<typename ValT, typename CfgT>
class walker_base{
    using value_type = ValT;
    using index_type = typename CfgT::index_type;
public:
    virtual ~walker_base(){}
    virtual void walk(const index_type& direction, const index_type& steps) = 0;
    virtual void step(const index_type& direction) = 0;
    virtual void step_back(const index_type& direction) = 0;
    virtual void reset(const index_type& direction) = 0;
    virtual void reset() = 0;
    virtual value_type operator*() const = 0;
    virtual std::unique_ptr<walker_base<ValT,CfgT>> clone()const = 0;
};

template<typename ValT, typename CfgT>
class walker_trivial_root_base : public walker_base<ValT, CfgT>{
    using value_type = ValT;
    using index_type = typename CfgT::index_type;
public:
    virtual ~walker_trivial_root_base(){}
    virtual value_type operator[](const index_type& idx) const = 0;
};


}   //end of namespace gtensor

#endif