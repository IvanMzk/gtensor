#ifndef IMPL_EWALKER_HPP_
#define IMPL_EWALKER_HPP_

namespace gtensor{

template<typename ValT, template<typename> typename Cfg, typename F, typename...Wks>
class ewalker_impl : public walker_impl_base<ValT, Cfg>{
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    
public:    
    
    void walk(index_type direction, index_type steps) override{}
    void step(index_type direction) override{}
    void step_back(index_type direction) override{}
    void reset(index_type direction) override{}
    void reset() override{}        
    value_type operator*() const override{return 0;}
};


}   //end of namespace gtensor


#endif