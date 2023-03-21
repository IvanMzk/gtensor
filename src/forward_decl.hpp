#ifndef FORWARD_DECL_HPP_
#define FORWARD_DECL_HPP_

namespace gtensor{

template<typename CfgT> class descriptor_base;

template<typename CfgT, typename StorT> class expression_template_storage_engine;
template<typename CfgT, typename F, typename...Operands> class expression_template_evaluating_engine;
template<typename CfgT, typename DescT, typename ParentT> class expression_template_viewing_engine;

template<typename EngineT> class storage_tensor;
template<typename EngineT> class evaluating_tensor;
template<typename DescT, typename EngineT> class viewing_tensor;

template<typename ValT, typename CfgT> class tensor_base;
template<typename ValT, typename CfgT, typename ImplT> class tensor;

class combiner;

}   //end of namespace gtensor


#endif