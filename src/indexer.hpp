#ifndef INDEXER_HPP_
#define INDEXER_HPP_

namespace gtensor{

namespace detail{
}   //end of namespace detail

//basic_indexer provides interface for random access of underlaying storage using subscription operator
//for storage engine it may wrapps its iterator, random access iterator required
//for viewing engine it is wrapper around its parent's indexer and use descriptor for index mapping
//subscription operator result can be used for assigning, if underlaying indexer's subscription result allows it
template<typename...> class basic_indexer;
template<typename IdxT, typename IndexerT>
class basic_indexer<IdxT, IndexerT>
{
public:
    using index_type = IdxT;
    using indexer_type = IndexerT;
    using result_type = decltype(std::declval<indexer_type>()[std::declval<index_type>()]);
    template<typename U>
    explicit basic_indexer(U&& indexer_):
        indexer{std::forward<U>(indexer_)}
    {}
    result_type operator[](const index_type& idx)const{return indexer[idx];}
private:
    indexer_type indexer;
};

template<typename IdxT ,typename IndexerT, typename DescT>
class basic_indexer<IdxT, IndexerT, DescT> : public basic_indexer<IdxT, IndexerT>
{
    using base_indexer_type = basic_indexer<IdxT, IndexerT>;
    using descriptor_type = DescT;
public:
    using typename base_indexer_type::index_type;
    using typename base_indexer_type::indexer_type;
    using typename base_indexer_type::result_type;
    template<typename U>
    basic_indexer(U&& indexer_, const descriptor_type& converter_):
        base_indexer_type{std::forward<U>(indexer_)},
        converter{&converter_}
    {}
    result_type operator[](const index_type& idx)const{return base_indexer_type::operator[](converter->convert(idx));}
private:
    const descriptor_type* converter;
};

//polymorphic indexer, can hold any indexer implementation
//subscription operator result type is always ValT
//IdxT is subscriptor type
//ValT is subscription result type
template<typename IdxT, typename ValT>
class poly_indexer{
    using value_type = ValT;
    using index_type = IdxT;

    class indexer_base{
    public:
        virtual ~indexer_base(){}
        virtual std::unique_ptr<indexer_base> clone()const = 0;
        virtual value_type operator[](const index_type&)const = 0;
        virtual void assign(const indexer_base&) = 0;
    };

    template<typename ImplT>
    class indexer_wrapper : public indexer_base
    {
        using impl_type = ImplT;
        impl_type impl_;
        std::unique_ptr<indexer_base> clone()const override{return std::make_unique<indexer_wrapper>(*this);}
        value_type operator[](const index_type& idx)const override{return impl_.operator[](idx);}
        void assign(const indexer_base& other) override{impl_ = static_cast<const indexer_wrapper&>(other).impl_;}
    public:
        indexer_wrapper(impl_type&& impl__):
            impl_{std::move(impl__)}
        {}
    };

    std::unique_ptr<indexer_base> impl_;
public:
    template<typename ImplT, std::enable_if_t<!std::is_convertible_v<std::decay_t<ImplT>, poly_indexer>,int> =0 >
    explicit poly_indexer(ImplT&& impl__):
        impl_{std::make_unique<indexer_wrapper<std::decay_t<ImplT>>>(std::forward<ImplT>(impl__))}
    {}
    poly_indexer() = default;
    poly_indexer(const poly_indexer& other):
        impl_{other.impl_->clone()}
    {}
    poly_indexer& operator=(const poly_indexer& other){
        impl_->assign(*other.impl_.get());
        return *this;
    }
    poly_indexer(poly_indexer&& other) = default;
    poly_indexer& operator=(poly_indexer&&) = default;

    value_type operator[](const index_type& idx)const{return impl_->operator[](idx);}
};

}   //end of namespace gtensor

#endif