/*
* GTensor - computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#ifndef MULTITHREADING_HPP_
#define MULTITHREADING_HPP_

#include <memory>
#include <thread>
#include <mutex>
#include <future>
#include <condition_variable>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>

namespace multithreading{

namespace detail{

template<typename T>
class element_
{
    using value_type = T;
public:
    //left buffer uninitialized
    element_(){}

    template<typename...Args>
    void emplace(Args&&...args){
        new(reinterpret_cast<void*>(buffer)) value_type{std::forward<Args>(args)...};
    }
    value_type&& move(){
        return std::move(*reinterpret_cast<value_type*>(buffer));
    }
    template<typename V>
    void move(V& v){
        v = std::move(*reinterpret_cast<value_type*>(buffer));
    }
    const value_type& get()const{
        return *reinterpret_cast<value_type*>(buffer);
    }
    value_type& get(){
        return *reinterpret_cast<value_type*>(buffer);
    }
    void destroy(){
        reinterpret_cast<value_type*>(buffer)->~value_type();
    }
private:
    alignas(value_type) std::byte buffer[sizeof(value_type)];
};

template<typename T>
class element
{
public:
    using value_type = T;
    ~element()
    {
        clear();
    }
    element(){}
    element(const element& other)
    {
        init(other);
    }
    element(element&& other)
    {
        init(std::move(other));
    }
    element& operator=(const element& other){
        clear();
        init(other);
        return *this;
    }
    element& operator=(element&& other){
        clear();
        init(std::move(other));
        return *this;
    }
    element& operator=(value_type&& v){
        element__.emplace(std::move(v));
        empty_ = false;
        return *this;
    }
    operator bool()const{return !empty_;}
    auto empty()const{return empty_;}
    auto& get()const{return element__.get();}
    auto& get(){return element__.get();}
private:
    void clear(){
        if (!empty_){
            element__.destroy();
        }
    }
    void init(const element& other){
        if (!other.empty_){
            element__.emplace(other.element__.get());
            empty_ = false;
        }
    }
    void init(element&& other){
        if (!other.empty_){
            element__.emplace(other.element__.move());
            empty_ = false;
        }
    }

    element_<value_type> element__{};
    bool empty_{true};
};

template<typename R>
class task_future{
    using result_type = R;
    bool sync_;
    std::future<result_type> f;
public:
    ~task_future(){
        if (sync_ && f.valid()){
            f.wait();
        }
    }
    task_future() = default;
    task_future(task_future&&) = default;
    task_future& operator=(task_future&&) = default;
    task_future(bool sync__, std::future<result_type>&& f_):
        sync_{sync__},
        f{std::move(f_)}
    {}
    operator bool(){return f.valid();}
    void wait()const{f.wait();}
    auto get(){return f.get();}
};

class task_v3_base
{
public:
    virtual ~task_v3_base(){}
    virtual void call() = 0;
};

template<typename F, typename...Args>
class task_v3_impl : public task_v3_base
{
    using args_type = decltype(std::make_tuple(std::declval<Args>()...));
    using result_type = std::decay_t<decltype(std::apply(std::declval<F>(),std::declval<args_type>()))>;
    F f;
    args_type args;
    std::promise<result_type> task_promise;
    void call() override {
            if constexpr(std::is_void_v<result_type>){
                std::apply(f, std::move(args));
                task_promise.set_value();
            }else{
                task_promise.set_value(std::apply(f, std::move(args)));
            }
        }
public:
    using future_type = task_future<result_type>;
    template<typename F_, typename...Args_>
    task_v3_impl(F_&& f_, Args_&&...args_):
            f{std::forward<F_>(f_)},
            args{std::make_tuple(std::forward<Args_>(args_)...)}
        {}
    auto get_future(bool sync = true){
            return future_type{sync, task_promise.get_future()};
        }
};

class task_v3
{
    std::unique_ptr<task_v3_base> impl;
public:
    task_v3() = default;
    void call(){
        impl->call();
    }
    template<typename F, typename...Args>
    auto set_task(bool sync, F&& f, Args&&...args){
        using impl_type = task_v3_impl<std::decay_t<F>, std::decay_t<Args>...>;
        impl = std::make_unique<impl_type>(std::forward<F>(f),std::forward<Args>(args)...);
        return static_cast<impl_type*>(impl.get())->get_future(sync);
    }
};

}   //end of namespace detail

//single thread bounded queue
template<typename T, typename Allocator = std::allocator<detail::element_<T>>>
class st_bounded_queue
{
    using element_type = typename std::allocator_traits<Allocator>::value_type;
public:
    using value_type = T;
    using allocator_type = Allocator;
    using size_type = typename std::allocator_traits<Allocator>::size_type;

    st_bounded_queue(size_type capacity__, const allocator_type& alloc = allocator_type()):
        capacity_{capacity__},
        allocator{alloc}
    {
        if (capacity_ == 0){
            throw std::invalid_argument("queue capacity must be > 0");
        }
        elements = allocator.allocate(capacity_+1);
    }
    ~st_bounded_queue()
    {
        clear();
        allocator.deallocate(elements, capacity_+1);
    }

    template<typename...Args>
    auto try_push(Args&&...args){
        value_type* res{nullptr};
        auto next_push_index = index(push_index+1);
        if (next_push_index==pop_index){
            return res;
        }else{
            elements[push_index].emplace(std::forward<Args>(args)...);
            res = &elements[push_index].get();
            push_index = next_push_index;
            return res;
        }
    }

    bool try_pop(value_type& v){
        return try_pop_(&v);
    }
    auto try_pop(){
        detail::element<value_type> v{};
        try_pop_(&v);
        return v;
    }
    bool pop(){
        return try_pop_(nullptr);
    }

    value_type* front(){return front_helper();}
    const value_type* front()const{return front_helper();}

    bool empty()const{return push_index == pop_index;}
    size_type size()const{return pop_index > push_index ? (capacity_+1+push_index-pop_index) : (push_index - pop_index);}
    size_type capacity()const{return capacity_;}

private:

    value_type* front_helper()const{
        return empty() ? nullptr : &elements[pop_index].get();
    }

    //call with nullptr arg will destroy front and update pop_index
    //call with valid pointer additionaly move element to v
    //in any case return true if queue no empty false otherwise
    template<typename V>
    bool try_pop_(V* v){
        if (empty()){
            return false;
        }else{
            if (v){
                elements[pop_index].move(*v);   //move assign from elements to v
            }
            elements[pop_index].destroy();
            pop_index = index(pop_index+1);
            return true;
        }
    }

    void clear(){
        while(!empty()){
            elements[pop_index].destroy();
            pop_index = index(pop_index+1);
        }
    }

    auto index(size_type cnt){return cnt%(capacity_+1);}

    size_type capacity_;
    allocator_type allocator;
    element_type* elements;
    size_type push_index{0};
    size_type pop_index{0};
};

//single allocation thread pool with bounded task queue
//allow different signatures and return types of task callable
//push task template method returns task_future<R>, where R is return type of callable given arguments types
class thread_pool_v3
{
    using task_type = detail::task_v3;
    using queue_type = st_bounded_queue<task_type>;
    using mutex_type = std::mutex;

public:

    ~thread_pool_v3()
    {
        stop();
    }
    thread_pool_v3(std::size_t n_workers):
        thread_pool_v3(n_workers, n_workers)
    {}
    thread_pool_v3(std::size_t n_workers, std::size_t n_tasks):
        workers(n_workers),
        tasks(n_tasks)
    {
        init();
    }

    //return task_future<R>, where R is return type of F called with args
    //std::reference_wrapper should be used to pass args by ref
    template<typename F, typename...Args>
    auto push(F&& f, Args&&...args){return push_<true>(std::forward<F>(f), std::forward<Args>(args)...);}
    template<typename F, typename...Args>
    auto push_async(F&& f, Args&&...args){return push_<false>(std::forward<F>(f), std::forward<Args>(args)...);}

private:

    template<bool Sync = true, typename F, typename...Args>
    auto push_(F&& f, Args&&...args){
        using future_type = decltype( std::declval<task_type>().set_task(Sync, std::forward<F>(f), std::forward<Args>(args)...));
        std::unique_lock<mutex_type> lock{guard};
        while(true){
            if (auto task = tasks.try_push()){
                future_type future = task->set_task(Sync, std::forward<F>(f), std::forward<Args>(args)...);
                lock.unlock();
                has_task.notify_one();
                return future;
            }else{
                has_slot.wait(lock);
            }
        }
    }

    void init(){
        std::for_each(workers.begin(),workers.end(),[this](auto& worker){worker=std::thread{&thread_pool_v3::worker_loop, this};});
    }

    void stop(){
        std::unique_lock<mutex_type> lock{guard};
        finish_workers.store(true);
        has_task.notify_all();
        lock.unlock();
        std::for_each(workers.begin(),workers.end(),[](auto& worker){worker.join();});
    }

    //problem is to use waiting not yealding in loop and have concurrent push and pop
    //conditional_variable must use same mutex to guard push and pop, even if queue is mpmc
    void worker_loop(){
        while(!finish_workers.load()){  //worker loop
            std::unique_lock<mutex_type> lock{guard};
            while(!finish_workers.load()){  //has_task conditional loop
                if (auto t = tasks.try_pop()){
                    lock.unlock();
                    has_slot.notify_one();
                    t.get().call();
                    break;
                }else{
                    has_task.wait(lock);
                }
            }
        }
    }

    std::vector<std::thread> workers;
    queue_type tasks;
    std::atomic<bool> finish_workers{false};
    mutex_type guard;
    std::condition_variable has_task;
    std::condition_variable has_slot;
};

template<typename ParSize>
class par_task_size
{
    using size_type = std::size_t;
    ParSize par_tasks_number_{0};
    ParSize par_task_size_{0};
    std::size_t rem_{0};
public:
    template<typename Size>
    par_task_size(const ParSize& tasks_number, const Size& max_par_tasks_number, const Size& min_tasks_per_par_task)
    {
        const ParSize par_tasks_number = std::min(static_cast<const ParSize&>(max_par_tasks_number), tasks_number/static_cast<const ParSize&>(min_tasks_per_par_task));
        if (par_tasks_number!=0){
            par_tasks_number_ = par_tasks_number;
            par_task_size_ = tasks_number/par_tasks_number;
            rem_ = static_cast<const size_type&>(tasks_number%par_tasks_number);
        }
    }
    size_type size()const{
        return static_cast<const size_type&>(par_tasks_number_);
    }
    ParSize operator[](size_type i)const{
        return i<rem_ ? par_task_size_+ParSize{1} : par_task_size_;
    }
};


inline constexpr std::size_t pool_workers_n = 16;
inline constexpr std::size_t pool_queue_size = 256;
inline auto& get_pool(){
    static thread_pool_v3 pool_{pool_workers_n, pool_queue_size};
    return pool_;
}

template<typename...> struct exec_policy_traits;
template<template<std::size_t> typename P, std::size_t V>
struct exec_policy_traits<P<V>>{
    using par_tasks = std::conditional_t<V==0, std::integral_constant<std::size_t,pool_workers_n>, std::integral_constant<std::size_t,V>>;
    using is_seq = std::bool_constant<par_tasks::value==1>;
};

template<std::size_t N> struct exec_pol : std::integral_constant<std::size_t,N>{};
template<typename> inline constexpr bool is_policy_v = false;
template<std::size_t N> inline constexpr bool is_policy_v<exec_pol<N>> = true;

template<typename Policy, typename It, typename Initial, typename BinaryF>
auto reduce(Policy, It first, It last, Initial initial, BinaryF f){
    if constexpr (std::is_convertible_v<typename std::iterator_traits<It>::iterator_category,std::random_access_iterator_tag> && !exec_policy_traits<Policy>::is_seq::value){ //parallelize
        using difference_type = typename std::iterator_traits<It>::difference_type;
        static constexpr std::size_t max_par_tasks_n = exec_policy_traits<Policy>::par_tasks::value;
        static constexpr std::size_t min_tasks_per_par_task = 2;

        par_task_size<difference_type> par_sizes{last-first,max_par_tasks_n,min_tasks_per_par_task};

        if (par_sizes.size()<2){
            return std::accumulate(first,last,initial,f);
        }

        auto body = [](auto first_, auto last_, auto f_){ //last-fist>=2 guaranteed by min_tasks_per_par_task = 2
            const auto& e0 = *first_;
            ++first_;
            const auto& e1 = *first_;
            return std::accumulate(++first_,last_,Initial(f_(e0,e1)),f_);
        };

        using future_type = decltype(get_pool().push(body,first,last,f));
        std::array<future_type, max_par_tasks_n> futures{};
        for (std::size_t i=0; i!=par_sizes.size(); ++i){
            const auto par_task_size = par_sizes[i];
            futures[i] = get_pool().push(body,first,first+par_task_size,f);
            first+=par_task_size;
        }
        return std::accumulate(futures.begin(),futures.begin()+par_sizes.size(),initial,[&f](const auto& init, auto& future){return f(init,future.get());});
    }else{
        return std::accumulate(first,last,initial,f);
    }
}

template<typename Policy, typename It1, typename It2, typename DstIt, typename BinaryF>
auto transform(Policy, It1 first1, It1 last1, It2 first2, DstIt dfirst, BinaryF f){
    if constexpr (
        std::is_convertible_v<typename std::iterator_traits<It1>::iterator_category,std::random_access_iterator_tag> &&
        std::is_convertible_v<typename std::iterator_traits<It2>::iterator_category,std::random_access_iterator_tag> &&
        std::is_convertible_v<typename std::iterator_traits<DstIt>::iterator_category,std::random_access_iterator_tag> &&
        !exec_policy_traits<Policy>::is_seq::value)
    { //parallelize
        using difference_type1 = typename std::iterator_traits<It1>::difference_type;
        using difference_type2 = typename std::iterator_traits<It2>::difference_type;
        using difference_type3 = typename std::iterator_traits<DstIt>::difference_type;
        static constexpr std::size_t max_par_tasks_n = exec_policy_traits<Policy>::par_tasks::value;
        static constexpr std::size_t min_tasks_per_par_task = 1;

        par_task_size<difference_type1> par_sizes{last1-first1,max_par_tasks_n,min_tasks_per_par_task};
        if (par_sizes.size()<2){
            return std::transform(first1,last1,first2,dfirst,f);
        }
        auto body = [](auto first1_, auto last1_, auto first2_, auto dfirst_, auto f_){
            for(;first1_!=last1_; ++first1_,++first2_,++dfirst_){
                *dfirst_ = f_(*first1_,*first2_);
            }
        };
        using future_type = decltype(get_pool().push(body,first1,last1,first2,dfirst,f));
        std::array<future_type, max_par_tasks_n> futures{};
        for (std::size_t i=0; i!=par_sizes.size(); ++i){
            const auto par_task_size = par_sizes[i];
            futures[i] = get_pool().push(body,first1,first1+par_task_size,first2,dfirst,f);
            first1+=par_task_size;
            first2+=static_cast<const difference_type2&>(par_task_size);
            dfirst+=static_cast<const difference_type3&>(par_task_size);
        }
        return dfirst;
    }else{
        return std::transform(first1,last1,first2,dfirst,f);
    }
}

template<typename Policy, typename DstIt, typename It, typename BinaryF>
void transform(Policy, DstIt first1, DstIt last1, It first2, BinaryF f){

    auto body = [](auto first1_, auto last1_, auto first2_, auto f_){
        for(;first1_!=last1_; ++first1_,++first2_){
            *first1_ = f_(*first1_,*first2_);
        }
    };

    if constexpr (
        std::is_convertible_v<typename std::iterator_traits<DstIt>::iterator_category,std::random_access_iterator_tag> &&
        std::is_convertible_v<typename std::iterator_traits<It>::iterator_category,std::random_access_iterator_tag> &&
        !exec_policy_traits<Policy>::is_seq::value)
    { //parallelize
        using difference_type1 = typename std::iterator_traits<DstIt>::difference_type;
        using difference_type2 = typename std::iterator_traits<It>::difference_type;
        static constexpr std::size_t max_par_tasks_n = exec_policy_traits<Policy>::par_tasks::value;
        static constexpr std::size_t min_tasks_per_par_task = 1;
        par_task_size<difference_type1> par_sizes{last1-first1,max_par_tasks_n,min_tasks_per_par_task};
        if (par_sizes.size()<2){
            body(first1,last1,first2,f);
            return;
        }
        using future_type = decltype(get_pool().push(body,first1,last1,first2,f));
        std::array<future_type, max_par_tasks_n> futures{};
        for (std::size_t i=0; i!=par_sizes.size(); ++i){
            const auto par_task_size = par_sizes[i];
            futures[i] = get_pool().push(body,first1,first1+par_task_size,first2,f);
            first1+=par_task_size;
            first2+=static_cast<const difference_type2&>(par_task_size);
        }
    }else{
        body(first1,last1,first2,f);
    }
}

template<typename Policy, typename It, typename DstIt>
auto copy(Policy, It first, It last, DstIt dfirst){
    if constexpr (std::is_convertible_v<typename std::iterator_traits<It>::iterator_category,std::random_access_iterator_tag> && !exec_policy_traits<Policy>::is_seq::value){ //parallelize
        using difference_type = typename std::iterator_traits<It>::difference_type;
        static constexpr std::size_t max_par_tasks_n = exec_policy_traits<Policy>::par_tasks::value;
        static constexpr std::size_t min_tasks_per_par_task = 1;
        par_task_size<difference_type> par_sizes{last-first,max_par_tasks_n,min_tasks_per_par_task};

        if (par_sizes.size()<2){
            return std::copy(first,last,dfirst);
        }

        auto body = [](auto first_, auto last_, auto dfirst_){
            std::copy(first_,last_,dfirst_);
        };

        using future_type = decltype(get_pool().push(body,first,last,dfirst));
        std::array<future_type, max_par_tasks_n> futures{};
        for (std::size_t i=0; i!=par_sizes.size(); ++i){
            const auto par_task_size = par_sizes[i];
            futures[i] = get_pool().push(body,first,first+par_task_size,dfirst);
            first+=par_task_size;
            dfirst+=par_task_size;
        }
        return dfirst;
    }else{
        return std::copy(first,last,dfirst);
    }
}

}   //end of namespace multithreading
#endif
