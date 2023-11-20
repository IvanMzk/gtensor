/*
* GTensor - computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#include "catch.hpp"
#include "benchmark_helpers.hpp"
#include "helpers_for_testing.hpp"
#include "tensor.hpp"
#include "tensor_math.hpp"


namespace benchmark_matmul_{

using gtensor::tensor;
using gtensor::basic_tensor;
using gtensor::detail::shape_to_str;
using benchmark_helpers::order_to_str;
using gtensor::config::c_order;
using gtensor::config::f_order;
using benchmark_helpers::timing;
using benchmark_helpers::statistic;

template<typename Tensor1, typename Tensor2>
struct bench_matmul_helper{

    using shape_type = typename Tensor1::shape_type;
    using layout1 = typename Tensor1::order;
    using layout2 = typename Tensor2::order;

    template<typename Shapes, typename Builder, typename Command>
    auto operator()(std::string mes, std::size_t n_iters, Shapes shapes, Builder builder, Command command){
        std::cout<<std::endl<<"layout1 "<<order_to_str(layout1{})<<" "<<"layout2 "<<order_to_str(layout2{})<<" "<<mes;
        std::vector<double> total_intervals{};
        for (auto shapes_it=shapes.begin(), shapes_last=shapes.end(); shapes_it!=shapes_last; ++shapes_it){
            const auto& shape1 = std::get<0>(*shapes_it);
            const auto& shape2 = std::get<1>(*shapes_it);
            auto t1 = builder(Tensor1(shape1));
            auto t2 = builder(Tensor2(shape2));
            std::vector<double> shape_intervals{};
            for (auto n=n_iters; n!=0; --n){
                //measure command time
                double dt = 0;
                dt = timing(command,t1,t2);
                shape_intervals.push_back(dt);
                total_intervals.push_back(dt);
            }
            std::cout<<std::endl<<"shape1 "<<shape_to_str(shape1)<<" shape2 "<<shape_to_str(shape2)<<" "<<statistic(shape_intervals);
        }
        std::cout<<std::endl<<"TOTAL "<<statistic(total_intervals);
    }
};

template<typename T=double, typename Shapes, typename Builder, typename Command>
auto bench_matmul(std::string mes, std::size_t n_iters, Shapes shapes, Builder builder, Command command){
    bench_matmul_helper<gtensor::tensor<T,c_order>,gtensor::tensor<T,c_order>>{}(mes,n_iters,shapes,builder,command);
    bench_matmul_helper<gtensor::tensor<T,f_order>,gtensor::tensor<T,f_order>>{}(mes,n_iters,shapes,builder,command);
    // bench_matmul_helper<gtensor::tensor<T,c_order>,gtensor::tensor<T,f_order>>{}(mes,n_iters,shapes,builder,command);
    // bench_matmul_helper<gtensor::tensor<T,f_order>,gtensor::tensor<T,c_order>>{}(mes,n_iters,shapes,builder,command);
}

template<typename...Ts, typename...Us>
auto matmul_2d(const basic_tensor<Ts...>& t1, const basic_tensor<Us...>& t2){
    using tensor_type1 = basic_tensor<Ts...>;
    using tensor_type2 = basic_tensor<Us...>;
    using value_type1 = typename tensor_type1::value_type;
    using value_type2 = typename tensor_type2::value_type;
    using order1 = typename basic_tensor<Ts...>::order;
    using order2 = typename basic_tensor<Us...>::order;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using res_order = std::conditional_t<std::is_same_v<order1,order2>,order1,gtensor::config::c_order>;
    using config_type = typename tensor_type1::config_type;
    using res_type = gtensor::detail::tensor_copy_type_t<std::decay_t<decltype(std::declval<value_type1>()*std::declval<value_type2>())>,res_order,config_type>;
    using shape_type = typename res_type::shape_type;
    const auto& shape1 = t1.shape();
    const auto& shape2 = t2.shape();
    res_type res({shape1[0],shape2[1]},0);
    const auto& res_shape = res.shape();
    const auto res_dim = gtensor::detail::make_dim(res_shape);
    const auto k = shape1[1];

    auto matmul_same_order = [k](auto res_it, auto it1, auto it2, const auto& m, const auto& n){
        for (auto res_last=res_it+m*n; res_it!=res_last; res_it+=n){
            for (auto [last1_,it2_]=std::make_tuple(it1+k,it2); it1!=last1_; ++it1){
                const auto e1 = *it1;
                for(auto [last2_,res_it_]=std::make_tuple(it2_+n,res_it); it2_!=last2_; ++it2_,++res_it_){
                    *res_it_+=e1**it2_;
                }
            }
        }
    };

    if constexpr (std::is_same_v<order1,order2>){
        auto a1 = t1.traverse_order_adapter(order1{});
        auto a2 = t2.traverse_order_adapter(order2{});
        auto a_res = res.traverse_order_adapter(res_order{});
        if constexpr (std::is_same_v<order1,c_order>){
            matmul_same_order(a_res.begin(),a1.begin(),a2.begin(),res_shape[0],res_shape[1]);
        }else{
            matmul_same_order(a_res.begin(),a2.begin(),a1.begin(),res_shape[1],res_shape[0]);
        }
    }else{
        if constexpr (std::is_same_v<order1,c_order>){
        }else{
        }
    }
    return res;
}

template<typename...Ts, typename...Us>
auto matmul_2d_tiled(const basic_tensor<Ts...>& t1, const basic_tensor<Us...>& t2){
    using tensor_type1 = basic_tensor<Ts...>;
    using tensor_type2 = basic_tensor<Us...>;
    using value_type1 = typename tensor_type1::value_type;
    using value_type2 = typename tensor_type2::value_type;
    using order1 = typename basic_tensor<Ts...>::order;
    using order2 = typename basic_tensor<Us...>::order;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using res_order = std::conditional_t<std::is_same_v<order1,order2>,order1,gtensor::config::c_order>;
    using config_type = typename tensor_type1::config_type;
    using res_type = gtensor::detail::tensor_copy_type_t<std::decay_t<decltype(std::declval<value_type1>()*std::declval<value_type2>())>,res_order,config_type>;
    using shape_type = typename res_type::shape_type;
    using index_type = typename res_type::index_type;
    const auto& shape1 = t1.shape();
    const auto& shape2 = t2.shape();
    res_type res({shape1[0],shape2[1]},0);
    const auto& res_shape = res.shape();
    const auto res_dim = gtensor::detail::make_dim(res_shape);
    const auto k = shape1[1];
    // const index_type k_ = 8;
    // const index_type n_ = 8;
    const index_type k_ = 16;
    const index_type n_ = 128;

    auto matmul_same_order = [k,k_](auto res_it, auto it1, auto it2, const auto& m, const auto& n, const auto& n_){

        for (index_type kk=0; kk<k; kk+=k_){
            for (index_type jj=0; jj<n; jj+=n_){
                //for (index_type i=0; i!=m; ++i){
                for (auto [res_it_, it1_, res_last_] = std::make_tuple(res_it,it1,res_it+m*n); res_it_!=res_last_; res_it_+=n,it1_+=k){
                    //for (auto r=kk; r!=std::min(kk+k_,k); ++r){
                    for (auto [it1__, it1_last__, it2_] = std::make_tuple(it1_+kk,it1_+std::min(kk+k_,k),it2+kk*n); it1__!=it1_last__; ++it1__,it2_+=n){
                        //const auto e1 = it1[i*k+r];
                        const auto e1 = *it1__;
                        //for(auto j=jj; j!=std::min(jj+n_,n); ++j){
                        for(auto [res_it__,res_last__,it2__] = std::make_tuple(res_it_+jj,res_it_+std::min(jj+n_,n),it2_+jj); res_it__!=res_last__; ++res_it__,++it2__){
                            //res_it[i*n+j]+=e1*it2[r*n+j];
                            *res_it__+=e1**it2__;
                        }
                    }
                }
            }
        }

        // for (index_type kk=0; kk<k; kk+=k_){
        //     for (index_type jj=0; jj<n; jj+=n_){
        //         for (index_type i=0; i!=m; ++i){
        //             for (auto r=kk; r!=std::min(kk+k_,k); ++r){
        //                 const auto e1 = it1[i*k+r];
        //                 for(auto j=jj; j!=std::min(jj+n_,n); ++j){
        //                     res_it[i*n+j]+=e1*it2[r*n+j];
        //                 }
        //             }
        //         }
        //     }
        // }

    };

    if constexpr (std::is_same_v<order1,order2>){
        auto a1 = t1.traverse_order_adapter(order1{});
        auto a2 = t2.traverse_order_adapter(order2{});
        auto a_res = res.traverse_order_adapter(res_order{});
        if constexpr (std::is_same_v<order1,c_order>){
            matmul_same_order(a_res.begin(),a1.begin(),a2.begin(),res_shape[0],res_shape[1],n_);
        }else{
            matmul_same_order(a_res.begin(),a2.begin(),a1.begin(),res_shape[1],res_shape[0],n_);
        }
    }else{
        if constexpr (std::is_same_v<order1,c_order>){
        }else{
        }
    }
    return res;
}

//precondition: idx<max
//postcondition: result > 0
template<typename IdxT>
auto make_size_helper(const IdxT& idx, const IdxT& block_size, const IdxT& max){
    return idx+block_size>max ? max-idx : block_size;
};

//2d f contigous to c contigous pack
template<typename It, typename DstIt, typename Size>
void naive_pack(It it, DstIt dit, const Size& rows_, const Size& cols_, const Size& stride_){
    using difference_type = typename std::iterator_traits<It>::difference_type;
    const auto rows = static_cast<difference_type>(rows_);
    const auto cols = static_cast<difference_type>(cols_);
    const auto stride = static_cast<difference_type>(stride_);
    for (auto it_last=it+rows; it!=it_last; ++it){
        for (auto it_=it, it_last_=it+stride*cols; it_!=it_last_; it_+=stride,++dit){
            *dit=*it_;
        }
    }
}
template<typename It, typename DstIt, typename Size>
void pack(It it, DstIt dit, const Size& rows_, const Size& cols_, const Size& stride_){
    using value_type = typename std::iterator_traits<DstIt>::value_type;
    static constexpr std::size_t cache_line_size = 64;
    static constexpr std::size_t n_ = cache_line_size/sizeof(value_type);
    if constexpr (n_ > 1){
        using difference_type = typename std::iterator_traits<It>::difference_type;
        const auto rows = static_cast<difference_type>(rows_);
        const auto cols = static_cast<difference_type>(cols_);
        const auto stride = static_cast<difference_type>(stride_);
        const auto n = static_cast<difference_type>(n_);
        for (difference_type r=0; r<rows; r+=n){
            const auto n_rows = make_size_helper(r,n,rows);
            for (auto it_=it+r, it_last_=it_+stride*cols; it_!=it_last_; it_+=stride,++dit){
                for (auto [it__,it_last__,dit_] = std::make_tuple(it_,it_+n_rows,dit); it__!=it_last__; ++it__,dit_+=cols){
                    *dit_ = *it__;
                }
            }
            dit+=cols*(n_rows-1);
        }
    }else{
        naive_pack(it,dit,rows_,cols_,stride_);
    }
}

template<typename...Ts, typename...Us>
auto matmul_2d_goto(const basic_tensor<Ts...>& t1, const basic_tensor<Us...>& t2){
    using tensor_type1 = basic_tensor<Ts...>;
    using tensor_type2 = basic_tensor<Us...>;
    using value_type1 = typename tensor_type1::value_type;
    using value_type2 = typename tensor_type2::value_type;
    using order1 = typename basic_tensor<Ts...>::order;
    using order2 = typename basic_tensor<Us...>::order;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using res_order = std::conditional_t<std::is_same_v<order1,order2>,order1,gtensor::config::c_order>;
    using config_type = typename tensor_type1::config_type;
    using res_type = gtensor::detail::tensor_copy_type_t<std::decay_t<decltype(std::declval<value_type1>()*std::declval<value_type2>())>,res_order,config_type>;
    using shape_type = typename res_type::shape_type;
    using index_type = typename res_type::index_type;
    const auto& shape1 = t1.shape();
    const auto& shape2 = t2.shape();
    res_type res({shape1[0],shape2[1]},0);
    const auto& res_shape = res.shape();
    const auto res_dim = gtensor::detail::make_dim(res_shape);

    const auto k = shape1[1];
    const auto m = res_shape[0];
    const auto n = res_shape[1];
    static constexpr std::size_t kc_size = 128;
    static constexpr std::size_t mc_size = 128;
    const index_type nc = 128;
    const index_type kc = kc_size;
    const index_type mc = mc_size;

    auto matmul_same_order = [k,kc](auto res_it, auto it1, auto it2, const auto& m, const auto& n, const auto& nc, const auto& mc){

        //got res tile, block A and panel B and sizes
        auto goto_micro_kernel_reference = [k,m](auto res_it, auto a_it, auto b_it, const auto& nc_, const auto& kc_, const auto& mc_){
            // for (index_type jr=0; jr!=nc_; ++jr){
            //     for (index_type ir=0; ir!=mc_; ++ir){
            //         decltype(auto) res = res_it[ir+jr*m];
            //         for (index_type kk=0; kk!=kc_; ++kk){
            //             res+=a_it[ir+kk*m]*b_it[kk+jr*k];   //dot
            //         }
            //     }
            // }

            for (auto res_last = res_it+nc_*m; res_it!=res_last; res_it+=m,b_it+=k){
                for (auto [res_it_,res_last_,a_it_] = std::make_tuple(res_it,res_it+mc_,a_it); res_it_!=res_last_; ++res_it_,++a_it_){
                    decltype(auto) res = *res_it_;
                    for (auto [b_it_,b_it_last_,a_it__] = std::make_tuple(b_it,b_it+kc_,a_it_); b_it_!=b_it_last_; ++b_it_,a_it__+=m){
                        res+=*a_it__**b_it_;
                    }
                }
            }
        };
        auto goto_micro_kernel_reference1 = [k,m](auto res_it, auto a_it, auto b_it, const auto& nc_, const auto& kc_, const auto& mc_){
            for (index_type jr=0; jr!=nc_; ++jr){
                for (index_type kk=0; kk!=kc_; ++kk){
                    const auto e = b_it[kk+jr*k];
                    for (index_type ir=0; ir!=mc_; ++ir){
                        res_it[ir+jr*m]+=a_it[ir+kk*m]*e;
                    }
                }
            }
        };
        auto goto_micro_kernel_reference1_a_packed = [k,m](auto res_it, auto a_it, auto b_it, const auto& nc_, const auto& kc_, const auto& mc_){
            for (index_type jr=0; jr!=nc_; ++jr){
                auto a_it_ = a_it;
                for (index_type kk=0; kk!=kc_; ++kk){
                    const auto e = b_it[kk+jr*k];
                    for (index_type ir=0; ir!=mc_; ++ir,++a_it_){
                        res_it[ir+jr*m]+=*a_it_*e;
                    }
                }
            }
        };
        auto goto_micro_kernel_a_packed = [k,m](auto res_it, auto a_it, auto b_it, const auto& nc_, const auto& kc_, const auto& mc_){
            // for (index_type jr=0; jr!=nc_; ++jr){
            //     for (index_type ir=0; ir!=mc_; ++ir){
            //         decltype(auto) res = res_it[ir+jr*m];
            //         for (index_type kk=0; kk!=kc_; ++kk){
            //             res+=a_it[ir*kc_+kk]*b_it[kk+jr*k];   //dot
            //         }
            //     }
            // }

            for (auto res_last = res_it+nc_*m; res_it!=res_last; res_it+=m,b_it+=k){
                for (auto [res_it_,res_last_,a_it_] = std::make_tuple(res_it,res_it+mc_,a_it); res_it_!=res_last_; ++res_it_){
                    decltype(auto) res = *res_it_;
                    for (auto b_it_=b_it,b_it_last_=b_it+kc_; b_it_!=b_it_last_; ++b_it_,++a_it_){
                        res+=*a_it_**b_it_;
                    }
                }
            }
        };



        auto make_size = [](const auto& idx, const auto& block_size, const auto& max){return idx+block_size>max ? max-idx : block_size;};


        // auto pack_a = [m](auto a_it, auto buf_it, const auto& mc_, const auto& kc_){
        //     for (index_type i=0; i!=mc_; ++i){
        //         for (index_type j=0; j!=kc_; ++j,++buf_it){
        //             *buf_it = a_it[i+j*m];
        //         }
        //     }
        // };
        auto pack_a = [m](auto a_it, auto buf_it, const auto& mc_, const auto& kc_){
            for (auto a_last = a_it+mc_; a_it!=a_last; ++a_it){
                for (auto a_it_=a_it, a_last_=a_it+kc_*m; a_it_!=a_last_; a_it_+=m,++buf_it){
                    *buf_it = *a_it_;
                }
            }
        };

        //pack block A columns to be contigous
        auto pack_a_ref1 = [m](auto a_it, auto buf_it, const auto& mc_, const auto& kc_){
            for (auto a_it_last=a_it+kc_*m; a_it!=a_it_last; a_it+=m){
                for (auto a_it_=a_it, a_it_last_=a_it+mc_; a_it_!=a_it_last_; ++a_it_,++buf_it){
                    *buf_it = *a_it_;
                }
            }
        };

        //buffer to pack a
        std::array<value_type1,kc_size*mc_size> a_buff;

        //The primary reason for the outer-most loop, indexed by jc, is to limit the amount of workspace required for panel B,
        //with a secondary reason to allow panel B to remain in the L3 cache
        //The primary advantage of constraining panel B to the L3 cache is that it is cheaper to access memory in terms of energy efficiency in the L3 cache rather than main memory.
        for (index_type jc=0; jc<n; jc+=nc){
            for (index_type pc=0; pc<k; pc+=kc){
                for (index_type ic=0; ic<m; ic+=mc){
                    //c
                    // res_it[ic*n+jc]
                    // it1[ic*k+pc]
                    // it2[pc*n+jc]
                    //f
                    // res_it[ic+jc*m]
                    // it1[ic+pc*m]
                    // it2[pc+jc*k]


                    //pack_a(it1+(ic+pc*m),a_buff.begin(),make_size(pc,kc,k),make_size(ic,mc,m));
                    //naive_pack(it1+(ic+pc*m),a_buff.begin(),make_size(ic,mc,m),make_size(pc,kc,k),m);
                    //pack(it1+(ic+pc*m),a_buff.begin(),make_size(ic,mc,m),make_size(pc,kc,k),m);
                    //goto_micro_kernel_a_packed(res_it+(ic+jc*m),a_buff.cbegin(),it2+(pc+jc*k),make_size(jc,nc,n),make_size(pc,kc,k),make_size(ic,mc,m));
                    //goto_micro_kernel_reference(res_it+(ic+jc*m),it1+(ic+pc*m),it2+(pc+jc*k),make_size(jc,nc,n),make_size(pc,kc,k),make_size(ic,mc,m));

                    goto_micro_kernel_reference1(res_it+(ic+jc*m),it1+(ic+pc*m),it2+(pc+jc*k),make_size(jc,nc,n),make_size(pc,kc,k),make_size(ic,mc,m));

                    //pack_a_ref1(it1+(ic+pc*m),a_buff.begin(),make_size(ic,mc,m),make_size(pc,kc,k));
                    //goto_micro_kernel_reference1_a_packed(res_it+(ic+jc*m),a_buff.cbegin(),it2+(pc+jc*k),make_size(jc,nc,n),make_size(pc,kc,k),make_size(ic,mc,m));
                }
            }
        }

    };

    if constexpr (std::is_same_v<order1,order2>){
        auto a1 = t1.traverse_order_adapter(order1{});
        auto a2 = t2.traverse_order_adapter(order2{});
        auto a_res = res.traverse_order_adapter(res_order{});
        if constexpr (std::is_same_v<order1,c_order>){
            matmul_same_order(a_res.begin(),a2.begin(),a1.begin(),res_shape[1],res_shape[0],mc,nc);
        }else{
            matmul_same_order(a_res.begin(),a1.begin(),a2.begin(),res_shape[0],res_shape[1],nc,mc);
        }
    }else{
        if constexpr (std::is_same_v<order1,c_order>){
        }else{
        }
    }
    return res;
}

template<typename...Ts, typename...Us>
auto matmul_2d_goto1(const basic_tensor<Ts...>& t1, const basic_tensor<Us...>& t2){
    using tensor_type1 = basic_tensor<Ts...>;
    using tensor_type2 = basic_tensor<Us...>;
    using value_type1 = typename tensor_type1::value_type;
    using value_type2 = typename tensor_type2::value_type;
    using order1 = typename basic_tensor<Ts...>::order;
    using order2 = typename basic_tensor<Us...>::order;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using res_order = std::conditional_t<std::is_same_v<order1,order2>,order1,gtensor::config::c_order>;
    using config_type = typename tensor_type1::config_type;
    using res_type = gtensor::detail::tensor_copy_type_t<std::decay_t<decltype(std::declval<value_type1>()*std::declval<value_type2>())>,res_order,config_type>;
    using shape_type = typename res_type::shape_type;
    using index_type = typename res_type::index_type;
    const auto& shape1 = t1.shape();
    const auto& shape2 = t2.shape();
    res_type res({shape1[0],shape2[1]},0);
    const auto& res_shape = res.shape();
    const auto res_dim = gtensor::detail::make_dim(res_shape);

    const auto k = shape1[1];
    const auto m = res_shape[0];
    const auto n = res_shape[1];
    static constexpr std::size_t mc_size = 128;
    static constexpr std::size_t nc_size = 128;
    static constexpr std::size_t kc_size = 128;
    const index_type mc = mc_size;
    const index_type nc = nc_size;
    const index_type kc = kc_size;

    const auto inner_stride_a = std::is_same_v<order1,c_order> ? index_type{1} : m;
    const auto outer_stride_a = std::is_same_v<order1,c_order> ? k : index_type{1};
    const auto inner_stride_b = std::is_same_v<order2,c_order> ? index_type{1} : k;
    const auto outer_stride_b = std::is_same_v<order2,c_order> ? n : index_type{1};
    const auto inner_stride_res = std::is_same_v<res_order,c_order> ? index_type{1} : m;
    const auto outer_stride_res = std::is_same_v<res_order,c_order> ? n : index_type{1};

    auto make_submatrix_size = [](const auto& idx, const auto& block_size, const auto& max){return idx+block_size>max ? max-idx : block_size;};

    auto fill_buf = [](auto it, auto dit, const auto& inner_stride, const auto& outer_stride, const auto& inner_size, const auto& outer_size){
        for (auto i=outer_size; i!=0; --i,it+=outer_stride){
            auto it_ = it;
            for (auto j=inner_size; j!=0; --j,it_+=inner_stride,++dit){
                *dit = *it_;
            }
        }
    };

    auto kernel = [](auto res_it, auto a_it, auto b_it, const auto& res_stride, const auto& mc_, const auto& nc_, const auto& kc_){
        for (index_type kk=0; kk!=kc_; ++kk){
            const auto jj = kk*mc_;
            const auto ii = kk*nc_;
            for (index_type jr=0; jr!=nc_; ++jr){
                const auto rr = jr*res_stride;
                const auto e = b_it[ii+jr];
                for (index_type ir=0; ir!=mc_; ++ir){
                    res_it[ir+rr]+=a_it[ir+jj]*e;
                }
            }
        }
    };

    benchmark_helpers::cpu_interval timer;
    double kernel_time = 0;

    //buffers
    std::array<value_type1,mc_size*kc_size> a_buf;
    std::array<value_type2,kc_size*nc_size> b_buf;
    //iterators
    auto res_it = res.traverse_order_adapter(res_order{}).begin();
    auto a_it = t1.traverse_order_adapter(order1{}).begin();
    auto b_it = t2.traverse_order_adapter(order2{}).begin();
    //The primary reason for the outer-most loop, indexed by jc, is to limit the amount of workspace required for panel B,
    //with a secondary reason to allow panel B to remain in the L3 cache
    //The primary advantage of constraining panel B to the L3 cache is that it is cheaper to access memory in terms of energy efficiency in the L3 cache rather than main memory.
    for (index_type jc=0; jc<n; jc+=nc){
            const auto nc_ = make_submatrix_size(jc,nc,n);
        for (index_type pc=0; pc<k; pc+=kc){
                const auto kc_ = make_submatrix_size(pc,kc,k);
                fill_buf(b_it+(pc*outer_stride_b+jc*inner_stride_b),b_buf.begin(),inner_stride_b,outer_stride_b,nc_,kc_);
            for (index_type ic=0; ic<m; ic+=mc){
                const auto mc_ = make_submatrix_size(ic,mc,m);
                fill_buf(a_it+(ic*outer_stride_a+pc*inner_stride_a),a_buf.begin(),outer_stride_a,inner_stride_a,mc_,kc_);
                timer.start();
                if constexpr (std::is_same_v<res_order,c_order>){
                    kernel(res_it+(jc+ic*n),b_buf.cbegin(),a_buf.cbegin(),n,nc_,mc_,kc_);
                }else{
                    kernel(res_it+(ic+jc*m),a_buf.cbegin(),b_buf.cbegin(),m,mc_,nc_,kc_);
                }
                timer.stop();
                kernel_time+=timer;
            }
        }
    }
    std::cout<<std::endl<<kernel_time;
    return res;
}

//add unroll
template<typename...Ts, typename...Us>
auto matmul_2d_goto2(const basic_tensor<Ts...>& t1, const basic_tensor<Us...>& t2){
    using tensor_type1 = basic_tensor<Ts...>;
    using tensor_type2 = basic_tensor<Us...>;
    using value_type1 = typename tensor_type1::value_type;
    using value_type2 = typename tensor_type2::value_type;
    using order1 = typename basic_tensor<Ts...>::order;
    using order2 = typename basic_tensor<Us...>::order;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using res_order = std::conditional_t<std::is_same_v<order1,order2>,order1,gtensor::config::c_order>;
    using config_type = typename tensor_type1::config_type;
    using res_type = gtensor::detail::tensor_copy_type_t<std::decay_t<decltype(std::declval<value_type1>()*std::declval<value_type2>())>,res_order,config_type>;
    using shape_type = typename res_type::shape_type;
    using index_type = typename res_type::index_type;
    const auto& shape1 = t1.shape();
    const auto& shape2 = t2.shape();
    res_type res({shape1[0],shape2[1]},0);
    const auto& res_shape = res.shape();
    const auto res_dim = gtensor::detail::make_dim(res_shape);

    const auto k = shape1[1];
    const auto m = res_shape[0];
    const auto n = res_shape[1];
    static constexpr std::size_t mc_size = 128;
    static constexpr std::size_t nc_size = 128;
    static constexpr std::size_t kc_size = 128;
    const index_type mc = mc_size;
    const index_type nc = nc_size;
    const index_type kc = kc_size;

    const auto inner_stride_a = std::is_same_v<order1,c_order> ? index_type{1} : m;
    const auto outer_stride_a = std::is_same_v<order1,c_order> ? k : index_type{1};
    const auto inner_stride_b = std::is_same_v<order2,c_order> ? index_type{1} : k;
    const auto outer_stride_b = std::is_same_v<order2,c_order> ? n : index_type{1};
    const auto inner_stride_res = std::is_same_v<res_order,c_order> ? index_type{1} : m;
    const auto outer_stride_res = std::is_same_v<res_order,c_order> ? n : index_type{1};

    auto make_submatrix_size = [](const auto& idx, const auto& block_size, const auto& max){return idx+block_size>max ? max-idx : block_size;};

    auto fill_buf = [](auto it, auto* dst, const auto& inner_stride, const auto& outer_stride, const auto& inner_size, const auto& outer_size){
        for (const auto it_last=it+outer_stride*outer_size; it!=it_last; it+=outer_stride){
            auto it_ = it;
            const auto dst_last = dst+static_cast<std::ptrdiff_t>(inner_size);
            if (inner_size > 3){
                for (const auto dst_last_=dst_last-3; dst<dst_last_; dst+=4){
                    *dst = *it_;
                    it_+=inner_stride;
                    *(dst+1) = *it_;
                    it_+=inner_stride;
                    *(dst+2) = *it_;
                    it_+=inner_stride;
                    *(dst+3) = *it_;
                    it_+=inner_stride;
                }
            }
            for (;dst!=dst_last; ++dst,it_+=inner_stride){
                *dst = *it_;
            }
        }
    };

    auto fill_res = [](const auto* buf, auto res_it, const auto& outer_stride, const auto& inner_size, const auto& outer_size){
        for (const auto res_last=res_it+outer_stride*outer_size; res_it!=res_last; res_it+=outer_stride){
            auto res_it_ = res_it;
            const auto buf_last = buf+static_cast<std::ptrdiff_t>(inner_size);
            if (inner_size > 3){
                for (const auto buf_last_=buf_last-3; buf<buf_last_; buf+=4,res_it_+=4){
                    *res_it_ += *buf;
                    *(res_it_+1) += *(buf+1);
                    *(res_it_+2) += *(buf+2);
                    *(res_it_+3) += *(buf+3);
                }
            }
            for (;buf!=buf_last; ++buf,++res_it_){
                *res_it_ += *buf;
            }
        }
    };

    auto kernel = [](auto* res_buf, const auto* const a_data, const auto* b_data, const std::ptrdiff_t& mc_, const std::ptrdiff_t& nc_, const std::ptrdiff_t& kc_){
        auto res_buf_ = res_buf;
        for (const auto b_last=b_data+nc_; b_data!=b_last; ++b_data){
            const auto e = *b_data;
            for (std::ptrdiff_t ir=0; ir!=mc_; ++ir,++res_buf_){
                *res_buf_=a_data[ir]*e;
            }
        }
        for (std::ptrdiff_t kk=1; kk!=kc_; ++kk){
            const auto a_data_ = a_data+kk*mc_;
            auto res_buf_ = res_buf;
            for (const auto b_last=b_data+nc_; b_data!=b_last; ++b_data){
                const auto e = *b_data;

                // auto a_data__=a_data_;
                // const auto a_last = a_data_+mc_;
                // if (mc_>7){
                //     for (const auto a_last__=a_last-7; a_data__<a_last__; a_data__+=8,res_buf_+=8){
                //         *res_buf_+=*a_data__*e;
                //         *(res_buf_+1)+=*(a_data__+1)*e;
                //         *(res_buf_+2)+=*(a_data__+2)*e;
                //         *(res_buf_+3)+=*(a_data__+3)*e;
                //         *(res_buf_+4)+=*(a_data__+4)*e;
                //         *(res_buf_+5)+=*(a_data__+5)*e;
                //         *(res_buf_+6)+=*(a_data__+6)*e;
                //         *(res_buf_+7)+=*(a_data__+7)*e;
                //     }
                // }
                // for (;a_data__!=a_last; ++a_data__,++res_buf_){
                //     *res_buf_+=*a_data__*e;
                // }

                // auto a_data__=a_data_;
                // const auto a_last = a_data_+mc_;
                // if (mc_>3){
                //     for (const auto a_last__=a_last-3; a_data__<a_last__; a_data__+=4,res_buf_+=4){
                //         *res_buf_+=*a_data__*e;
                //         *(res_buf_+1)+=*(a_data__+1)*e;
                //         *(res_buf_+2)+=*(a_data__+2)*e;
                //         *(res_buf_+3)+=*(a_data__+3)*e;
                //     }
                // }
                // for (;a_data__!=a_last; ++a_data__,++res_buf_){
                //     *res_buf_+=*a_data__*e;
                // }

                // auto a_data__=a_data_;
                // const auto a_last = a_data_+mc_;
                // if (mc_>3){
                //     for (const auto a_last__=a_last-3; a_data__<a_last__; a_data__+=4){
                //         *res_buf_+=*a_data__*e;
                //         ++res_buf_;
                //         *res_buf_+=*(a_data__+1)*e;
                //         ++res_buf_;
                //         *res_buf_+=*(a_data__+2)*e;
                //         ++res_buf_;
                //         *res_buf_+=*(a_data__+3)*e;
                //         ++res_buf_;
                //     }
                // }
                // for (;a_data__!=a_last; ++a_data__,++res_buf_){
                //     *res_buf_+=*a_data__*e;
                // }

                // auto a_data__=a_data_;
                // for (const auto a_last__=a_data__+mc_; a_data__!=a_last__; ++a_data__,++res_buf_){
                //     *res_buf_+=*a_data__*e;
                // }

                // auto a_data__=a_data_;
                // for (const auto res_last=res_buf_+mc_; res_buf_!=res_last; ++res_buf_,++a_data__){
                //     *res_buf_+=*a_data__*e;
                // }
                // std::ptrdiff_t ir=0;
                // for (const auto ir_last=mc_-3; ir<ir_last; ir+=4,res_buf_+=4){
                //     *res_buf_+=a_data_[ir]*e;
                //     *(res_buf_+1)+=a_data_[ir+1]*e;
                //     *(res_buf_+2)+=a_data_[ir+2]*e;
                //     *(res_buf_+3)+=a_data_[ir+3]*e;
                // }
                // for (;ir!=mc_; ++ir,++res_buf_){
                //     *res_buf_+=a_data_[ir]*e;
                // }

                for (std::ptrdiff_t ir=0; ir!=mc_; ++ir,++res_buf_){
                    *res_buf_+=a_data_[ir]*e;
                }
            }
        }
    };

    // benchmark_helpers::cpu_interval timer;
    // double kernel_time = 0;

    //buffers
    std::array<value_type1,mc_size*kc_size> a_buf;
    std::array<value_type2,kc_size*nc_size> b_buf;
    std::array<value_type2,mc_size*nc_size> res_buf;
    //iterators
    auto res_it = res.traverse_order_adapter(res_order{}).begin();
    auto a_it = t1.traverse_order_adapter(order1{}).begin();
    auto b_it = t2.traverse_order_adapter(order2{}).begin();
    //The primary reason for the outer-most loop, indexed by jc, is to limit the amount of workspace required for panel B,
    //with a secondary reason to allow panel B to remain in the L3 cache
    //The primary advantage of constraining panel B to the L3 cache is that it is cheaper to access memory in terms of energy efficiency in the L3 cache rather than main memory.
    for (index_type jc=0; jc<n; jc+=nc){
            const auto nc_ = make_submatrix_size(jc,nc,n);
        for (index_type pc=0; pc<k; pc+=kc){
                const auto kc_ = make_submatrix_size(pc,kc,k);
                fill_buf(b_it+(pc*outer_stride_b+jc*inner_stride_b),b_buf.data(),inner_stride_b,outer_stride_b,nc_,kc_);
            for (index_type ic=0; ic<m; ic+=mc){
                const auto mc_ = make_submatrix_size(ic,mc,m);
                fill_buf(a_it+(ic*outer_stride_a+pc*inner_stride_a),a_buf.data(),outer_stride_a,inner_stride_a,mc_,kc_);
                //timer.start();
                //std::fill(res_buf.begin(),res_buf.end(),0);
                if constexpr (std::is_same_v<res_order,c_order>){
                    kernel(res_buf.data(),b_buf.cbegin(),a_buf.cbegin(),nc_,mc_,kc_);
                    fill_res(res_buf.data(),res_it+(jc+ic*n),n,nc_,mc_);
                }else{
                    kernel(res_buf.data(),a_buf.cbegin(),b_buf.cbegin(),mc_,nc_,kc_);
                    // std::copy(res_buf.begin(),res_buf.begin()+mc_*nc_,std::ostream_iterator<double>{std::cout,","});
                    // std::cout<<std::endl;
                    fill_res(res_buf.data(),res_it+(ic+jc*m),m,mc_,nc_);
                }
                //timer.stop();
                //kernel_time+=timer;
            }
        }
    }
    //std::cout<<std::endl<<kernel_time;
    return res;
}


}   //end of namespace benchmark_matmul_

TEST_CASE("benchmark_pack","[benchmark_tensor]")
{
    using benchmark_matmul_::pack;
    using benchmark_matmul_::naive_pack;
    using benchmark_helpers::timing;
    using benchmark_helpers::statistic;
    using gtensor::tensor;
    using gtensor::config::c_order;
    using gtensor::config::f_order;

    //(30,40)
    const auto aa = tensor<double,f_order>{{1,1,1,3,3,2,3,2,2,2,0,2,4,0,4,3,2,3,2,3,2,4,3,4,0,2,3,4,1,0,0,1,2,0,4,0,2,3,2,4},{4,4,2,2,4,1,1,1,4,4,1,2,3,4,0,4,3,4,3,2,2,1,3,2,1,4,1,0,1,0,1,2,4,1,4,3,1,4,4,0},{1,1,1,0,2,0,3,3,1,4,3,1,1,3,1,1,4,2,1,2,3,1,4,1,0,3,2,1,2,0,2,2,1,3,4,4,2,1,3,4},{0,3,2,3,1,2,0,4,3,3,4,3,0,1,0,0,3,0,0,0,0,0,3,1,1,2,2,4,1,3,0,1,2,0,3,1,3,1,3,2},{4,0,4,0,2,3,2,0,4,3,0,1,4,2,0,1,1,1,2,4,1,3,2,4,0,3,4,3,3,0,4,4,0,1,2,0,4,1,1,2},{3,3,1,2,2,1,4,0,1,4,0,2,3,4,1,3,0,0,1,3,4,4,1,4,3,1,1,1,1,2,0,2,2,1,3,4,4,0,0,3},{4,0,2,0,2,2,4,1,4,0,0,0,4,4,3,1,4,4,1,4,4,0,0,4,0,4,2,4,4,0,4,1,1,4,3,1,0,3,2,0},{0,0,3,0,0,1,1,3,2,1,0,4,3,2,3,1,2,1,4,4,1,0,1,3,2,1,1,2,0,3,1,3,2,0,2,2,1,3,3,1},{0,1,1,1,0,2,1,2,1,4,0,0,3,0,2,3,4,0,3,0,4,1,4,3,2,3,2,1,3,3,3,0,4,3,4,1,0,1,4,2},{3,4,1,1,1,2,1,3,3,1,3,3,2,2,1,2,1,0,4,3,0,4,3,1,0,3,2,4,3,3,0,1,2,1,3,2,3,2,3,3},{0,4,1,3,4,4,3,1,1,3,1,4,0,1,1,4,0,2,0,4,4,3,2,3,0,3,4,4,2,2,4,1,3,0,2,4,0,1,3,3},{0,2,4,1,2,0,2,1,0,4,1,0,3,4,3,2,0,1,3,1,3,3,2,4,1,0,0,1,4,4,4,3,3,2,0,4,3,4,3,4},{0,3,3,3,2,2,4,2,4,4,0,0,1,0,4,0,0,1,3,3,0,2,2,0,2,4,1,1,3,4,4,2,2,3,2,2,0,3,1,0},{0,0,2,3,3,0,3,0,1,4,2,3,3,0,2,4,3,1,2,1,4,1,0,3,1,4,2,3,3,4,0,1,4,2,4,4,1,4,2,0},{3,0,1,4,1,3,1,3,0,3,3,3,3,4,3,3,2,4,0,2,1,1,2,1,0,1,0,2,2,0,3,1,3,1,3,2,1,3,4,3},{4,4,1,0,2,3,1,3,2,1,1,0,2,0,2,3,0,1,2,2,4,1,3,3,4,4,2,4,0,3,2,4,4,0,2,2,3,2,1,0},{1,1,4,1,2,4,0,4,0,4,4,1,0,0,3,4,4,1,2,2,4,0,2,3,2,0,2,2,4,4,3,0,4,0,4,2,3,0,0,0},{2,1,2,4,2,4,1,4,3,0,0,0,2,1,4,1,1,0,4,1,1,4,4,3,3,0,2,2,1,4,3,3,4,3,2,0,2,3,1,4},{1,2,1,4,1,1,3,4,2,4,4,3,3,0,0,1,3,3,1,4,1,2,2,3,3,3,1,0,1,1,3,1,4,1,0,1,1,2,0,0},{2,4,3,1,3,1,1,1,2,3,0,1,4,4,2,4,1,2,0,0,0,3,1,2,3,3,4,3,1,1,1,1,0,1,0,1,0,2,4,3},{0,2,0,0,1,1,3,2,0,0,3,4,4,3,4,4,2,1,2,4,4,1,3,0,2,0,1,0,1,2,4,1,3,1,4,4,0,1,1,0},{1,0,3,1,3,4,3,4,1,0,0,1,4,4,1,4,4,4,0,3,2,4,4,0,2,1,4,3,0,2,0,3,2,0,0,2,0,1,0,3},{1,1,2,0,0,4,0,2,3,0,2,1,1,4,0,0,1,4,3,1,1,4,4,2,1,0,1,4,1,3,3,3,0,2,1,3,1,4,4,2},{4,0,2,4,4,3,1,1,2,2,1,2,0,2,4,1,4,1,2,0,3,3,1,2,3,3,2,0,2,0,0,4,2,1,4,0,4,2,4,2},{3,3,3,4,2,0,1,1,4,3,0,4,1,4,1,4,2,3,0,1,1,3,0,3,1,1,1,4,4,3,0,0,4,3,2,2,4,4,2,3},{1,4,2,0,3,3,3,2,1,2,1,1,0,0,0,3,3,4,1,0,1,1,3,2,3,1,1,2,2,3,3,1,3,0,2,0,3,0,3,4},{0,0,4,2,2,2,3,2,1,0,0,3,4,4,2,1,2,4,1,2,2,2,1,4,0,2,0,1,2,2,1,3,4,3,3,0,3,0,4,3},{0,2,3,3,3,2,1,0,3,1,0,4,0,4,2,2,2,4,1,1,4,1,1,4,4,0,3,2,3,3,4,0,2,1,4,0,1,1,2,3},{3,1,3,2,4,1,4,1,3,0,3,1,2,0,1,1,1,2,4,3,0,2,0,4,4,0,0,1,0,2,0,0,4,4,0,0,3,4,1,4},{0,4,1,2,2,0,3,1,3,4,1,2,1,0,4,4,0,4,0,0,0,3,0,4,4,3,0,3,3,0,2,0,1,3,4,2,3,3,4,3}};
    const auto expected = tensor<double>{3,3,2,3,2,2,2,0,2,4,2,4,1,1,1,4,4,1,2,3,0,2,0,3,3,1,4,3,1,1,3,1,2,0,4,3,3,4,3,0,0,2,3,2,0,4,3,0,1,4,2,2,1,4,0,1,4,0,2,3,0,2,2,4,1,4,0,0,0,4,0,0,1,1,3,2,1,0,4,3,1,0,2,1,2,1,4,0,0,3,1,1,2,1,3,3,1,3,3,2,3,4,4,3,1,1,3,1,4,0,1,2,0,2,1,0,4,1,0,3,3,2,2,4,2,4,4,0,0,1,3,3,0,3,0,1,4,2,3,3,4,1,3,1,3,0,3,3,3,3,0,2,3,1,3,2,1,1,0,2,1,2,4,0,4,0,4,4,1,0,4,2,4,1,4,3,0,0,0,2,4,1,1,3,4,2,4,4,3,3,1,3,1,1,1,2,3,0,1,4};
    SECTION("naiv_pack")
    {
        tensor<double> buf(200,0);
        naive_pack(aa.traverse_order_adapter(f_order{}).begin()+90,buf.begin(),20,10,30);
        REQUIRE(buf==expected);
    }
    SECTION("pack")
    {
        tensor<double> buf(200,0);
        pack(aa.traverse_order_adapter(f_order{}).begin()+90,buf.begin(),20,10,30);
        REQUIRE(buf==expected);
    }
    SECTION("pack1")
    {
        tensor<double> buf(aa.size(),0);
        pack(aa.traverse_order_adapter(f_order{}).begin(),buf.begin(),30,40,30);
        REQUIRE(buf==aa.flatten());
    }

    tensor<double,f_order> t({10000,10000},1);
    std::vector<double> buf(128*128,0);

    auto bench_naive_pack = [](auto it, auto dit, auto rows, auto cols, auto stride){
        naive_pack(it,dit,rows,cols,stride);
        return *dit;
    };
    auto bench_pack = [](auto it, auto dit, auto rows, auto cols, auto stride){
        pack(it,dit,rows,cols,stride);
        return *dit;
    };

    //auto dt = timing(bench_naive_pack, t.traverse_order_adapter(f_order{}).begin(),buf.begin(),128,128,10000);
    //std::cout<<std::endl<<dt;



}

TEST_CASE("benchmark_matmul","[benchmark_tensor]")
{
    //using value_type = std::complex<double>;
    //using value_type = std::int64_t;
    //using value_type = int;
    using value_type = double;
    //using value_type = float;
    using gtensor::tensor;
    using tensor_type = tensor<value_type>;
    using helpers_for_testing::generate_lehmer;
    using benchmark_helpers::timing;
    using benchmark_matmul_::bench_matmul;

    auto builder = [](auto&& t_){
        generate_lehmer(t_.begin(),t_.end(),[](const auto& e){return e%5;},123);
        return t_.clone_shallow();
    };

    auto shapes = std::vector<std::pair<std::vector<int>,std::vector<int>>>{
        // //1d x nd
        // std::make_pair(std::vector<int>{10000},std::vector<int>{10000,10000}),
        // //nd x 1d
        // std::make_pair(std::vector<int>{10000,10000},std::vector<int>{10000}),
        // //nd x nd
        //std::make_pair(std::vector<int>{1000,1000},std::vector<int>{1000,1000})
        //std::make_pair(std::vector<int>{1281,1000},std::vector<int>{1000,1282}),
        //std::make_pair(std::vector<int>{1282,1000},std::vector<int>{1000,1281}),
        //std::make_pair(std::vector<int>{1284,1000},std::vector<int>{1000,1283}),
        //std::make_pair(std::vector<int>{1283,1000},std::vector<int>{1000,1284})
        //std::make_pair(std::vector<int>{2000,2000},std::vector<int>{2000,2000})
        //std::make_pair(std::vector<int>{4000,4000},std::vector<int>{4000,4000})
        //std::make_pair(std::vector<int>{4567,4765},std::vector<int>{4765,4321})
        //std::make_pair(std::vector<int>{200,100000},std::vector<int>{100000,300})
        //std::make_pair(std::vector<int>{6000,6000},std::vector<int>{6000,6000})
        std::make_pair(std::vector<int>{10000,10000},std::vector<int>{10000,10000})
        //std::make_pair(std::vector<int>{3,2,300,1000},std::vector<int>{2,1000,900})
        //std::make_pair(std::vector<int>{100,100,200,100},std::vector<int>{100,100,300})
    };
    const auto n_iters = 1;
    //bench_matmul("bench matmul",n_iters,shapes,builder,command_matmul);


    using benchmark_matmul_::matmul_2d;
    using benchmark_matmul_::matmul_2d_tiled;
    using benchmark_matmul_::matmul_2d_goto;
    using benchmark_matmul_::matmul_2d_goto1;
    using benchmark_matmul_::matmul_2d_goto2;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    //auto aa = tensor<value_type>{{1,1,1,3,3,2,3,2,2,2,0,2,4,0,4,3,2,3,2,3,2,4,3,4,0,2,3,4,1,0,0,1,2,0,4,0,2,3,2,4},{4,4,2,2,4,1,1,1,4,4,1,2,3,4,0,4,3,4,3,2,2,1,3,2,1,4,1,0,1,0,1,2,4,1,4,3,1,4,4,0},{1,1,1,0,2,0,3,3,1,4,3,1,1,3,1,1,4,2,1,2,3,1,4,1,0,3,2,1,2,0,2,2,1,3,4,4,2,1,3,4},{0,3,2,3,1,2,0,4,3,3,4,3,0,1,0,0,3,0,0,0,0,0,3,1,1,2,2,4,1,3,0,1,2,0,3,1,3,1,3,2},{4,0,4,0,2,3,2,0,4,3,0,1,4,2,0,1,1,1,2,4,1,3,2,4,0,3,4,3,3,0,4,4,0,1,2,0,4,1,1,2},{3,3,1,2,2,1,4,0,1,4,0,2,3,4,1,3,0,0,1,3,4,4,1,4,3,1,1,1,1,2,0,2,2,1,3,4,4,0,0,3},{4,0,2,0,2,2,4,1,4,0,0,0,4,4,3,1,4,4,1,4,4,0,0,4,0,4,2,4,4,0,4,1,1,4,3,1,0,3,2,0},{0,0,3,0,0,1,1,3,2,1,0,4,3,2,3,1,2,1,4,4,1,0,1,3,2,1,1,2,0,3,1,3,2,0,2,2,1,3,3,1},{0,1,1,1,0,2,1,2,1,4,0,0,3,0,2,3,4,0,3,0,4,1,4,3,2,3,2,1,3,3,3,0,4,3,4,1,0,1,4,2},{3,4,1,1,1,2,1,3,3,1,3,3,2,2,1,2,1,0,4,3,0,4,3,1,0,3,2,4,3,3,0,1,2,1,3,2,3,2,3,3},{0,4,1,3,4,4,3,1,1,3,1,4,0,1,1,4,0,2,0,4,4,3,2,3,0,3,4,4,2,2,4,1,3,0,2,4,0,1,3,3},{0,2,4,1,2,0,2,1,0,4,1,0,3,4,3,2,0,1,3,1,3,3,2,4,1,0,0,1,4,4,4,3,3,2,0,4,3,4,3,4},{0,3,3,3,2,2,4,2,4,4,0,0,1,0,4,0,0,1,3,3,0,2,2,0,2,4,1,1,3,4,4,2,2,3,2,2,0,3,1,0},{0,0,2,3,3,0,3,0,1,4,2,3,3,0,2,4,3,1,2,1,4,1,0,3,1,4,2,3,3,4,0,1,4,2,4,4,1,4,2,0},{3,0,1,4,1,3,1,3,0,3,3,3,3,4,3,3,2,4,0,2,1,1,2,1,0,1,0,2,2,0,3,1,3,1,3,2,1,3,4,3},{4,4,1,0,2,3,1,3,2,1,1,0,2,0,2,3,0,1,2,2,4,1,3,3,4,4,2,4,0,3,2,4,4,0,2,2,3,2,1,0},{1,1,4,1,2,4,0,4,0,4,4,1,0,0,3,4,4,1,2,2,4,0,2,3,2,0,2,2,4,4,3,0,4,0,4,2,3,0,0,0},{2,1,2,4,2,4,1,4,3,0,0,0,2,1,4,1,1,0,4,1,1,4,4,3,3,0,2,2,1,4,3,3,4,3,2,0,2,3,1,4},{1,2,1,4,1,1,3,4,2,4,4,3,3,0,0,1,3,3,1,4,1,2,2,3,3,3,1,0,1,1,3,1,4,1,0,1,1,2,0,0},{2,4,3,1,3,1,1,1,2,3,0,1,4,4,2,4,1,2,0,0,0,3,1,2,3,3,4,3,1,1,1,1,0,1,0,1,0,2,4,3},{0,2,0,0,1,1,3,2,0,0,3,4,4,3,4,4,2,1,2,4,4,1,3,0,2,0,1,0,1,2,4,1,3,1,4,4,0,1,1,0},{1,0,3,1,3,4,3,4,1,0,0,1,4,4,1,4,4,4,0,3,2,4,4,0,2,1,4,3,0,2,0,3,2,0,0,2,0,1,0,3},{1,1,2,0,0,4,0,2,3,0,2,1,1,4,0,0,1,4,3,1,1,4,4,2,1,0,1,4,1,3,3,3,0,2,1,3,1,4,4,2},{4,0,2,4,4,3,1,1,2,2,1,2,0,2,4,1,4,1,2,0,3,3,1,2,3,3,2,0,2,0,0,4,2,1,4,0,4,2,4,2},{3,3,3,4,2,0,1,1,4,3,0,4,1,4,1,4,2,3,0,1,1,3,0,3,1,1,1,4,4,3,0,0,4,3,2,2,4,4,2,3},{1,4,2,0,3,3,3,2,1,2,1,1,0,0,0,3,3,4,1,0,1,1,3,2,3,1,1,2,2,3,3,1,3,0,2,0,3,0,3,4},{0,0,4,2,2,2,3,2,1,0,0,3,4,4,2,1,2,4,1,2,2,2,1,4,0,2,0,1,2,2,1,3,4,3,3,0,3,0,4,3},{0,2,3,3,3,2,1,0,3,1,0,4,0,4,2,2,2,4,1,1,4,1,1,4,4,0,3,2,3,3,4,0,2,1,4,0,1,1,2,3},{3,1,3,2,4,1,4,1,3,0,3,1,2,0,1,1,1,2,4,3,0,2,0,4,4,0,0,1,0,2,0,0,4,4,0,0,3,4,1,4},{0,4,1,2,2,0,3,1,3,4,1,2,1,0,4,4,0,4,0,0,0,3,0,4,4,3,0,3,3,0,2,0,1,3,4,2,3,3,4,3}};
    //auto bb = tensor<value_type>{{1,1,1,3,3,2,3,2,2,2,0,2,4,0,4,3,2,3,2,3},{2,4,3,4,0,2,3,4,1,0,0,1,2,0,4,0,2,3,2,4},{4,4,2,2,4,1,1,1,4,4,1,2,3,4,0,4,3,4,3,2},{2,1,3,2,1,4,1,0,1,0,1,2,4,1,4,3,1,4,4,0},{1,1,1,0,2,0,3,3,1,4,3,1,1,3,1,1,4,2,1,2},{3,1,4,1,0,3,2,1,2,0,2,2,1,3,4,4,2,1,3,4},{0,3,2,3,1,2,0,4,3,3,4,3,0,1,0,0,3,0,0,0},{0,0,3,1,1,2,2,4,1,3,0,1,2,0,3,1,3,1,3,2},{4,0,4,0,2,3,2,0,4,3,0,1,4,2,0,1,1,1,2,4},{1,3,2,4,0,3,4,3,3,0,4,4,0,1,2,0,4,1,1,2},{3,3,1,2,2,1,4,0,1,4,0,2,3,4,1,3,0,0,1,3},{4,4,1,4,3,1,1,1,1,2,0,2,2,1,3,4,4,0,0,3},{4,0,2,0,2,2,4,1,4,0,0,0,4,4,3,1,4,4,1,4},{4,0,0,4,0,4,2,4,4,0,4,1,1,4,3,1,0,3,2,0},{0,0,3,0,0,1,1,3,2,1,0,4,3,2,3,1,2,1,4,4},{1,0,1,3,2,1,1,2,0,3,1,3,2,0,2,2,1,3,3,1},{0,1,1,1,0,2,1,2,1,4,0,0,3,0,2,3,4,0,3,0},{4,1,4,3,2,3,2,1,3,3,3,0,4,3,4,1,0,1,4,2},{3,4,1,1,1,2,1,3,3,1,3,3,2,2,1,2,1,0,4,3},{0,4,3,1,0,3,2,4,3,3,0,1,2,1,3,2,3,2,3,3},{0,4,1,3,4,4,3,1,1,3,1,4,0,1,1,4,0,2,0,4},{4,3,2,3,0,3,4,4,2,2,4,1,3,0,2,4,0,1,3,3},{0,2,4,1,2,0,2,1,0,4,1,0,3,4,3,2,0,1,3,1},{3,3,2,4,1,0,0,1,4,4,4,3,3,2,0,4,3,4,3,4},{0,3,3,3,2,2,4,2,4,4,0,0,1,0,4,0,0,1,3,3},{0,2,2,0,2,4,1,1,3,4,4,2,2,3,2,2,0,3,1,0},{0,0,2,3,3,0,3,0,1,4,2,3,3,0,2,4,3,1,2,1},{4,1,0,3,1,4,2,3,3,4,0,1,4,2,4,4,1,4,2,0},{3,0,1,4,1,3,1,3,0,3,3,3,3,4,3,3,2,4,0,2},{1,1,2,1,0,1,0,2,2,0,3,1,3,1,3,2,1,3,4,3},{4,4,1,0,2,3,1,3,2,1,1,0,2,0,2,3,0,1,2,2},{4,1,3,3,4,4,2,4,0,3,2,4,4,0,2,2,3,2,1,0},{1,1,4,1,2,4,0,4,0,4,4,1,0,0,3,4,4,1,2,2},{4,0,2,3,2,0,2,2,4,4,3,0,4,0,4,2,3,0,0,0},{2,1,2,4,2,4,1,4,3,0,0,0,2,1,4,1,1,0,4,1},{1,4,4,3,3,0,2,2,1,4,3,3,4,3,2,0,2,3,1,4},{1,2,1,4,1,1,3,4,2,4,4,3,3,0,0,1,3,3,1,4},{1,2,2,3,3,3,1,0,1,1,3,1,4,1,0,1,1,2,0,0},{2,4,3,1,3,1,1,1,2,3,0,1,4,4,2,4,1,2,0,0},{0,3,1,2,3,3,4,3,1,1,1,1,0,1,0,1,0,2,4,3}};
    //auto rr = tensor<value_type>{{147,148,178,179,134,188,161,185,168,197,140,142,207,132,180,184,158,158,192,173},{176,174,204,203,165,208,172,194,185,217,166,144,229,159,208,179,170,173,178,176},{125,157,158,178,139,165,159,184,148,205,140,129,180,133,163,148,139,124,144,140},{118,121,140,144,102,140,126,133,117,161,89,99,165,108,150,145,121,119,137,125},{181,150,154,176,141,181,164,179,186,200,150,144,208,133,157,188,159,168,156,171},{137,168,157,212,124,174,170,204,170,177,163,155,167,107,170,147,151,166,158,191},{179,135,165,175,144,203,141,180,207,218,146,132,225,153,192,190,156,167,156,157},{131,140,147,134,116,142,106,149,149,157,101,114,172,111,142,144,137,122,147,145},{121,142,164,151,124,164,129,162,148,179,132,122,171,118,176,174,135,130,161,148},{164,162,170,186,128,182,166,197,160,195,136,135,218,138,193,184,143,161,176,182},{159,197,191,201,149,197,164,197,154,214,160,162,194,138,203,210,157,174,176,187},{174,186,171,199,146,179,160,209,170,192,190,162,205,150,162,173,152,187,166,189},{143,149,186,142,111,176,128,179,168,166,149,130,189,120,176,139,139,135,161,153},{144,158,167,192,149,181,136,169,162,212,168,161,205,136,177,184,170,165,151,158},{160,140,168,175,134,190,150,172,149,169,124,124,200,144,200,175,138,149,167,149},{133,156,185,173,145,184,160,189,157,213,135,146,198,109,194,174,142,167,172,189},{131,152,165,178,121,165,143,185,142,205,131,153,177,123,186,192,161,146,188,192},{162,143,196,162,135,187,158,197,168,190,148,132,215,114,196,192,147,154,214,187},{134,156,174,154,113,169,145,156,150,190,131,116,176,105,171,156,152,125,152,159},{146,128,146,164,124,149,160,151,161,168,122,115,185,128,166,147,126,162,148,140},{126,150,155,151,121,151,131,176,132,164,100,117,162,115,180,144,130,111,148,166},{137,119,172,156,129,171,162,175,147,201,131,115,185,128,184,169,144,146,183,148},{184,148,164,162,127,163,143,152,157,173,132,101,212,142,167,170,96,135,159,148},{141,144,169,182,146,187,159,179,156,201,143,154,201,121,178,187,152,144,170,159},{188,151,174,231,144,200,161,197,180,211,172,145,231,130,200,192,168,195,177,177},{123,147,155,159,115,150,136,173,131,182,125,105,154,105,160,149,125,127,163,152},{177,141,167,171,136,178,130,186,176,193,154,123,192,144,173,183,159,155,164,154},{169,157,161,193,138,182,142,169,173,185,134,124,182,128,186,187,129,151,185,173},{137,149,153,152,124,140,141,158,169,194,140,108,176,103,135,147,144,128,156,167},{159,160,175,203,129,172,155,184,179,196,147,134,203,122,179,144,134,154,161,167}};

    auto aa = tensor<value_type>{{1,1,1,3,3,2,3,2,2,2},{0,2,4,0,4,3,2,3,2,3},{2,4,3,4,0,2,3,4,1,0},{0,1,2,0,4,0,2,3,2,4},{4,4,2,2,4,1,1,1,4,4},{1,2,3,4,0,4,3,4,3,2},{2,1,3,2,1,4,1,0,1,0},{1,2,4,1,4,3,1,4,4,0},{1,1,1,0,2,0,3,3,1,4},{3,1,1,3,1,1,4,2,1,2},{3,1,4,1,0,3,2,1,2,0},{2,2,1,3,4,4,2,1,3,4},{0,3,2,3,1,2,0,4,3,3},{4,3,0,1,0,0,3,0,0,0},{0,0,3,1,1,2,2,4,1,3},{0,1,2,0,3,1,3,1,3,2},{4,0,4,0,2,3,2,0,4,3},{0,1,4,2,0,1,1,1,2,4},{1,3,2,4,0,3,4,3,3,0},{4,4,0,1,2,0,4,1,1,2},{3,3,1,2,2,1,4,0,1,4},{0,2,3,4,1,3,0,0,1,3},{4,4,1,4,3,1,1,1,1,2},{0,2,2,1,3,4,4,0,0,3},{4,0,2,0,2,2,4,1,4,0},{0,0,4,4,3,1,4,4,1,4},{4,0,0,4,0,4,2,4,4,0},{4,1,1,4,3,1,0,3,2,0},{0,0,3,0,0,1,1,3,2,1},{0,4,3,2,3,1,2,1,4,4},{1,0,1,3,2,1,1,2,0,3},{1,3,2,0,2,2,1,3,3,1},{0,1,1,1,0,2,1,2,1,4},{0,0,3,0,2,3,4,0,3,0},{4,1,4,3,2,3,2,1,3,3},{3,0,4,3,4,1,0,1,4,2},{3,4,1,1,1,2,1,3,3,1},{3,3,2,2,1,2,1,0,4,3},{0,4,3,1,0,3,2,4,3,3},{0,1,2,1,3,2,3,2,3,3},{0,4,1,3,4,4,3,1,1,3},{1,4,0,1,1,4,0,2,0,4},{4,3,2,3,0,3,4,4,2,2},{4,1,3,0,2,4,0,1,3,3},{0,2,4,1,2,0,2,1,0,4},{1,0,3,4,3,2,0,1,3,1},{3,3,2,4,1,0,0,1,4,4},{4,3,3,2,0,4,3,4,3,4},{0,3,3,3,2,2,4,2,4,4},{0,0,1,0,4,0,0,1,3,3},{0,2,2,0,2,4,1,1,3,4},{4,2,2,3,2,2,0,3,1,0},{0,0,2,3,3,0,3,0,1,4},{2,3,3,0,2,4,3,1,2,1},{4,1,0,3,1,4,2,3,3,4},{0,1,4,2,4,4,1,4,2,0},{3,0,1,4,1,3,1,3,0,3},{3,3,3,4,3,3,2,4,0,2},{1,1,2,1,0,1,0,2,2,0},{3,1,3,1,3,2,1,3,4,3},{4,4,1,0,2,3,1,3,2,1},{1,0,2,0,2,3,0,1,2,2},{4,1,3,3,4,4,2,4,0,3},{2,4,4,0,2,2,3,2,1,0},{1,1,4,1,2,4,0,4,0,4},{4,1,0,0,3,4,4,1,2,2},{4,0,2,3,2,0,2,2,4,4},{3,0,4,0,4,2,3,0,0,0},{2,1,2,4,2,4,1,4,3,0},{0,0,2,1,4,1,1,0,4,1},{1,4,4,3,3,0,2,2,1,4},{3,3,4,3,2,0,2,3,1,4},{1,2,1,4,1,1,3,4,2,4},{4,3,3,0,0,1,3,3,1,4},{1,2,2,3,3,3,1,0,1,1},{3,1,4,1,0,1,1,2,0,0},{2,4,3,1,3,1,1,1,2,3},{0,1,4,4,2,4,1,2,0,0},{0,3,1,2,3,3,4,3,1,1},{1,1,0,1,0,1,0,2,4,3},{0,2,0,0,1,1,3,2,0,0},{3,4,4,3,4,4,2,1,2,4},{4,1,3,0,2,0,1,0,1,2},{4,1,3,1,4,4,0,1,1,0},{1,0,3,1,3,4,3,4,1,0},{0,1,4,4,1,4,4,4,0,3},{2,4,4,0,2,1,4,3,0,2},{0,3,2,0,0,2,0,1,0,3},{1,1,2,0,0,4,0,2,3,0},{2,1,1,4,0,0,1,4,3,1},{1,4,4,2,1,0,1,4,1,3},{3,3,0,2,1,3,1,4,4,2},{4,0,2,4,4,3,1,1,2,2},{1,2,0,2,4,1,4,1,2,0},{3,3,1,2,3,3,2,0,2,0},{0,4,2,1,4,0,4,2,4,2},{3,3,3,4,2,0,1,1,4,3},{0,4,1,4,1,4,2,3,0,1},{1,3,0,3,1,1,1,4,4,3},{0,0,4,3,2,2,4,4,2,3},{1,4,2,0,3,3,3,2,1,2},{1,1,0,0,0,3,3,4,1,0},{1,1,3,2,3,1,1,2,2,3},{3,1,3,0,2,0,3,0,3,4},{0,0,4,2,2,2,3,2,1,0},{0,3,4,4,2,1,2,4,1,2},{2,2,1,4,0,2,0,1,2,2},{1,3,4,3,3,0,3,0,4,3},{0,2,3,3,3,2,1,0,3,1},{0,4,0,4,2,2,2,4,1,1},{4,1,1,4,4,0,3,2,3,3},{4,0,2,1,4,0,1,1,2,3},{3,1,3,2,4,1,4,1,3,0},{3,1,2,0,1,1,1,2,4,3},{0,2,0,4,4,0,0,1,0,2},{0,0,4,4,0,0,3,4,1,4},{0,4,1,2,2,0,3,1,3,4},{1,2,1,0,4,4,0,4,0,0},{0,3,0,4,4,3,0,3,3,0},{2,0,1,3,4,2,3,3,4,3},{1,0,2,1,4,1,1,1,2,2},{3,1,3,0,1,2,0,1,0,3},{2,0,4,4,2,4,0,2,3,1},{3,0,1,2,1,1,4,1,3,3},{3,2,0,0,3,1,0,3,0,4},{3,0,3,0,2,3,3,3,3,1},{1,3,1,2,4,2,2,3,4,0},{0,0,4,2,1,1,4,2,2,2},{3,4,0,4,4,0,3,2,4,1},{4,1,0,2,1,3,2,3,0,3},{0,0,0,1,4,1,4,1,1,4},{2,1,4,2,1,0,1,4,4,2},{0,0,1,0,4,4,2,3,2,2},{3,1,1,3,2,0,4,1,0,4},{4,2,1,1,4,0,2,1,1,1},{1,3,3,3,3,1,4,2,1,0},{0,2,4,3,0,3,4,0,4,3},{1,1,4,2,1,1,4,1,4,4},{2,0,3,4,4,1,0,1,4,2},{0,1,3,1,3,4,1,3,2,4},{1,2,1,0,4,4,0,1,4,4},{3,2,4,4,1,0,3,4,0,3},{0,4,0,1,4,4,1,3,1,3},{3,1,2,3,3,3,1,3,0,4},{2,4,4,3,3,4,3,1,4,4},{0,2,3,0,3,1,0,0,4,4},{0,3,0,4,0,3,4,4,3,1},{2,2,3,2,0,4,3,4,0,0},{0,1,4,4,3,2,3,2,2,4},{4,1,3,1,3,0,2,3,0,3},{2,0,3,2,3,3,2,2,1,4},{1,1,4,2,4,4,4,4,2,4},{4,4,3,2,3,1,3,0,0,3},{4,4,2,1,3,1,1,3,2,2},{3,2,4,1,4,3,0,2,0,2},{0,1,4,0,3,0,1,2,4,3},{3,0,4,0,3,3,0,0,3,1},{2,0,0,3,0,3,3,1,3,4},{4,0,0,0,4,3,1,0,0,1},{2,1,2,2,4,4,1,4,3,0},{2,4,3,2,3,1,3,2,2,4},{0,4,1,4,0,2,1,1,2,0},{3,4,3,1,1,1,1,4,4,3},{2,0,1,2,0,3,4,4,0,2},{3,3,2,2,2,4,4,0,4,3},{4,2,0,0,0,4,1,3,0,0},{0,3,2,4,2,3,0,0,4,4},{2,3,2,1,0,2,0,0,4,1},{2,1,0,1,1,4,0,2,4,2},{2,0,4,3,4,2,4,3,3,3},{3,0,0,0,0,1,3,3,3,4},{0,3,0,4,3,2,4,4,1,4},{2,0,2,1,1,4,1,3,3,0},{0,1,4,1,3,1,4,3,4,1},{1,2,3,0,4,2,4,1,4,3},{0,0,3,2,0,1,1,3,2,2},{0,0,0,1,1,0,2,4,3,1},{3,4,1,3,3,3,4,3,0,3},{1,0,3,0,2,2,3,1,1,2},{3,1,1,3,4,0,2,3,3,4},{2,0,3,0,2,0,1,2,4,3},{1,0,1,0,4,4,4,2,3,3},{0,1,1,1,0,1,4,0,3,2},{0,0,2,1,4,3,1,3,3,0},{0,2,0,3,4,4,4,1,0,1},{2,0,2,0,0,2,2,4,4,1},{1,1,1,4,4,1,1,4,1,4},{3,1,1,1,0,3,2,2,3,4},{2,2,0,1,1,2,1,4,0,0},{2,2,3,0,3,2,4,3,0,4},{3,1,0,3,2,0,4,2,4,1},{2,2,4,1,2,4,2,1,2,4},{3,4,0,0,3,4,4,3,3,1},{4,1,2,1,3,1,2,0,1,1},{2,4,1,1,2,0,3,3,2,2},{2,4,0,4,2,2,4,2,0,1},{0,0,4,0,2,0,2,4,2,0},{3,3,4,3,0,1,3,4,2,0},{3,1,2,4,1,1,4,3,2,2},{0,3,2,0,2,4,4,0,0,0},{3,0,3,0,4,0,2,1,2,0},{0,4,1,1,2,3,2,3,0,3},{1,3,1,2,1,0,0,4,2,4},{0,3,3,3,1,1,1,4,3,0},{3,1,0,2,2,2,4,1,3,3},{1,3,1,2,1,4,3,1,1,3},{3,2,1,0,4,1,2,4,2,4},{1,3,2,3,3,2,1,4,2,1},{1,3,3,2,0,4,1,1,3,2},{4,2,2,4,2,4,4,0,3,1},{0,2,2,0,2,2,2,1,1,2},{0,1,1,0,4,4,0,2,3,0},{2,2,1,3,0,4,0,3,1,3},{1,0,3,3,2,4,3,0,0,1},{3,2,1,4,3,1,2,3,1,1},{4,1,1,3,0,1,3,1,3,0},{4,3,0,4,2,3,0,1,4,1},{2,1,0,0,4,2,0,0,2,3},{2,0,4,3,1,3,2,1,1,4},{0,1,2,0,2,0,4,1,2,1},{1,4,4,1,3,4,2,3,4,4},{2,4,1,4,4,2,1,3,0,1},{1,0,4,4,2,1,4,2,0,4},{3,2,3,0,2,3,4,0,4,0},{0,1,2,4,3,4,2,4,0,4},{1,3,3,4,3,1,4,0,1,3},{0,0,3,3,0,1,2,2,1,0},{0,4,4,3,4,4,2,4,4,2},{0,2,4,3,2,1,1,2,1,3},{1,0,1,3,4,2,2,4,2,1},{1,1,0,0,2,4,0,4,2,4},{2,1,4,2,4,1,0,0,0,2},{0,4,2,4,0,3,0,1,3,0},{2,0,3,3,1,2,4,4,2,3},{1,2,3,1,0,0,0,2,2,0},{4,2,0,0,1,3,2,0,0,1},{0,1,0,3,3,3,3,0,3,4},{4,1,1,1,1,1,1,0,3,0},{1,2,0,2,1,3,3,3,4,1},{1,0,2,4,3,4,4,1,1,3},{0,3,4,2,0,4,2,2,2,4},{1,1,4,3,0,2,3,0,4,3},{0,1,2,1,1,1,4,2,1,0},{1,3,0,1,3,4,2,1,1,3},{2,4,0,1,0,1,1,1,4,2},{3,4,4,1,1,0,4,1,0,4},{0,4,3,4,1,1,1,1,2,4},{1,2,3,0,1,4,0,0,0,1},{2,2,1,0,4,2,1,3,0,4},{0,0,4,4,3,2,2,3,0,3},{4,2,1,2,2,4,4,3,0,1},{0,2,1,3,4,3,0,2,1,2},{1,4,3,2,4,3,2,1,1,3},{2,3,2,2,4,4,3,0,2,3},{0,2,0,2,2,0,1,2,2,2},{0,2,2,4,3,4,0,1,3,3},{0,2,0,2,1,2,3,1,1,2},{3,1,2,1,4,0,1,4,1,4},{0,0,4,2,1,3,0,2,1,4},{3,0,3,4,3,4,2,0,1,3}};
    auto bb = tensor<value_type>{{1,1,1,3,3,2,3,2,2,2,0,2,4,0,4,3,2,3,2,3,2,4,3,4,0,2,3,4,1,0},{0,1,2,0,4,0,2,3,2,4,4,4,2,2,4,1,1,1,4,4,1,2,3,4,0,4,3,4,3,2},{2,1,3,2,1,4,1,0,1,0,1,2,4,1,4,3,1,4,4,0,1,1,1,0,2,0,3,3,1,4},{3,1,1,3,1,1,4,2,1,2,3,1,4,1,0,3,2,1,2,0,2,2,1,3,4,4,2,1,3,4},{0,3,2,3,1,2,0,4,3,3,4,3,0,1,0,0,3,0,0,0,0,0,3,1,1,2,2,4,1,3},{0,1,2,0,3,1,3,1,3,2,4,0,4,0,2,3,2,0,4,3,0,1,4,2,0,1,1,1,2,4},{1,3,2,4,0,3,4,3,3,0,4,4,0,1,2,0,4,1,1,2,3,3,1,2,2,1,4,0,1,4},{0,2,3,4,1,3,0,0,1,3,4,4,1,4,3,1,1,1,1,2,0,2,2,1,3,4,4,0,0,3},{4,0,2,0,2,2,4,1,4,0,0,0,4,4,3,1,4,4,1,4,4,0,0,4,0,4,2,4,4,0},{4,1,1,4,3,1,0,3,2,0,0,0,3,0,0,1,1,3,2,1,0,4,3,2,3,1,2,1,4,4}};
    auto rr = tensor<value_type>{{31,32,37,51,32,38,44,42,46,31,54,40,46,28,34,28,47,30,35,33,27,36,40,44,35,47,51,38,40,61},{30,36,50,52,41,49,33,42,52,35,60,48,52,34,49,31,46,40,49,38,20,35,51,39,34,44,59,50,42,73},{27,32,47,52,41,45,55,36,42,44,67,58,60,38,61,42,43,37,56,46,32,46,44,54,40,61,67,43,40,68},{30,31,37,52,29,39,20,39,41,25,42,40,33,30,31,16,38,34,27,26,17,32,37,31,33,38,49,38,35,55},{47,34,47,58,60,45,53,60,63,45,52,50,73,37,59,40,57,56,54,55,37,52,59,71,33,66,67,77,63,63},{42,33,52,57,46,51,62,38,54,38,67,48,74,42,59,47,53,46,60,51,36,48,49,58,46,63,68,43,53,80},{19,18,29,25,30,29,39,23,33,23,37,23,50,13,37,35,31,26,42,28,19,24,34,33,17,27,34,35,28,45},{29,34,54,46,41,53,44,36,55,42,63,51,60,46,60,37,52,43,49,46,29,28,47,47,30,58,62,60,41,64},{26,28,31,51,27,34,22,35,35,21,37,38,29,24,30,15,33,30,25,27,17,38,34,31,31,33,47,27,30,49},{30,30,33,55,31,38,47,39,40,27,46,42,46,23,39,30,43,33,35,34,30,46,37,46,35,43,54,33,35,55},{24,20,35,32,32,39,44,22,37,21,35,31,55,21,51,37,36,38,45,36,27,31,33,38,19,31,45,40,29,45},{43,36,45,57,53,42,55,57,63,41,62,41,69,30,44,40,58,42,52,48,31,47,60,62,37,58,59,58,60,75},{37,24,42,44,43,36,38,33,42,37,51,38,59,40,45,34,37,39,47,41,23,36,42,48,38,59,53,42,49,61},{10,17,17,27,25,18,34,28,24,22,27,33,26,10,34,18,25,19,25,30,22,33,25,37,10,27,35,29,19,22},{27,26,37,48,26,40,25,24,33,21,42,34,41,27,35,26,31,32,35,25,15,33,34,26,36,33,47,23,29,59},{27,26,33,37,25,35,31,34,42,18,38,33,33,26,32,16,41,31,27,29,24,24,29,32,22,32,42,37,33,46},{42,26,41,46,44,48,49,38,55,20,32,30,69,24,54,40,51,55,48,44,34,41,45,50,23,36,53,58,46,54},{39,17,31,38,30,33,29,25,31,13,26,22,51,21,33,29,28,41,40,23,20,32,28,31,33,30,40,31,40,53},{33,31,46,47,39,44,64,37,50,37,66,50,61,40,56,39,52,37,52,50,39,41,40,57,37,61,63,41,46,67},{23,31,32,49,40,32,44,49,44,35,47,51,39,23,46,23,43,32,36,44,30,48,43,54,23,46,56,47,37,45},{35,32,34,55,43,35,47,53,48,30,47,44,50,19,41,29,46,38,43,40,30,52,47,54,32,43,56,46,47,60},{34,18,30,33,36,26,36,31,33,25,39,21,57,16,29,36,29,31,47,24,17,30,37,37,32,36,36,35,44,59},{31,30,36,51,48,33,48,51,45,46,53,47,59,25,46,38,43,36,47,41,28,47,51,60,32,58,56,58,47,56},{23,33,36,44,35,34,38,45,46,27,57,38,41,14,32,26,42,24,44,31,18,36,47,36,28,29,46,34,38,69},{28,28,37,42,31,45,52,34,51,21,38,38,49,28,51,29,53,41,33,44,38,34,34,47,17,38,52,48,32,41},{44,42,50,77,32,57,43,46,48,31,64,53,56,35,41,36,51,44,45,27,28,49,45,41,59,51,69,37,45,89},{34,26,40,48,40,42,64,30,50,36,52,36,68,38,52,44,52,38,42,52,38,42,42,60,32,62,56,40,42,52},{26,26,34,47,34,36,42,34,38,40,45,39,53,30,41,36,40,32,33,33,26,34,39,48,30,55,49,48,33,44},{19,14,27,26,16,30,18,9,22,11,23,22,30,24,31,18,21,27,24,20,14,17,17,17,20,23,32,19,18,33},{46,31,46,49,48,42,46,51,56,34,53,45,61,38,49,31,51,49,51,45,33,40,46,56,36,57,60,60,60,68},{25,22,24,44,23,26,23,29,26,22,34,25,35,15,18,23,26,22,25,15,12,31,31,27,33,31,35,22,28,48},{22,24,39,33,37,35,33,30,42,33,46,40,44,35,48,25,37,33,39,42,22,27,38,41,20,46,48,45,35,45},{26,16,24,33,28,22,21,23,27,16,28,19,36,17,23,20,22,25,30,24,11,30,30,27,26,27,32,18,32,44},{22,24,33,28,20,37,40,26,42,12,39,28,36,21,35,21,43,28,31,29,27,18,25,28,16,23,38,32,27,46},{47,32,47,59,50,52,59,46,57,33,49,41,80,29,58,50,55,56,58,46,37,51,53,60,38,52,64,61,54,71},{44,27,41,50,38,47,44,39,49,29,37,33,67,31,45,40,49,51,41,32,32,33,40,48,33,49,52,62,46,55},{25,24,39,37,46,34,44,35,45,40,48,44,54,36,56,32,40,37,45,52,28,38,44,55,21,56,54,51,41,44},{42,21,36,38,49,34,51,41,50,29,36,31,67,28,50,37,45,48,49,48,34,41,43,59,24,50,50,57,54,49},{35,28,49,45,48,43,44,35,49,36,58,47,61,42,59,35,42,44,57,52,27,42,47,51,35,56,62,44,50,68},{34,31,40,48,33,41,38,40,49,25,49,38,45,31,37,24,47,36,36,35,26,33,39,40,32,42,51,40,42,61},{30,38,44,51,48,36,49,57,56,45,74,49,54,27,40,33,51,28,53,43,24,42,58,54,36,54,57,49,53,79},{20,21,30,33,47,19,27,36,36,37,47,30,46,18,34,28,25,22,46,39,8,38,51,42,23,40,39,33,41,53},{37,37,51,65,52,52,65,46,56,44,67,59,72,39,68,47,55,48,60,59,39,61,56,67,42,65,76,50,50,74},{34,23,39,40,49,40,41,35,50,29,35,28,68,23,52,41,42,47,50,45,24,39,51,49,20,39,48,56,44,51},{29,25,32,45,28,34,20,34,30,19,35,35,37,17,31,22,27,34,37,18,14,34,33,26,33,26,44,33,33,57},{35,22,34,38,29,36,40,30,39,26,39,25,56,26,32,35,40,35,36,24,25,23,32,38,31,43,40,44,39,52},{51,21,36,48,49,35,49,43,46,32,34,33,71,33,47,39,43,53,47,43,35,46,41,62,36,60,54,58,59,52},{47,37,57,68,63,57,65,49,64,44,65,56,86,42,75,53,58,60,70,66,39,66,65,72,44,67,80,59,61,82},{51,37,52,61,48,51,59,53,62,34,64,51,68,42,54,37,60,52,56,50,40,49,49,61,46,62,70,54,63,82},{26,18,23,30,21,24,13,28,32,15,21,18,26,21,16,10,29,26,14,17,13,15,24,23,18,27,27,34,29,31},{33,23,37,34,43,32,34,36,48,25,42,26,53,25,38,28,39,36,45,40,19,31,45,41,23,37,42,42,47,57},{21,25,36,43,38,36,40,31,36,41,47,41,55,27,48,39,35,32,42,36,22,36,43,46,27,50,50,47,31,47},{36,27,27,50,22,32,30,40,35,15,35,28,36,15,17,20,37,30,26,14,21,33,29,30,37,29,40,29,37,57},{23,30,43,38,43,41,47,39,51,33,55,44,54,26,55,34,46,36,52,47,27,36,48,47,20,40,54,50,39,59},{43,31,42,60,53,42,58,46,57,38,53,38,73,32,50,44,53,45,50,54,33,56,57,65,38,60,61,47,55,66},{23,34,50,46,34,49,38,32,47,40,66,46,54,37,48,37,45,32,47,34,20,25,46,36,34,49,55,46,34,70},{30,26,32,54,36,34,39,33,35,32,45,31,56,19,33,39,34,30,40,29,18,45,45,42,39,43,47,29,36,60},{31,41,52,68,50,51,51,50,51,53,75,61,68,34,58,48,49,40,60,44,26,54,62,57,47,63,72,54,46,83},{16,10,22,18,19,22,22,10,20,16,21,19,32,21,30,20,19,23,24,22,15,15,17,23,14,27,27,24,19,24},{41,31,47,55,46,50,44,41,55,34,46,42,66,38,55,38,51,52,46,46,31,42,49,53,33,54,61,59,48,60},{19,28,41,40,50,36,42,39,48,45,53,48,54,32,59,34,41,35,48,54,24,41,53,55,18,53,56,55,38,47},{21,16,26,25,27,26,22,21,32,17,26,16,39,16,27,23,27,26,29,24,12,19,31,25,15,23,28,31,27,37},{33,44,53,75,51,56,49,52,56,50,72,57,71,30,56,50,53,43,58,43,24,57,68,55,47,57,72,55,45,86},{17,31,44,40,38,43,40,36,43,36,56,54,46,29,59,31,40,35,50,42,25,35,43,42,22,41,58,50,31,56},{28,29,44,52,42,43,25,31,39,34,51,37,58,25,44,39,31,37,52,31,9,40,53,33,38,38,52,37,37,72},{24,34,37,49,42,39,50,47,56,32,52,41,49,21,45,30,53,32,39,48,29,44,52,52,20,40,53,46,38,54},{51,29,39,63,41,47,50,44,51,26,35,37,66,33,46,37,52,55,38,40,38,50,41,57,40,54,60,53,51,56},{14,30,33,41,23,41,31,33,37,22,40,38,36,11,38,27,38,28,33,21,19,27,36,26,18,19,43,42,18,48},{31,30,46,48,40,45,54,33,49,42,62,42,66,39,51,44,49,36,48,44,30,35,46,52,36,61,57,46,42,64},{28,20,28,27,21,30,29,29,39,16,29,21,35,24,24,17,39,29,21,22,23,12,23,29,17,31,31,41,32,36},{40,35,46,61,45,45,39,51,46,39,57,54,58,32,49,35,43,46,53,35,26,48,49,50,45,54,65,55,51,75},{42,35,47,68,47,50,43,48,46,39,53,55,65,33,56,41,44,52,54,39,29,56,51,54,47,56,70,55,49,73},{42,33,42,64,40,42,47,44,46,35,57,47,56,37,42,33,46,40,43,40,30,51,44,53,49,60,63,36,50,71},{33,30,42,58,47,45,40,40,44,31,43,50,57,28,60,35,39,49,51,47,27,57,49,51,33,44,65,46,41,59},{23,24,31,33,33,28,38,36,38,31,47,30,47,17,31,31,36,24,40,27,19,26,39,38,24,37,38,41,36,53},{15,17,28,32,23,33,26,15,21,20,27,31,40,16,42,30,21,30,35,22,16,28,26,25,20,24,39,30,16,36},{32,28,40,44,46,37,36,45,46,36,46,44,54,28,49,31,40,42,48,40,24,39,47,49,27,47,54,58,46,56},{21,26,38,38,28,37,38,26,33,32,56,34,52,21,36,39,33,24,47,22,16,25,38,30,34,37,43,32,30,66},{20,37,43,49,35,39,44,44,48,40,71,53,40,32,42,26,47,23,42,40,24,36,46,44,33,50,57,37,37,68},{32,11,23,26,30,21,28,21,32,16,19,15,41,27,28,19,28,32,24,33,21,25,24,37,19,38,31,29,37,28},{3,19,20,23,14,18,19,20,21,19,36,31,10,16,22,7,21,7,17,21,11,18,20,19,13,22,29,13,12,29},{46,42,57,66,65,54,61,64,68,51,73,57,85,33,65,53,61,55,73,55,34,58,72,70,43,64,75,75,66,91},{23,19,25,36,29,31,25,29,30,18,19,28,40,12,37,25,28,36,30,24,19,32,31,32,16,23,38,43,25,32},{17,27,37,37,39,38,37,34,43,37,46,35,55,18,46,39,39,31,44,34,18,29,49,40,17,36,44,54,29,49},{17,35,45,49,28,48,38,30,44,33,62,46,44,30,45,32,44,27,40,33,20,30,43,32,31,39,54,34,25,65},{36,39,51,67,38,53,50,40,47,35,72,51,63,31,48,44,47,38,58,35,25,50,51,43,54,49,67,31,44,93},{22,37,47,56,40,48,37,43,44,37,60,62,45,30,59,30,41,39,51,41,24,47,48,43,33,43,67,47,34,67},{16,12,22,20,30,16,14,20,21,19,26,20,32,12,27,19,13,21,35,23,5,24,30,23,16,21,27,23,27,37},{17,12,29,15,29,26,31,12,32,20,30,18,44,24,39,27,27,26,35,35,17,16,28,30,10,30,30,30,26,32},{33,20,32,44,28,34,41,24,32,28,37,34,49,36,39,30,35,35,30,33,29,34,26,44,35,54,48,32,35,42},{32,28,44,52,41,42,31,35,36,37,50,51,53,36,53,33,32,43,50,36,21,43,42,43,40,52,61,45,41,63},{34,27,43,46,51,38,52,39,53,43,54,43,64,42,56,37,48,41,47,58,32,44,50,63,29,66,59,51,50,53},{37,32,39,56,42,43,51,46,51,37,50,36,67,23,41,44,51,40,44,35,29,42,51,53,35,50,53,55,45,63},{19,32,32,41,25,32,42,43,44,31,54,44,29,26,31,17,47,20,25,32,28,27,33,41,23,43,46,39,31,47},{21,27,34,34,40,31,49,42,47,37,51,39,50,22,44,32,45,28,42,42,28,32,44,51,17,45,46,52,38,47},{35,37,47,51,39,45,46,52,57,36,61,57,44,43,50,21,56,41,40,46,36,36,41,53,32,58,64,57,49,62},{50,27,42,53,48,43,54,47,51,35,43,42,72,36,53,41,50,55,50,44,39,46,43,63,38,62,61,64,58,59},{20,29,39,41,40,30,45,37,39,44,69,45,50,28,41,35,36,20,51,39,19,37,47,46,35,53,51,32,40,68},{39,25,39,47,43,34,44,38,46,37,49,40,55,43,44,29,43,39,39,47,30,40,40,56,36,65,55,43,51,53},{41,38,50,67,32,56,46,39,49,28,61,49,57,37,46,36,51,44,46,33,30,44,42,41,51,49,66,36,43,82},{20,34,43,44,43,38,38,45,49,39,62,51,44,28,49,27,43,30,48,44,21,38,51,45,25,44,56,47,39,63},{8,22,29,31,22,28,30,18,30,24,44,34,26,25,35,18,29,15,26,34,16,26,29,28,18,32,39,15,17,38},{34,27,37,48,33,39,31,36,40,27,41,35,49,27,36,29,38,38,37,27,21,33,38,37,34,41,48,43,39,57},{40,26,34,49,36,41,38,41,46,16,27,34,50,22,43,26,44,49,36,35,31,42,36,44,26,33,51,49,42,48},{21,27,36,40,18,41,34,24,33,20,46,36,38,23,35,27,36,27,34,20,21,23,27,24,30,29,44,28,23,56},{34,34,48,58,37,47,41,39,41,40,64,54,56,38,49,37,41,39,51,33,25,41,43,44,48,57,64,43,43,76},{32,15,26,32,36,23,41,28,32,27,33,22,55,21,33,34,30,31,39,32,23,33,33,45,27,44,37,35,41,43},{49,32,45,53,42,47,53,51,55,29,49,46,63,35,50,34,55,53,49,40,40,41,40,56,38,54,62,62,57,67},{32,23,35,32,32,33,41,35,42,27,44,30,51,26,35,30,41,33,40,29,26,22,33,40,26,42,41,47,42,53},{22,31,39,46,37,31,42,40,40,46,68,50,43,36,39,28,39,21,41,39,22,36,43,48,37,61,54,35,40,62},{45,38,42,70,42,48,55,56,56,38,53,50,61,34,45,36,59,47,38,41,39,52,48,62,43,62,66,58,51,64},{32,27,31,51,33,38,30,40,41,25,29,33,46,20,35,27,40,40,28,27,23,37,39,40,26,36,46,50,34,43},{31,36,43,53,32,50,53,45,53,31,53,50,51,31,50,32,57,41,38,38,38,36,39,49,29,47,60,56,37,57},{36,20,34,40,38,37,36,30,43,21,26,29,53,30,46,28,39,46,35,41,28,36,35,45,22,41,47,46,40,39},{20,22,21,36,23,17,20,36,25,31,40,28,27,16,11,17,25,13,21,12,10,22,30,29,29,38,30,30,30,43},{43,29,40,64,26,47,36,30,33,20,44,40,52,31,37,33,36,43,40,22,25,45,31,34,54,43,58,24,39,72},{39,28,36,46,40,32,41,48,47,29,47,42,45,32,38,21,44,38,39,40,30,40,38,51,33,51,53,45,52,57},{3,28,36,33,32,30,20,28,35,42,57,40,32,25,36,24,29,13,34,31,5,21,46,28,18,38,40,35,20,48},{24,28,39,36,38,30,43,39,46,47,64,40,49,38,36,30,44,22,38,39,23,23,43,49,29,63,46,47,43,55},{44,38,46,65,40,50,53,50,60,35,58,45,60,39,43,34,61,44,38,43,36,44,48,56,42,60,63,51,51,69},{25,24,29,38,24,32,24,32,36,21,33,27,35,20,25,20,35,28,24,20,17,22,31,28,23,30,36,38,29,43},{21,17,26,34,33,28,20,24,27,20,23,23,44,10,35,29,21,32,37,24,10,33,37,28,19,21,35,34,26,40},{38,25,43,44,39,45,50,30,46,32,48,30,73,30,47,48,45,43,51,35,28,32,43,46,35,49,50,49,44,64},{39,27,32,52,32,39,49,39,46,18,35,33,50,24,38,28,48,41,32,37,35,44,34,48,31,40,51,37,41,50},{19,25,28,46,38,26,16,37,33,34,36,35,35,19,31,21,26,26,29,30,8,39,46,36,24,37,42,37,30,42},{28,31,44,49,35,50,45,32,50,27,47,42,54,32,54,34,49,42,41,43,30,37,42,43,26,41,57,45,33,55},{27,33,45,43,39,41,48,43,54,43,63,50,49,43,49,28,53,33,39,47,32,29,43,53,27,62,57,55,43,55},{34,28,38,49,22,45,39,29,38,15,42,37,44,27,38,27,41,38,36,25,28,33,27,31,37,33,51,29,33,61},{38,37,44,57,46,42,61,58,59,48,64,58,57,43,52,32,61,41,41,52,43,45,47,70,35,74,67,65,54,58},{24,28,32,53,40,33,39,37,39,34,46,37,50,19,39,34,36,29,39,38,19,49,49,46,31,42,50,32,34,54},{27,32,28,51,23,31,27,44,41,19,43,33,25,17,16,12,41,22,20,21,18,33,35,30,31,29,41,26,33,55},{41,25,44,51,35,49,40,28,42,27,38,41,60,42,54,35,42,52,41,38,32,37,33,45,37,53,59,48,41,53},{20,31,38,42,30,37,29,34,46,29,53,34,37,27,31,22,42,23,31,32,15,25,43,31,25,36,43,33,31,57},{34,32,30,62,32,36,40,47,39,25,42,41,43,16,31,27,41,34,33,27,26,51,40,44,39,38,53,34,38,59},{19,29,30,45,32,33,33,42,39,33,40,43,36,20,38,23,39,29,28,31,23,35,39,42,20,39,47,49,28,39},{24,36,43,51,31,44,47,43,44,37,64,56,44,31,47,30,47,31,43,34,30,36,39,44,35,49,60,45,35,65},{49,27,44,45,41,45,61,40,54,20,49,35,69,31,50,39,53,50,56,44,40,41,38,53,37,46,57,45,57,72},{51,30,44,56,38,50,52,42,53,18,42,39,63,33,49,34,53,55,47,40,39,45,37,50,40,45,61,47,53,68},{44,26,38,48,35,42,44,39,47,29,39,30,63,31,37,37,48,45,37,29,31,30,37,47,35,51,48,56,47,55},{34,31,45,50,42,43,33,38,49,32,54,36,57,30,41,34,42,38,48,36,17,36,51,39,36,43,52,42,45,72},{35,26,39,37,49,33,36,44,56,33,45,28,57,29,39,29,46,38,43,45,21,31,52,49,21,46,44,55,52,55},{38,36,46,72,39,52,45,42,40,37,56,57,61,32,54,42,42,47,51,34,29,57,46,49,52,54,71,43,41,75},{20,33,41,43,47,30,32,47,49,47,67,45,44,30,38,26,40,22,45,43,13,35,57,46,28,52,50,44,44,65},{33,34,41,63,45,42,38,45,45,40,55,42,61,23,41,41,41,37,48,34,18,50,57,47,42,48,57,44,43,72},{54,41,60,64,65,57,70,63,74,46,73,56,89,41,69,52,68,61,74,62,43,57,67,75,44,69,78,75,73,92},{38,19,33,31,37,31,26,35,43,19,27,23,48,26,34,22,36,42,36,31,21,24,34,37,21,35,38,50,46,45},{32,31,43,48,38,38,59,38,49,38,68,48,53,42,47,32,50,30,45,50,35,41,40,56,39,64,60,32,47,66},{17,30,43,46,35,43,45,27,38,36,61,48,52,28,54,39,37,29,51,40,22,40,44,40,32,43,57,31,28,64},{47,37,48,65,39,51,48,48,52,31,60,45,64,32,42,39,52,46,51,32,30,45,47,47,51,51,63,46,53,85},{27,33,38,62,35,45,29,40,38,32,42,48,46,23,45,31,37,40,37,29,20,47,45,39,35,39,58,45,30,56},{36,33,41,59,39,45,38,42,47,29,49,37,58,22,39,37,44,40,45,31,21,44,50,41,39,39,54,42,42,71},{43,48,61,77,49,64,53,55,66,42,78,60,70,40,58,44,63,50,60,47,31,54,64,54,52,58,78,54,54,98},{31,35,40,57,48,41,46,55,48,39,53,53,57,19,52,37,45,42,53,40,28,53,54,55,32,45,62,59,45,65},{28,32,44,52,50,42,41,46,49,46,53,54,57,35,59,35,44,43,48,49,27,46,53,57,28,58,63,62,43,54},{22,31,42,48,43,42,30,39,42,40,51,43,56,21,48,39,36,36,50,32,14,37,54,39,28,39,52,54,34,62},{37,24,39,41,30,42,26,31,42,19,32,33,45,34,40,22,39,45,33,29,24,25,30,33,28,38,47,47,39,49},{27,20,34,30,34,38,34,27,42,21,28,23,55,19,43,34,38,40,39,31,22,23,37,35,14,28,38,52,32,41},{42,23,30,47,37,32,51,37,46,19,37,23,57,22,32,32,45,37,37,39,31,44,38,50,33,42,45,30,48,55},{9,23,21,32,28,23,25,33,34,26,32,24,31,5,24,22,31,16,23,24,11,26,40,30,9,21,29,36,19,32},{25,34,48,48,40,47,46,37,53,44,64,46,58,39,51,38,51,34,44,44,26,31,50,48,30,57,57,52,38,62},{41,38,49,63,51,48,48,56,56,41,61,57,62,35,56,36,52,49,55,47,32,53,55,59,41,58,70,60,55,75},{23,16,28,22,32,20,43,27,31,31,45,30,45,26,35,28,30,22,40,34,24,24,28,43,23,47,37,33,38,43},{41,29,50,53,54,48,47,41,53,41,50,52,69,46,68,39,47,56,55,57,34,49,50,62,34,65,69,62,53,59},{22,30,35,54,28,39,40,29,36,26,51,40,42,23,38,30,37,26,36,33,21,44,39,36,36,37,52,18,27,60},{45,35,48,53,56,47,69,56,68,36,60,46,75,32,60,43,64,51,60,60,43,52,57,70,31,57,66,63,62,72},{5,19,27,28,35,24,32,21,30,33,40,32,39,17,43,29,25,18,36,40,13,33,41,37,11,33,38,28,19,33},{48,22,38,38,49,31,49,44,51,32,46,26,70,30,38,38,45,43,52,41,29,35,45,56,34,55,46,53,63,64},{29,11,28,17,38,24,40,24,37,22,25,21,53,25,44,29,32,37,40,41,27,24,29,45,11,39,35,46,40,30},{29,17,31,28,40,27,40,27,44,27,35,20,54,28,38,30,38,32,36,44,23,28,39,46,17,44,37,39,41,39},{47,44,55,75,41,63,56,52,62,35,65,55,68,39,54,42,64,53,50,41,38,50,52,54,50,57,74,56,51,85},{34,23,30,49,33,35,36,31,41,17,28,30,43,27,38,22,39,39,27,40,27,44,34,43,27,38,48,29,36,41},{36,42,46,69,43,42,48,56,53,45,76,57,50,37,39,30,52,31,45,42,27,52,54,55,51,64,67,38,52,83},{22,21,36,32,31,37,40,20,40,26,41,28,51,29,44,33,38,31,37,38,23,25,35,37,20,39,42,35,29,44},{35,35,49,52,29,54,45,37,52,26,55,50,48,42,51,27,54,44,39,38,35,31,34,41,35,48,62,47,39,64},{39,37,48,53,42,50,48,51,62,29,55,48,54,35,51,28,59,47,45,46,35,39,47,51,30,47,62,58,50,67},{29,17,30,36,21,33,26,16,26,15,29,24,41,26,31,25,26,32,30,21,18,25,22,25,31,32,38,22,28,45},{21,19,26,34,15,28,24,18,28,17,31,28,23,32,25,11,30,22,13,25,20,20,17,26,24,37,36,18,22,31},{30,44,49,69,53,46,55,60,57,52,78,64,60,31,55,40,54,36,57,51,29,60,65,63,43,62,73,52,50,82},{22,25,31,39,23,36,28,28,35,15,35,30,35,16,32,22,34,29,31,23,18,28,31,25,23,21,40,29,25,49},{44,36,42,68,42,46,44,52,53,37,50,47,57,36,42,32,53,46,36,39,32,49,48,56,43,60,63,54,50,63},{37,21,34,42,30,40,29,28,40,16,23,28,47,30,40,24,38,46,29,31,26,30,29,36,25,36,45,44,36,40},{31,37,43,53,37,45,44,46,59,28,57,40,47,29,39,26,56,34,37,42,27,37,49,44,29,41,54,42,42,66},{29,18,24,29,21,26,38,27,35,8,28,23,32,20,27,15,36,28,25,29,28,26,19,33,20,27,35,23,33,38},{20,27,38,35,25,38,31,27,42,29,49,33,39,32,34,24,41,25,29,29,19,16,34,30,23,40,41,38,28,49},{17,36,35,45,31,31,44,47,46,37,69,43,36,19,27,25,46,13,37,31,20,32,46,40,30,41,46,32,35,67},{28,21,37,38,29,41,38,19,40,20,34,32,47,36,48,27,39,39,32,41,28,30,29,38,23,41,48,33,30,40},{36,35,40,65,37,40,33,46,43,40,57,44,50,32,31,31,42,33,36,28,19,43,48,44,48,56,56,40,44,70},{38,23,35,46,44,36,45,35,47,24,36,29,60,26,45,34,42,43,43,46,28,46,44,51,28,43,50,39,46,52},{6,21,27,32,26,24,24,21,25,33,43,36,28,23,34,21,23,14,27,30,11,27,33,30,19,37,38,23,17,35},{28,40,46,65,41,49,35,48,49,34,59,55,47,26,49,30,45,39,47,38,21,51,54,42,38,40,65,43,38,74},{36,30,35,52,31,40,55,42,49,28,45,43,47,35,42,26,54,38,28,42,41,40,32,55,31,55,56,44,41,46},{39,31,46,51,50,46,46,44,54,31,51,39,69,25,53,42,47,48,59,44,26,46,55,50,33,42,58,53,51,73},{23,39,49,50,52,44,57,52,64,48,72,59,54,39,62,32,59,35,50,64,34,46,59,64,23,61,67,57,46,62},{21,25,28,40,30,33,35,36,37,25,33,34,41,14,37,27,37,31,31,28,23,33,36,38,18,30,42,45,27,39},{26,31,39,49,39,37,39,43,44,37,52,53,41,35,49,23,42,35,38,44,28,42,41,50,29,53,59,46,39,51},{22,35,37,52,39,33,52,49,44,44,68,54,45,26,42,31,45,23,44,41,28,46,47,54,35,55,58,39,40,64},{18,24,36,38,14,42,20,16,28,18,36,38,28,32,38,18,30,30,24,20,18,18,20,18,26,30,44,28,16,42},{31,31,48,54,39,51,54,33,43,38,57,57,62,40,66,42,45,46,53,46,36,46,40,53,38,58,69,48,38,62},{39,34,42,64,36,48,56,42,47,32,54,49,59,33,49,38,51,43,43,40,37,51,41,54,44,55,65,41,43,66},{8,27,32,26,28,28,36,33,38,26,54,38,30,14,36,21,35,15,40,32,17,24,37,30,14,24,39,30,25,52},{19,26,31,39,21,39,28,30,36,21,31,36,33,21,37,21,38,32,23,23,23,23,28,29,17,28,42,45,20,35},{19,30,38,43,41,30,30,40,40,39,60,45,40,26,39,25,33,23,45,38,13,38,49,40,30,44,50,34,38,63},{33,22,34,46,39,30,26,33,34,33,39,37,46,34,38,25,29,36,36,35,18,39,38,43,35,52,49,37,42,49},{28,24,42,38,32,39,40,26,37,35,52,44,50,42,49,31,37,35,42,37,27,27,31,42,33,56,53,41,37,53},{37,31,35,53,39,38,53,47,53,27,46,38,52,26,40,29,53,38,36,44,35,46,43,55,30,47,54,43,46,55},{28,28,36,42,43,31,46,42,46,32,55,37,52,21,40,32,41,29,49,42,23,42,48,48,29,42,49,36,45,64},{31,37,44,63,45,45,33,49,52,40,53,52,48,35,48,27,47,41,38,44,23,48,54,50,34,53,63,52,42,60},{27,32,45,49,40,41,41,39,45,45,63,50,53,39,48,34,43,33,45,40,24,35,46,48,36,60,58,48,41,63},{34,20,39,31,45,34,48,31,45,29,45,30,65,28,50,39,39,40,55,45,27,34,42,49,25,45,47,45,48,57},{40,35,45,54,49,47,74,52,62,38,62,46,75,28,57,48,63,45,57,53,44,50,53,68,33,57,64,59,54,70},{18,22,29,30,27,27,24,29,33,21,38,30,31,18,30,18,29,23,32,26,14,24,32,27,19,26,36,30,28,45},{14,22,33,22,29,28,27,26,41,30,45,26,36,27,31,21,36,19,29,32,14,11,36,30,12,36,32,39,28,40},{29,21,34,41,44,29,39,30,37,35,46,29,60,24,40,39,31,31,48,39,17,41,47,46,32,47,45,33,42,57},{23,27,32,40,26,35,42,32,37,22,48,29,47,11,30,34,38,24,41,22,20,30,37,31,29,26,40,29,30,62},{27,33,38,56,36,39,45,43,42,42,57,49,50,30,42,34,44,31,38,35,27,42,44,50,37,56,57,45,37,58},{30,21,29,39,30,34,54,30,39,23,34,33,51,25,44,32,43,36,33,39,37,37,28,50,23,44,47,39,34,37},{36,23,36,38,51,31,59,43,52,43,48,34,70,32,49,42,49,39,47,52,35,39,48,67,24,64,50,60,53,47},{22,20,23,30,33,21,22,36,38,24,28,20,35,14,22,18,32,24,24,27,13,24,38,34,13,29,29,41,33,34},{41,27,38,54,38,43,43,36,42,22,41,30,65,18,40,42,40,44,50,29,24,45,44,41,40,35,51,38,44,70},{20,24,28,34,16,32,28,28,33,13,34,34,22,22,29,11,35,25,21,23,23,22,20,25,20,25,39,27,23,39},{46,39,61,59,62,56,55,54,69,46,71,56,79,46,69,45,60,57,69,61,34,50,65,65,40,66,75,69,65,86},{21,35,41,52,43,35,41,48,43,53,69,54,50,30,43,35,41,25,46,37,20,40,53,51,36,60,57,50,40,65},{41,36,41,69,30,49,42,43,41,24,52,44,54,22,36,36,44,41,44,22,26,49,41,38,52,39,60,33,41,80},{29,29,42,37,39,45,57,39,56,26,47,42,56,29,58,33,55,43,46,50,39,34,40,52,16,41,55,56,40,50},{34,38,46,65,41,44,40,45,47,41,70,45,58,29,36,39,44,31,50,32,17,46,56,43,51,52,59,34,46,87},{39,36,41,58,39,42,51,54,49,33,59,49,55,24,41,34,50,39,49,33,32,46,45,51,42,48,60,49,50,75},{21,17,26,31,13,30,30,14,21,14,32,25,34,20,27,24,25,23,27,15,19,20,16,21,28,27,34,17,20,42},{43,43,65,61,57,59,60,54,69,54,85,63,78,53,68,47,64,51,68,58,36,44,63,65,46,76,78,69,63,92},{34,26,38,47,33,38,31,34,35,28,45,37,51,26,37,32,33,37,44,24,19,34,37,35,39,41,49,39,40,64},{26,34,40,54,28,42,38,37,44,36,58,43,43,34,34,28,46,27,30,30,23,31,40,39,37,51,52,37,33,60},{25,24,35,41,41,30,25,31,42,32,44,28,46,28,34,26,33,28,36,39,11,34,48,38,26,42,42,32,38,52},{24,24,30,40,29,33,23,34,31,26,34,30,44,12,30,30,29,31,36,15,13,27,37,28,26,27,38,45,29,50},{28,15,33,20,38,24,47,26,36,33,46,28,57,30,42,35,33,29,48,39,26,23,32,47,23,51,39,41,44,47},{41,36,47,68,36,54,51,39,49,29,56,48,61,35,50,39,51,46,46,39,33,51,44,48,48,51,67,37,43,75},{18,11,25,20,21,25,22,12,20,18,22,25,34,24,36,21,19,28,28,23,17,17,17,25,16,30,32,30,21,26},{10,19,21,27,33,20,33,30,32,25,32,27,35,7,34,24,28,19,32,34,16,33,38,37,8,24,33,32,23,31},{40,29,33,46,37,31,47,48,52,25,49,28,50,23,25,26,50,31,36,35,28,36,42,48,33,44,44,38,52,63},{22,14,22,24,28,25,38,24,33,19,20,22,42,18,37,25,33,31,26,33,27,25,25,40,9,32,33,41,27,21},{30,27,39,40,37,36,52,35,50,32,54,39,50,38,45,28,49,32,38,49,33,34,38,52,27,55,52,38,43,52},{37,37,41,60,36,44,53,48,52,30,62,39,58,21,34,38,53,33,46,32,28,44,49,46,43,43,55,37,46,80},{40,27,46,46,48,42,46,37,48,30,54,38,68,30,52,41,41,45,62,44,25,44,49,48,38,46,57,42,53,76},{49,23,39,44,37,43,55,35,48,16,37,29,67,28,46,38,48,51,49,38,38,39,33,49,35,42,52,44,52,62},{15,24,29,34,15,32,31,23,29,17,41,36,24,22,31,16,32,20,25,23,21,23,21,24,23,27,40,20,19,43},{22,29,34,39,43,27,37,45,47,36,55,36,44,20,34,26,40,23,42,40,17,36,51,45,23,41,44,40,42,58},{30,15,28,25,41,22,41,32,40,27,31,29,47,30,43,23,35,35,36,47,29,32,32,52,15,48,41,44,44,31},{34,33,41,59,44,44,41,48,43,30,47,54,53,22,55,33,40,47,53,39,28,56,47,49,36,40,65,49,43,66},{43,24,38,45,44,33,42,42,41,32,47,37,61,29,41,35,37,42,52,35,26,41,41,50,40,52,52,46,55,66},{11,14,25,16,30,21,22,19,26,21,31,19,39,8,32,27,19,20,40,24,7,19,34,23,10,17,26,30,23,39},{21,32,36,52,40,34,21,43,41,37,49,42,39,22,35,24,34,28,36,32,10,40,52,37,29,39,49,41,34,57},{34,34,42,61,29,46,34,37,38,30,56,41,52,25,33,36,39,34,43,19,18,38,42,32,49,41,54,33,37,78},{20,38,43,58,43,44,53,45,50,43,67,54,54,25,53,39,49,30,49,47,27,51,56,53,32,49,63,42,35,67},{23,27,34,39,35,28,30,38,39,38,54,33,44,24,27,28,35,21,37,27,13,26,44,37,30,45,40,39,38,58},{31,36,46,51,49,41,43,53,53,43,65,50,58,27,48,36,47,37,56,41,23,42,57,51,34,50,59,57,50,75},{35,37,45,52,51,42,54,57,61,40,64,46,63,25,48,38,56,39,55,47,30,45,59,58,31,50,59,59,54,74},{23,19,24,32,24,21,24,29,29,24,34,28,28,25,22,14,28,21,21,24,17,23,25,32,24,39,34,28,31,36},{40,26,40,41,45,34,46,42,50,36,54,29,66,29,36,39,45,36,50,37,24,32,48,50,35,53,46,49,55,68},{21,22,25,33,26,22,34,32,33,22,42,29,31,18,24,18,32,18,29,28,19,29,30,34,24,33,36,22,32,46},{31,34,40,64,38,44,25,43,42,36,45,47,46,30,41,28,39,40,34,31,18,45,48,41,38,47,58,47,36,58},{34,20,34,41,32,35,25,24,32,19,34,21,54,19,31,34,27,36,43,21,12,31,37,27,35,29,39,29,37,61},{39,32,39,56,42,43,52,46,50,31,51,33,69,16,39,46,49,40,51,32,27,45,52,49,38,41,52,48,47,73}};

    //std::cout<<std::endl<<aa;
    //std::cout<<std::endl<<bb;

    //std::cout<<std::endl<<matmul_2d_goto(aa.copy(f_order{}),bb.copy(f_order{}));
    //std::cout<<std::endl<<matmul_2d_goto(aa,bb);

    //REQUIRE(matmul_2d(tensor<value_type,c_order>{{1,2,4,2},{3,4,2,0},{5,3,1,1}},tensor<value_type,c_order>{{2,1,2},{0,3,1},{1,1,4},{4,3,3}})==tensor<value_type>{{14,17,26},{8,17,18},{15,18,20}});
    //REQUIRE(matmul_2d(tensor<value_type,f_order>{{1,2,4,2},{3,4,2,0},{5,3,1,1}},tensor<value_type,f_order>{{2,1,2},{0,3,1},{1,1,4},{4,3,3}})==tensor<value_type>{{14,17,26},{8,17,18},{15,18,20}});
    // REQUIRE(matmul_2d_tiled(aa,bb)==rr);
    // REQUIRE(matmul_2d_tiled(aa.copy(f_order{}),bb.copy(f_order{}))==rr);

    // REQUIRE(matmul_2d_tiled(tensor<value_type,c_order>{{1,2,4,2},{3,4,2,0},{5,3,1,1}},tensor<value_type,c_order>{{2,1,2},{0,3,1},{1,1,4},{4,3,3}})==tensor<value_type>{{14,17,26},{8,17,18},{15,18,20}});
    //REQUIRE(matmul_2d_tiled(tensor<value_type,f_order>{{1,2,4,2},{3,4,2,0},{5,3,1,1}},tensor<value_type,f_order>{{2,1,2},{0,3,1},{1,1,4},{4,3,3}})==tensor<value_type>{{14,17,26},{8,17,18},{15,18,20}});

    //REQUIRE(matmul_2d_goto(aa.copy(f_order{}),bb.copy(f_order{}))==rr);
    //REQUIRE(matmul_2d_goto(aa,bb)==rr);

    // REQUIRE(matmul_2d_goto1(aa.copy(c_order{}),bb.copy(c_order{}))==rr);
    // REQUIRE(matmul_2d_goto1(aa.copy(f_order{}),bb.copy(f_order{}))==rr);
    // REQUIRE(matmul_2d_goto1(aa.copy(c_order{}),bb.copy(f_order{}))==rr);
    // REQUIRE(matmul_2d_goto1(aa.copy(f_order{}),bb.copy(c_order{}))==rr);

    // REQUIRE(matmul_2d_goto2(aa.copy(c_order{}),bb.copy(c_order{}))==rr);
    // REQUIRE(matmul_2d_goto2(aa.copy(f_order{}),bb.copy(f_order{}))==rr);
    // REQUIRE(matmul_2d_goto2(aa.copy(c_order{}),bb.copy(f_order{}))==rr);
    // REQUIRE(matmul_2d_goto2(aa.copy(f_order{}),bb.copy(c_order{}))==rr);

    //REQUIRE(matmul(tensor<double,c_order>{{1,2,4,2},{3,4,2,0},{5,3,1,1}},tensor<double,c_order>{{2,1,2},{0,3,1},{1,1,4},{4,3,3}})==tensor<double>{{14,17,26},{8,17,18},{15,18,20}});
    //REQUIRE(matmul(tensor<double,f_order>{{1,2,4,2},{3,4,2,0},{5,3,1,1}},tensor<double,f_order>{{2,1,2},{0,3,1},{1,1,4},{4,3,3}})==tensor<double>{{14,17,26},{8,17,18},{15,18,20}});

    REQUIRE(matmul(aa.copy(c_order{}),bb.copy(c_order{}))==rr);
    REQUIRE(matmul(aa.copy(f_order{}),bb.copy(f_order{}))==rr);
    REQUIRE(matmul(aa.copy(c_order{}),bb.copy(f_order{}))==rr);
    REQUIRE(matmul(aa.copy(f_order{}),bb.copy(c_order{}))==rr);
    REQUIRE(matmul(multithreading::exec_pol<4>{},aa.copy(c_order{}),bb.copy(c_order{}))==rr);
    REQUIRE(matmul(multithreading::exec_pol<4>{},aa.copy(f_order{}),bb.copy(f_order{}))==rr);
    REQUIRE(matmul(multithreading::exec_pol<4>{},aa.copy(c_order{}),bb.copy(f_order{}))==rr);
    REQUIRE(matmul(multithreading::exec_pol<4>{},aa.copy(f_order{}),bb.copy(c_order{}))==rr);

    auto command_matmul = [](const auto& t1, const auto& t2){
        auto r = matmul(t1,t2);
        return std::abs(*r.begin());
    };
    auto command_matmul_par = [](const auto& t1, const auto& t2){
        auto r = matmul(multithreading::exec_pol<16>{}, t1,t2);
        return std::abs(*r.begin());
    };
    auto command_matmul_2d = [](const auto& t1, const auto& t2){
        auto r = matmul_2d(t1,t2);
        return *r.begin();
    };
    auto command_matmul_2d_tiled = [](const auto& t1, const auto& t2){
        auto r = matmul_2d_tiled(t1,t2);
        return *r.begin();
    };
    auto command_matmul_2d_goto = [](const auto& t1, const auto& t2){
        auto r = matmul_2d_goto(t1,t2);
        return *r.begin();
    };
    auto command_matmul_2d_goto1 = [](const auto& t1, const auto& t2){
        auto r = matmul_2d_goto1(t1,t2);
        return *r.begin();
    };
    auto command_matmul_2d_goto2 = [](const auto& t1, const auto& t2){
        auto r = matmul_2d_goto2(t1,t2);
        return *r.begin();
    };
    //bench_matmul("bench matmul_2d_goto2",n_iters,shapes,builder,command_matmul_2d_goto2);
    //bench_matmul("bench matmul_2d_goto1",n_iters,shapes,builder,command_matmul_2d_goto1);
    //bench_matmul("bench matmul_2d",n_iters,shapes,builder,command_matmul_2d);
    //bench_matmul("bench matmul_2d_tiled",n_iters,shapes,builder,command_matmul_2d_tiled);
    //bench_matmul("bench matmul_2d_goto",n_iters,shapes,builder,command_matmul_2d_goto);
    //bench_matmul("bench matmul",n_iters,shapes,builder,command_matmul);
    bench_matmul<value_type>("bench matmul_par",n_iters,shapes,builder,command_matmul_par);

}