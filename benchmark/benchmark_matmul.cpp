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

template<typename Shapes, typename Builder, typename Command>
auto bench_matmul(std::string mes, std::size_t n_iters, Shapes shapes, Builder builder, Command command){
    using value_type = double;
    bench_matmul_helper<gtensor::tensor<value_type,c_order>,gtensor::tensor<value_type,c_order>>{}(mes,n_iters,shapes,builder,command);
    bench_matmul_helper<gtensor::tensor<value_type,f_order>,gtensor::tensor<value_type,f_order>>{}(mes,n_iters,shapes,builder,command);
    // bench_matmul_helper<gtensor::tensor<value_type,c_order>,gtensor::tensor<value_type,f_order>>{}(mes,n_iters,shapes,builder,command);
    // bench_matmul_helper<gtensor::tensor<value_type,f_order>,gtensor::tensor<value_type,c_order>>{}(mes,n_iters,shapes,builder,command);
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
    static constexpr std::size_t kc_size = 256;
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

    auto kernel_res_c = [n](auto res_it, auto a_it, auto b_it, const auto& mc_, const auto& kc_, const auto& nc_){
        for (index_type kk=0; kk!=kc_; ++kk){
            const auto jj = kk*mc_;
            const auto ii = kk*nc_;
            for (index_type ir=0; ir!=mc_; ++ir){
                const auto rr = ir*n;
                const auto e = a_it[ir+jj];
                for (index_type jr=0; jr!=nc_; ++jr){
                    res_it[rr+jr]+=e*b_it[jr+ii];
                }
            }
        }
    };
    auto kernel_res_f = [m](auto res_it, auto a_it, auto b_it, const auto& mc_, const auto& kc_, const auto& nc_){
        for (index_type kk=0; kk!=kc_; ++kk){
            const auto jj = kk*mc_;
            const auto ii = kk*nc_;
            for (index_type jr=0; jr!=nc_; ++jr){
                const auto rr = jr*m;
                const auto e = b_it[ii+jr];
                for (index_type ir=0; ir!=mc_; ++ir){
                    res_it[ir+rr]+=a_it[ir+jj]*e;
                }
            }
        }
    };

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
        for (index_type pc=0; pc<k; pc+=kc){
                const auto kc_ = make_submatrix_size(pc,kc,k);
                const auto nc_ = make_submatrix_size(jc,nc,n);
                fill_buf(b_it+(pc*outer_stride_b+jc*inner_stride_b),b_buf.begin(),inner_stride_b,outer_stride_b,nc_,kc_);
            for (index_type ic=0; ic<m; ic+=mc){
                const auto mc_ = make_submatrix_size(ic,mc,m);
                fill_buf(a_it+(ic*outer_stride_a+pc*inner_stride_a),a_buf.begin(),outer_stride_a,inner_stride_a,mc_,kc_);
                if constexpr (std::is_same_v<res_order,c_order>){
                    kernel_res_c(res_it+(ic*n+jc),a_buf.cbegin(),b_buf.cbegin(),mc_,kc_,nc_);
                }else{
                    kernel_res_f(res_it+(ic+jc*m),a_buf.cbegin(),b_buf.cbegin(),mc_,kc_,nc_);
                }
            }
        }
    }

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
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = tensor<value_type>;
    using helpers_for_testing::generate_lehmer;
    using benchmark_helpers::timing;
    using benchmark_matmul_::bench_matmul;

    auto builder = [](auto&& t_){
        generate_lehmer(t_.begin(),t_.end(),[](const auto& e){return e%5;},123);
        return t_.clone_shallow();
    };

    auto command_matmul = [](const auto& t1, const auto& t2){
        auto r = matmul(t1,t2);
        return *r.begin();
    };

    auto shapes = std::vector<std::pair<std::vector<int>,std::vector<int>>>{
        // //1d x nd
        // std::make_pair(std::vector<int>{10000},std::vector<int>{10000,10000}),
        // //nd x 1d
        // std::make_pair(std::vector<int>{10000,10000},std::vector<int>{10000}),
        // //nd x nd
        std::make_pair(std::vector<int>{1000,1000},std::vector<int>{1000,1000}),
        std::make_pair(std::vector<int>{2000,2000},std::vector<int>{2000,2000})
        //std::make_pair(std::vector<int>{4000,4000},std::vector<int>{4000,4000})
        //std::make_pair(std::vector<int>{6000,6000},std::vector<int>{6000,6000})
        //std::make_pair(std::vector<int>{2000,1000},std::vector<int>{1000,3000}),
        //std::make_pair(std::vector<int>{10000,10000},std::vector<int>{10000,10000})
        //std::make_pair(std::vector<int>{3,2,300,1000},std::vector<int>{2,1000,900})
        //std::make_pair(std::vector<int>{100,100,200,100},std::vector<int>{100,100,300})
    };
    const auto n_iters = 10;
    //bench_matmul("bench matmul",n_iters,shapes,builder,command_matmul);


    using benchmark_matmul_::matmul_2d;
    using benchmark_matmul_::matmul_2d_tiled;
    using benchmark_matmul_::matmul_2d_goto;
    using benchmark_matmul_::matmul_2d_goto1;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    auto aa = tensor<double>{{1,1,1,3,3,2,3,2,2,2,0,2,4,0,4,3,2,3,2,3,2,4,3,4,0,2,3,4,1,0,0,1,2,0,4,0,2,3,2,4},{4,4,2,2,4,1,1,1,4,4,1,2,3,4,0,4,3,4,3,2,2,1,3,2,1,4,1,0,1,0,1,2,4,1,4,3,1,4,4,0},{1,1,1,0,2,0,3,3,1,4,3,1,1,3,1,1,4,2,1,2,3,1,4,1,0,3,2,1,2,0,2,2,1,3,4,4,2,1,3,4},{0,3,2,3,1,2,0,4,3,3,4,3,0,1,0,0,3,0,0,0,0,0,3,1,1,2,2,4,1,3,0,1,2,0,3,1,3,1,3,2},{4,0,4,0,2,3,2,0,4,3,0,1,4,2,0,1,1,1,2,4,1,3,2,4,0,3,4,3,3,0,4,4,0,1,2,0,4,1,1,2},{3,3,1,2,2,1,4,0,1,4,0,2,3,4,1,3,0,0,1,3,4,4,1,4,3,1,1,1,1,2,0,2,2,1,3,4,4,0,0,3},{4,0,2,0,2,2,4,1,4,0,0,0,4,4,3,1,4,4,1,4,4,0,0,4,0,4,2,4,4,0,4,1,1,4,3,1,0,3,2,0},{0,0,3,0,0,1,1,3,2,1,0,4,3,2,3,1,2,1,4,4,1,0,1,3,2,1,1,2,0,3,1,3,2,0,2,2,1,3,3,1},{0,1,1,1,0,2,1,2,1,4,0,0,3,0,2,3,4,0,3,0,4,1,4,3,2,3,2,1,3,3,3,0,4,3,4,1,0,1,4,2},{3,4,1,1,1,2,1,3,3,1,3,3,2,2,1,2,1,0,4,3,0,4,3,1,0,3,2,4,3,3,0,1,2,1,3,2,3,2,3,3},{0,4,1,3,4,4,3,1,1,3,1,4,0,1,1,4,0,2,0,4,4,3,2,3,0,3,4,4,2,2,4,1,3,0,2,4,0,1,3,3},{0,2,4,1,2,0,2,1,0,4,1,0,3,4,3,2,0,1,3,1,3,3,2,4,1,0,0,1,4,4,4,3,3,2,0,4,3,4,3,4},{0,3,3,3,2,2,4,2,4,4,0,0,1,0,4,0,0,1,3,3,0,2,2,0,2,4,1,1,3,4,4,2,2,3,2,2,0,3,1,0},{0,0,2,3,3,0,3,0,1,4,2,3,3,0,2,4,3,1,2,1,4,1,0,3,1,4,2,3,3,4,0,1,4,2,4,4,1,4,2,0},{3,0,1,4,1,3,1,3,0,3,3,3,3,4,3,3,2,4,0,2,1,1,2,1,0,1,0,2,2,0,3,1,3,1,3,2,1,3,4,3},{4,4,1,0,2,3,1,3,2,1,1,0,2,0,2,3,0,1,2,2,4,1,3,3,4,4,2,4,0,3,2,4,4,0,2,2,3,2,1,0},{1,1,4,1,2,4,0,4,0,4,4,1,0,0,3,4,4,1,2,2,4,0,2,3,2,0,2,2,4,4,3,0,4,0,4,2,3,0,0,0},{2,1,2,4,2,4,1,4,3,0,0,0,2,1,4,1,1,0,4,1,1,4,4,3,3,0,2,2,1,4,3,3,4,3,2,0,2,3,1,4},{1,2,1,4,1,1,3,4,2,4,4,3,3,0,0,1,3,3,1,4,1,2,2,3,3,3,1,0,1,1,3,1,4,1,0,1,1,2,0,0},{2,4,3,1,3,1,1,1,2,3,0,1,4,4,2,4,1,2,0,0,0,3,1,2,3,3,4,3,1,1,1,1,0,1,0,1,0,2,4,3},{0,2,0,0,1,1,3,2,0,0,3,4,4,3,4,4,2,1,2,4,4,1,3,0,2,0,1,0,1,2,4,1,3,1,4,4,0,1,1,0},{1,0,3,1,3,4,3,4,1,0,0,1,4,4,1,4,4,4,0,3,2,4,4,0,2,1,4,3,0,2,0,3,2,0,0,2,0,1,0,3},{1,1,2,0,0,4,0,2,3,0,2,1,1,4,0,0,1,4,3,1,1,4,4,2,1,0,1,4,1,3,3,3,0,2,1,3,1,4,4,2},{4,0,2,4,4,3,1,1,2,2,1,2,0,2,4,1,4,1,2,0,3,3,1,2,3,3,2,0,2,0,0,4,2,1,4,0,4,2,4,2},{3,3,3,4,2,0,1,1,4,3,0,4,1,4,1,4,2,3,0,1,1,3,0,3,1,1,1,4,4,3,0,0,4,3,2,2,4,4,2,3},{1,4,2,0,3,3,3,2,1,2,1,1,0,0,0,3,3,4,1,0,1,1,3,2,3,1,1,2,2,3,3,1,3,0,2,0,3,0,3,4},{0,0,4,2,2,2,3,2,1,0,0,3,4,4,2,1,2,4,1,2,2,2,1,4,0,2,0,1,2,2,1,3,4,3,3,0,3,0,4,3},{0,2,3,3,3,2,1,0,3,1,0,4,0,4,2,2,2,4,1,1,4,1,1,4,4,0,3,2,3,3,4,0,2,1,4,0,1,1,2,3},{3,1,3,2,4,1,4,1,3,0,3,1,2,0,1,1,1,2,4,3,0,2,0,4,4,0,0,1,0,2,0,0,4,4,0,0,3,4,1,4},{0,4,1,2,2,0,3,1,3,4,1,2,1,0,4,4,0,4,0,0,0,3,0,4,4,3,0,3,3,0,2,0,1,3,4,2,3,3,4,3}};
    auto bb = tensor<double>{{1,1,1,3,3,2,3,2,2,2,0,2,4,0,4,3,2,3,2,3},{2,4,3,4,0,2,3,4,1,0,0,1,2,0,4,0,2,3,2,4},{4,4,2,2,4,1,1,1,4,4,1,2,3,4,0,4,3,4,3,2},{2,1,3,2,1,4,1,0,1,0,1,2,4,1,4,3,1,4,4,0},{1,1,1,0,2,0,3,3,1,4,3,1,1,3,1,1,4,2,1,2},{3,1,4,1,0,3,2,1,2,0,2,2,1,3,4,4,2,1,3,4},{0,3,2,3,1,2,0,4,3,3,4,3,0,1,0,0,3,0,0,0},{0,0,3,1,1,2,2,4,1,3,0,1,2,0,3,1,3,1,3,2},{4,0,4,0,2,3,2,0,4,3,0,1,4,2,0,1,1,1,2,4},{1,3,2,4,0,3,4,3,3,0,4,4,0,1,2,0,4,1,1,2},{3,3,1,2,2,1,4,0,1,4,0,2,3,4,1,3,0,0,1,3},{4,4,1,4,3,1,1,1,1,2,0,2,2,1,3,4,4,0,0,3},{4,0,2,0,2,2,4,1,4,0,0,0,4,4,3,1,4,4,1,4},{4,0,0,4,0,4,2,4,4,0,4,1,1,4,3,1,0,3,2,0},{0,0,3,0,0,1,1,3,2,1,0,4,3,2,3,1,2,1,4,4},{1,0,1,3,2,1,1,2,0,3,1,3,2,0,2,2,1,3,3,1},{0,1,1,1,0,2,1,2,1,4,0,0,3,0,2,3,4,0,3,0},{4,1,4,3,2,3,2,1,3,3,3,0,4,3,4,1,0,1,4,2},{3,4,1,1,1,2,1,3,3,1,3,3,2,2,1,2,1,0,4,3},{0,4,3,1,0,3,2,4,3,3,0,1,2,1,3,2,3,2,3,3},{0,4,1,3,4,4,3,1,1,3,1,4,0,1,1,4,0,2,0,4},{4,3,2,3,0,3,4,4,2,2,4,1,3,0,2,4,0,1,3,3},{0,2,4,1,2,0,2,1,0,4,1,0,3,4,3,2,0,1,3,1},{3,3,2,4,1,0,0,1,4,4,4,3,3,2,0,4,3,4,3,4},{0,3,3,3,2,2,4,2,4,4,0,0,1,0,4,0,0,1,3,3},{0,2,2,0,2,4,1,1,3,4,4,2,2,3,2,2,0,3,1,0},{0,0,2,3,3,0,3,0,1,4,2,3,3,0,2,4,3,1,2,1},{4,1,0,3,1,4,2,3,3,4,0,1,4,2,4,4,1,4,2,0},{3,0,1,4,1,3,1,3,0,3,3,3,3,4,3,3,2,4,0,2},{1,1,2,1,0,1,0,2,2,0,3,1,3,1,3,2,1,3,4,3},{4,4,1,0,2,3,1,3,2,1,1,0,2,0,2,3,0,1,2,2},{4,1,3,3,4,4,2,4,0,3,2,4,4,0,2,2,3,2,1,0},{1,1,4,1,2,4,0,4,0,4,4,1,0,0,3,4,4,1,2,2},{4,0,2,3,2,0,2,2,4,4,3,0,4,0,4,2,3,0,0,0},{2,1,2,4,2,4,1,4,3,0,0,0,2,1,4,1,1,0,4,1},{1,4,4,3,3,0,2,2,1,4,3,3,4,3,2,0,2,3,1,4},{1,2,1,4,1,1,3,4,2,4,4,3,3,0,0,1,3,3,1,4},{1,2,2,3,3,3,1,0,1,1,3,1,4,1,0,1,1,2,0,0},{2,4,3,1,3,1,1,1,2,3,0,1,4,4,2,4,1,2,0,0},{0,3,1,2,3,3,4,3,1,1,1,1,0,1,0,1,0,2,4,3}};
    auto rr = tensor<double>{{147,148,178,179,134,188,161,185,168,197,140,142,207,132,180,184,158,158,192,173},{176,174,204,203,165,208,172,194,185,217,166,144,229,159,208,179,170,173,178,176},{125,157,158,178,139,165,159,184,148,205,140,129,180,133,163,148,139,124,144,140},{118,121,140,144,102,140,126,133,117,161,89,99,165,108,150,145,121,119,137,125},{181,150,154,176,141,181,164,179,186,200,150,144,208,133,157,188,159,168,156,171},{137,168,157,212,124,174,170,204,170,177,163,155,167,107,170,147,151,166,158,191},{179,135,165,175,144,203,141,180,207,218,146,132,225,153,192,190,156,167,156,157},{131,140,147,134,116,142,106,149,149,157,101,114,172,111,142,144,137,122,147,145},{121,142,164,151,124,164,129,162,148,179,132,122,171,118,176,174,135,130,161,148},{164,162,170,186,128,182,166,197,160,195,136,135,218,138,193,184,143,161,176,182},{159,197,191,201,149,197,164,197,154,214,160,162,194,138,203,210,157,174,176,187},{174,186,171,199,146,179,160,209,170,192,190,162,205,150,162,173,152,187,166,189},{143,149,186,142,111,176,128,179,168,166,149,130,189,120,176,139,139,135,161,153},{144,158,167,192,149,181,136,169,162,212,168,161,205,136,177,184,170,165,151,158},{160,140,168,175,134,190,150,172,149,169,124,124,200,144,200,175,138,149,167,149},{133,156,185,173,145,184,160,189,157,213,135,146,198,109,194,174,142,167,172,189},{131,152,165,178,121,165,143,185,142,205,131,153,177,123,186,192,161,146,188,192},{162,143,196,162,135,187,158,197,168,190,148,132,215,114,196,192,147,154,214,187},{134,156,174,154,113,169,145,156,150,190,131,116,176,105,171,156,152,125,152,159},{146,128,146,164,124,149,160,151,161,168,122,115,185,128,166,147,126,162,148,140},{126,150,155,151,121,151,131,176,132,164,100,117,162,115,180,144,130,111,148,166},{137,119,172,156,129,171,162,175,147,201,131,115,185,128,184,169,144,146,183,148},{184,148,164,162,127,163,143,152,157,173,132,101,212,142,167,170,96,135,159,148},{141,144,169,182,146,187,159,179,156,201,143,154,201,121,178,187,152,144,170,159},{188,151,174,231,144,200,161,197,180,211,172,145,231,130,200,192,168,195,177,177},{123,147,155,159,115,150,136,173,131,182,125,105,154,105,160,149,125,127,163,152},{177,141,167,171,136,178,130,186,176,193,154,123,192,144,173,183,159,155,164,154},{169,157,161,193,138,182,142,169,173,185,134,124,182,128,186,187,129,151,185,173},{137,149,153,152,124,140,141,158,169,194,140,108,176,103,135,147,144,128,156,167},{159,160,175,203,129,172,155,184,179,196,147,134,203,122,179,144,134,154,161,167}};

    //std::cout<<std::endl<<matmul_2d_goto(aa.copy(f_order{}),bb.copy(f_order{}));
    //std::cout<<std::endl<<matmul_2d_goto(aa,bb);

    //REQUIRE(matmul_2d(tensor<double,c_order>{{1,2,4,2},{3,4,2,0},{5,3,1,1}},tensor<double,c_order>{{2,1,2},{0,3,1},{1,1,4},{4,3,3}})==tensor<double>{{14,17,26},{8,17,18},{15,18,20}});
    //REQUIRE(matmul_2d(tensor<double,f_order>{{1,2,4,2},{3,4,2,0},{5,3,1,1}},tensor<double,f_order>{{2,1,2},{0,3,1},{1,1,4},{4,3,3}})==tensor<double>{{14,17,26},{8,17,18},{15,18,20}});
    // REQUIRE(matmul_2d_tiled(aa,bb)==rr);
    // REQUIRE(matmul_2d_tiled(aa.copy(f_order{}),bb.copy(f_order{}))==rr);

    // REQUIRE(matmul_2d_tiled(tensor<double,c_order>{{1,2,4,2},{3,4,2,0},{5,3,1,1}},tensor<double,c_order>{{2,1,2},{0,3,1},{1,1,4},{4,3,3}})==tensor<double>{{14,17,26},{8,17,18},{15,18,20}});
    //REQUIRE(matmul_2d_tiled(tensor<double,f_order>{{1,2,4,2},{3,4,2,0},{5,3,1,1}},tensor<double,f_order>{{2,1,2},{0,3,1},{1,1,4},{4,3,3}})==tensor<double>{{14,17,26},{8,17,18},{15,18,20}});

    //REQUIRE(matmul_2d_goto(aa.copy(f_order{}),bb.copy(f_order{}))==rr);
    //REQUIRE(matmul_2d_goto(aa,bb)==rr);

    REQUIRE(matmul_2d_goto1(aa.copy(c_order{}),bb.copy(c_order{}))==rr);
    REQUIRE(matmul_2d_goto1(aa.copy(f_order{}),bb.copy(f_order{}))==rr);
    REQUIRE(matmul_2d_goto1(aa.copy(c_order{}),bb.copy(f_order{}))==rr);
    REQUIRE(matmul_2d_goto1(aa.copy(f_order{}),bb.copy(c_order{}))==rr);

    //REQUIRE(matmul(tensor<double,c_order>{{1,2,4,2},{3,4,2,0},{5,3,1,1}},tensor<double,c_order>{{2,1,2},{0,3,1},{1,1,4},{4,3,3}})==tensor<double>{{14,17,26},{8,17,18},{15,18,20}});
    //REQUIRE(matmul(aa,bb)==rr);

    REQUIRE(matmul(aa.copy(c_order{}),bb.copy(c_order{}))==rr);
    REQUIRE(matmul(aa.copy(f_order{}),bb.copy(f_order{}))==rr);
    REQUIRE(matmul(aa.copy(c_order{}),bb.copy(f_order{}))==rr);
    REQUIRE(matmul(aa.copy(f_order{}),bb.copy(c_order{}))==rr);

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
    //bench_matmul("bench matmul_2d_goto1",n_iters,shapes,builder,command_matmul_2d_goto1);
    //bench_matmul("bench matmul_2d",n_iters,shapes,builder,command_matmul_2d);
    //bench_matmul("bench matmul_2d_tiled",n_iters,shapes,builder,command_matmul_2d_tiled);
    //bench_matmul("bench matmul_2d_goto",n_iters,shapes,builder,command_matmul_2d_goto);
    bench_matmul("bench matmul",n_iters,shapes,builder,command_matmul);

}