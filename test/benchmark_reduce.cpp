#include "benchmark_helpers.hpp"
#include "helpers_for_testing.hpp"
#include "tensor.hpp"
#include "tensor_math.hpp"
#include "statistic.hpp"

namespace benchmark_expression_template_helpers{

using gtensor::basic_tensor;
using gtensor::tensor;

template<typename Axes, typename...Ts>
auto reduce_sum_(const basic_tensor<Ts...>& parent, const Axes& axes_, bool keep_dims=false){
    using parent_type = basic_tensor<Ts...>;
    using order = typename parent_type::order;
    using config_type = typename parent_type::config_type;
    using value_type = typename parent_type::value_type;
    using traverse_order = typename parent_type::traverse_order;
    const auto pdim = parent.dim();
    const auto& pshape = parent.shape();
    auto axes = gtensor::detail::make_axes<config_type>(pdim,axes_);
    gtensor::detail::check_reduce_args(pshape, axes);
    auto res = tensor<value_type,order,config_type>{gtensor::detail::make_reduce_shape(pshape, axes, keep_dims)};
    if (!res.empty()){
        auto axes_iterator_maker = gtensor::detail::make_axes_iterator_maker<config_type>(pshape,axes,traverse_order{});
        auto traverser = axes_iterator_maker.create_forward_traverser(parent.create_walker(),std::true_type{});
        auto a = res.traverse_order_adapter(order{});
        auto res_it = a.begin();
        do{
            *res_it = std::accumulate(
                axes_iterator_maker.begin_complement(traverser.walker(),std::false_type{}),
                axes_iterator_maker.end_complement(traverser.walker(),std::false_type{}),
                value_type{0}
            );
            ++res_it;
        }while(traverser.template next<order>());
    }
    return res;
}



template<typename Axes, typename...Ts>
auto reduce_sum(const basic_tensor<Ts...>& parent, const Axes& axes_, bool keep_dims=false){
    return reduce_sum_(parent,axes_,keep_dims);
}

template<typename...Ts, typename U>
auto reduce_sum(const basic_tensor<Ts...>& parent, std::initializer_list<U> axes_, bool keep_dims=false){
    return reduce_sum_(parent,axes_,keep_dims);
}

}   //end of namespace benchmark_expression_template_helpers


// TEMPLATE_TEST_CASE("test_reduce_sum","[benchmark_tensor]",
//     gtensor::config::c_order,
//     gtensor::config::f_order
// )
// {
//     using value_type = double;
//     using gtensor::tensor;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using tensor_type = gtensor::tensor<value_type,TestType>;
//     //using config_type = typename tensor_type::config_type;
//     //using order = typename tensor_type::order;
//     using benchmark_expression_template_helpers::reduce_sum;

//     auto t = tensor_type{{{{{7,5,8,5},{0,5,5,1},{3,8,0,8}}},{{{0,0,2,5},{1,2,3,0},{6,7,3,7}}}},{{{{4,8,0,7},{0,0,2,4},{1,5,8,5}}},{{{6,8,4,8},{4,1,3,2},{7,0,6,2}}}},{{{{7,3,6,4},{2,6,4,7},{0,3,3,1}}},{{{2,1,3,0},{4,7,4,4},{7,6,3,3}}}}};

//     REQUIRE(reduce_sum(tensor_type{1,2,3,4,5},std::vector<int>{0}) == tensor_type(15));
//     REQUIRE(reduce_sum(t,std::vector<int>{0}) == tensor_type{{{{18,16,14,16},{2,11,11,12},{4,16,11,14}}},{{{8,9,9,13},{9,10,10,6},{20,13,12,12}}}});
//     REQUIRE(reduce_sum(t,std::vector<int>{0,1}) == tensor_type{{{26,25,23,29},{11,21,21,18},{24,29,23,26}}});
//     REQUIRE(reduce_sum(t,std::vector<int>{0,2}) == tensor_type{{{18,16,14,16},{2,11,11,12},{4,16,11,14}},{{8,9,9,13},{9,10,10,6},{20,13,12,12}}});
//     REQUIRE(reduce_sum(t,std::vector<int>{1,2,3}) == tensor_type{{17,27,21,26},{22,22,23,28},{22,26,23,19}});
//     REQUIRE(reduce_sum(t,std::vector<int>{2,3}) == tensor_type{{{10,18,13,14},{7,9,8,12}},{{5,13,10,16},{17,9,13,12}},{{9,12,13,12},{13,14,10,7}}});
// }

// TEMPLATE_TEST_CASE("benchmark_axes_iterator","[benchmark_tensor]",
//     gtensor::config::c_order,
//     gtensor::config::f_order
// )
// {
//     using value_type = double;
//     using gtensor::tensor;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using tensor_type = gtensor::tensor<value_type,TestType>;
//     using config_type = typename tensor_type::config_type;
//     using order = typename tensor_type::order;
//     using benchmark_helpers::make_asymmetric_tree;
//     using benchmark_helpers::benchmark;

//     auto t = tensor_type({50,50,50,50},2);
//     auto axes_01 = gtensor::detail::make_axes<config_type>(t.dim(),std::vector<int>{0,1});
//     auto axes_23 = gtensor::detail::make_axes<config_type>(t.dim(),std::vector<int>{2,3});
//     auto axes_iterator_maker_01 = gtensor::detail::make_axes_iterator_maker<config_type>(t.shape(),axes_01,order{});
//     auto axes_iterator_maker_23 = gtensor::detail::make_axes_iterator_maker<config_type>(t.shape(),axes_23,order{});

//     auto bench_iteration_deref = [](const auto& t, const auto& it_maker){
//         value_type v{0};
//         auto it = it_maker.begin(t.create_walker());
//         auto last = it_maker.end(t.create_walker());
//         for (;it!=last; ++it){
//             v += *it;
//         }
//         return v;
//     };

//     benchmark("bench_axes_iterator_01",bench_iteration_deref,t,axes_iterator_maker_01);
//     benchmark("bench_axes_iterator_23",bench_iteration_deref,t,axes_iterator_maker_23);

// }

// TEMPLATE_TEST_CASE("benchmark_reduce_stdev","[benchmark_tensor]",
//     gtensor::config::c_order,
//     gtensor::config::f_order
// )
// {
//     using value_type = double;
//     using gtensor::tensor;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using tensor_type = gtensor::tensor<value_type,TestType>;
//     using benchmark_helpers::benchmark;

//     auto bench_sum = [](const auto& t, auto...axes){
//         if constexpr (sizeof...(axes) == 1){
//             auto tmp = gtensor::std(t,axes...);
//             return tmp.size();
//         }else{
//             auto tmp = gtensor::std(t,{axes...});
//             return tmp.size();
//         }
//     };

//     //auto t = tensor_type({10,5,100,1000},2);
//     auto t = tensor_type({50,50,50,50},2);

//     //like over flatten
//     //benchmark("std_all_10E6",bench_sum,t);

//     //single axis
//     // benchmark("std_axis_10E6",bench_sum,t,0);
//     // benchmark("std_axis_10E6",bench_sum,t,1);
//     // benchmark("std_axis_10E6",bench_sum,t,2);
//     // benchmark("std_axis_10E6",bench_sum,t,3);

//     //axes
//     benchmark("std_axes2_10E6",bench_sum,t,0,1);
//     benchmark("std_axes2_10E6",bench_sum,t,0,2);
//     benchmark("std_axes2_10E6",bench_sum,t,0,3);
//     benchmark("std_axes2_10E6",bench_sum,t,1,2);
//     benchmark("std_axes2_10E6",bench_sum,t,1,3);
//     benchmark("std_axes2_10E6",bench_sum,t,2,3);


//     // benchmark("std_axes2_10E6",bench_sum,t,0,2);
//     // benchmark("std_axes3_10E6",bench_sum,t,0,1,2);
//     // benchmark("std_axes4_10E6",bench_sum,t,0,1,2,3);
// }


// TEMPLATE_TEST_CASE("benchmark_tensor_sum","[benchmark_tensor]",
//     gtensor::config::c_order,
//     gtensor::config::f_order
// )
// {
//     using value_type = double;
//     using gtensor::tensor;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using tensor_type = gtensor::tensor<value_type,TestType>;
//     using benchmark_helpers::benchmark;

//     auto bench_sum = [](const auto& t, auto...axes){
//         auto tmp = t.sum({axes...});
//         return tmp.size();
//     };

//     //auto t = tensor_type({50,5,10,10,5,50},2);
//     //auto t = tensor_type({50,50,50,50},2);
//     auto t = tensor_type({500,3,1000,2000},2);
//     //auto t = tensor_type({100000000},2);

//     //like over flatten
//     //benchmark("sum_all_10E6",bench_sum,t);

//     //single axis
//     benchmark("sum_axis_10E6",bench_sum,t,0);
//     //benchmark("sum_axis_10E6",bench_sum,t,1);
//     //benchmark("sum_axis_10E6",bench_sum,t,2);
//     // benchmark("sum_axis_10E6",bench_sum,t,3);
//     // benchmark("sum_axis_10E6",bench_sum,t,4);
//     // benchmark("sum_axis_10E6",bench_sum,t,5);

//     //axes
//     // benchmark("sum_axes01_10E6",bench_sum,t,0,1);
//     // benchmark("sum_axes02_10E6",bench_sum,t,0,2);
//     // benchmark("sum_axes03_10E6",bench_sum,t,0,3);
//     // benchmark("sum_axes12_10E6",bench_sum,t,1,2);
//     // benchmark("sum_axes13_10E6",bench_sum,t,1,3);
//     // benchmark("sum_axes23_10E6",bench_sum,t,2,3);

//     // benchmark("sum_axes012_10E6",bench_sum,t,0,1,2);
//     // benchmark("sum_axes024_10E6",bench_sum,t,0,2,4);
//     // benchmark("sum_axes135_10E6",bench_sum,t,1,3,5);
//     // benchmark("sum_axes543_10E6",bench_sum,t,5,4,3);
// }


// TEMPLATE_TEST_CASE("benchmark_tensor_big","[benchmark_tensor]",
//     gtensor::config::c_order,
//     gtensor::config::f_order
// )
// {
//     using value_type = double;
//     using gtensor::tensor;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using tensor_type = gtensor::tensor<value_type,TestType>;
//     using shape_type = typename tensor_type::shape_type;
//     using benchmark_helpers::benchmark;
//     using benchmark_helpers::cpu_timer;
//     using benchmark_helpers::order_to_str;
//     using gtensor::detail::shape_to_str;

//     const auto axis=0;
//     std::vector<shape_type> shapes{
//         shape_type{100000000,3,1,2},
//         shape_type{10000000,3,1,20},
//         shape_type{1000000,3,10,20},
//         shape_type{100000,3,100,20},
//         shape_type{10000,3,100,200},
//         shape_type{1000,3,1000,200},
//         shape_type{100,3,1000,2000},
//         shape_type{50,6,1000,2000}
//     };
//     for (auto it=shapes.begin(), last=shapes.end(); it!=last; ++it){
//         auto t = tensor_type(*it,2);
//         std::cout<<std::endl<<order_to_str(typename tensor_type::order{})<<" "<<shape_to_str(t.shape())<<" axes "<<axis;
//         //mean
//         {
//             auto start = cpu_timer{};
//             auto t_mean = t.mean(axis);
//             auto stop = cpu_timer{};
//             std::cout<<std::endl<<"mean "<<stop-start<<" ms";
//         }
//         //std
//         {
//             auto start = cpu_timer{};
//             auto t_std = t.stdev(axis);
//             auto stop = cpu_timer{};
//             std::cout<<std::endl<<"std "<<stop-start<<" ms";
//         }
//     }
// }

TEMPLATE_TEST_CASE("test_expression_big","[benchmark_tensor]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using gtensor::tensor;
    using gtensor::tensor_close;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using tensor_type = gtensor::tensor<value_type,TestType>;
    using shape_type = typename tensor_type::shape_type;
    using benchmark_helpers::benchmark;
    using benchmark_helpers::cpu_timer;
    using benchmark_helpers::order_to_str;
    using gtensor::detail::shape_to_str;
    using benchmark_helpers::axes_to_str;
    using benchmark_helpers::make_asymmetric_tree;
    using helpers_for_testing::generate_lehmer;

    auto t1 = tensor_type(shape_type{100,1000,1000});
    auto t2 = tensor_type(shape_type{1000,1000});
    generate_lehmer(t1.begin(),t1.end(),123);
    generate_lehmer(t2.begin(),t2.end(),456);
    auto t = (t1+t2)*(t1-t2)/(t1*t2);
    REQUIRE(tensor_close(t.mean(), tensor<double>(-1.4611760878752456),1E-6,1E-6));
    REQUIRE(tensor_close(t.mean({1,2}),
            tensor<double>{-5.35539869e-01,1.08250989e+00,5.43758367e-01,-5.17622501e-01,-5.04011927e-01,-4.82679268e+00,4.09168553e-01,-3.54736887e+00,-2.00296408e-01,-1.77344374e-01,-2.58693331e-01,1.94368892e+00,7.65005786e-01,-1.43434565e+01,1.44088371e-01,-1.28539956e+00,7.75160651e-01,-2.28205846e+00,9.69762350e-01,2.30817317e-01,-9.43928772e-04,8.91365019e-01,-4.50719207e+00,-7.06661252e+00,1.63648873e+00,-6.21188195e-01,1.89105888e-01,-3.27709481e+00,1.10329001e-01,-2.90737067e+00,-1.24251996e+00,-1.10932262e+00,-1.26885591e+00,-5.15516533e-01,5.21639884e-01,-7.07365606e+00,-4.19547221e+00,8.10425560e-01,1.77609669e+00,4.95367300e-02,-7.14505508e-01,2.01581598e-01,-2.82936871e-01,2.81225117e-02,-1.53922507e+00,8.13045196e-01,-1.05154522e+00,-1.55849452e-01,-1.06036515e+00,-2.99006854e+00,-3.92536581e+00,-5.68026801e-01,-1.17389864e+00,-5.64707480e-01,5.65173652e-01,-8.97804810e-01,7.17410255e-01,3.96100006e-01,-1.39119192e+00,-1.08081330e+00,-1.79465227e-01,-5.93282656e-01,1.06335752e+00,8.76044341e-01,-1.14962509e+00,-1.48097810e+01,1.27335057e+00,7.30128061e-01,-1.25545978e+01,-3.54134672e+00,1.57904548e+00,3.63418042e-01,4.96351392e-01,-2.77612098e-01,-7.79032624e+00,-1.77501720e-01,4.73141057e-01,1.09132034e+00,-4.71031092e-02,9.99786927e-01,-1.65714428e-01,-1.07759352e+01,-1.29319167e+00,4.32733385e-01,-2.92991186e+00,-9.35307890e-01,1.01734006e+00,1.65203946e-01,-5.95137810e-01,-2.20694033e+00,9.86984574e-01,5.30246732e-01,-1.69672286e+00,-1.63490047e+00,6.58019055e-01,-2.42410416e+00,6.85240738e-01,-2.02708624e+01,-3.25351528e+00,-6.14618144e+00},
            1E-6,1E-6
        )
    );
    REQUIRE(tensor_close(t.stdev(), tensor<double>(3961.020094286836),1E-6,1E-6));
    REQUIRE(tensor_close(t.stdev({1,2}),
            tensor<double>{1504.70664559,934.55784304,1685.01381661,1443.93453632,1010.74607378,3030.57254466,1308.70265382,3138.070689,729.33591362,874.62768811,1296.23618438,1154.47420614,1218.02869016,14101.23096864,1105.99696494,2050.67989266,766.26431761,1586.03843649,843.79747457,1124.66127437,657.5992524,1404.88793274,4442.69593625,7934.98838096,803.70581387,1395.50492401,1384.34632355,2201.5947884,512.08668492,3032.51204304,2394.62790505,755.22417564,1042.17263223,1202.87785473,1162.42533752,3787.94750657,4015.06677365,681.31730375,1070.23461493,1313.82366475,1110.59864412,1186.6593341,1331.72376387,826.14303397,1907.81325945,1515.5561326,856.60494795,976.30659226,1774.15931722,3379.83020145,3853.99252069,1326.27889455,1701.52736547,998.96763201,886.75547419,1086.43109748,1015.40007823,819.4723255,1461.86622163,1680.1272538,1065.4736938,1033.44180946,1290.83411612,1086.17654201,827.43832645,15800.12923277,1049.71148916,699.71482742,13210.89503032,2979.51545944,1111.9466047,1378.89929855,1341.55851545,1217.19094234,7629.29273845,854.27843712,742.04332766,945.33246036,1321.95325978,1118.9240963,1333.14268384,10351.12136019,1745.93943005,1290.18720926,1859.54615928,1008.700286,1171.80905143,1449.08331808,753.49060283,1937.10495624,732.19345342,1132.02104209,1960.83896858,1553.62501088,1088.36187278,3196.31904967,790.58849569,19707.55206705,3558.53230023,7900.50976964},
            1E-6,1E-6
        )
    );
}

// TEMPLATE_TEST_CASE("benchmark_tensor_big","[benchmark_tensor]",
//     gtensor::config::c_order,
//     gtensor::config::f_order
// )
// {
//     using value_type = double;
//     using gtensor::tensor;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using tensor_type = gtensor::tensor<value_type,TestType>;
//     using benchmark_helpers::benchmark;
//     using benchmark_helpers::cpu_timer;
//     using benchmark_helpers::order_to_str;
//     using gtensor::detail::shape_to_str;
//     using benchmark_helpers::axes_to_str;
//     using benchmark_helpers::make_asymmetric_tree;

//     //const auto axes=2;
//     const auto axes=std::vector<int>{1,2};

//     //auto t = tensor_type({100,1000,1000},2);

//     auto t1 = tensor_type({100,1000,1000},2);
//     auto t2 = tensor_type({1000,1000},1);
//     auto t = make_asymmetric_tree<50>(t1,t2);


//     std::cout<<std::endl<<order_to_str(typename tensor_type::order{})<<" "<<shape_to_str(t.shape())<<" axes "<<axes_to_str(axes);
//     //mean
//     {
//         auto start = cpu_timer{};
//         auto t_mean = t.mean(axes);
//         auto stop = cpu_timer{};
//         std::cout<<std::endl<<"mean "<<stop-start<<" ms";
//     }
//     //std
//     {
//         auto start = cpu_timer{};
//         auto t_std = t.stdev(axes);
//         auto stop = cpu_timer{};
//         std::cout<<std::endl<<"std "<<stop-start<<" ms";
//     }
// }
