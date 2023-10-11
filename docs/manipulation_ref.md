# Manipulation

GTensor **manipulation** module resides in **manipulation.hpp** header.
It provides routines to combine and split tensors.

## stack

Join a sequence of tensors along a new axis, tensors must have the same shape.

```cpp
//takes variadic tensors to join
template<typename DimT, typename...Us, typename...Ts>
auto stack(const DimT& axis, const basic_tensor<Us...>& t, const Ts&...ts);

//takes container of tensors to join
template<typename DimT, typename Container>
auto stack(const DimT& axis, const Container& ts);
```

```cpp
using gtensor::tensor;
auto res1 = stack(0,tensor<double>{{1,2},{3,4}},tensor<double>{{5,6},{7,8}},tensor<double>{{9,10},{11,12}});
std::vector<tensor<double>> v{};
v.push_back(tensor<double>{{1,2},{3,4}});
v.push_back(tensor<double>{{5,6},{7,8}});
v.push_back(tensor<double>{{9,10},{11,12}});
auto res2 = stack(1,v);
std::cout<<std::endl<<res1; //[(3,2,2){{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}}}]
std::cout<<std::endl<<res2; //[(2,3,2){{{1,2},{5,6},{9,10}},{{3,4},{7,8},{11,12}}}]
```

## concatenate

Join a sequence of tensors along existing axis, tensors must have the same shape except concatenate axis.
As `stack` has two versions: one which take variadic tensors and container of tensors.

```cpp
using gtensor::tensor;
auto res1 = concatenate(0,tensor<double>{{1,2}},tensor<double>{{3,4},{5,6}},tensor<double>{{7,8},{9,10},{11,12}});
std::vector<tensor<double>> v{};
v.push_back(tensor<double>{{1},{2}});
v.push_back(tensor<double>{{3,4},{5,6}});
v.push_back(tensor<double>{{7,8,9},{10,11,12}});
auto res2 = concatenate(1,v);
std::cout<<std::endl<<res1; //[(6,2){{1,2},{3,4},{5,6},{7,8},{9,10},{11,12}}]
std::cout<<std::endl<<res2; //[(2,6){{1,3,4,7,8,9},{2,5,6,10,11,12}}]
```

## vstack

`vstack` is equivalent to concatenation along the first axis.
1d tensors of shape (n) are reshaped to (1,n) before concatenation.
Has two versions: one which take variadic tensors and container of tensors.

```cpp
using gtensor::tensor;
auto res = vstack(tensor<double>{1,2},tensor<double>{{3,4},{5,6}},tensor<double>{{7,8},{9,10},{11,12}});
std::cout<<std::endl<<res;  //[(6,2){{1,2},{3,4},{5,6},{7,8},{9,10},{11,12}}]
```

## hstack

`hstack` is equivalent to concatenation along the second axis.
For 1d tensors is equivalent to concatenation along the first axis.
Has two versions: one which take variadic tensors and container of tensors.

```cpp
using gtensor::tensor;
auto res1 = hstack(tensor<double>{{1},{2}},tensor<double>{{3,4},{5,6}},tensor<double>{{7,8,9},{10,11,12}});
auto res2 = hstack(tensor<double>{1,2},tensor<double>{3,4,5,6},tensor<double>{7,8,9,10,11,12});
std::cout<<std::endl<<res1; //[(2,6){{1,3,4,7,8,9},{2,5,6,10,11,12}}]
std::cout<<std::endl<<res2; //[(12){1,2,3,4,5,6,7,8,9,10,11,12}]
```

## block

Assemble a tensor from nested sequences of blocks.
Has two versions: one which takes nested tuple of blocks and nested std::initializer_list of blocks.
Follows rules of **numpy block** routine.

```cpp
using gtensor::tensor;
auto res1 = block(std::make_tuple(std::make_tuple(tensor<double>{{1,2},{3,4}},tensor<double>{{5},{6}}),std::make_tuple(tensor<double>{{7,8,9},{10,11,12}})));
auto res2 = block({{tensor<double>{{1,2},{3,4}},tensor<double>{{5},{6}}},{tensor<double>{{7,8,9},{10,11,12}}}});
std::cout<<std::endl<<res1; //[(4,3){{1,2,5},{3,4,6},{7,8,9},{10,11,12}}]
std::cout<<std::endl<<res2; //[(4,3){{1,2,5},{3,4,6},{7,8,9},{10,11,12}}]
```

## split

Split tensor along given axis and return container of slice views.
By default split along first axis.

```cpp
//split points determined using split_points parameter that is container of split points or std::initializer_list
template<typename...Ts, typename IdxContainer, typename DimT=int>
auto split(const basic_tensor<Ts...>& t, const IdxContainer& split_points, const DimT& axis=0);
template<typename...Ts, typename IdxT, typename DimT=int>
auto split(const basic_tensor<Ts...>& t, std::initializer_list<IdxT> split_points, const DimT& axis=0);
//split points determined by dividing tensor by equal parts
template<typename...Ts, typename DimT=int>
auto split(const basic_tensor<Ts...>& t, const typename basic_tensor<Ts...>::index_type& parts_number, const DimT& axis=0);
```

```cpp
gtensor::tensor<double> t{1,2,3,4,5,6,7,8,9,10};
auto res1 = split(t,{2,5});    //split using split points
auto res2 = split(t,2);        //split by equal parts
std::cout<<std::endl<<res1[0]; //[(2){1,2}]
std::cout<<std::endl<<res1[1]; //[(3){3,4,5}]
std::cout<<std::endl<<res1[2]; //[(5){6,7,8,9,10}]
std::cout<<std::endl<<res2[0]; //[(5){1,2,3,4,5}]
std::cout<<std::endl<<res2[1]; //[(5){6,7,8,9,10}]
```

## vsplit

Equivalent to `split` along the first axis.

```cpp
gtensor::tensor<double> t{{1,2},{3,4},{5,6},{7,8},{9,10},{11,12}};
auto res1 = vsplit(t,{2,4});    //split using split points
auto res2 = vsplit(t,2);        //split by equal parts
std::cout<<std::endl<<res1[0];  //[(2,2){{1,2},{3,4}}]
std::cout<<std::endl<<res1[1];  //[(2,2){{5,6},{7,8}}]
std::cout<<std::endl<<res1[2];  //[(2,2){{9,10},{11,12}}]
std::cout<<std::endl<<res2[0];  //[(3,2){{1,2},{3,4},{5,6}}]
std::cout<<std::endl<<res2[1];  //[(3,2){{7,8},{9,10},{11,12}}]
```

## hsplit

Equivalent to `split` along second axis.
For 1d tensors is equivalent to `split` along first axis.

```cpp
gtensor::tensor<double> t{{1,2,3,4,5,6},{7,8,9,10,11,12}};
auto res1 = hsplit(t,{2,4});    //split using split points
auto res2 = hsplit(t,2);        //split by equal parts
std::cout<<std::endl<<res1[0];  //[(2,2){{1,2},{7,8}}]
std::cout<<std::endl<<res1[1];  //[(2,2){{3,4},{9,10}}]
std::cout<<std::endl<<res1[2];  //[(2,2){{5,6},{11,12}}]
std::cout<<std::endl<<res2[0];  //[(2,3){{1,2,3},{7,8,9}}]
std::cout<<std::endl<<res2[1];  //[(2,3){{4,5,6},{10,11,12}}]
```
