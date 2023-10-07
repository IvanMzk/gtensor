# GTensor quick reference

GTensor library is divided into modules.
Each module is defined in header file which has to be included to use module.

| Module | Header file | Contents |
|---|---|---|
| [**tensor**](/docs/tensor_qref.md) | tensor.hpp | tensor and basic_tensor class templates, tensor operators and views |
| [**builder**](/docs/builder_qref.md) | builder.hpp | tensor creation routines: empty, full, zeros, ones, identity, eye,  arrange, linspace, logspace, geomspace,... |
| [**manipulation**](/docs/manipulation_qref.md) | manipulation.hpp | tensor manipulation routines: stack, concatenate, vstack, hstack, block, split, vsplit, hsplit |
| [**tensor_math**](/docs/tensor_math_qref.md) | tensor_math.hpp | tensor math routines: abs, exp, log, pow, sin, cos, all, any, min, max, sum, prod, cumsum, cumprod, diff, gradient,... |
| [**reduce**](/docs/reduce_qref.md) | reduce.hpp | routines to make various tensor reductions and transformations |
| [**statistic**](/docs/statistic_qref.md) | statistic.hpp | tensor statistic routines: ptp, mean, var, stdev, median, quantile, average, moving_average, moving_mean, histogram,... |
| [**sort_search**](/docs/sort_search_qref.md) | sort_search.hpp | tensor sorting, searching and counting routines: sort, argsort, partition, argpartition, argmin, argmax, count_nonzero, unique,... |
| [**indexing**](/docs/indexing_qref.md) | indexing.hpp | tensor indexing routines: take, take_along_axis |
| [**random**](/docs/random_qref.md) | random.hpp | implements pseudo-random number generator which provides ability to draw samples from probability distributions and routines to make random permutations and choice  |
