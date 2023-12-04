# Reference

GTensor library is divided into modules.
Each module is defined in header file which has to be included to use module.

| Module | Header file | Contents |
|---|---|---|
| [tensor](/docs/tensor_ref.md)<br/> [tensor_operators](/docs/tensor_operators_ref.md) | tensor.hpp | tensor and basic_tensor class templates, tensor operators and views |
| [builder](/docs/builder_ref.md) | builder.hpp | tensor creation routines: empty, full, zeros, ones, identity, eye,  arrange, linspace, logspace, geomspace,... |
| [manipulation](/docs/manipulation_ref.md) | manipulation.hpp | tensor manipulation routines: stack, concatenate, vstack, hstack, block, split, vsplit, hsplit |
| [tensor_math](/docs/tensor_math_ref.md) | tensor_math.hpp | tensor math routines: abs, exp, log, pow, sin, cos, all, any, min, max, sum, prod, cumsum, cumprod, diff, gradient, matmul... |
| [reduce](/docs/reduce_ref.md) | reduce.hpp | routines to make various tensor reductions and transformations |
| [statistic](/docs/statistic_ref.md) | statistic.hpp | tensor statistic routines: ptp, mean, var, stdev, median, quantile, average, moving_average, moving_mean, histogram, cov, corrcoef... |
| [sort_search](/docs/sort_search_ref.md) | sort_search.hpp | tensor sorting, searching and counting routines: sort, argsort, partition, argpartition, argmin, argmax, count_nonzero, unique,... |
| [indexing](/docs/indexing_ref.md) | indexing.hpp | tensor indexing routines: take, take_along_axis |
| [random](/docs/random_ref.md) | random.hpp | implements pseudo-random number generator which provides ability to draw samples from probability distributions and routines to make random permutations and choice  |