# GTensor quick reference

GTensor library is divided into modules.
Each module is defined in header file which has to be included to use module.

| Module       | Header file      | Contents                                                                                                                                                          |
|--------------|------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| tensor       | tensor.hpp       | tensor and basic_tensor class templates, tensor operators and views                                                                                               |
| builder      | builder.hpp      | tensor creation routines: empty, full, zeros, ones, identity, eye,  arrange, linspace, logspace, geomspace,...                                                    |
| manipulation | manipulation.hpp | tensor manipulation routines: stack, concatenate, vstack, hstack, block, split, vsplit, hsplit                                                                    |
| tensor_math  | tensor_math.hpp  | tensor math routines: abs, exp, log, pow, sin, cos, all, any, min, max, sum, prod, cumsum, cumprod, diff, gradient,...                                            |
| reduce       | reduce.hpp       | routines to make various tensor reductions and transformations                                                                                                    |
| statistic    | statistic.hpp    | tensor statistic routines: ptp, mean, var, stdev, median, quantile, average, moving_average, moving_mean, histigram,...                                           |
| sort_search  | sort_search.hpp  | tensor sorting, searching and counting routines: sort, argsort, partition, argpartition, argmin, argmax, count_nonzero, unique,...                                |
| indexing     | indexing.hpp     | tensor indexing routines: take, take_along_axis                                                                                                                   |
| random       | random.hpp       | implements pseudo-random number generator which provides ability to draw samples from probability distributions, routines to make random permutations and choice  |


1. [`tensor` and `basic_tensor` classes](/docs/tensor_basic_tensor_quick_reference.md)