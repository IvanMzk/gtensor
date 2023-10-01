/*
* GTensor - computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#ifndef EXCEPTION_HPP_
#define EXCEPTION_HPP_

#include <exception>

namespace gtensor{

//base for all gtensor exception classes
class gtensor_error : public std::runtime_error
{
public:
    explicit gtensor_error(const char* what):
        runtime_error(what)
    {}
};

//should be raised when callable argument has invalid value
class value_error : public gtensor_error
{
public:
    explicit value_error(const char* what):
        gtensor_error(what)
    {}
};

//should be raised when axis has invalid value
class axis_error : public value_error
{
public:
    explicit axis_error(const char* what):
        value_error(what)
    {}
};

//should be raised when subscript is out of range
class index_error : public value_error
{
public:
    explicit index_error(const char* what):
        value_error(what)
    {}
};


}   //end of namespace gtensor
#endif