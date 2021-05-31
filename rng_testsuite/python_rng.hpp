#pragma once

#include "python_wrapper.hpp"
#include <string>
#include <algorithm>
#include <vector>

template <typename T>
struct value_converter { /*T convert(ScopedPyObject py_val);*/ };

template <>
struct value_converter<double> { double convert(ScopedPyObject py_val) { return PyFloat_AsDouble(py_val); } };

template <>
struct value_converter<float> { float convert(ScopedPyObject py_val) { return static_cast<float>(PyFloat_AsDouble(py_val)); } };

template <>
struct value_converter<uint8_t> { uint8_t convert(ScopedPyObject py_val) { return static_cast<uint8_t>(PyLong_AsLong(py_val)); } };

template <>
struct value_converter<uint16_t> { uint16_t convert(ScopedPyObject py_val) { return static_cast<uint16_t>(PyLong_AsLong(py_val)); } };

template <>
struct value_converter<uint32_t> { uint32_t convert(ScopedPyObject py_val) { return static_cast<uint32_t>(PyLong_AsLong(py_val)); } };

template <>
struct value_converter<uint64_t> { uint64_t convert(ScopedPyObject py_val) { return static_cast<uint64_t>(PyLong_AsLongLong(py_val)); } };


class PythonRNGFunctions
{
private:
    ScopedPyObject _create;
    ScopedPyObject _uniform;
    ScopedPyObject _randomBits;
    ScopedPyObject _toString;
public:
    typedef ScopedPyObject PRNGContext;
private:
    static ScopedPyObject LoadPythonModule(const std::string& moduleFilePath);
public:
    PythonRNGFunctions(
        ScopedPyObject createFunction,
        ScopedPyObject uniformFunction,
        ScopedPyObject randomBitsFunction,
        ScopedPyObject toStringFunction)
            : _create(std::move(createFunction))
            , _uniform(std::move(uniformFunction))
            , _randomBits(std::move(randomBitsFunction))
            , _toString(std::move(toStringFunction))
    { }

    static PythonRNGFunctions LoadFromModule(const std::string& moduleFilePath);

    PRNGContext prngKey(const std::vector<uint8_t>& seed) const;

    template <typename T>
    std::pair<std::vector<T>, ScopedPyObject> processReturn(const ScopedPyObject& py_ret, size_t expectedCount) const
    {
        if (!PyTuple_Check(py_ret) || PyTuple_Size(py_ret) != 2)
        {
            PyErr_Print();
            throw std::string("valued returned is not a tuple of length 2");
        }
        auto py_randoms = ScopedPyObject::borrow(PyTuple_GetItem(py_ret, 0));
        auto py_newCtx = ScopedPyObject::borrow(PyTuple_GetItem(py_ret, 1));
        py_newCtx.makeOwned();

        auto py_fastRandoms = ScopedPyObject::own(PySequence_Fast(py_randoms, ""));
        if (!py_fastRandoms.isValid())
        {
            PyErr_Print();
            throw std::string("failed to convert returned sequence");
        }

        size_t returnedCount = PySequence_Fast_GET_SIZE(static_cast<PyObject*>(py_fastRandoms));
        if (returnedCount != expectedCount) throw std::string("did not receive expected amount of samples");
        std::vector<T> randoms(returnedCount);
        value_converter<T> converter;
        for (size_t i = 0; i < returnedCount; ++i)
        {
            auto py_item = ScopedPyObject::borrow(PySequence_Fast_GET_ITEM(static_cast<PyObject*>(py_fastRandoms), i));
            PyErr_Clear();
            T val = converter.convert(py_item);
            if (PyErr_Occurred()) throw std::string("returned value cannot be converted");
            randoms[i] = val;
        }
        return std::make_pair(randoms, py_newCtx);
    }

    template <typename T>
    std::pair<std::vector<T>, ScopedPyObject> uniform(PRNGContext& ctx, int count) const
    {
        auto py_count = ScopedPyObject::own(PyLong_FromLong(count));
        auto py_ret = ScopedPyObject::own(
            PyObject_CallFunctionObjArgs(_uniform, static_cast<PyObject*>(ctx), static_cast<PyObject*>(py_count), nullptr)
        );
        if (!py_ret.isValid())
        {
            PyErr_Print();
            throw std::string("failed calling uniform_and_state_update");
        }
        return processReturn<double>(py_ret, count);
    }

    template <typename T>
    std::pair<std::vector<T>, ScopedPyObject> randomBits(PRNGContext& ctx, int count) const
    {
        auto py_count = ScopedPyObject::own(PyLong_FromLong(count));
        auto py_ret = ScopedPyObject::own(
            PyObject_CallFunctionObjArgs(_randomBits, static_cast<PyObject*>(ctx), static_cast<PyObject*>(py_count), nullptr)
        );
        if (!py_ret.isValid())
        {
            PyErr_Print();
            throw std::string("failed calling uniform_and_state_update");
        }
        return processReturn<T>(py_ret, count);
    }

    std::wstring toString(PRNGContext ctx) const;
};
