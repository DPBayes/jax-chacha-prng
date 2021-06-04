// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2021 Aalto University

#include "python_rng.hpp"

PythonRNGFunctions::PRNGContext PythonRNGFunctions::prngKey(const std::vector<uint8_t>& seed) const
{
    auto py_seed = ScopedPyObject::own(PyBytes_FromStringAndSize(
        reinterpret_cast<const char*>(seed.data()), seed.size())
    );
    auto py_ret = ScopedPyObject::own(
        PyObject_CallFunctionObjArgs(_create,
            static_cast<PyObject*>(py_seed),
            nullptr
        )
    );
    if (!py_ret.isValid())
    {
        PyErr_Print();
        throw std::string("failed calling PRNGKey");
    }
    return py_ret;
}

std::wstring PythonRNGFunctions::toString(PRNGContext ctx) const
{
    auto py_ret = ScopedPyObject::own(
        PyObject_CallFunctionObjArgs(_toString,
            static_cast<PyObject*>(ctx),
            nullptr
        )
    );
    if (!PyUnicode_Check(py_ret))
    {
        PyErr_Print();
        throw std::string("failed calling to_string");
    }

    Py_ssize_t length = 0;
    PyErr_Clear();
    wchar_t* rawString = PyUnicode_AsWideCharString(py_ret, &length);
    if (PyErr_Occurred())
    {
        PyErr_Print();
        throw std::string("failed to convert to string");
    }
    std::wstring str(rawString);
    PyMem_Free(rawString);
    return str;
}


ScopedPyObject PythonRNGFunctions::LoadPythonModule(const std::string& moduleFilePath)
{
    std::string moduleBasePath;
    std::string fileName = moduleFilePath;
    size_t separatorPos = moduleFilePath.find_last_of('/');
    if (separatorPos != std::string::npos)
    {
        std::string moduleBasePath = moduleFilePath.substr(0, separatorPos);
        fileName = moduleFilePath.substr(separatorPos + 1);

        auto py_moduleBasePath = ScopedPyObject::own(PyUnicode_FromString(moduleBasePath.c_str()));
        auto py_pythonPath = ScopedPyObject::borrow(PySys_GetObject("path"));
        PyList_Insert(py_pythonPath, 0, py_moduleBasePath);
    }


    separatorPos = fileName.find_last_of(".");
    if (separatorPos != std::string::npos)
    {
        fileName = fileName.substr(0, separatorPos);
    }
    auto py_chachaModule = ScopedPyObject::own(PyImport_ImportModule(fileName.c_str()));
    if (!py_chachaModule.isValid())
    {
        throw std::string("cannot load module");
    }
    return py_chachaModule;
}


PythonRNGFunctions PythonRNGFunctions::LoadFromModule(const std::string& moduleFilePath)
{
    auto module = LoadPythonModule(moduleFilePath);
    auto py_create = ScopedPyObject::own(PyObject_GetAttrString(module, "create_context"));
    if (!py_create.isValid())
    {
        PyErr_Print();
        throw std::string("could not load 'create_context' function!");
    }

    auto py_uniform = ScopedPyObject::own(PyObject_GetAttrString(module, "uniform_and_state_update"));
    if (!py_uniform.isValid())
    {
        PyErr_Print();
        throw std::string("could not load 'uniform_and_state_update' function!");
    }

    auto py_bits = ScopedPyObject::own(PyObject_GetAttrString(module, "bits_and_state_update"));
    if (!py_bits.isValid())
    {
        PyErr_Print();
        throw std::string("could not load 'bits_and_state_update' function!");
    }

    auto py_toString = ScopedPyObject::own(PyObject_GetAttrString(module, "to_string"));
    if (!py_toString.isValid())
    {
        PyErr_Print();
        throw std::string("could not load 'to_string' function!");
    }

    return PythonRNGFunctions(py_create, py_uniform, py_bits, py_toString);
}
