// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2021 Aalto University

// Simple C++ Wrapper for Python objects

#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string>

class PythonContext
{
private:
    bool _controlling;
public:
    PythonContext();
    ~PythonContext();
    static PythonContext initialize();
    PythonContext(const PythonContext&) = delete;
    PythonContext(PythonContext&& other);
};

class ScopedPyObject
{
private:
    PyObject* _object;
    bool _owns;
private:
    ScopedPyObject(PyObject* object, bool owns);

public:
    ScopedPyObject();
    ScopedPyObject(const ScopedPyObject& other);
    ScopedPyObject(ScopedPyObject&& other);
    static ScopedPyObject own(PyObject* object);
    static ScopedPyObject borrow(PyObject* object);
    ~ScopedPyObject();

    void swap(ScopedPyObject& other);

    ScopedPyObject& operator=(ScopedPyObject other);

    operator PyObject*() const;
    bool operator==(const ScopedPyObject& other);

    void makeOwned();

    std::string toString() const;

    bool isValid() const;
};
