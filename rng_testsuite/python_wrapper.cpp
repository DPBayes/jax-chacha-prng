#include "python_wrapper.hpp"
#include <utility>
#include <sstream>

ScopedPyObject::ScopedPyObject(PyObject* object, bool owns) : _object(object), _owns(owns) { }

ScopedPyObject::ScopedPyObject() : _object(nullptr), _owns(false) { }

ScopedPyObject::ScopedPyObject(const ScopedPyObject& other) : _object(other._object), _owns(other._owns)
{
    if (_owns)
    {
        Py_XINCREF(_object);
    }
}

ScopedPyObject::ScopedPyObject(ScopedPyObject&& other) : _object()
{
    swap(other);
}


ScopedPyObject ScopedPyObject::own(PyObject* object) { return ScopedPyObject(object, true); }

ScopedPyObject ScopedPyObject::borrow(PyObject* object) { return ScopedPyObject(object, false); }

ScopedPyObject::~ScopedPyObject()
{
    if (_owns) Py_XDECREF(_object);
    _object = nullptr;
    _owns = false;
}

void ScopedPyObject::swap(ScopedPyObject& other)
{
    std::swap(_object, other._object);
    std::swap(_owns, other._owns);
}

ScopedPyObject& ScopedPyObject::operator=(ScopedPyObject other)
{
    swap(other);
    return *this;
}

ScopedPyObject::operator PyObject*() const { return _object; }

bool ScopedPyObject::operator==(const ScopedPyObject& other) { return _object == other._object; }

void ScopedPyObject::makeOwned()
{
    if (!_owns) {
        Py_XINCREF(_object);
        _owns = true;
    }
}

std::string ScopedPyObject::toString() const
{
    std::stringstream ss;
    ss << "ScopedPyObject(" << _object << (_owns ? ", owned" : ", borrowed") << ")";
    return ss.str();
}

bool ScopedPyObject::isValid() const { return _object != nullptr; }

PythonContext::PythonContext() : _controlling(false) { }

PythonContext::~PythonContext()
{
    if (_controlling) Py_Finalize();
}

PythonContext PythonContext::initialize()
{
    Py_Initialize();
    PythonContext env;
    env._controlling = true;
    return env;
}

PythonContext::PythonContext(PythonContext&& other) : PythonContext()
{
    std::swap(_controlling, other._controlling);
}
