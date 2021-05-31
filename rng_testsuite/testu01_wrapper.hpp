#pragma once

extern "C" {
#include "TestU01.h"
}

#include "python_rng.hpp"


// actual typesizes expected by TestU01 framework
typedef uint32_t testu01_bitsequence_t; //.. although it specifies "unsigned long", it wants 32 bits always
typedef double testu01_uniform_t;

struct PythonRNGState
{
    const PythonRNGFunctions functions;
    const size_t valuesPerBatch;
    const std::string name;

    PythonRNGFunctions::PRNGContext ctx;
    std::vector<testu01_uniform_t> uniformBuffer;
    size_t uniformBufferCursor;
    std::vector<testu01_bitsequence_t> bitBuffer;
    size_t bitBufferCursor;
    size_t numCalls;
    PythonRNGState(PythonRNGFunctions functions, PythonRNGFunctions::PRNGContext ctx, std::string name, size_t valuesPerBatch)
        : functions(functions)
        , valuesPerBatch(valuesPerBatch)
        , name(name)
        , ctx(ctx)
        , uniformBuffer()
        , uniformBufferCursor(0)
        , bitBuffer()
        , bitBufferCursor(0)
        , numCalls(0)
    { }
};

unif01_Gen CreatePythonPRNG(
    const PythonRNGFunctions& pythonRNGFunctions, const std::vector<uint8_t>& seed, std::string name, size_t valuesPerBatch
);
void ClearPythonRNG(unif01_Gen& gen);
