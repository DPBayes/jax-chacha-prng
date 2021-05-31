#include "testu01_wrapper.hpp"

#include <iostream>

unsigned long pythonRNG_GetBits(void* param, void* state)
{
    auto rngState = static_cast<PythonRNGState*>(state);
    if (rngState->bitBufferCursor >= rngState->bitBuffer.size())
    {
        auto randomsAndState = rngState->functions.randomBits<testu01_bitsequence_t>(
            rngState->ctx, rngState->valuesPerBatch
        );
        rngState->bitBuffer = std::move(randomsAndState.first);
        rngState->ctx = std::move(randomsAndState.second);
        rngState->bitBufferCursor = 0;
    }
    testu01_bitsequence_t val = rngState->bitBuffer[rngState->bitBufferCursor];
    rngState->bitBufferCursor++;
    rngState->numCalls++;

    return val;
}

testu01_uniform_t pythonRNG_GetU01(void* param, void* state)
{
    // bitsequence_t bits = pythonRNG_GetBits(param, state);
    // return unif01_INV32 * bits;
    auto rngState = static_cast<PythonRNGState*>(state);
    if (rngState->uniformBufferCursor >= rngState->uniformBuffer.size())
    {
        auto randomsAndState = rngState->functions.uniform<testu01_uniform_t>(rngState->ctx, rngState->valuesPerBatch);
        rngState->uniformBuffer = std::move(randomsAndState.first);
        rngState->ctx = std::move(randomsAndState.second);
        rngState->uniformBufferCursor = 0;
    }
    testu01_uniform_t val = rngState->uniformBuffer[rngState->uniformBufferCursor];
    rngState->uniformBufferCursor++;
    rngState->numCalls++;

    return val;
}

void pythonRNG_write(void* state)
{
    auto rngState = static_cast<PythonRNGState*>(state);

    std::cout << rngState->name << " (numCalls = " << rngState->numCalls << ")" << std::endl;
    std::wcout << rngState->functions.toString(rngState->ctx) << std::endl;
}

unif01_Gen CreatePythonPRNG(const PythonRNGFunctions& rngFunctions, ScopedPyObject ctx, std::string name, size_t valuesPerBatch)
{
    ctx.makeOwned();
    PythonRNGState* state = new PythonRNGState(rngFunctions, ctx, name, valuesPerBatch);
    unif01_Gen gen;
    gen.GetBits = pythonRNG_GetBits;
    gen.GetU01 = pythonRNG_GetU01;
    gen.Write = pythonRNG_write;
    size_t nameLength = name.length();
    gen.name = new char[nameLength + 1];
    memset(gen.name, 0, nameLength + 1);
    strncpy(gen.name, name.c_str(), nameLength);
    gen.param = nullptr;
    gen.state = state;
    return gen;
}

unif01_Gen CreatePythonPRNG(
    const PythonRNGFunctions& pythonRNGFunctions, const std::vector<uint8_t>& seed, std::string name, size_t valuesPerBatch
)
{
    PythonRNGFunctions::PRNGContext ctx = pythonRNGFunctions.prngKey(seed);
    return CreatePythonPRNG(pythonRNGFunctions, ctx, name, valuesPerBatch);
}

void ClearPythonRNG(unif01_Gen& gen)
{
    delete static_cast<PythonRNGState*>(gen.state);
    delete[] gen.name;
}
