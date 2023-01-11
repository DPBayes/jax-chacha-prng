// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2021 Aalto University

#include <iostream>
#include <array>
#include <vector>

#include "tests.hpp"
#include "gpu_kernel.hpp"

template <size_t size>
class DeviceBuffer
{
private:
    uint32_t* _ptr;
public:
    DeviceBuffer() : _ptr(nullptr)
    {
        if (hipMalloc(&_ptr, size * sizeof(uint32_t)) != hipSuccess)
            throw std::runtime_error("Could not allocate device memory!");
    }

    DeviceBuffer(const std::array<uint32_t, size>& data) : DeviceBuffer()
    {
        write(data);
    }

    ~DeviceBuffer()
    {
        if (_ptr != nullptr)
        {
            if (hipFree(_ptr) == hipSuccess)
                _ptr = nullptr;
        }
    }

    void write(const std::array<uint32_t, size>& data)
    {
        if (hipMemcpy(_ptr, data.data(), size * sizeof(uint32_t), hipMemcpyHostToDevice) != hipSuccess)
            throw std::runtime_error("Failed to copy data to device memory!");
    }

    void read(std::array<uint32_t, size>& buffer)
    {
        if (hipMemcpy(buffer.data(), _ptr, size * sizeof(uint32_t), hipMemcpyDeviceToHost) != hipSuccess)
            throw std::runtime_error("Failed to copy data from device memory!");

    }

    std::array<uint32_t, size> read()
    {
        std::array<uint32_t, size> buffer;
        read(buffer);
        return buffer;
    }

    uint32_t* raw() { return _ptr; }
};

int test_chacha20_block_with_shuffle()
{
    int num_fails = 0;
    DeviceBuffer<ChaChaStateSizeInWords> input_device;
    DeviceBuffer<ChaChaStateSizeInWords> result_device;
    std::array<uint32_t, ChaChaStateSizeInWords> result;

    for (int i = 0; i < test_vector_states.size(); ++i)
    {
        std::cout << "test_chacha20_block_with_shuffle(vector " << i << "): ";
        input_device.write(test_vector_states[i]);

        chacha20_block_with_shuffle<<<1, 4>>>(result_device.raw(), input_device.raw(), /*num_threads=*/4);
        result_device.read(result);
        num_fails += test_assert(result == test_vector_expected[i]);
    }

    return num_fails;
}

int test_gpu_chacha20_block()
{
    int num_fails = 0;
    std::cout << "test_gpu_chacha20_block(single input): ";
    DeviceBuffer<ChaChaStateSizeInWords> single_block_input_device(test_vector_states[0]);
    DeviceBuffer<ChaChaStateSizeInWords> single_block_output_device;
    uint32_t num_inputs = 1;

    std::array<void*, 2> buffers = { single_block_input_device.raw(), single_block_output_device.raw() };

    gpu_chacha20_block(
        /*stream=*/nullptr, buffers.data(), /*opaque=*/reinterpret_cast<const char*>(&num_inputs),
        /*opaque_length=*/sizeof(num_inputs)
    );

    std::array<uint32_t, ChaChaStateSizeInWords> single_block_output = single_block_output_device.read();
    num_fails += test_assert(single_block_output == test_vector_expected[0]);

    std::cout << "test_gpu_chacha20_block(double input, same): ";
    std::array<uint32_t, 4*ChaChaStateSizeInWords> multiple_block_input;
    std::array<uint32_t, 4*ChaChaStateSizeInWords> multiple_block_output;
    num_inputs = 2;
    auto it = std::copy(test_vector_states[0].cbegin(), test_vector_states[0].cend(),
        multiple_block_input.begin());
    it = std::copy(test_vector_states[0].cbegin(), test_vector_states[0].cend(), it);

    DeviceBuffer<4*ChaChaStateSizeInWords> multiple_block_input_device(multiple_block_input);
    DeviceBuffer<4*ChaChaStateSizeInWords> multiple_block_output_device;

    buffers = { multiple_block_input_device.raw(), multiple_block_output_device.raw() };

    gpu_chacha20_block(
        /*stream=*/nullptr, buffers.data(), /*opaque=*/reinterpret_cast<const char*>(&num_inputs),
        /*opaque_length=*/sizeof(num_inputs)
    );

    multiple_block_output_device.read(multiple_block_output);

    num_fails += test_assert(
        std::equal(test_vector_expected[0].cbegin(), test_vector_expected[0].cend(),
            multiple_block_output.cbegin()
        ) &&
        std::equal(test_vector_expected[0].cbegin(), test_vector_expected[0].cend(),
            multiple_block_output.cbegin() + ChaChaStateSizeInWords
        )
    );

    std::cout << "test_gpu_chacha20_block(double input, diff): ";
    num_inputs = 2;
    it = std::copy(test_vector_states[1].cbegin(), test_vector_states[1].cend(),
        multiple_block_input.begin());
    it = std::copy(test_vector_states[3].cbegin(), test_vector_states[3].cend(), it);
    multiple_block_input_device.write(multiple_block_input);

    buffers = { multiple_block_input_device.raw(), multiple_block_output_device.raw() };

    gpu_chacha20_block(
        /*stream=*/nullptr, buffers.data(), /*opaque=*/reinterpret_cast<const char*>(&num_inputs),
        /*opaque_length=*/sizeof(num_inputs)
    );

    multiple_block_output_device.read(multiple_block_output);

    num_fails += test_assert(
        std::equal(test_vector_expected[1].cbegin(), test_vector_expected[1].cend(),
            multiple_block_output.cbegin()
        ) &&
        std::equal(test_vector_expected[3].cbegin(), test_vector_expected[3].cend(),
            multiple_block_output.cbegin() + ChaChaStateSizeInWords
        )
    );

    std::cout << "test_gpu_chacha20_block(quad input): ";
    num_inputs = 4;
    it = std::copy(test_vector_states[1].cbegin(), test_vector_states[1].cend(),
        multiple_block_input.begin());
    it = std::copy(test_vector_states[3].cbegin(), test_vector_states[3].cend(), it);
    it = std::copy(test_vector_states[2].cbegin(), test_vector_states[2].cend(), it);
    it = std::copy(test_vector_states[5].cbegin(), test_vector_states[5].cend(), it);
    multiple_block_input_device.write(multiple_block_input);

    buffers = { multiple_block_input_device.raw(), multiple_block_output_device.raw() };

    gpu_chacha20_block(
        /*stream=*/nullptr, buffers.data(), /*opaque=*/reinterpret_cast<const char*>(&num_inputs),
        /*opaque_length=*/sizeof(num_inputs)
    );

    multiple_block_output_device.read(multiple_block_output);

    num_fails += test_assert(
        std::equal(test_vector_expected[1].cbegin(), test_vector_expected[1].cend(),
            multiple_block_output.cbegin()
        ) &&
        std::equal(test_vector_expected[3].cbegin(), test_vector_expected[3].cend(),
            multiple_block_output.cbegin() + ChaChaStateSizeInWords
        ) &&
        std::equal(test_vector_expected[2].cbegin(), test_vector_expected[2].cend(),
            multiple_block_output.cbegin() + 2*ChaChaStateSizeInWords
        ) &&
        std::equal(test_vector_expected[5].cbegin(), test_vector_expected[5].cend(),
            multiple_block_output.cbegin() + 3*ChaChaStateSizeInWords
        )
    );

    return num_fails;
}


int main(int argc, const char** argv)
{
    try
    {
        int num_fails = 0;
        num_fails += test_chacha20_block_with_shuffle();
        num_fails += test_gpu_chacha20_block();
        return (num_fails > 0);
    }
    catch (std::runtime_error& err)
    {
        std::cout << err.what() << std::endl;
        return -1;
    }
}
