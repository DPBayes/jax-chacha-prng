// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2021 Aalto University

#include <iostream>

#include "tests.hpp"
#include "cpu_kernel.hpp"

#ifdef ARCH_INTEL
inline __m128i from_array(std::array<uint32_t, 4>& vals)
{
    return _mm_load_si128(reinterpret_cast<__m128i*>(vals.data()));
}

inline std::array<uint32_t, 4> to_array(__m128i x)
{
    std::array<uint32_t, 4> vals;
    _mm_store_si128(reinterpret_cast<__m128i*>(vals.data()), x);
    return vals;
}
#endif

#ifdef ARCH_ARM

inline uint32x4_t from_array(std::array<uint32_t, 4>& vals)
{
    return vld1q_u32(vals.data());
}

inline std::array<uint32_t, 4> to_array(uint32x4_t x)
{
    std::array<uint32_t, 4> vals;
    vst1q_u32(vals.data(), x);
    return vals;
}

#endif

int test_vectorize_unvectorize()
{
    std::cout << "test_vectorize_unvectorize(16 element array): ";
    int num_fails = 0;
    std::array<uint32_t, 16> vals({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    std::array<uint32_t, 16> expected(vals);
    std::array<uint32_t, 16> result;

    VectorizedState state(vals.data());
    state.unvectorize(result.data());

    num_fails += test_assert(result == expected);

    std::cout << "test_vectorize_unvectorize(4 vectors): ";

    std::array<uint32_t, 4> first_row({1, 2, 3, 4});
    std::array<uint32_t, 4> second_row({5, 6, 7, 8});
    std::array<uint32_t, 4> third_row({9, 10, 11, 12});
    std::array<uint32_t, 4> fourth_row({13, 14, 15, 16});

    state = VectorizedState(
        from_array(first_row), from_array(second_row), from_array(third_row), from_array(fourth_row)
    );
    state.unvectorize(result.data());
    num_fails += test_assert(result == expected);

    return num_fails;
}

int test_rotate_elements_left()
{
    int num_fails = 0;
    std::array<uint32_t, 4> vals({1, 2, 3, 4});
    std::array<uint32_t, 4> expected;
    std::array<uint32_t, 4> result;

    auto vec_vals = from_array(vals);

    std::cout << "test_rotate_elements_left(0): ";
    expected = vals;
    result = to_array(rotate_elements_left<0>(vec_vals));
    num_fails += test_assert(result == expected);

    std::cout << "test_rotate_elements_left(1): ";
    expected = {2, 3, 4, 1};
    result = to_array(rotate_elements_left<1>(vec_vals));
    num_fails += test_assert(result == expected);

    std::cout << "test_rotate_elements_left(3): ";
    expected = {4, 1, 2, 3};
    result = to_array(rotate_elements_left<3>(vec_vals));
    num_fails += test_assert(result == expected);

    return num_fails;
}

// int test_rotate_left()
// {
//     int num_fails = 0;

//     std::array<uint32_t, 4> vals({ 1, 0xffffffff, 0x0f0f0f0f, 0xcccccccc });
//     __m128i vec_vals = from_array(vals);

//     std::cout << "test_rotate_left(x, 0):\t\t";
//     std::array<uint32_t, 4> expected(vals);

//     __m128i vec_result = rotate_left(vec_vals, 0);
//     std::array<uint32_t, 4> result = to_array<uint32_t>(vec_result);

//     num_fails += test_assert(vals == result);

//     std::cout << "test_rotate_left(x, 1):\t\t";
//     expected = { 2, 0xffffffff, 0x1e1e1e1e, 0x99999999 };

//     vec_result = rotate_left(vec_vals, 1);
//     result = to_array<uint32_t>(vec_result);

//     num_fails += test_assert(result == expected);

//     std::cout << "test_rotate_left(x, 32):\t";
//     expected = vals;

//     vec_result = rotate_left(vec_vals, 32);
//     result = to_array<uint32_t>(vec_result);

//     num_fails += test_assert(result == expected);

//     return num_fails;
// }

int test_chacha20_block()
{
    int num_fails = 0;
    for (int i = 0; i < test_vector_states.size(); ++i)
    {
        std::array<uint32_t, ChaChaStateSizeInWords> result;

        std::cout << "test_chacha20_block(vector " << i << "): ";
        chacha20_block(result.data(), test_vector_states[i].data());
        num_fails += test_assert(result == test_vector_expected[i]);
    }

    return num_fails;
}

int test_cpu_chacha20_block()
{
    int num_fails = 0;
    std::cout << "test_cpu_chacha20_block(single input): ";
    std::array<uint32_t, ChaChaStateSizeInWords> single_block_input = test_vector_states[0];
    std::array<uint32_t, ChaChaStateSizeInWords> single_block_output;
    uint32_t num_inputs = 1;

    std::array<const void*, 2> inputs({&num_inputs, single_block_input.data()});

    cpu_chacha20_block(single_block_output.data(), inputs.data());

    num_fails += test_assert(single_block_output == test_vector_expected[0]);

    std::cout << "test_cpu_chacha20_block(double input, same): ";
    std::array<uint32_t, 4*ChaChaStateSizeInWords> multiple_block_input;
    std::array<uint32_t, 4*ChaChaStateSizeInWords> multiple_block_output;
    num_inputs = 2;
    auto it = std::copy(test_vector_states[0].cbegin(), test_vector_states[0].cend(),
        multiple_block_input.begin());
    it = std::copy(test_vector_states[0].cbegin(), test_vector_states[0].cend(), it);

    inputs = {&num_inputs, multiple_block_input.data()};

    cpu_chacha20_block(multiple_block_output.data(), inputs.data());

    num_fails += test_assert(
        std::equal(test_vector_expected[0].cbegin(), test_vector_expected[0].cend(),
            multiple_block_output.cbegin()
        ) &&
        std::equal(test_vector_expected[0].cbegin(), test_vector_expected[0].cend(),
            multiple_block_output.cbegin() + ChaChaStateSizeInWords
        )
    );

    std::cout << "test_cpu_chacha20_block(double input, diff): ";
    num_inputs = 2;
    it = std::copy(test_vector_states[1].cbegin(), test_vector_states[1].cend(),
        multiple_block_input.begin());
    it = std::copy(test_vector_states[3].cbegin(), test_vector_states[3].cend(), it);

    inputs = {&num_inputs, multiple_block_input.data()};

    cpu_chacha20_block(multiple_block_output.data(), inputs.data());

    num_fails += test_assert(
        std::equal(test_vector_expected[1].cbegin(), test_vector_expected[1].cend(),
            multiple_block_output.cbegin()
        ) &&
        std::equal(test_vector_expected[3].cbegin(), test_vector_expected[3].cend(),
            multiple_block_output.cbegin() + ChaChaStateSizeInWords
        )
    );

    std::cout << "test_cpu_chacha20_block(quad input): ";
    num_inputs = 4;
    it = std::copy(test_vector_states[1].cbegin(), test_vector_states[1].cend(),
        multiple_block_input.begin());
    it = std::copy(test_vector_states[3].cbegin(), test_vector_states[3].cend(), it);
    it = std::copy(test_vector_states[2].cbegin(), test_vector_states[2].cend(), it);
    it = std::copy(test_vector_states[5].cbegin(), test_vector_states[5].cend(), it);

    inputs = {&num_inputs, multiple_block_input.data()};

    cpu_chacha20_block(multiple_block_output.data(), inputs.data());

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
    int num_fails = 0;
    // test_rotate_left();
    num_fails += test_vectorize_unvectorize();
    num_fails += test_rotate_elements_left();
    num_fails += test_cpu_chacha20_block();
    num_fails += test_cpu_chacha20_block();
    return (num_fails > 0);
}
