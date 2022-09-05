// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2021 Aalto University

#include <iostream>

#include "tests.hpp"
#include "cpu_kernel.hpp"

inline StateRow from_array(const std::array<uint32_t, 4>& vals)
{
    return StateRow(vals.data());
}

inline std::array<uint32_t, 4> to_array(const StateRow& x)
{
    std::array<uint32_t, 4> vals;
    x.unvectorize(vals.data());
    return vals;
}

template <size_t len>
void print_array(const std::array<uint32_t, len>& arr)
{
    for (size_t i = 0; i < len; ++i)
    {
        std::cout << arr[i] << ", ";
    }
    std::cout << std::endl;
}


template <size_t len>
void print_array(const uint32_t arr[len])
{
    for (size_t i = 0; i < len; ++i)
    {
        std::cout << arr[i] << ", ";
    }
    std::cout << std::endl;
}

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

    std::array<uint32_t, 4> first_row_vals({1, 2, 3, 4});
    std::array<uint32_t, 4> second_row_vals({5, 6, 7, 8});
    std::array<uint32_t, 4> third_row_vals({9, 10, 11, 12});
    std::array<uint32_t, 4> fourth_row_vals({13, 14, 15, 16});

    // auto first_row = from_array(first_row_vals);
    // auto second_row = from_array(second_row_vals);
    auto third_row = from_array(third_row_vals);
    auto fourth_row = from_array(fourth_row_vals);

    state = VectorizedState(
        from_array(first_row_vals), from_array(second_row_vals), third_row, fourth_row
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

    auto vals_vec = from_array(vals);

    std::cout << "test_rotate_elements_left(0): ";
    expected = vals;
    result = to_array(vals_vec.rotate_elements_left<0>());
    num_fails += test_assert(result == expected);

    std::cout << "test_rotate_elements_left(1): ";
    expected = {2, 3, 4, 1};
    result = to_array(vals_vec.rotate_elements_left<1>());
    num_fails += test_assert(result == expected);

    std::cout << "test_rotate_elements_left(2): ";
    expected = {3, 4, 1, 2};
    result = to_array(vals_vec.rotate_elements_left<2>());
    num_fails += test_assert(result == expected);

    std::cout << "test_rotate_elements_left(3): ";
    expected = {4, 1, 2, 3};
    result = to_array(vals_vec.rotate_elements_left<3>());
    num_fails += test_assert(result == expected);

    return num_fails;
}

int test_rotate_left()
{
    int num_fails = 0;

    std::array<uint32_t, 4> vals({ 0x80000001, 0xffffffff, 0x0f0f0f0f, 0xcccccccc });
    auto vec_vals = from_array(vals);

    std::cout << "test_rotate_left(x, 1): ";
    std::array<uint32_t, 4> expected = { 3, 0xffffffff, 0x1e1e1e1e, 0x99999999 };

    auto vec_result = vec_vals << 1;
    auto result = to_array(vec_result);

    num_fails += test_assert(result == expected);

    std::cout << "test_rotate_left(x, 15): ";
    expected = { 0x0000c000, 0xffffffff, 0x87878787, 0x66666666 };

    vec_result = vec_vals << 15;
    result = to_array(vec_result);

    num_fails += test_assert(result == expected);

    std::cout << "test_rotate_left(x, 16): ";
    expected = { 0x00018000, 0xffffffff, 0x0f0f0f0f, 0xcccccccc };

    vec_result = vec_vals << 16;
    result = to_array(vec_result);

    num_fails += test_assert(result == expected);

    std::cout << "test_rotate_left(x, 31): ";
    expected = { 0xc0000000, 0xffffffff, 0x87878787, 0x66666666 };

    vec_result = vec_vals << 31;
    result = to_array(vec_result);

    num_fails += test_assert(result == expected);

    return num_fails;
}

int test_add()
{
    int num_fails = 0;

    std::array<uint32_t, 4> left_vals({ 1, 0xffffffff, 0x0f0f0f0f, 0xcccccccc });
    std::array<uint32_t, 4> right_vals({ 1, 1, 0x0f0f0f0f, 0x00cc00cc });

    std::array<uint32_t, 4> expected({ 2, 0, 0x1e1e1e1e , 0xcd98cd98});

    std::cout << "test_add(): ";
    auto left = from_array(left_vals);
    auto right = from_array(right_vals);
    auto vec_result = left + right;
    auto result = to_array(vec_result);

    num_fails += test_assert(result == expected);

    return num_fails;
}

int test_xor()
{
    int num_fails = 0;

    std::array<uint32_t, 4> left_vals({ 1, 0xffffffff, 0x0f0f0f0f, 0xcccccccc });
    std::array<uint32_t, 4> right_vals({ 1, 1, 0xf0f0f0f0, 0x00cc00cc });

    std::array<uint32_t, 4> expected({ 0, 0xfffffffe, 0xffffffff , 0xcc00cc00 });

    std::cout << "test_xor(): ";
    auto left = from_array(left_vals);
    auto right = from_array(right_vals);
    auto vec_result = left ^ right;
    auto result = to_array(vec_result);

    num_fails += test_assert(result == expected);

    return num_fails;
}

int test_pack_diagonals()
{
    int num_fails = 0;

    std::cout << "test_pack_diagonals(): ";
    std::array<uint32_t, 16> vals({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    std::array<uint32_t, 16> expected({1, 2, 3, 4, 6, 7, 8, 5, 11, 12, 9, 10, 16, 13, 14, 15});
    std::array<uint32_t, 16> result;

    auto vec = VectorizedState(vals.data());
    pack_diagonals(vec, vec);
    vec.unvectorize(result.data());
    num_fails += test_assert(result == expected);

    return num_fails;
}


int test_unpack_diagonals()
{
    int num_fails = 0;

    std::cout << "test_unpack_diagonals(): ";
    std::array<uint32_t, 16> expected({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    std::array<uint32_t, 16> vals({1, 2, 3, 4, 6, 7, 8, 5, 11, 12, 9, 10, 16, 13, 14, 15});
    std::array<uint32_t, 16> result;

    auto vec = VectorizedState(vals.data());
    unpack_diagonals(vec, vec);
    vec.unvectorize(result.data());
    num_fails += test_assert(result == expected);

    return num_fails;
}

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
    num_fails += test_rotate_left();
    num_fails += test_add();
    num_fails += test_xor();
    num_fails += test_pack_diagonals();
    num_fails += test_unpack_diagonals();
    num_fails += test_cpu_chacha20_block();
    num_fails += test_cpu_chacha20_block();
    return (num_fails > 0);
}
