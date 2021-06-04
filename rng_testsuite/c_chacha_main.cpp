// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2021 Aalto University

#include "chacha_reference/chacharng.hpp"
#include "argparse.h"

const int RANDOMNESS_BATCH_SIZE = 16000;

const int RETURN_SUCCESS = 0;
const int RETURN_ARGUMENT_ERROR = 1;
const int RETURN_OTHER_ERROR = 2;

std::vector<uint8_t> ParseSeed(const std::string& hexSeed)
{
    std::vector<uint8_t> bytes(KEY_LENGTH_IN_BYTES);
    for (size_t i = 0; i < hexSeed.length() && i < KEY_LENGTH_IN_BYTES; i += 2)
    {
        auto byteString = hexSeed.substr(i, 2);
        bytes[i] = static_cast<uint8_t>(std::stoul(byteString, nullptr, 16));
    }
    return bytes;
}


enum TestBattery
{
    SmallCrush,
    Crush,
    BigCrush,
    FIPS1402
};

TestBattery ParseTestBattery(const std::string& testBatteryArg)
{
    if (testBatteryArg == "small-crush") return TestBattery::SmallCrush;
    if (testBatteryArg == "crush") return TestBattery::Crush;
    if (testBatteryArg == "big-crush") return TestBattery::BigCrush;
    if (testBatteryArg == "fips-140-2") return TestBattery::FIPS1402;
    throw std::invalid_argument("testBatteryArg");
}

int main(int argc, const char** argv)
{
    argparse::ArgumentParser parser("chacha_rng_tests", "Test suite for ChaCha20 based RNG (using Bernstein's reference implementation) based on TestU01");
    parser.add_argument()
        .name("test-battery")
        .description("Which TestU01 battery to run. One of small-crush, crush, big-crush, fips-140-2 .")
        .required(true)
        .position(0);
    parser.add_argument()
        .name("--seed")
        .description("Seed for the RNG as hex-string")
        .required(false);
    parser.enable_help();

    auto err = parser.parse(argc, argv);
    if (err)
    {
        std::cerr << err << std::endl;
        return RETURN_ARGUMENT_ERROR;
    }

    if (parser.exists("help"))
    {
        parser.print_help();
        return RETURN_SUCCESS;
    }

    TestBattery battery;
    std::string testBatteryArg = parser.get<std::string>("test-battery");
    try
    {
        battery = ParseTestBattery(testBatteryArg);
    }
    catch (const std::invalid_argument&)
    {
        std::cerr << "test battery must be one of small-crush, crush, big-crush, fips-140-2 ." << std::endl;
        return RETURN_ARGUMENT_ERROR;
    }

    std::vector<uint8_t> seed(KEY_LENGTH_IN_BYTES);
    std::fill(seed.begin(), seed.end(), 0);
    if (parser.exists("seed"))
    {
        std::string seedArg = parser.get<std::string>("seed");
        try
        {
            seed = ParseSeed(seedArg);
        }
        catch (const std::invalid_argument& e)
        {
            std::cerr << "Could not parse seed: " << seedArg << std::endl;
            std::cerr << e.what() << std::endl;
            return RETURN_ARGUMENT_ERROR;
        }
    }

    try
    {
        auto chachaGen = chacha_CreateRNG(seed, RANDOMNESS_BATCH_SIZE);

        switch (battery)
        {
            case TestBattery::SmallCrush: { bbattery_SmallCrush(&chachaGen); break; }
            case TestBattery::Crush: { bbattery_Crush(&chachaGen); break; }
            case TestBattery::BigCrush: { bbattery_Crush(&chachaGen); break; }
            case TestBattery::FIPS1402: { bbattery_FIPS_140_2(&chachaGen); break; }
        }
        chacha_ClearRNG(chachaGen);
    }
    catch (const std::string& e)
    {
        std::cerr << e << std::endl;
        return RETURN_OTHER_ERROR;
    }

    return RETURN_SUCCESS;
}

