#pragma once
// Based on https://github.com/rygorous/ryg_rans/blob/master/rans64.h

#include <random>
#include <stdexcept>
#include <vector>
#include <inttypes.h>
#include <stdio.h>
#include <iostream>

typedef __uint128_t uint128_t;
constexpr uint128_t ONE = 1;
constexpr uint128_t RANS_L = ONE << 31;
constexpr int S_BITS = 64;

struct ANSBitstream {
    std::vector<uint64_t> stream;
    uint128_t tip;
    int mass_bits; // distribution masses assumed to sum to 1<<mass_bits

    ANSBitstream(int mass_bits_) : tip(RANS_L), mass_bits(mass_bits_) {
        if (mass_bits < 1 || mass_bits > 63) {
            throw std::runtime_error("mass_bits must be in [1, 63]");
        }
    }

    void print_stream() {
        for (int i = 0; i < stream.size(); ++i){
            printf("%" PRIu64 "\n", uint64_t (stream[i] & ((ONE << S_BITS) - 1)));
        }
        
    }

    void encode(uint64_t pmf, uint64_t cdf) {
        //For debug
        // printf("%" PRIu64 "\n", peek());
        // printf("%" PRIu64, pmf);
        // printf("     ");
        // printf("%" PRIu64, cdf);
        // printf("\n\n");
        
        if (tip >= ((RANS_L >> mass_bits) << S_BITS) * pmf) {
            stream.push_back(tip);
            tip >>= S_BITS;
        }
        tip = ((tip / pmf) << mass_bits) + (tip % pmf) + cdf;
        
    }

    void decode(uint64_t peeked, uint64_t pmf, uint64_t cdf) {
        tip = pmf * (tip >> mass_bits) + peeked - cdf;
        if (tip < RANS_L) {
            if (stream.empty()) {
                throw std::runtime_error("Empty bitstream!");
            }
            tip = (tip << S_BITS) | stream.back();
            stream.pop_back();
        }
    }

    uint64_t peek() const {
        return tip & ((ONE << S_BITS) - 1);
    }

    size_t tip_length() const {
        // count bits in the tip
        size_t size = 0;
        uint128_t tip_copy = tip;
        while (tip_copy != 0) {
            ++size;
            tip_copy >>= 1;
        }
        return size;
    }

    size_t length() const { // length in bits
        return S_BITS * stream.size() + tip_length();
    }
};
