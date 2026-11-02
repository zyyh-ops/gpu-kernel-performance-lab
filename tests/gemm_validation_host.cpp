#include "gemm_validation.h"
#include <cassert>
#include <limits>

int main() {
    // Non-symmetric operands detect accidental operand swaps / transposes.
    const std::vector<float> a{1,2,3,4}, b{5,6,7,8};
    const auto ref=cpu_gemm_reference(a,b,2);
    assert((ref==std::vector<double>{19,22,43,50}));
    assert(compare_gemm({19,22,43,50},ref,0.005,0.001).passed());
    assert(compare_gemm({23,34,31,46},ref,0.005,0.001).mismatches==4);
    assert(compare_gemm({19,43,22,50},ref,0.005,0.001).mismatches==2);

    // Absolute tolerance near zero, relative tolerance away from zero.
    auto inside=compare_gemm({0.004f,100.09f},{0.,100.},0.005,0.001);
    assert(inside.passed() && inside.max_scaled_error<1.);
    auto outside=compare_gemm({0.006f,100.2f},{0.,100.},0.005,0.001);
    assert(!outside.passed() && outside.mismatches==2 && outside.max_scaled_error>1.);

    // NaN poison, missing writes and a broken reference must never pass.
    const float nan=std::numeric_limits<float>::quiet_NaN();
    const float inf=std::numeric_limits<float>::infinity();
    auto bad=compare_gemm({nan,inf,1.f},{0.,0.,std::numeric_limits<double>::quiet_NaN()},0.005,0.001);
    assert(bad.mismatches==3 && bad.nonfinite==3);
    assert(std::isfinite(bad.max_abs) && std::isfinite(bad.max_scaled_error));
    assert(!compare_gemm({0,0,0,0},ref,0.005,0.001).passed());

    bool rejected=false;
    try { compare_gemm({1},{},0.005,0.001); }
    catch(const std::invalid_argument&) { rejected=true; }
    assert(rejected);
    rejected=false;
    try { compare_gemm({1},{1},0.,0.001); }
    catch(const std::invalid_argument&) { rejected=true; }
    assert(rejected);
}
