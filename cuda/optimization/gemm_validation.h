#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

// Shared by the CUDA runner and host-only regression tests.
struct ErrorMetrics {
    double max_abs = 0.;
    double max_scaled_error = 0.; // |actual-reference| / (atol + rtol*|reference|)
    std::size_t mismatches = 0;
    std::size_t nonfinite = 0;
    bool passed() const { return mismatches == 0; }
};

inline ErrorMetrics compare_gemm(const std::vector<float>& actual,
                                 const std::vector<double>& reference,
                                 double atol, double rtol) {
    if (actual.empty() || actual.size() != reference.size())
        throw std::invalid_argument("invalid comparison size");
    if (!std::isfinite(atol) || !std::isfinite(rtol) || atol <= 0 || rtol < 0)
        throw std::invalid_argument("invalid comparison tolerance");
    ErrorMetrics result;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double got = actual[i], ref = reference[i];
        if (!std::isfinite(got) || !std::isfinite(ref)) {
            ++result.nonfinite;
            ++result.mismatches;
            continue;
        }
        const double error = std::abs(got-ref);
        const double limit = atol + rtol*std::abs(ref);
        result.max_abs = std::max(result.max_abs, error);
        result.max_scaled_error = std::max(result.max_scaled_error, error/limit);
        if (error > limit) ++result.mismatches;
    }
    return result;
}

inline std::vector<double> cpu_gemm_reference(const std::vector<float>& a,
                                             const std::vector<float>& b, int n) {
    if (n <= 0 || a.size() != std::size_t(n)*n || b.size() != a.size())
        throw std::invalid_argument("invalid reference matrix size");
    std::vector<double> result(a.size(), 0.);
    for (int i=0; i<n; ++i)
        for (int j=0; j<n; ++j)
            for (int k=0; k<n; ++k)
                result[std::size_t(i)*n+j] +=
                    double(a[std::size_t(i)*n+k])*double(b[std::size_t(k)*n+j]);
    return result;
}
