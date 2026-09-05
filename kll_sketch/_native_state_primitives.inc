constexpr std::uint64_t U64_MAX = std::numeric_limits<std::uint64_t>::max();
constexpr std::uint64_t GAMMA = 0x9E3779B97F4A7C15ULL;
constexpr std::uint64_t M1 = 0xBF58476D1CE4E5B9ULL;
constexpr std::uint64_t M2 = 0x94D049BB133111EBULL;
constexpr int MAX_CAPACITY_DEPTH = 39;
constexpr const char* CAPSULE_NAME = "kll_sketch._native.SketchState";

inline std::uint64_t next_u64(std::uint64_t& state) {
    state += GAMMA;
    std::uint64_t z = state;
    z = (z ^ (z >> 30U)) * M1;
    z = (z ^ (z >> 27U)) * M2;
    return z ^ (z >> 31U);
}
inline int next_bit(std::uint64_t& state) { return static_cast<int>(next_u64(state) & 1ULL); }

inline void insertion_sort(double* data, std::size_t n) {
    for (std::size_t i = 1; i < n; ++i) {
        const double key = data[i];
        std::size_t j = i;
        while (j > 0 && key < data[j - 1]) {
            data[j] = data[j - 1];
            --j;
        }
        data[j] = key;
    }
}

inline void sort_level(std::vector<double>& values) {
    if (values.size() <= 32U) {
        insertion_sort(values.data(), values.size());
        return;
    }
    if (std::is_sorted(values.begin(), values.end())) return;
    bool has_zero = false;
    for (double value : values) {
        if (value == 0.0) { has_zero = true; break; }
    }
    if (has_zero) std::stable_sort(values.begin(), values.end());
    else std::sort(values.begin(), values.end());
}

bool py_double(PyObject* obj, double& out) {
    out = PyFloat_AsDouble(obj);
    if (PyErr_Occurred()) return false;
    if (!std::isfinite(out)) {
        PyErr_SetString(PyExc_ValueError, "native value must be finite");
        return false;
    }
    return true;
}

bool parse_levels(PyObject* obj, std::vector<std::vector<double>>& levels) {
    PyObject* outer = PySequence_Fast(obj, "levels must be a sequence");
    if (!outer) return false;
    const Py_ssize_t count = PySequence_Fast_GET_SIZE(outer);
    levels.reserve(static_cast<std::size_t>(count));
    for (Py_ssize_t h = 0; h < count; ++h) {
        PyObject* inner = PySequence_Fast(PySequence_Fast_GET_ITEM(outer, h), "each level must be a sequence");
        if (!inner) { Py_DECREF(outer); return false; }
        const Py_ssize_t n = PySequence_Fast_GET_SIZE(inner);
        std::vector<double> level;
        level.reserve(static_cast<std::size_t>(n));
        for (Py_ssize_t i = 0; i < n; ++i) {
            double value = 0.0;
            if (!py_double(PySequence_Fast_GET_ITEM(inner, i), value)) {
                Py_DECREF(inner); Py_DECREF(outer); return false;
            }
            level.push_back(value);
        }
        Py_DECREF(inner);
        levels.push_back(std::move(level));
    }
    Py_DECREF(outer);
    return true;
}

PyObject* levels_to_python(const std::vector<std::vector<double>>& levels) {
    PyObject* outer = PyList_New(static_cast<Py_ssize_t>(levels.size()));
    if (!outer) return nullptr;
    for (Py_ssize_t h = 0; h < static_cast<Py_ssize_t>(levels.size()); ++h) {
        const auto& src = levels[static_cast<std::size_t>(h)];
        PyObject* inner = PyList_New(static_cast<Py_ssize_t>(src.size()));
        if (!inner) { Py_DECREF(outer); return nullptr; }
        for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(src.size()); ++i) {
            PyObject* value = PyFloat_FromDouble(src[static_cast<std::size_t>(i)]);
            if (!value) { Py_DECREF(inner); Py_DECREF(outer); return nullptr; }
            PyList_SET_ITEM(inner, i, value);
        }
        PyList_SET_ITEM(outer, h, inner);
    }
    return outer;
}

bool level_capacity(std::uint64_t k, std::size_t h, std::size_t level_count,
                    std::uint64_t minimum, std::uint64_t& out) {
    if (h >= level_count) {
        PyErr_SetString(PyExc_IndexError, "native level out of range");
        return false;
    }
    const int depth = static_cast<int>(level_count - h - 1U);
    if (depth > MAX_CAPACITY_DEPTH) {
        PyErr_SetString(PyExc_OverflowError, "native capacity depth exceeded");
        return false;
    }
    std::uint64_t numerator = k;
    std::uint64_t denominator = 1;
    for (int i = 0; i < depth; ++i) {
        if (numerator > U64_MAX / 2ULL || denominator > U64_MAX / 3ULL) {
            PyErr_SetString(PyExc_OverflowError, "native capacity arithmetic exceeded");
            return false;
        }
        numerator *= 2ULL;
        denominator *= 3ULL;
    }
    const std::uint64_t half = denominator / 2ULL;
    if (numerator > U64_MAX - half) {
        PyErr_SetString(PyExc_OverflowError, "native capacity rounding exceeded");
        return false;
    }
    out = std::max(minimum, (numerator + half) / denominator);
    return true;
}

#if KLL_STATE_AVX2
inline bool cpu_avx2() {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
}

__attribute__((target("avx2")))
bool scan_values_avx2(const double* data, std::size_t n, double& lo, double& hi) {
    const __m256d sign = _mm256_set1_pd(-0.0);
    const __m256d maxf = _mm256_set1_pd(std::numeric_limits<double>::max());
    const __m256d zero = _mm256_setzero_pd();
    __m256d vlo = _mm256_set1_pd(std::numeric_limits<double>::infinity());
    __m256d vhi = _mm256_set1_pd(-std::numeric_limits<double>::infinity());
    bool saw_zero = false;
    std::size_t i = 0;
    for (; i + 4U <= n; i += 4U) {
        const __m256d x = _mm256_loadu_pd(data + i);
        const __m256d ax = _mm256_andnot_pd(sign, x);
        const __m256d finite = _mm256_cmp_pd(ax, maxf, _CMP_LE_OQ);
        if (_mm256_movemask_pd(finite) != 0xF) return false;
        const __m256d is_zero = _mm256_cmp_pd(ax, zero, _CMP_EQ_OQ);
        saw_zero = saw_zero || (_mm256_movemask_pd(is_zero) != 0);
        vlo = _mm256_min_pd(vlo, x);
        vhi = _mm256_max_pd(vhi, x);
    }

    alignas(32) double los[4];
    alignas(32) double his[4];
    _mm256_store_pd(los, vlo);
    _mm256_store_pd(his, vhi);
    double local_lo = los[0];
    double local_hi = his[0];
    for (int lane = 1; lane < 4; ++lane) {
        if (los[lane] < local_lo) local_lo = los[lane];
        if (his[lane] > local_hi) local_hi = his[lane];
    }
    for (; i < n; ++i) {
        const double x = data[i];
        if (!std::isfinite(x)) return false;
        saw_zero = saw_zero || (x == 0.0);
        if (x < local_lo) local_lo = x;
        if (x > local_hi) local_hi = x;
    }

    if (!saw_zero) {
        lo = local_lo;
        hi = local_hi;
        return true;
    }

    // AVX min/max is free to choose either sign for equal zero values.  The
    // canonical Python implementation retains the first equal extremum, so
    // only zero-containing batches need this rare scalar tie-preserving pass.
    lo = data[0];
    hi = data[0];
    for (std::size_t j = 1; j < n; ++j) {
        const double x = data[j];
        if (x < lo) lo = x;
        if (x > hi) hi = x;
    }
    return true;
}
#endif

bool scan_values(const double* data, std::size_t n, double& lo, double& hi) {
    if (n == 0U) return true;
#if KLL_STATE_AVX2
    if (cpu_avx2()) return scan_values_avx2(data, n, lo, hi);
#endif
    lo = data[0];
    hi = data[0];
    if (!std::isfinite(data[0])) return false;
    for (std::size_t i = 1; i < n; ++i) {
        const double x = data[i];
        if (!std::isfinite(x)) return false;
        if (x < lo) lo = x;
        if (x > hi) hi = x;
    }
    return true;
}
