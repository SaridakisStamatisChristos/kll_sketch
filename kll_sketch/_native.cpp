#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define KLL_X86 1
#else
#define KLL_X86 0
#endif

#if KLL_X86 && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define KLL_RUNTIME_AVX2 1
#else
#define KLL_RUNTIME_AVX2 0
#endif

namespace {

constexpr std::uint64_t U64_MASK = std::numeric_limits<std::uint64_t>::max();
constexpr std::uint64_t SPLITMIX_GAMMA = 0x9E3779B97F4A7C15ULL;
constexpr std::uint64_t SPLITMIX_M1 = 0xBF58476D1CE4E5B9ULL;
constexpr std::uint64_t SPLITMIX_M2 = 0x94D049BB133111EBULL;
constexpr int MAX_NATIVE_CAPACITY_DEPTH = 39;

inline std::uint64_t next_u64(std::uint64_t& state) {
    state += SPLITMIX_GAMMA;
    std::uint64_t z = state;
    z = ((z ^ (z >> 30U)) * SPLITMIX_M1);
    z = ((z ^ (z >> 27U)) * SPLITMIX_M2);
    return z ^ (z >> 31U);
}

inline int next_bit(std::uint64_t& state) {
    return static_cast<int>(next_u64(state) & 1ULL);
}

bool scan_scalar(const double* data, std::size_t n, double& min_value, double& max_value) {
    if (n == 0) {
        return true;
    }
    double lo = data[0];
    double hi = data[0];
    if (!std::isfinite(lo)) {
        return false;
    }
    for (std::size_t i = 1; i < n; ++i) {
        const double value = data[i];
        if (!std::isfinite(value)) {
            return false;
        }
        lo = std::min(lo, value);
        hi = std::max(hi, value);
    }
    min_value = lo;
    max_value = hi;
    return true;
}

#if KLL_RUNTIME_AVX2
__attribute__((target("avx2")))
bool scan_avx2(const double* data, std::size_t n, double& min_value, double& max_value) {
    if (n == 0) {
        return true;
    }
    std::size_t i = 0;
    const __m256d sign_mask = _mm256_set1_pd(-0.0);
    const __m256d max_finite = _mm256_set1_pd(std::numeric_limits<double>::max());
    __m256d vmin = _mm256_set1_pd(std::numeric_limits<double>::infinity());
    __m256d vmax = _mm256_set1_pd(-std::numeric_limits<double>::infinity());
    for (; i + 4 <= n; i += 4) {
        const __m256d values = _mm256_loadu_pd(data + i);
        const __m256d abs_values = _mm256_andnot_pd(sign_mask, values);
        const __m256d finite = _mm256_cmp_pd(abs_values, max_finite, _CMP_LE_OQ);
        if (_mm256_movemask_pd(finite) != 0xF) {
            return false;
        }
        vmin = _mm256_min_pd(vmin, values);
        vmax = _mm256_max_pd(vmax, values);
    }
    alignas(32) double mins[4];
    alignas(32) double maxs[4];
    _mm256_store_pd(mins, vmin);
    _mm256_store_pd(maxs, vmax);
    double lo = std::min(std::min(mins[0], mins[1]), std::min(mins[2], mins[3]));
    double hi = std::max(std::max(maxs[0], maxs[1]), std::max(maxs[2], maxs[3]));
    for (; i < n; ++i) {
        const double value = data[i];
        if (!std::isfinite(value)) {
            return false;
        }
        lo = std::min(lo, value);
        hi = std::max(hi, value);
    }
    min_value = lo;
    max_value = hi;
    return true;
}

bool cpu_has_avx2() {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
}
#endif

bool scan_buffer(const double* data, std::size_t n, double& min_value, double& max_value) {
#if KLL_RUNTIME_AVX2
    if (cpu_has_avx2()) {
        return scan_avx2(data, n, min_value, max_value);
    }
#endif
    return scan_scalar(data, n, min_value, max_value);
}

bool py_to_double(PyObject* obj, double& out) {
    out = PyFloat_AsDouble(obj);
    if (PyErr_Occurred()) {
        return false;
    }
    if (!std::isfinite(out)) {
        PyErr_SetString(PyExc_ValueError, "native batch contains a non-finite value");
        return false;
    }
    return true;
}

PyObject* vector_to_float_list(const std::vector<double>& values) {
    PyObject* list = PyList_New(static_cast<Py_ssize_t>(values.size()));
    if (!list) {
        return nullptr;
    }
    for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(values.size()); ++i) {
        PyObject* value = PyFloat_FromDouble(values[static_cast<std::size_t>(i)]);
        if (!value) {
            Py_DECREF(list);
            return nullptr;
        }
        PyList_SET_ITEM(list, i, value);
    }
    return list;
}

bool parse_levels(PyObject* levels_obj, std::vector<std::vector<double>>& levels) {
    PyObject* outer = PySequence_Fast(levels_obj, "levels must be a sequence");
    if (!outer) {
        return false;
    }
    const Py_ssize_t level_count = PySequence_Fast_GET_SIZE(outer);
    levels.reserve(static_cast<std::size_t>(level_count));
    for (Py_ssize_t h = 0; h < level_count; ++h) {
        PyObject* inner_obj = PySequence_Fast_GET_ITEM(outer, h);
        PyObject* inner = PySequence_Fast(inner_obj, "each level must be a sequence");
        if (!inner) {
            Py_DECREF(outer);
            return false;
        }
        const Py_ssize_t count = PySequence_Fast_GET_SIZE(inner);
        std::vector<double> level;
        level.reserve(static_cast<std::size_t>(count));
        for (Py_ssize_t i = 0; i < count; ++i) {
            double value = 0.0;
            if (!py_to_double(PySequence_Fast_GET_ITEM(inner, i), value)) {
                Py_DECREF(inner);
                Py_DECREF(outer);
                return false;
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
    if (!outer) {
        return nullptr;
    }
    for (Py_ssize_t h = 0; h < static_cast<Py_ssize_t>(levels.size()); ++h) {
        PyObject* inner = vector_to_float_list(levels[static_cast<std::size_t>(h)]);
        if (!inner) {
            Py_DECREF(outer);
            return nullptr;
        }
        PyList_SET_ITEM(outer, h, inner);
    }
    return outer;
}

bool level_capacity(std::uint64_t k, std::size_t level, std::size_t level_count,
                    std::uint64_t min_level_capacity, std::uint64_t& out) {
    if (level >= level_count) {
        PyErr_SetString(PyExc_IndexError, "native level out of range");
        return false;
    }
    const int depth = static_cast<int>(level_count - level - 1U);
    if (depth > MAX_NATIVE_CAPACITY_DEPTH) {
        PyErr_SetString(PyExc_OverflowError, "native capacity depth exceeded; use Python fallback");
        return false;
    }
    std::uint64_t numerator = k;
    std::uint64_t denominator = 1;
    for (int i = 0; i < depth; ++i) {
        if (numerator > U64_MASK / 2ULL || denominator > U64_MASK / 3ULL) {
            PyErr_SetString(PyExc_OverflowError, "native capacity arithmetic exceeded; use Python fallback");
            return false;
        }
        numerator *= 2ULL;
        denominator *= 3ULL;
    }
    const std::uint64_t rounded = (numerator + denominator / 2ULL) / denominator;
    out = std::max(min_level_capacity, rounded);
    return true;
}

bool total_capacity(std::uint64_t k, const std::vector<std::vector<double>>& levels,
                    std::uint64_t min_level_capacity, std::uint64_t& out) {
    std::uint64_t total = 0;
    for (std::size_t h = 0; h < levels.size(); ++h) {
        std::uint64_t cap = 0;
        if (!level_capacity(k, h, levels.size(), min_level_capacity, cap)) {
            return false;
        }
        if (total > U64_MASK - cap) {
            PyErr_SetString(PyExc_OverflowError, "native total capacity overflow");
            return false;
        }
        total += cap;
    }
    out = total;
    return true;
}

bool find_overfull(std::uint64_t k, const std::vector<std::vector<double>>& levels,
                   std::uint64_t min_level_capacity, std::size_t& level_out, bool& found) {
    found = false;
    for (std::size_t h = 0; h < levels.size(); ++h) {
        std::uint64_t cap = 0;
        if (!level_capacity(k, h, levels.size(), min_level_capacity, cap)) {
            return false;
        }
        if (levels[h].size() > cap) {
            level_out = h;
            found = true;
            return true;
        }
    }
    return true;
}

bool compact_level_native(std::vector<std::vector<double>>& levels, std::size_t level,
                          std::uint64_t& rng_state, std::uint64_t& retained,
                          std::uint64_t& compactions, int max_levels) {
    std::vector<double>& items = levels[level];
    if (items.size() < 2U) {
        PyErr_SetString(PyExc_RuntimeError, "native attempted to compact a non-compactable level");
        return false;
    }
    std::sort(items.begin(), items.end());
    const std::size_t old_count = items.size();
    std::vector<double> leftover;
    std::size_t start = 0;
    std::size_t stop = items.size();
    if (items.size() & 1U) {
        if (next_bit(rng_state)) {
            leftover.push_back(items.back());
            --stop;
        } else {
            leftover.push_back(items.front());
            ++start;
        }
    }
    const int offset = next_bit(rng_state);
    std::vector<double> promoted;
    for (std::size_t i = start + static_cast<std::size_t>(offset); i < stop; i += 2U) {
        promoted.push_back(items[i]);
    }
    if (promoted.empty()) {
        PyErr_SetString(PyExc_RuntimeError, "native compaction produced no promoted items");
        return false;
    }
    items = std::move(leftover);
    if (level + 2U > levels.size()) {
        if (level + 2U > static_cast<std::size_t>(max_levels)) {
            PyErr_SetString(PyExc_OverflowError, "native sketch exceeds maximum supported level count");
            return false;
        }
        levels.resize(level + 2U);
    }
    levels[level + 1U].insert(levels[level + 1U].end(), promoted.begin(), promoted.end());
    retained = retained + items.size() + promoted.size() - old_count;
    if (compactions == U64_MASK) {
        PyErr_SetString(PyExc_OverflowError, "native compaction count overflow");
        return false;
    }
    ++compactions;
    return true;
}

bool compress_native(std::uint64_t k, std::vector<std::vector<double>>& levels,
                     std::uint64_t min_level_capacity, int max_levels,
                     std::uint64_t& rng_state, std::uint64_t& retained,
                     std::uint64_t& compactions) {
    int guard = 0;
    while (true) {
        std::uint64_t capacity = 0;
        if (!total_capacity(k, levels, min_level_capacity, capacity)) {
            return false;
        }
        if (retained <= capacity) {
            return true;
        }
        std::size_t level = 0;
        bool found = false;
        if (!find_overfull(k, levels, min_level_capacity, level, found)) {
            return false;
        }
        if (!found) {
            PyErr_SetString(PyExc_RuntimeError, "native KLL capacity accounting became inconsistent");
            return false;
        }
        if (!compact_level_native(levels, level, rng_state, retained, compactions, max_levels)) {
            return false;
        }
        if (++guard > 10000) {
            PyErr_SetString(PyExc_RuntimeError, "native KLL compaction did not converge");
            return false;
        }
    }
}

bool extract_values(PyObject* values_obj, std::vector<double>& values, double& batch_min,
                    double& batch_max, bool& has_values) {
    has_values = false;
    Py_buffer view{};
    if (PyObject_GetBuffer(values_obj, &view, PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) == 0) {
        const bool one_dimensional = view.ndim == 1;
        const bool contiguous = PyBuffer_IsContiguous(&view, 'C') != 0;
        const bool doubles = view.itemsize == static_cast<Py_ssize_t>(sizeof(double)) && view.format &&
                             std::strcmp(view.format, "d") == 0;
        if (one_dimensional && contiguous && doubles && view.len >= 0 && view.len % 8 == 0) {
            const std::size_t count = static_cast<std::size_t>(view.len / 8);
            values.resize(count);
            if (count) {
                std::memcpy(values.data(), view.buf, count * sizeof(double));
                if (!scan_buffer(values.data(), count, batch_min, batch_max)) {
                    PyBuffer_Release(&view);
                    PyErr_SetString(PyExc_ValueError, "native batch contains a non-finite value");
                    return false;
                }
                has_values = true;
            }
            PyBuffer_Release(&view);
            return true;
        }
        PyBuffer_Release(&view);
    } else {
        PyErr_Clear();
    }

    PyObject* seq = PySequence_Fast(values_obj, "native batch requires a finite sequence or contiguous double buffer");
    if (!seq) {
        return false;
    }
    const Py_ssize_t count = PySequence_Fast_GET_SIZE(seq);
    values.reserve(static_cast<std::size_t>(count));
    for (Py_ssize_t i = 0; i < count; ++i) {
        double value = 0.0;
        if (!py_to_double(PySequence_Fast_GET_ITEM(seq, i), value)) {
            Py_DECREF(seq);
            return false;
        }
        if (!has_values) {
            batch_min = batch_max = value;
            has_values = true;
        } else {
            batch_min = std::min(batch_min, value);
            batch_max = std::max(batch_max, value);
        }
        values.push_back(value);
    }
    Py_DECREF(seq);
    return true;
}

PyObject* py_info(PyObject*, PyObject*) {
    PyObject* result = PyDict_New();
    if (!result) {
        return nullptr;
    }
    PyObject* available = Py_True;
    Py_INCREF(available);
    PyDict_SetItemString(result, "available", available);
    Py_DECREF(available);
#if KLL_RUNTIME_AVX2
    const char* simd = cpu_has_avx2() ? "avx2-runtime" : "scalar";
#else
    const char* simd = "scalar";
#endif
#if defined(__clang__)
    const char* compiler = "clang";
#elif defined(__GNUC__)
    const char* compiler = "gcc";
#elif defined(_MSC_VER)
    const char* compiler = "msvc";
#else
    const char* compiler = "unknown";
#endif
    PyObject* simd_obj = PyUnicode_FromString(simd);
    PyObject* compiler_obj = PyUnicode_FromString(compiler);
    PyObject* api_obj = PyLong_FromLong(1);
    if (!simd_obj || !compiler_obj || !api_obj) {
        Py_XDECREF(simd_obj);
        Py_XDECREF(compiler_obj);
        Py_XDECREF(api_obj);
        Py_DECREF(result);
        return nullptr;
    }
    PyDict_SetItemString(result, "simd", simd_obj);
    PyDict_SetItemString(result, "compiler", compiler_obj);
    PyDict_SetItemString(result, "api_version", api_obj);
    Py_DECREF(simd_obj);
    Py_DECREF(compiler_obj);
    Py_DECREF(api_obj);
    return result;
}

PyObject* py_compact_level(PyObject*, PyObject* args) {
    PyObject* items_obj = nullptr;
    int keep_high = 0;
    int offset = 0;
    if (!PyArg_ParseTuple(args, "Opp:compact_level", &items_obj, &keep_high, &offset)) {
        return nullptr;
    }
    if (offset != 0 && offset != 1) {
        PyErr_SetString(PyExc_ValueError, "offset must be 0 or 1");
        return nullptr;
    }
    PyObject* seq = PySequence_Fast(items_obj, "items must be a sequence");
    if (!seq) {
        return nullptr;
    }
    const Py_ssize_t count = PySequence_Fast_GET_SIZE(seq);
    std::vector<double> items;
    items.reserve(static_cast<std::size_t>(count));
    for (Py_ssize_t i = 0; i < count; ++i) {
        double value = 0.0;
        if (!py_to_double(PySequence_Fast_GET_ITEM(seq, i), value)) {
            Py_DECREF(seq);
            return nullptr;
        }
        items.push_back(value);
    }
    Py_DECREF(seq);
    if (items.size() < 2U) {
        PyErr_SetString(PyExc_ValueError, "at least two items are required");
        return nullptr;
    }
    std::sort(items.begin(), items.end());
    std::vector<double> leftover;
    std::size_t start = 0;
    std::size_t stop = items.size();
    if (items.size() & 1U) {
        if (keep_high) {
            leftover.push_back(items.back());
            --stop;
        } else {
            leftover.push_back(items.front());
            ++start;
        }
    }
    std::vector<double> promoted;
    for (std::size_t i = start + static_cast<std::size_t>(offset); i < stop; i += 2U) {
        promoted.push_back(items[i]);
    }
    PyObject* left_obj = vector_to_float_list(leftover);
    PyObject* promoted_obj = vector_to_float_list(promoted);
    if (!left_obj || !promoted_obj) {
        Py_XDECREF(left_obj);
        Py_XDECREF(promoted_obj);
        return nullptr;
    }
    PyObject* result = PyTuple_Pack(2, left_obj, promoted_obj);
    Py_DECREF(left_obj);
    Py_DECREF(promoted_obj);
    return result;
}

PyObject* py_materialize(PyObject*, PyObject* args) {
    PyObject* levels_obj = nullptr;
    PyObject* n_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OO:materialize", &levels_obj, &n_obj)) {
        return nullptr;
    }
    const unsigned long long expected_n = PyLong_AsUnsignedLongLong(n_obj);
    if (PyErr_Occurred()) {
        return nullptr;
    }
    std::vector<std::vector<double>> levels;
    if (!parse_levels(levels_obj, levels)) {
        return nullptr;
    }
    std::vector<std::pair<double, std::uint64_t>> weighted;
    std::size_t retained = 0;
    for (const auto& level : levels) {
        retained += level.size();
    }
    weighted.reserve(retained);
    for (std::size_t h = 0; h < levels.size(); ++h) {
        if (h >= 64U) {
            PyErr_SetString(PyExc_OverflowError, "native materialization level exceeds uint64 weight");
            return nullptr;
        }
        const std::uint64_t weight = 1ULL << h;
        for (double value : levels[h]) {
            weighted.emplace_back(value, weight);
        }
    }
    Py_BEGIN_ALLOW_THREADS
    std::sort(weighted.begin(), weighted.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    Py_END_ALLOW_THREADS

    PyObject* values = PyList_New(static_cast<Py_ssize_t>(weighted.size()));
    PyObject* prefix = PyList_New(static_cast<Py_ssize_t>(weighted.size()));
    if (!values || !prefix) {
        Py_XDECREF(values);
        Py_XDECREF(prefix);
        return nullptr;
    }
    std::uint64_t cumulative = 0;
    for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(weighted.size()); ++i) {
        const auto& pair = weighted[static_cast<std::size_t>(i)];
        if (cumulative > U64_MASK - pair.second) {
            Py_DECREF(values);
            Py_DECREF(prefix);
            PyErr_SetString(PyExc_OverflowError, "native materialized weight overflow");
            return nullptr;
        }
        cumulative += pair.second;
        PyObject* value_obj = PyFloat_FromDouble(pair.first);
        PyObject* prefix_obj = PyLong_FromUnsignedLongLong(cumulative);
        if (!value_obj || !prefix_obj) {
            Py_XDECREF(value_obj);
            Py_XDECREF(prefix_obj);
            Py_DECREF(values);
            Py_DECREF(prefix);
            return nullptr;
        }
        PyList_SET_ITEM(values, i, value_obj);
        PyList_SET_ITEM(prefix, i, prefix_obj);
    }
    if (cumulative != static_cast<std::uint64_t>(expected_n)) {
        Py_DECREF(values);
        Py_DECREF(prefix);
        PyErr_SetString(PyExc_RuntimeError, "native materialized KLL weight does not equal n");
        return nullptr;
    }
    PyObject* result = PyTuple_Pack(2, values, prefix);
    Py_DECREF(values);
    Py_DECREF(prefix);
    return result;
}

PyObject* py_ingest_batch(PyObject*, PyObject* args) {
    PyObject* levels_obj = nullptr;
    PyObject* n_obj = nullptr;
    PyObject* k_obj = nullptr;
    PyObject* rng_obj = nullptr;
    PyObject* compactions_obj = nullptr;
    PyObject* retained_obj = nullptr;
    PyObject* min_obj = nullptr;
    PyObject* max_obj = nullptr;
    PyObject* values_obj = nullptr;
    int min_level_capacity = 0;
    int max_levels = 0;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOii:ingest_batch", &levels_obj, &n_obj, &k_obj, &rng_obj,
                          &compactions_obj, &retained_obj, &min_obj, &max_obj, &values_obj,
                          &min_level_capacity, &max_levels)) {
        return nullptr;
    }
    std::uint64_t n = PyLong_AsUnsignedLongLong(n_obj);
    if (PyErr_Occurred()) return nullptr;
    const std::uint64_t k = PyLong_AsUnsignedLongLong(k_obj);
    if (PyErr_Occurred()) return nullptr;
    std::uint64_t rng_state = PyLong_AsUnsignedLongLong(rng_obj);
    if (PyErr_Occurred()) return nullptr;
    std::uint64_t compactions = PyLong_AsUnsignedLongLong(compactions_obj);
    if (PyErr_Occurred()) return nullptr;
    std::uint64_t retained = PyLong_AsUnsignedLongLong(retained_obj);
    if (PyErr_Occurred()) return nullptr;
    if (min_level_capacity <= 0 || max_levels <= 0 || max_levels > 64) {
        PyErr_SetString(PyExc_ValueError, "invalid native KLL limits");
        return nullptr;
    }

    bool has_existing = min_obj != Py_None;
    double min_value = 0.0;
    double max_value = 0.0;
    if (has_existing) {
        if (!py_to_double(min_obj, min_value) || !py_to_double(max_obj, max_value)) {
            return nullptr;
        }
    }

    std::vector<std::vector<double>> levels;
    if (!parse_levels(levels_obj, levels)) {
        return nullptr;
    }
    if (levels.empty() || levels.size() > static_cast<std::size_t>(max_levels)) {
        PyErr_SetString(PyExc_ValueError, "invalid native KLL level count");
        return nullptr;
    }

    std::vector<double> values;
    double batch_min = 0.0;
    double batch_max = 0.0;
    bool has_batch = false;
    if (!extract_values(values_obj, values, batch_min, batch_max, has_batch)) {
        return nullptr;
    }
    if (values.size() > static_cast<std::size_t>(U64_MASK - n)) {
        PyErr_SetString(PyExc_OverflowError, "total sketch weight exceeds uint64 serialization range");
        return nullptr;
    }
    if (has_batch) {
        if (!has_existing) {
            min_value = batch_min;
            max_value = batch_max;
            has_existing = true;
        } else {
            min_value = std::min(min_value, batch_min);
            max_value = std::max(max_value, batch_max);
        }
    }

    bool ok = true;
    Py_BEGIN_ALLOW_THREADS
    for (double value : values) {
        levels[0].push_back(value);
        ++n;
        ++retained;
        if (!compress_native(k, levels, static_cast<std::uint64_t>(min_level_capacity), max_levels,
                             rng_state, retained, compactions)) {
            ok = false;
            break;
        }
    }
    Py_END_ALLOW_THREADS
    if (!ok) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "native KLL batch ingestion failed");
        }
        return nullptr;
    }

    PyObject* py_levels = levels_to_python(levels);
    PyObject* py_n = PyLong_FromUnsignedLongLong(n);
    PyObject* py_retained = PyLong_FromUnsignedLongLong(retained);
    PyObject* py_rng = PyLong_FromUnsignedLongLong(rng_state);
    PyObject* py_compactions = PyLong_FromUnsignedLongLong(compactions);
    PyObject* py_min = has_existing ? PyFloat_FromDouble(min_value) : (Py_INCREF(Py_None), Py_None);
    PyObject* py_max = has_existing ? PyFloat_FromDouble(max_value) : (Py_INCREF(Py_None), Py_None);
    if (!py_levels || !py_n || !py_retained || !py_rng || !py_compactions || !py_min || !py_max) {
        Py_XDECREF(py_levels); Py_XDECREF(py_n); Py_XDECREF(py_retained); Py_XDECREF(py_rng);
        Py_XDECREF(py_compactions); Py_XDECREF(py_min); Py_XDECREF(py_max);
        return nullptr;
    }
    PyObject* result = PyTuple_Pack(7, py_levels, py_n, py_retained, py_rng, py_compactions, py_min, py_max);
    Py_DECREF(py_levels); Py_DECREF(py_n); Py_DECREF(py_retained); Py_DECREF(py_rng);
    Py_DECREF(py_compactions); Py_DECREF(py_min); Py_DECREF(py_max);
    return result;
}

PyObject* py_ranks_many(PyObject*, PyObject* args) {
    PyObject* values_obj = nullptr;
    PyObject* prefix_obj = nullptr;
    PyObject* xs_obj = nullptr;
    int inclusive = 1;
    if (!PyArg_ParseTuple(args, "OOOp:ranks_many", &values_obj, &prefix_obj, &xs_obj, &inclusive)) {
        return nullptr;
    }
    PyObject* values_seq = PySequence_Fast(values_obj, "values must be a sequence");
    PyObject* prefix_seq = PySequence_Fast(prefix_obj, "prefix must be a sequence");
    PyObject* xs_seq = PySequence_Fast(xs_obj, "xs must be a sequence");
    if (!values_seq || !prefix_seq || !xs_seq) {
        Py_XDECREF(values_seq); Py_XDECREF(prefix_seq); Py_XDECREF(xs_seq);
        return nullptr;
    }
    const Py_ssize_t count = PySequence_Fast_GET_SIZE(values_seq);
    if (PySequence_Fast_GET_SIZE(prefix_seq) != count) {
        Py_DECREF(values_seq); Py_DECREF(prefix_seq); Py_DECREF(xs_seq);
        PyErr_SetString(PyExc_ValueError, "values/prefix length mismatch");
        return nullptr;
    }
    std::vector<double> values;
    std::vector<std::uint64_t> prefix;
    values.reserve(static_cast<std::size_t>(count));
    prefix.reserve(static_cast<std::size_t>(count));
    for (Py_ssize_t i = 0; i < count; ++i) {
        double value = 0.0;
        if (!py_to_double(PySequence_Fast_GET_ITEM(values_seq, i), value)) {
            Py_DECREF(values_seq); Py_DECREF(prefix_seq); Py_DECREF(xs_seq);
            return nullptr;
        }
        const auto p = PyLong_AsUnsignedLongLong(PySequence_Fast_GET_ITEM(prefix_seq, i));
        if (PyErr_Occurred()) {
            Py_DECREF(values_seq); Py_DECREF(prefix_seq); Py_DECREF(xs_seq);
            return nullptr;
        }
        values.push_back(value);
        prefix.push_back(p);
    }
    const Py_ssize_t xs_count = PySequence_Fast_GET_SIZE(xs_seq);
    PyObject* out = PyList_New(xs_count);
    if (!out) {
        Py_DECREF(values_seq); Py_DECREF(prefix_seq); Py_DECREF(xs_seq);
        return nullptr;
    }
    for (Py_ssize_t i = 0; i < xs_count; ++i) {
        double x = 0.0;
        if (!py_to_double(PySequence_Fast_GET_ITEM(xs_seq, i), x)) {
            Py_DECREF(values_seq); Py_DECREF(prefix_seq); Py_DECREF(xs_seq); Py_DECREF(out);
            return nullptr;
        }
        auto it = inclusive ? std::upper_bound(values.begin(), values.end(), x)
                            : std::lower_bound(values.begin(), values.end(), x);
        double rank = 0.0;
        if (it != values.begin()) {
            const std::size_t pos = static_cast<std::size_t>(it - values.begin() - 1);
            rank = static_cast<double>(prefix[pos]);
        }
        PyObject* rank_obj = PyFloat_FromDouble(rank);
        if (!rank_obj) {
            Py_DECREF(values_seq); Py_DECREF(prefix_seq); Py_DECREF(xs_seq); Py_DECREF(out);
            return nullptr;
        }
        PyList_SET_ITEM(out, i, rank_obj);
    }
    Py_DECREF(values_seq); Py_DECREF(prefix_seq); Py_DECREF(xs_seq);
    return out;
}

PyObject* py_quantiles_many(PyObject*, PyObject* args) {
    PyObject* values_obj = nullptr;
    PyObject* prefix_obj = nullptr;
    PyObject* n_obj = nullptr;
    PyObject* qs_obj = nullptr;
    double min_value = 0.0;
    double max_value = 0.0;
    if (!PyArg_ParseTuple(args, "OOOOdd:quantiles_many", &values_obj, &prefix_obj, &n_obj, &qs_obj,
                          &min_value, &max_value)) {
        return nullptr;
    }
    const std::uint64_t n = PyLong_AsUnsignedLongLong(n_obj);
    if (PyErr_Occurred()) return nullptr;
    PyObject* values_seq = PySequence_Fast(values_obj, "values must be a sequence");
    PyObject* prefix_seq = PySequence_Fast(prefix_obj, "prefix must be a sequence");
    PyObject* qs_seq = PySequence_Fast(qs_obj, "qs must be a sequence");
    if (!values_seq || !prefix_seq || !qs_seq) {
        Py_XDECREF(values_seq); Py_XDECREF(prefix_seq); Py_XDECREF(qs_seq);
        return nullptr;
    }
    const Py_ssize_t count = PySequence_Fast_GET_SIZE(values_seq);
    if (count == 0 || PySequence_Fast_GET_SIZE(prefix_seq) != count) {
        Py_DECREF(values_seq); Py_DECREF(prefix_seq); Py_DECREF(qs_seq);
        PyErr_SetString(PyExc_ValueError, "invalid native query view");
        return nullptr;
    }
    std::vector<double> values;
    std::vector<std::uint64_t> prefix;
    values.reserve(static_cast<std::size_t>(count));
    prefix.reserve(static_cast<std::size_t>(count));
    for (Py_ssize_t i = 0; i < count; ++i) {
        double value = 0.0;
        if (!py_to_double(PySequence_Fast_GET_ITEM(values_seq, i), value)) {
            Py_DECREF(values_seq); Py_DECREF(prefix_seq); Py_DECREF(qs_seq);
            return nullptr;
        }
        const auto p = PyLong_AsUnsignedLongLong(PySequence_Fast_GET_ITEM(prefix_seq, i));
        if (PyErr_Occurred()) {
            Py_DECREF(values_seq); Py_DECREF(prefix_seq); Py_DECREF(qs_seq);
            return nullptr;
        }
        values.push_back(value);
        prefix.push_back(p);
    }
    const Py_ssize_t qcount = PySequence_Fast_GET_SIZE(qs_seq);
    PyObject* out = PyList_New(qcount);
    if (!out) {
        Py_DECREF(values_seq); Py_DECREF(prefix_seq); Py_DECREF(qs_seq);
        return nullptr;
    }
    for (Py_ssize_t i = 0; i < qcount; ++i) {
        double q = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(qs_seq, i));
        if (PyErr_Occurred() || !std::isfinite(q) || q < 0.0 || q > 1.0) {
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "q must be in [0,1]");
            Py_DECREF(values_seq); Py_DECREF(prefix_seq); Py_DECREF(qs_seq); Py_DECREF(out);
            return nullptr;
        }
        double answer = 0.0;
        if (q <= 0.0) {
            answer = min_value;
        } else if (q >= 1.0) {
            answer = max_value;
        } else {
            const double target = q * static_cast<double>(n - 1ULL);
            auto it = std::upper_bound(prefix.begin(), prefix.end(), target,
                [](double lhs, std::uint64_t rhs) { return lhs < static_cast<double>(rhs); });
            std::size_t pos = static_cast<std::size_t>(it - prefix.begin());
            if (pos >= values.size()) pos = values.size() - 1U;
            answer = values[pos];
        }
        PyObject* answer_obj = PyFloat_FromDouble(answer);
        if (!answer_obj) {
            Py_DECREF(values_seq); Py_DECREF(prefix_seq); Py_DECREF(qs_seq); Py_DECREF(out);
            return nullptr;
        }
        PyList_SET_ITEM(out, i, answer_obj);
    }
    Py_DECREF(values_seq); Py_DECREF(prefix_seq); Py_DECREF(qs_seq);
    return out;
}

PyMethodDef methods[] = {
    {"info", reinterpret_cast<PyCFunction>(py_info), METH_NOARGS, "Return native backend diagnostics."},
    {"compact_level", py_compact_level, METH_VARARGS, "Sort and compact one KLL level."},
    {"materialize", py_materialize, METH_VARARGS, "Materialize sorted values and cumulative weights."},
    {"ingest_batch", py_ingest_batch, METH_VARARGS, "Bulk-ingest a finite sequence using native KLL compaction."},
    {"ranks_many", py_ranks_many, METH_VARARGS, "Compute batched ranks over a materialized view."},
    {"quantiles_many", py_quantiles_many, METH_VARARGS, "Compute batched quantiles over a materialized view."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "Optional C++17 acceleration helpers for kll-sketch.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__native(void) {
    return PyModule_Create(&module);
}
