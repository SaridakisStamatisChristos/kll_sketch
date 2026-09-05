// Persistent native KLL state engine, optimized for hot public API paths.
//
// This implementation deliberately keeps the canonical Python/KLL2 state
// semantics.  Levels are sorted in place only at the exact moment Python's
// reference compactor would sort them; after compaction the source level is
// replaced by its boundary leftover and promotions are appended to the next
// raw level in the same order as the Python implementation.

struct QueryView {
    std::vector<double> values;
    std::vector<std::uint64_t> prefix;
};

struct SketchState {
    std::vector<std::vector<double>> levels;
    std::vector<unsigned char> level_has_zero;
    std::vector<std::uint64_t> capacities;
    std::uint64_t total_capacity = 0;
    std::uint64_t n = 0;
    std::uint64_t k = 0;
    std::uint64_t rng_state = 0;
    std::uint64_t compactions = 0;
    std::uint64_t retained = 0;
    std::uint64_t min_level_capacity = 8;
    int max_levels = 64;
    bool has_values = false;
    double min_value = 0.0;
    double max_value = 0.0;
    bool query_valid = false;
    QueryView query;
};

void capsule_destructor(PyObject* capsule) {
    void* ptr = PyCapsule_GetPointer(capsule, CAPSULE_NAME);
    if (!ptr) {
        PyErr_Clear();
        return;
    }
    delete static_cast<SketchState*>(ptr);
}

inline SketchState* state_from_capsule(PyObject* capsule) {
    return static_cast<SketchState*>(PyCapsule_GetPointer(capsule, CAPSULE_NAME));
}

inline bool vector_has_zero(const std::vector<double>& values) {
    for (double value : values) {
        if (value == 0.0) return true;
    }
    return false;
}

inline void sort_level_fast(std::vector<double>& values, bool has_zero) {
    if (values.size() <= 32U) {
        insertion_sort(values.data(), values.size());
        return;
    }
    if (std::is_sorted(values.begin(), values.end())) return;
    // Python's list.sort is stable.  Equal nonzero doubles have identical bit
    // patterns for our purposes, but +0.0/-0.0 do not.  Pay for stable_sort
    // only on levels known to contain zero so byte-level parity is preserved.
    if (has_zero) std::stable_sort(values.begin(), values.end());
    else std::sort(values.begin(), values.end());
}

#if KLL_STATE_AVX2
__attribute__((target("avx2")))
bool scan_values_avx2_fused(const double* data, std::size_t n,
                            double& lo, double& hi, bool& has_zero) {
    if (n == 0U) {
        has_zero = false;
        return true;
    }
    const __m256d sign = _mm256_set1_pd(-0.0);
    const __m256d maxf = _mm256_set1_pd(std::numeric_limits<double>::max());
    const __m256d zero = _mm256_setzero_pd();
    __m256d vlo = _mm256_set1_pd(std::numeric_limits<double>::infinity());
    __m256d vhi = _mm256_set1_pd(-std::numeric_limits<double>::infinity());
    bool zero_seen = false;
    std::size_t i = 0;
    for (; i + 4U <= n; i += 4U) {
        const __m256d x = _mm256_loadu_pd(data + i);
        const __m256d ax = _mm256_andnot_pd(sign, x);
        const __m256d finite = _mm256_cmp_pd(ax, maxf, _CMP_LE_OQ);
        if (_mm256_movemask_pd(finite) != 0xF) return false;
        const __m256d is_zero = _mm256_cmp_pd(ax, zero, _CMP_EQ_OQ);
        zero_seen = zero_seen || (_mm256_movemask_pd(is_zero) != 0);
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
        if (x == 0.0) zero_seen = true;
        if (x < local_lo) local_lo = x;
        if (x > local_hi) local_hi = x;
    }

    has_zero = zero_seen;
    if (!zero_seen) {
        lo = local_lo;
        hi = local_hi;
        return true;
    }

    // SIMD min/max does not preserve Python's first-value tie semantics for
    // signed zero.  Zero-containing batches are uncommon in numeric streams;
    // rescan only that case to keep exact +0.0/-0.0 extrema behavior.
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

bool scan_values_fast(const double* data, std::size_t n,
                      double& lo, double& hi, bool& has_zero) {
    if (n == 0U) {
        has_zero = false;
        return true;
    }
#if KLL_STATE_AVX2
    if (cpu_avx2()) return scan_values_avx2_fused(data, n, lo, hi, has_zero);
#endif
    lo = data[0];
    hi = data[0];
    has_zero = (data[0] == 0.0);
    if (!std::isfinite(data[0])) return false;
    for (std::size_t i = 1; i < n; ++i) {
        const double x = data[i];
        if (!std::isfinite(x)) return false;
        has_zero = has_zero || (x == 0.0);
        if (x < lo) lo = x;
        if (x > hi) hi = x;
    }
    return true;
}

bool rebuild_capacities(SketchState& s) {
    s.capacities.resize(s.levels.size());
    s.total_capacity = 0;
    for (std::size_t h = 0; h < s.levels.size(); ++h) {
        std::uint64_t cap = 0;
        if (!level_capacity(s.k, h, s.levels.size(), s.min_level_capacity, cap)) return false;
        s.capacities[h] = cap;
        if (s.total_capacity > U64_MAX - cap) {
            PyErr_SetString(PyExc_OverflowError, "native total capacity overflow");
            return false;
        }
        s.total_capacity += cap;
    }
    return true;
}

bool ensure_levels(SketchState& s, std::size_t count) {
    if (count <= s.levels.size()) return true;
    if (count > static_cast<std::size_t>(s.max_levels)) {
        PyErr_SetString(PyExc_OverflowError, "native sketch exceeds maximum supported level count");
        return false;
    }
    s.levels.resize(count);
    s.level_has_zero.resize(count, 0U);
    return rebuild_capacities(s);
}

bool compact_level(SketchState& s, std::size_t h) {
    if (h >= s.levels.size() || s.levels[h].size() < 2U) {
        PyErr_SetString(PyExc_RuntimeError, "native state attempted invalid compaction");
        return false;
    }
    if (!ensure_levels(s, h + 2U)) return false;

    auto& source = s.levels[h];
    sort_level_fast(source, s.level_has_zero[h] != 0U);
    const std::size_t old_count = source.size();
    std::size_t start = 0;
    std::size_t stop = old_count;
    bool has_leftover = false;
    double leftover = 0.0;
    if (old_count & 1U) {
        has_leftover = true;
        if (next_bit(s.rng_state)) {
            leftover = source.back();
            --stop;
        } else {
            leftover = source.front();
            ++start;
        }
    }
    const int offset = next_bit(s.rng_state);
    const std::size_t promoted_count = (stop - start) / 2U;
    if (!promoted_count) {
        PyErr_SetString(PyExc_RuntimeError, "native compaction produced no promoted items");
        return false;
    }

    std::vector<double> promoted;
    promoted.reserve(promoted_count);
    bool promoted_has_zero = false;
    for (std::size_t i = start + static_cast<std::size_t>(offset); i < stop; i += 2U) {
        const double value = source[i];
        promoted.push_back(value);
        promoted_has_zero = promoted_has_zero || (value == 0.0);
    }

    auto& next = s.levels[h + 1U];
    next.insert(next.end(), promoted.begin(), promoted.end());
    if (promoted_has_zero) s.level_has_zero[h + 1U] = 1U;

    source.clear();
    if (has_leftover) source.push_back(leftover);
    s.level_has_zero[h] = static_cast<unsigned char>(has_leftover && leftover == 0.0);

    if (s.retained < old_count) {
        PyErr_SetString(PyExc_RuntimeError, "native retained-item accounting underflow");
        return false;
    }
    s.retained = s.retained - static_cast<std::uint64_t>(old_count)
               + static_cast<std::uint64_t>(promoted_count + (has_leftover ? 1U : 0U));
    if (s.compactions == U64_MAX) {
        PyErr_SetString(PyExc_OverflowError, "native compaction count overflow");
        return false;
    }
    ++s.compactions;
    return true;
}

bool compress(SketchState& s) {
    int guard = 0;
    while (s.retained > s.total_capacity) {
        std::size_t h = 0;
        bool found = false;
        for (; h < s.levels.size(); ++h) {
            if (s.levels[h].size() > s.capacities[h]) {
                found = true;
                break;
            }
        }
        if (!found) {
            PyErr_SetString(PyExc_RuntimeError, "native KLL capacity accounting became inconsistent");
            return false;
        }
        if (!compact_level(s, h)) return false;
        if (++guard > 10000) {
            PyErr_SetString(PyExc_RuntimeError, "native KLL compaction did not converge");
            return false;
        }
    }
    return true;
}

bool ingest_span(SketchState& s, const double* data, std::size_t count,
                 double batch_min, double batch_max, bool batch_has_zero) {
    if (count > static_cast<std::size_t>(U64_MAX - s.n)) {
        PyErr_SetString(PyExc_OverflowError, "total sketch weight exceeds uint64 serialization range");
        return false;
    }
    if (count) {
        if (!s.has_values) {
            s.has_values = true;
            s.min_value = batch_min;
            s.max_value = batch_max;
        } else {
            if (batch_min < s.min_value) s.min_value = batch_min;
            if (batch_max > s.max_value) s.max_value = batch_max;
        }
    }

    std::size_t pos = 0;
    while (pos < count) {
        if (s.retained > s.total_capacity && !compress(s)) return false;
        const std::uint64_t room = s.total_capacity - s.retained;
        const std::uint64_t trigger = room == U64_MAX ? U64_MAX : room + 1ULL;
        const std::size_t remaining = count - pos;
        const std::size_t chunk = trigger < static_cast<std::uint64_t>(remaining)
            ? static_cast<std::size_t>(trigger) : remaining;
        s.levels[0].insert(s.levels[0].end(), data + pos, data + pos + chunk);
        if (batch_has_zero) s.level_has_zero[0] = 1U;
        pos += chunk;
        s.n += static_cast<std::uint64_t>(chunk);
        s.retained += static_cast<std::uint64_t>(chunk);
        if (s.retained > s.total_capacity && !compress(s)) return false;
    }
    if (count) s.query_valid = false;
    return true;
}

bool build_query(SketchState& s) {
    if (s.query_valid) return true;
    std::vector<std::pair<double, std::uint64_t>> weighted;
    weighted.reserve(static_cast<std::size_t>(s.retained));
    for (std::size_t h = 0; h < s.levels.size(); ++h) {
        if (h >= 64U) {
            PyErr_SetString(PyExc_OverflowError, "native query level exceeds uint64 weight");
            return false;
        }
        const std::uint64_t weight = 1ULL << h;
        for (double value : s.levels[h]) weighted.emplace_back(value, weight);
    }
    std::stable_sort(weighted.begin(), weighted.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });
    s.query.values.resize(weighted.size());
    s.query.prefix.resize(weighted.size());
    std::uint64_t cumulative = 0;
    for (std::size_t i = 0; i < weighted.size(); ++i) {
        if (cumulative > U64_MAX - weighted[i].second) {
            PyErr_SetString(PyExc_OverflowError, "native query weight overflow");
            return false;
        }
        cumulative += weighted[i].second;
        s.query.values[i] = weighted[i].first;
        s.query.prefix[i] = cumulative;
    }
    if (cumulative != s.n) {
        PyErr_SetString(PyExc_RuntimeError, "native materialized KLL weight does not equal n");
        return false;
    }
    s.query_valid = true;
    return true;
}

PyObject* stats(const SketchState& s) {
    PyObject* a = PyLong_FromUnsignedLongLong(s.n);
    PyObject* b = PyLong_FromUnsignedLongLong(s.retained);
    PyObject* c = PyLong_FromUnsignedLongLong(s.rng_state);
    PyObject* d = PyLong_FromUnsignedLongLong(s.compactions);
    PyObject* e = s.has_values ? PyFloat_FromDouble(s.min_value) : Py_NewRef(Py_None);
    PyObject* f = s.has_values ? PyFloat_FromDouble(s.max_value) : Py_NewRef(Py_None);
    PyObject* g = PyLong_FromSize_t(s.levels.size());
    if (!a || !b || !c || !d || !e || !f || !g) {
        Py_XDECREF(a); Py_XDECREF(b); Py_XDECREF(c); Py_XDECREF(d);
        Py_XDECREF(e); Py_XDECREF(f); Py_XDECREF(g);
        return nullptr;
    }
    PyObject* out = PyTuple_Pack(7, a, b, c, d, e, f, g);
    Py_DECREF(a); Py_DECREF(b); Py_DECREF(c); Py_DECREF(d);
    Py_DECREF(e); Py_DECREF(f); Py_DECREF(g);
    return out;
}

PyObject* state_create(PyObject*, PyObject* args) {
    PyObject *levels_obj, *n_obj, *k_obj, *rng_obj, *comp_obj, *ret_obj, *min_obj, *max_obj;
    int mincap, maxlevels;
    if (!PyArg_ParseTuple(args, "OOOOOOOOii:state_create", &levels_obj, &n_obj, &k_obj, &rng_obj,
                          &comp_obj, &ret_obj, &min_obj, &max_obj, &mincap, &maxlevels)) return nullptr;
    auto* s = new (std::nothrow) SketchState();
    if (!s) return PyErr_NoMemory();
    s->n = PyLong_AsUnsignedLongLong(n_obj);
    if (PyErr_Occurred()) { delete s; return nullptr; }
    s->k = PyLong_AsUnsignedLongLong(k_obj);
    if (PyErr_Occurred()) { delete s; return nullptr; }
    s->rng_state = PyLong_AsUnsignedLongLong(rng_obj);
    if (PyErr_Occurred()) { delete s; return nullptr; }
    s->compactions = PyLong_AsUnsignedLongLong(comp_obj);
    if (PyErr_Occurred()) { delete s; return nullptr; }
    s->retained = PyLong_AsUnsignedLongLong(ret_obj);
    if (PyErr_Occurred()) { delete s; return nullptr; }
    if (mincap <= 0 || maxlevels <= 0 || maxlevels > 64) {
        delete s;
        PyErr_SetString(PyExc_ValueError, "invalid native KLL limits");
        return nullptr;
    }
    s->min_level_capacity = static_cast<std::uint64_t>(mincap);
    s->max_levels = maxlevels;
    s->has_values = min_obj != Py_None;
    if (s->has_values && (!py_double(min_obj, s->min_value) || !py_double(max_obj, s->max_value))) {
        delete s;
        return nullptr;
    }
    if (!parse_levels(levels_obj, s->levels)) {
        delete s;
        return nullptr;
    }
    if (s->levels.empty() || s->levels.size() > static_cast<std::size_t>(maxlevels)) {
        delete s;
        PyErr_SetString(PyExc_ValueError, "invalid native KLL level count");
        return nullptr;
    }
    s->level_has_zero.resize(s->levels.size(), 0U);
    for (std::size_t h = 0; h < s->levels.size(); ++h) {
        s->level_has_zero[h] = static_cast<unsigned char>(vector_has_zero(s->levels[h]));
    }
    if (!rebuild_capacities(*s)) {
        delete s;
        return nullptr;
    }
    PyObject* capsule = PyCapsule_New(s, CAPSULE_NAME, capsule_destructor);
    if (!capsule) {
        delete s;
        return nullptr;
    }
    return capsule;
}

PyObject* state_stats(PyObject*, PyObject* args) {
    PyObject* capsule;
    if (!PyArg_ParseTuple(args, "O:state_stats", &capsule)) return nullptr;
    SketchState* s = state_from_capsule(capsule);
    if (!s) return nullptr;
    return stats(*s);
}

PyObject* state_export(PyObject*, PyObject* args) {
    PyObject* capsule;
    if (!PyArg_ParseTuple(args, "O:state_export", &capsule)) return nullptr;
    SketchState* s = state_from_capsule(capsule);
    if (!s) return nullptr;
    PyObject* levels = levels_to_python(s->levels);
    PyObject* state_stats_obj = stats(*s);
    if (!levels || !state_stats_obj) {
        Py_XDECREF(levels);
        Py_XDECREF(state_stats_obj);
        return nullptr;
    }
    PyObject* out = PyTuple_Pack(2, levels, state_stats_obj);
    Py_DECREF(levels);
    Py_DECREF(state_stats_obj);
    return out;
}

PyObject* state_ingest(PyObject*, PyObject* args) {
    PyObject *capsule, *values_obj;
    if (!PyArg_ParseTuple(args, "OO:state_ingest", &capsule, &values_obj)) return nullptr;
    SketchState* s = state_from_capsule(capsule);
    if (!s) return nullptr;
    SketchState backup = *s;

    Py_buffer view{};
    if (PyObject_GetBuffer(values_obj, &view, PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) == 0) {
        const bool compatible = view.ndim == 1 && PyBuffer_IsContiguous(&view, 'C') &&
            view.itemsize == static_cast<Py_ssize_t>(sizeof(double)) && view.format &&
            std::strcmp(view.format, "d") == 0 && view.len >= 0 && view.len % 8 == 0;
        if (compatible) {
            const std::size_t count = static_cast<std::size_t>(view.len / 8);
            const double* data = static_cast<const double*>(view.buf);
            double lo = 0.0, hi = 0.0;
            bool has_zero = false;
            if (count && !scan_values_fast(data, count, lo, hi, has_zero)) {
                PyBuffer_Release(&view);
                PyErr_SetString(PyExc_ValueError, "native batch contains a non-finite value");
                return nullptr;
            }
            const bool ok = ingest_span(*s, data, count, lo, hi, has_zero);
            PyBuffer_Release(&view);
            if (!ok) {
                *s = std::move(backup);
                return nullptr;
            }
            return stats(*s);
        }
        PyBuffer_Release(&view);
    } else {
        PyErr_Clear();
    }

    PyObject* seq = PySequence_Fast(values_obj, "native batch requires a finite sequence");
    if (!seq) return nullptr;
    const Py_ssize_t count = PySequence_Fast_GET_SIZE(seq);
    std::vector<double> values;
    values.reserve(static_cast<std::size_t>(count));
    double lo = 0.0, hi = 0.0;
    bool has_zero = false;
    for (Py_ssize_t i = 0; i < count; ++i) {
        double value = 0.0;
        if (!py_double(PySequence_Fast_GET_ITEM(seq, i), value)) {
            Py_DECREF(seq);
            return nullptr;
        }
        if (i == 0) lo = hi = value;
        else {
            if (value < lo) lo = value;
            if (value > hi) hi = value;
        }
        has_zero = has_zero || (value == 0.0);
        values.push_back(value);
    }
    Py_DECREF(seq);
    if (!ingest_span(*s, values.data(), values.size(), lo, hi, has_zero)) {
        *s = std::move(backup);
        return nullptr;
    }
    return stats(*s);
}

PyObject* state_merge(PyObject*, PyObject* args) {
    PyObject *dst_obj, *src_obj;
    if (!PyArg_ParseTuple(args, "OO:state_merge", &dst_obj, &src_obj)) return nullptr;
    SketchState* dst = state_from_capsule(dst_obj);
    if (!dst) return nullptr;
    SketchState* src = state_from_capsule(src_obj);
    if (!src) return nullptr;
    if (dst == src) {
        PyErr_SetString(PyExc_ValueError, "cannot merge native state with itself");
        return nullptr;
    }
    if (!src->n) return stats(*dst);
    if (src->n > U64_MAX - dst->n || src->retained > U64_MAX - dst->retained) {
        PyErr_SetString(PyExc_OverflowError, "native merge exceeds uint64 range");
        return nullptr;
    }

    SketchState backup = *dst;
    if (!ensure_levels(*dst, std::max(dst->levels.size(), src->levels.size()))) {
        *dst = std::move(backup);
        return nullptr;
    }
    for (std::size_t h = 0; h < src->levels.size(); ++h) {
        if (src->levels[h].empty()) continue;
        dst->levels[h].insert(dst->levels[h].end(), src->levels[h].begin(), src->levels[h].end());
        if (src->level_has_zero[h]) dst->level_has_zero[h] = 1U;
    }
    dst->n += src->n;
    dst->retained += src->retained;
    if (src->has_values) {
        if (!dst->has_values) {
            dst->has_values = true;
            dst->min_value = src->min_value;
            dst->max_value = src->max_value;
        } else {
            if (src->min_value < dst->min_value) dst->min_value = src->min_value;
            if (src->max_value > dst->max_value) dst->max_value = src->max_value;
        }
    }
    dst->query_valid = false;
    if (!compress(*dst)) {
        *dst = std::move(backup);
        return nullptr;
    }
    return stats(*dst);
}

PyObject* state_quantiles(PyObject*, PyObject* args) {
    PyObject *capsule, *qs_obj;
    if (!PyArg_ParseTuple(args, "OO:state_quantiles", &capsule, &qs_obj)) return nullptr;
    SketchState* s = state_from_capsule(capsule);
    if (!s) return nullptr;
    if (!s->n) {
        PyErr_SetString(PyExc_ValueError, "empty sketch");
        return nullptr;
    }
    if (!build_query(*s)) return nullptr;
    PyObject* seq = PySequence_Fast(qs_obj, "probabilities must be a sequence");
    if (!seq) return nullptr;
    const Py_ssize_t count = PySequence_Fast_GET_SIZE(seq);
    PyObject* out = PyList_New(count);
    if (!out) {
        Py_DECREF(seq);
        return nullptr;
    }

    double previous_q = -1.0;
    std::size_t lower_hint = 0;
    for (Py_ssize_t i = 0; i < count; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
        const double q = PyFloat_CheckExact(item) ? PyFloat_AS_DOUBLE(item) : PyFloat_AsDouble(item);
        if (PyErr_Occurred() || !std::isfinite(q) || q < 0.0 || q > 1.0) {
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "probability must be in [0,1]");
            Py_DECREF(seq);
            Py_DECREF(out);
            return nullptr;
        }
        double answer = 0.0;
        if (q <= 0.0) {
            answer = s->min_value;
        } else if (q >= 1.0) {
            answer = s->max_value;
            if (q >= previous_q && !s->query.values.empty()) lower_hint = s->query.values.size() - 1U;
        } else {
            const double target = q * static_cast<double>(s->n - 1ULL);
            auto begin = s->query.prefix.begin();
            if (q >= previous_q && lower_hint < s->query.prefix.size()) {
                begin += static_cast<std::ptrdiff_t>(lower_hint);
            } else {
                lower_hint = 0;
            }
            auto it = std::upper_bound(begin, s->query.prefix.end(), target,
                [](double lhs, std::uint64_t rhs) { return lhs < static_cast<double>(rhs); });
            std::size_t pos = static_cast<std::size_t>(it - s->query.prefix.begin());
            if (pos >= s->query.values.size()) pos = s->query.values.size() - 1U;
            answer = s->query.values[pos];
            if (q >= previous_q) lower_hint = pos;
        }
        previous_q = q;
        PyObject* value = PyFloat_FromDouble(answer);
        if (!value) {
            Py_DECREF(seq);
            Py_DECREF(out);
            return nullptr;
        }
        PyList_SET_ITEM(out, i, value);
    }
    Py_DECREF(seq);
    return out;
}

PyObject* state_ranks(PyObject*, PyObject* args) {
    PyObject *capsule, *xs_obj;
    int inclusive = 1;
    if (!PyArg_ParseTuple(args, "OOp:state_ranks", &capsule, &xs_obj, &inclusive)) return nullptr;
    SketchState* s = state_from_capsule(capsule);
    if (!s) return nullptr;
    if (!build_query(*s)) return nullptr;
    PyObject* seq = PySequence_Fast(xs_obj, "xs must be a sequence");
    if (!seq) return nullptr;
    const Py_ssize_t count = PySequence_Fast_GET_SIZE(seq);
    PyObject* out = PyList_New(count);
    if (!out) {
        Py_DECREF(seq);
        return nullptr;
    }
    for (Py_ssize_t i = 0; i < count; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
        const double x = PyFloat_CheckExact(item) ? PyFloat_AS_DOUBLE(item) : PyFloat_AsDouble(item);
        if (PyErr_Occurred() || !std::isfinite(x)) {
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "x must be finite");
            Py_DECREF(seq);
            Py_DECREF(out);
            return nullptr;
        }
        auto it = inclusive ? std::upper_bound(s->query.values.begin(), s->query.values.end(), x)
                            : std::lower_bound(s->query.values.begin(), s->query.values.end(), x);
        double rank = 0.0;
        if (it != s->query.values.begin()) {
            rank = static_cast<double>(s->query.prefix[
                static_cast<std::size_t>(it - s->query.values.begin() - 1)]);
        }
        PyObject* value = PyFloat_FromDouble(rank);
        if (!value) {
            Py_DECREF(seq);
            Py_DECREF(out);
            return nullptr;
        }
        PyList_SET_ITEM(out, i, value);
    }
    Py_DECREF(seq);
    return out;
}

static PyMethodDef addon_methods[] = {
    {"state_create", state_create, METH_VARARGS, "Create persistent native KLL state."},
    {"state_stats", state_stats, METH_VARARGS, "Return persistent native KLL state statistics."},
    {"state_export", state_export, METH_VARARGS, "Export persistent native KLL state."},
    {"state_ingest", state_ingest, METH_VARARGS, "Bulk ingest into persistent native KLL state."},
    {"state_merge", state_merge, METH_VARARGS, "Merge persistent native KLL states."},
    {"state_quantiles", state_quantiles, METH_VARARGS, "Query persistent native KLL state."},
    {"state_ranks", state_ranks, METH_VARARGS, "Rank-query persistent native KLL state."},
    {nullptr, nullptr, 0, nullptr},
};
