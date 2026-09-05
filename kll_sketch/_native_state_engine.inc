struct QueryView {
    std::vector<double> values;
    std::vector<std::uint64_t> prefix;
};

struct SketchState {
    std::vector<std::vector<double>> levels;
    std::vector<std::vector<double>> sorted;
    std::vector<std::uint64_t> capacities;
    std::uint64_t total_capacity = 0;
    std::uint64_t n = 0;
    std::uint64_t k = 0;
    // v3.2 merge metadata cache. Zero means "not loaded from the owning KLL
    // object yet"; valid public k/min_k values are always >= 40.
    std::uint64_t min_k_cache = 0;
    std::uint64_t rng_state = 0;
    std::uint64_t compactions = 0;
    std::uint64_t retained = 0;
    std::uint64_t min_level_capacity = 8;
    int max_levels = 64;
    bool has_values = false;
    double min_value = 0.0;
    double max_value = 0.0;
    bool level0_sorted = true;
    bool query_valid = false;
    QueryView query;
};

void capsule_destructor(PyObject* capsule) {
    void* ptr = PyCapsule_GetPointer(capsule, CAPSULE_NAME);
    if (!ptr) { PyErr_Clear(); return; }
    delete static_cast<SketchState*>(ptr);
}
inline SketchState* state_from_capsule(PyObject* capsule) {
    return static_cast<SketchState*>(PyCapsule_GetPointer(capsule, CAPSULE_NAME));
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
    s.sorted.resize(count);
    return rebuild_capacities(s);
}

// std::inplace_merge is permitted to acquire a temporary buffer on every call.
// Merge-heavy KLL workloads hit this path repeatedly across levels, and the
// allocator traffic shows up disproportionately in cold 8-way merges. Keep one
// scratch vector per calling thread instead. CPython calls this native backend
// with the GIL held, while thread_local storage also keeps independent Python
// threads isolated. Equal values choose the existing destination run first,
// exactly matching stable-merge semantics (including +0.0/-0.0 ordering).
inline std::vector<double>& stable_merge_scratch() {
    thread_local std::vector<double> scratch;
    return scratch;
}

template <class RandomIt>
inline void stable_append_sorted_range(std::vector<double>& dst, RandomIt first, RandomIt last) {
    if (first == last) return;
    if (dst.empty()) {
        dst.insert(dst.end(), first, last);
        return;
    }

    const std::size_t incoming = static_cast<std::size_t>(last - first);
    if (incoming > std::numeric_limits<std::size_t>::max() - dst.size()) {
        throw std::bad_alloc();
    }
    const std::size_t total = dst.size() + incoming;
    auto& scratch = stable_merge_scratch();
    scratch.clear();
    if (scratch.capacity() < total) scratch.reserve(total);

    auto left = dst.cbegin();
    const auto left_end = dst.cend();
    auto right = first;
    while (left != left_end && right != last) {
        if (*right < *left) {
            scratch.push_back(*right);
            ++right;
        } else {
            scratch.push_back(*left);
            ++left;
        }
    }
    scratch.insert(scratch.end(), left, left_end);
    scratch.insert(scratch.end(), right, last);
    dst.swap(scratch);
}

inline void stable_append_sorted(std::vector<double>& dst, const std::vector<double>& incoming) {
    stable_append_sorted_range(dst, incoming.begin(), incoming.end());
}

bool compact_level(SketchState& s, std::size_t h) {
    if (h >= s.levels.size() || s.levels[h].size() < 2U) {
        PyErr_SetString(PyExc_RuntimeError, "native state attempted invalid compaction");
        return false;
    }
    if (!ensure_levels(s, h + 2U)) return false;
    if (h == 0U && !s.level0_sorted) {
        s.sorted[0] = s.levels[0];
        sort_level(s.sorted[0]);
        s.level0_sorted = true;
    }
    if (s.sorted[h].size() != s.levels[h].size()) {
        PyErr_SetString(PyExc_RuntimeError, "native sorted shadow is out of sync");
        return false;
    }

    auto& source = s.sorted[h];
    const std::size_t old_count = source.size();
    std::size_t start = 0;
    std::size_t stop = old_count;
    bool has_leftover = false;
    double leftover = 0.0;
    if (old_count & 1U) {
        has_leftover = true;
        if (next_bit(s.rng_state)) { leftover = source.back(); --stop; }
        else { leftover = source.front(); ++start; }
    }
    const int offset = next_bit(s.rng_state);
    const std::size_t promoted_count = (stop - start) / 2U;
    if (!promoted_count) {
        PyErr_SetString(PyExc_RuntimeError, "native compaction produced no promoted items");
        return false;
    }

    // Write selected items directly into the canonical raw next level. The
    // selected sequence is already sorted because `source` is sorted. Reuse
    // that freshly appended raw run to update the sorted shadow, eliminating
    // the old temporary `promoted` allocation and its extra population pass.
    auto& raw_next = s.levels[h + 1U];
    const std::size_t raw_next_old = raw_next.size();
    raw_next.reserve(raw_next_old + promoted_count);
    for (std::size_t i = start + static_cast<std::size_t>(offset); i < stop; i += 2U) {
        raw_next.push_back(source[i]);
    }
    stable_append_sorted_range(
        s.sorted[h + 1U],
        raw_next.begin() + static_cast<std::ptrdiff_t>(raw_next_old),
        raw_next.end());

    auto& raw = s.levels[h];
    raw.clear();
    source.clear();
    if (has_leftover) { raw.push_back(leftover); source.push_back(leftover); }
    if (h == 0U) s.level0_sorted = true;

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
            if (s.levels[h].size() > s.capacities[h]) { found = true; break; }
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
                 double batch_min, double batch_max) {
    if (count > static_cast<std::size_t>(U64_MAX - s.n)) {
        PyErr_SetString(PyExc_OverflowError, "total sketch weight exceeds uint64 serialization range");
        return false;
    }
    if (count) {
        if (!s.has_values) {
            s.has_values = true; s.min_value = batch_min; s.max_value = batch_max;
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
        s.level0_sorted = false;
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
        Py_XDECREF(e); Py_XDECREF(f); Py_XDECREF(g); return nullptr;
    }
    PyObject* out = PyTuple_Pack(7, a,b,c,d,e,f,g);
    Py_DECREF(a); Py_DECREF(b); Py_DECREF(c); Py_DECREF(d); Py_DECREF(e); Py_DECREF(f); Py_DECREF(g);
    return out;
}