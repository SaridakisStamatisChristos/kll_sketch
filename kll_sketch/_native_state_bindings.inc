PyObject* state_create(PyObject*, PyObject* args) {
    PyObject *levels_obj, *n_obj, *k_obj, *rng_obj, *comp_obj, *ret_obj, *min_obj, *max_obj;
    int mincap, maxlevels;
    if (!PyArg_ParseTuple(args, "OOOOOOOOii:state_create", &levels_obj,&n_obj,&k_obj,&rng_obj,
                          &comp_obj,&ret_obj,&min_obj,&max_obj,&mincap,&maxlevels)) return nullptr;
    auto* s = new (std::nothrow) SketchState();
    if (!s) return PyErr_NoMemory();
    s->n = PyLong_AsUnsignedLongLong(n_obj); if (PyErr_Occurred()) { delete s; return nullptr; }
    s->k = PyLong_AsUnsignedLongLong(k_obj); if (PyErr_Occurred()) { delete s; return nullptr; }
    s->rng_state = PyLong_AsUnsignedLongLong(rng_obj); if (PyErr_Occurred()) { delete s; return nullptr; }
    s->compactions = PyLong_AsUnsignedLongLong(comp_obj); if (PyErr_Occurred()) { delete s; return nullptr; }
    s->retained = PyLong_AsUnsignedLongLong(ret_obj); if (PyErr_Occurred()) { delete s; return nullptr; }
    if (mincap <= 0 || maxlevels <= 0 || maxlevels > 64) {
        delete s; PyErr_SetString(PyExc_ValueError, "invalid native KLL limits"); return nullptr;
    }
    s->min_level_capacity = static_cast<std::uint64_t>(mincap);
    s->max_levels = maxlevels;
    s->has_values = min_obj != Py_None;
    if (s->has_values && (!py_double(min_obj, s->min_value) || !py_double(max_obj, s->max_value))) {
        delete s; return nullptr;
    }
    if (!parse_levels(levels_obj, s->levels)) { delete s; return nullptr; }
    if (s->levels.empty() || s->levels.size() > static_cast<std::size_t>(maxlevels)) {
        delete s; PyErr_SetString(PyExc_ValueError, "invalid native KLL level count"); return nullptr;
    }
    s->sorted = s->levels;
    for (auto& level : s->sorted) sort_level(level);
    s->level0_sorted = true;
    if (!rebuild_capacities(*s)) { delete s; return nullptr; }
    PyObject* capsule = PyCapsule_New(s, CAPSULE_NAME, capsule_destructor);
    if (!capsule) { delete s; return nullptr; }
    return capsule;
}

PyObject* state_stats(PyObject*, PyObject* args) {
    PyObject* capsule;
    if (!PyArg_ParseTuple(args, "O:state_stats", &capsule)) return nullptr;
    SketchState* s = state_from_capsule(capsule); if (!s) return nullptr;
    return stats(*s);
}

PyObject* state_export(PyObject*, PyObject* args) {
    PyObject* capsule;
    if (!PyArg_ParseTuple(args, "O:state_export", &capsule)) return nullptr;
    SketchState* s = state_from_capsule(capsule); if (!s) return nullptr;
    PyObject* levels = levels_to_python(s->levels);
    PyObject* state_stats_obj = stats(*s);
    if (!levels || !state_stats_obj) { Py_XDECREF(levels); Py_XDECREF(state_stats_obj); return nullptr; }
    PyObject* out = PyTuple_Pack(2, levels, state_stats_obj);
    Py_DECREF(levels); Py_DECREF(state_stats_obj); return out;
}

PyObject* state_ingest(PyObject*, PyObject* args) {
    PyObject *capsule, *values_obj;
    if (!PyArg_ParseTuple(args, "OO:state_ingest", &capsule, &values_obj)) return nullptr;
    SketchState* s = state_from_capsule(capsule); if (!s) return nullptr;
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
            if (count && !scan_values(data, count, lo, hi)) {
                PyBuffer_Release(&view); PyErr_SetString(PyExc_ValueError, "native batch contains a non-finite value"); return nullptr;
            }
            const bool ok = ingest_span(*s, data, count, lo, hi);
            PyBuffer_Release(&view);
            if (!ok) { *s = std::move(backup); return nullptr; }
            return stats(*s);
        }
        PyBuffer_Release(&view);
    } else PyErr_Clear();

    PyObject* seq = PySequence_Fast(values_obj, "native batch requires a finite sequence");
    if (!seq) return nullptr;
    const Py_ssize_t count = PySequence_Fast_GET_SIZE(seq);
    std::vector<double> values;
    values.reserve(static_cast<std::size_t>(count));
    double lo = 0.0, hi = 0.0;
    for (Py_ssize_t i = 0; i < count; ++i) {
        double value = 0.0;
        if (!py_double(PySequence_Fast_GET_ITEM(seq, i), value)) { Py_DECREF(seq); return nullptr; }
        if (i == 0) lo = hi = value;
        else { if (value < lo) lo = value; if (value > hi) hi = value; }
        values.push_back(value);
    }
    Py_DECREF(seq);
    if (!ingest_span(*s, values.data(), values.size(), lo, hi)) { *s = std::move(backup); return nullptr; }
    return stats(*s);
}

// Merge rollback is journaled lazily.  The old implementation deep-copied the
// entire destination before every merge, even though successful merges cannot
// hit a semantic failure after the inexpensive preflight checks.  Here we copy
// only raw levels that are actually destroyed by compaction.  On a failure,
// untouched levels are restored by truncating appended tails and sorted shadows
// are deterministically rebuilt from the restored raw representation.
struct MergeUndo {
    std::size_t level_count;
    std::vector<std::size_t> sizes;
    std::vector<std::vector<double>> compacted_originals;
    std::vector<unsigned char> captured;
    std::uint64_t n;
    std::uint64_t retained;
    std::uint64_t rng_state;
    std::uint64_t compactions;
    bool has_values;
    double min_value;
    double max_value;
    bool level0_sorted;
};

MergeUndo make_merge_undo(const SketchState& s) {
    MergeUndo undo;
    undo.level_count = s.levels.size();
    undo.sizes.reserve(undo.level_count);
    for (const auto& level : s.levels) undo.sizes.push_back(level.size());
    undo.compacted_originals.resize(undo.level_count);
    undo.captured.resize(undo.level_count, 0U);
    undo.n = s.n;
    undo.retained = s.retained;
    undo.rng_state = s.rng_state;
    undo.compactions = s.compactions;
    undo.has_values = s.has_values;
    undo.min_value = s.min_value;
    undo.max_value = s.max_value;
    undo.level0_sorted = s.level0_sorted;
    return undo;
}

void capture_merge_level(const SketchState& s, MergeUndo& undo, std::size_t h) {
    if (h >= undo.level_count || undo.captured[h]) return;
    const std::size_t old_size = undo.sizes[h];
    const auto& level = s.levels[h];
    const std::size_t available = std::min(old_size, level.size());
    undo.compacted_originals[h].assign(level.begin(), level.begin() + static_cast<std::ptrdiff_t>(available));
    undo.captured[h] = 1U;
}

bool rollback_merge(SketchState& s, MergeUndo& undo) {
    if (s.levels.size() < undo.level_count) s.levels.resize(undo.level_count);
    for (std::size_t h = 0; h < undo.level_count; ++h) {
        if (undo.captured[h]) s.levels[h] = std::move(undo.compacted_originals[h]);
        else if (s.levels[h].size() >= undo.sizes[h]) s.levels[h].resize(undo.sizes[h]);
        else {
            PyErr_SetString(PyExc_RuntimeError, "native merge rollback lost original level data");
            return false;
        }
    }
    s.levels.resize(undo.level_count);
    s.sorted = s.levels;
    for (auto& level : s.sorted) sort_level(level);
    s.n = undo.n;
    s.retained = undo.retained;
    s.rng_state = undo.rng_state;
    s.compactions = undo.compactions;
    s.has_values = undo.has_values;
    s.min_value = undo.min_value;
    s.max_value = undo.max_value;
    s.level0_sorted = undo.level0_sorted;
    return rebuild_capacities(s);
}

bool compress_merge(SketchState& s, MergeUndo& undo) {
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
        capture_merge_level(s, undo, h);
        if (!compact_level(s, h)) return false;
        if (++guard > 10000) {
            PyErr_SetString(PyExc_RuntimeError, "native KLL compaction did not converge");
            return false;
        }
    }
    return true;
}

PyObject* state_merge(PyObject*, PyObject* args) {
    PyObject *dst_obj, *src_obj;
    if (!PyArg_ParseTuple(args, "OO:state_merge", &dst_obj, &src_obj)) return nullptr;
    SketchState* dst = state_from_capsule(dst_obj); if (!dst) return nullptr;
    SketchState* src = state_from_capsule(src_obj); if (!src) return nullptr;
    if (dst == src) { PyErr_SetString(PyExc_ValueError, "cannot merge native state with itself"); return nullptr; }
    if (!src->n) return stats(*dst);
    if (src->n > U64_MAX - dst->n || src->retained > U64_MAX - dst->retained) {
        PyErr_SetString(PyExc_OverflowError, "native merge exceeds uint64 range"); return nullptr;
    }

    MergeUndo undo = make_merge_undo(*dst);
    if (!ensure_levels(*dst, std::max(dst->levels.size(), src->levels.size()))) {
        rollback_merge(*dst, undo); return nullptr;
    }
    for (std::size_t h = 0; h < src->levels.size(); ++h) {
        if (src->levels[h].empty()) continue;
        dst->levels[h].insert(dst->levels[h].end(), src->levels[h].begin(), src->levels[h].end());
        if (h == 0U) dst->level0_sorted = false;
        else stable_append_sorted(dst->sorted[h], src->sorted[h]);
    }
    dst->n += src->n;
    dst->retained += src->retained;
    if (src->has_values) {
        if (!dst->has_values) { dst->has_values = true; dst->min_value = src->min_value; dst->max_value = src->max_value; }
        else { if (src->min_value < dst->min_value) dst->min_value = src->min_value;
               if (src->max_value > dst->max_value) dst->max_value = src->max_value; }
    }
    if (!compress_merge(*dst, undo)) {
        PyObject *type = nullptr, *value = nullptr, *traceback = nullptr;
        PyErr_Fetch(&type, &value, &traceback);
        const bool restored = rollback_merge(*dst, undo);
        if (restored) PyErr_Restore(type, value, traceback);
        else { Py_XDECREF(type); Py_XDECREF(value); Py_XDECREF(traceback); }
        return nullptr;
    }
    dst->query_valid = false;
    return stats(*dst);
}

PyObject* state_quantiles(PyObject*, PyObject* args) {
    PyObject *capsule, *qs_obj;
    if (!PyArg_ParseTuple(args, "OO:state_quantiles", &capsule, &qs_obj)) return nullptr;
    SketchState* s = state_from_capsule(capsule); if (!s) return nullptr;
    if (!s->n) { PyErr_SetString(PyExc_ValueError, "empty sketch"); return nullptr; }
    if (!build_query(*s)) return nullptr;
    PyObject* seq = PySequence_Fast(qs_obj, "probabilities must be a sequence");
    if (!seq) return nullptr;
    const Py_ssize_t count = PySequence_Fast_GET_SIZE(seq);
    PyObject* out = PyList_New(count);
    if (!out) { Py_DECREF(seq); return nullptr; }
    double prev_q = -1.0;
    std::size_t lower_hint = 0;
    for (Py_ssize_t i = 0; i < count; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
        const double q = PyFloat_CheckExact(item) ? PyFloat_AS_DOUBLE(item) : PyFloat_AsDouble(item);
        if (PyErr_Occurred() || !std::isfinite(q) || q < 0.0 || q > 1.0) {
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "probability must be in [0,1]");
            Py_DECREF(seq); Py_DECREF(out); return nullptr;
        }
        double answer;
        if (q <= 0.0) answer = s->min_value;
        else if (q >= 1.0) {
            answer = s->max_value;
            if (q >= prev_q && !s->query.values.empty()) lower_hint = s->query.values.size() - 1U;
        } else {
            const double target = q * static_cast<double>(s->n - 1ULL);
            auto begin = s->query.prefix.begin();
            if (q >= prev_q && lower_hint < s->query.prefix.size())
                begin += static_cast<std::ptrdiff_t>(lower_hint);
            else lower_hint = 0;
            auto it = std::upper_bound(begin, s->query.prefix.end(), target,
                [](double lhs, std::uint64_t rhs) { return lhs < static_cast<double>(rhs); });
            std::size_t pos = static_cast<std::size_t>(it - s->query.prefix.begin());
            if (pos >= s->query.values.size()) pos = s->query.values.size() - 1U;
            answer = s->query.values[pos];
            if (q >= prev_q) lower_hint = pos;
        }
        prev_q = q;
        PyObject* value = PyFloat_FromDouble(answer);
        if (!value) { Py_DECREF(seq); Py_DECREF(out); return nullptr; }
        PyList_SET_ITEM(out, i, value);
    }
    Py_DECREF(seq); return out;
}

PyObject* state_ranks(PyObject*, PyObject* args) {
    PyObject *capsule, *xs_obj; int inclusive = 1;
    if (!PyArg_ParseTuple(args, "OOp:state_ranks", &capsule, &xs_obj, &inclusive)) return nullptr;
    SketchState* s = state_from_capsule(capsule); if (!s) return nullptr;
    if (!build_query(*s)) return nullptr;
    PyObject* seq = PySequence_Fast(xs_obj, "xs must be a sequence");
    if (!seq) return nullptr;
    const Py_ssize_t count = PySequence_Fast_GET_SIZE(seq);
    PyObject* out = PyList_New(count);
    if (!out) { Py_DECREF(seq); return nullptr; }
    for (Py_ssize_t i = 0; i < count; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
        const double x = PyFloat_CheckExact(item) ? PyFloat_AS_DOUBLE(item) : PyFloat_AsDouble(item);
        if (PyErr_Occurred() || !std::isfinite(x)) {
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "x must be finite");
            Py_DECREF(seq); Py_DECREF(out); return nullptr;
        }
        auto it = inclusive ? std::upper_bound(s->query.values.begin(), s->query.values.end(), x)
                            : std::lower_bound(s->query.values.begin(), s->query.values.end(), x);
        double rank = 0.0;
        if (it != s->query.values.begin()) rank = static_cast<double>(s->query.prefix[static_cast<std::size_t>(it - s->query.values.begin() - 1)]);
        PyObject* value = PyFloat_FromDouble(rank);
        if (!value) { Py_DECREF(seq); Py_DECREF(out); return nullptr; }
        PyList_SET_ITEM(out, i, value);
    }
    Py_DECREF(seq); return out;
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
