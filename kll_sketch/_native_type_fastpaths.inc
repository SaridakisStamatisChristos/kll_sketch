// Direct descriptors for the hottest KLL public methods. The Python runtime
// dispatcher remains the canonical semantic fallback; these descriptors only
// bypass Python framing when resident native state is already available.

static PyObject* g_kll_type = nullptr;
static PyObject* g_state_handle_type = nullptr;
static PyObject* g_fallback_quantiles_at = nullptr;
static PyObject* g_fallback_merge = nullptr;
static int g_type_fastpaths_enabled = 1;
constexpr std::uint64_t MAX_EXACT_DOUBLE_INTEGER = 1ULL << 53U;

inline PyObject* call_fallback_vector(
    PyObject* callable,
    PyObject* self,
    PyObject* const* args,
    Py_ssize_t nargs,
    PyObject* kwnames
) {
    if (!callable) {
        PyErr_SetString(PyExc_RuntimeError, "native fast-path fallback is not installed");
        return nullptr;
    }
    PyObject* positional = PyTuple_New(nargs + 1);
    if (!positional) return nullptr;
    PyTuple_SET_ITEM(positional, 0, Py_NewRef(self));
    for (Py_ssize_t i = 0; i < nargs; ++i) {
        PyTuple_SET_ITEM(positional, i + 1, Py_NewRef(args[i]));
    }

    PyObject* kwargs = nullptr;
    const Py_ssize_t nkw = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;
    if (nkw) {
        kwargs = PyDict_New();
        if (!kwargs) {
            Py_DECREF(positional);
            return nullptr;
        }
        for (Py_ssize_t i = 0; i < nkw; ++i) {
            PyObject* key = PyTuple_GET_ITEM(kwnames, i);
            PyObject* value = args[nargs + i];
            if (PyDict_SetItem(kwargs, key, value) < 0) {
                Py_DECREF(kwargs);
                Py_DECREF(positional);
                return nullptr;
            }
        }
    }

    PyObject* out = PyObject_Call(callable, positional, kwargs);
    Py_XDECREF(kwargs);
    Py_DECREF(positional);
    return out;
}

inline PyObject* call_quantiles_fallback(
    PyObject* self,
    PyObject* const* args,
    Py_ssize_t nargs,
    PyObject* kwnames
) {
    return call_fallback_vector(g_fallback_quantiles_at, self, args, nargs, kwnames);
}

inline PyObject* call_merge_fallback(
    PyObject* self,
    PyObject* const* args,
    Py_ssize_t nargs,
    PyObject* kwnames
) {
    return call_fallback_vector(g_fallback_merge, self, args, nargs, kwnames);
}

inline SketchState* resident_state_from_kll(PyObject* self, PyObject** capsule_out) {
    *capsule_out = nullptr;

    // Runtime stores the raw state capsule in `_cache_prefix` while a resident
    // state is active. Nonresident sketches keep the ordinary Python list in
    // that slot. One lookup + an exact capsule check replaces the old chain:
    // cache_generation -> handle -> handle.capsule.
    PyObject* capsule = PyObject_GetAttrString(self, "_cache_prefix");
    if (!capsule) return nullptr;
    if (!PyCapsule_CheckExact(capsule)) {
        Py_DECREF(capsule);
        return nullptr;
    }

    SketchState* state = state_from_capsule(capsule);
    if (!state) {
        Py_DECREF(capsule);
        return nullptr;
    }
    *capsule_out = capsule;
    return state;
}

inline bool read_u64_attr(PyObject* self, const char* name, std::uint64_t& out) {
    PyObject* obj = PyObject_GetAttrString(self, name);
    if (!obj) return false;
    out = PyLong_AsUnsignedLongLong(obj);
    Py_DECREF(obj);
    return !PyErr_Occurred();
}

inline bool read_positive_int_attr(PyObject* self, const char* name, int& out) {
    PyObject* obj = PyObject_GetAttrString(self, name);
    if (!obj) return false;
    const long value = PyLong_AsLong(obj);
    Py_DECREF(obj);
    if (PyErr_Occurred()) return false;
    if (value <= 0 || value > std::numeric_limits<int>::max()) {
        PyErr_SetString(PyExc_ValueError, "invalid native KLL limit");
        return false;
    }
    out = static_cast<int>(value);
    return true;
}

SketchState* bootstrap_empty_resident_state(PyObject* self, PyObject** capsule_out) {
    *capsule_out = nullptr;
    if (!g_state_handle_type) return nullptr;

    std::uint64_t n = 0;
    std::uint64_t retained = 0;
    if (!read_u64_attr(self, "_n", n) || !read_u64_attr(self, "_num_retained", retained)) {
        return nullptr;
    }
    if (n != 0ULL || retained != 0ULL) return nullptr;

    std::uint64_t k = 0;
    std::uint64_t rng_state = 0;
    std::uint64_t compactions = 0;
    int min_level_capacity = 0;
    int max_levels = 0;
    if (!read_u64_attr(self, "_k", k) ||
        !read_u64_attr(self, "_rng_state", rng_state) ||
        !read_u64_attr(self, "_compaction_count", compactions) ||
        !read_positive_int_attr(self, "_MIN_LEVEL_CAPACITY", min_level_capacity) ||
        !read_positive_int_attr(self, "_MAX_LEVELS", max_levels)) {
        return nullptr;
    }
    if (max_levels > 64) {
        PyErr_SetString(PyExc_ValueError, "invalid native maximum level count");
        return nullptr;
    }

    auto* state = new (std::nothrow) SketchState();
    if (!state) {
        PyErr_NoMemory();
        return nullptr;
    }
    state->n = 0;
    state->k = k;
    state->rng_state = rng_state;
    state->compactions = compactions;
    state->retained = 0;
    state->min_level_capacity = static_cast<std::uint64_t>(min_level_capacity);
    state->max_levels = max_levels;
    state->has_values = false;
    state->levels.resize(1);
    state->sorted.resize(1);
    state->level0_sorted = true;
    state->query_valid = false;
    if (!rebuild_capacities(*state)) {
        delete state;
        return nullptr;
    }

    PyObject* capsule = PyCapsule_New(state, CAPSULE_NAME, capsule_destructor);
    if (!capsule) {
        delete state;
        return nullptr;
    }
    PyObject* handle = PyObject_CallOneArg(g_state_handle_type, capsule);
    if (!handle) {
        Py_DECREF(capsule);
        return nullptr;
    }
    PyObject* sentinel = PyLong_FromLong(-2L);
    if (!sentinel) {
        Py_DECREF(handle);
        Py_DECREF(capsule);
        return nullptr;
    }

    // Commit the same resident representation used by `_store_state` in the
    // Python dispatcher. This is only the empty-destination bootstrap; any
    // non-empty/nonresident object still takes the canonical Python fallback.
    if (PyObject_SetAttrString(self, "_cache_values", handle) < 0 ||
        PyObject_SetAttrString(self, "_cache_prefix", capsule) < 0 ||
        PyObject_SetAttrString(self, "_cache_generation", sentinel) < 0) {
        Py_DECREF(sentinel);
        Py_DECREF(handle);
        Py_DECREF(capsule);
        return nullptr;
    }
    Py_DECREF(sentinel);
    Py_DECREF(handle);
    *capsule_out = capsule;  // owned reference transferred to caller
    return state;
}

// -------------------------------------------------------------------------
// Journal-free resident merge
// -------------------------------------------------------------------------
// Compaction *sizes* are independent of random parity and retained values. We
// therefore prove the complete structural merge before touching dst. The hot
// execution can then use the ordinary compactor without copying rollback state.
// The transactional journaled state_merge() remains available as the reference
// fallback for nonresident/error paths.

struct HotMergePlan {
    std::size_t initial_levels = 0;
    std::size_t final_levels = 0;
    std::uint64_t final_retained = 0;
    std::uint64_t final_compactions = 0;
};

bool hot_plan_capacities(
    const SketchState& dst,
    std::size_t level_count,
    std::array<std::uint64_t, 64>& capacities,
    std::uint64_t& total_capacity
) {
    if (level_count == 0U || level_count > static_cast<std::size_t>(dst.max_levels) || level_count > 64U) {
        PyErr_SetString(PyExc_OverflowError, "native merge exceeds maximum supported level count");
        return false;
    }
    total_capacity = 0;
    for (std::size_t h = 0; h < level_count; ++h) {
        std::uint64_t cap = 0;
        if (!level_capacity(dst.k, h, level_count, dst.min_level_capacity, cap)) return false;
        capacities[h] = cap;
        if (total_capacity > U64_MAX - cap) {
            PyErr_SetString(PyExc_OverflowError, "native merge total capacity overflow");
            return false;
        }
        total_capacity += cap;
    }
    return true;
}

bool preflight_resident_merge(
    const SketchState& dst,
    const SketchState& src,
    HotMergePlan& plan
) {
    if (src.n > U64_MAX - dst.n || src.retained > U64_MAX - dst.retained) {
        PyErr_SetString(PyExc_OverflowError, "native merge exceeds uint64 range");
        return false;
    }

    plan.initial_levels = std::max(dst.levels.size(), src.levels.size());
    if (plan.initial_levels == 0U || plan.initial_levels > static_cast<std::size_t>(dst.max_levels) ||
        plan.initial_levels > 64U) {
        PyErr_SetString(PyExc_OverflowError, "native merge exceeds maximum supported level count");
        return false;
    }

    std::array<std::uint64_t, 64> sizes{};
    std::array<std::uint64_t, 64> capacities{};
    for (std::size_t h = 0; h < plan.initial_levels; ++h) {
        const std::uint64_t a = h < dst.levels.size()
            ? static_cast<std::uint64_t>(dst.levels[h].size()) : 0ULL;
        const std::uint64_t b = h < src.levels.size()
            ? static_cast<std::uint64_t>(src.levels[h].size()) : 0ULL;
        if (b > U64_MAX - a) {
            PyErr_SetString(PyExc_OverflowError, "native merge level size overflow");
            return false;
        }
        sizes[h] = a + b;
    }

    std::size_t level_count = plan.initial_levels;
    std::uint64_t total_capacity = 0;
    // Common resident case: source hierarchy already fits in destination.
    // Reuse the exact capacity table maintained by the live sketch instead of
    // recomputing level_capacity() for every merge. Recompute only when the
    // source initially has more levels or a simulated top-level compaction
    // changes the geometry.
    if (level_count == dst.levels.size() && dst.capacities.size() == level_count) {
        std::copy(dst.capacities.begin(), dst.capacities.end(), capacities.begin());
        total_capacity = dst.total_capacity;
    } else if (!hot_plan_capacities(dst, level_count, capacities, total_capacity)) {
        return false;
    }

    std::uint64_t retained = dst.retained + src.retained;
    std::uint64_t compactions = dst.compactions;
    int guard = 0;
    while (retained > total_capacity) {
        std::size_t h = 0;
        bool found = false;
        for (; h < level_count; ++h) {
            if (sizes[h] > capacities[h]) {
                found = true;
                break;
            }
        }
        if (!found) {
            PyErr_SetString(PyExc_RuntimeError, "native KLL capacity accounting became inconsistent");
            return false;
        }

        // compact_level() grows the hierarchy before consuming the selected
        // level, so capacity geometry for subsequent iterations changes here.
        if (h + 2U > level_count) {
            level_count = h + 2U;
            if (level_count > static_cast<std::size_t>(dst.max_levels) || level_count > 64U) {
                PyErr_SetString(PyExc_OverflowError, "native sketch exceeds maximum supported level count");
                return false;
            }
            if (!hot_plan_capacities(dst, level_count, capacities, total_capacity)) return false;
        }

        const std::uint64_t old_count = sizes[h];
        if (old_count < 2ULL) {
            PyErr_SetString(PyExc_RuntimeError, "native state attempted invalid compaction");
            return false;
        }
        const std::uint64_t leftover = old_count & 1ULL;
        const std::uint64_t promoted = old_count / 2ULL;
        if (sizes[h + 1U] > U64_MAX - promoted) {
            PyErr_SetString(PyExc_OverflowError, "native merge promoted level overflow");
            return false;
        }
        sizes[h] = leftover;
        sizes[h + 1U] += promoted;
        retained = retained - old_count + leftover + promoted;
        if (compactions == U64_MAX) {
            PyErr_SetString(PyExc_OverflowError, "native compaction count overflow");
            return false;
        }
        ++compactions;
        if (++guard > 10000) {
            PyErr_SetString(PyExc_RuntimeError, "native KLL compaction did not converge");
            return false;
        }
    }

    plan.final_levels = level_count;
    plan.final_retained = retained;
    plan.final_compactions = compactions;
    return true;
}

bool merge_resident_states_preflighted(
    SketchState& dst,
    const SketchState& src,
    bool& mutation_started
) {
    mutation_started = false;
    if (!src.n) return true;

    HotMergePlan plan;
    if (!preflight_resident_merge(dst, src, plan)) return false;

    // This hierarchy growth is part of canonical merge semantics and occurs
    // before source levels are appended.
    if (!ensure_levels(dst, plan.initial_levels)) return false;

    mutation_started = true;
    for (std::size_t h = 0; h < src.levels.size(); ++h) {
        if (src.levels[h].empty()) continue;
        dst.levels[h].insert(dst.levels[h].end(), src.levels[h].begin(), src.levels[h].end());
        if (h == 0U) dst.level0_sorted = false;
        else stable_append_sorted(dst.sorted[h], src.sorted[h]);
    }
    dst.n += src.n;
    dst.retained += src.retained;
    if (src.has_values) {
        if (!dst.has_values) {
            dst.has_values = true;
            dst.min_value = src.min_value;
            dst.max_value = src.max_value;
        } else {
            if (src.min_value < dst.min_value) dst.min_value = src.min_value;
            if (src.max_value > dst.max_value) dst.max_value = src.max_value;
        }
    }

    if (!compress(dst)) return false;
    if (dst.levels.size() != plan.final_levels ||
        dst.retained != plan.final_retained ||
        dst.compactions != plan.final_compactions) {
        PyErr_SetString(PyExc_RuntimeError, "native merge diverged from structural preflight");
        return false;
    }
    dst.query_valid = false;
    return true;
}

PyObject* state_merge_retained(PyObject*, PyObject* args) {
    PyObject *dst_obj = nullptr, *src_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OO:state_merge_retained", &dst_obj, &src_obj)) return nullptr;
    SketchState* dst = state_from_capsule(dst_obj);
    if (!dst) return nullptr;
    SketchState* src = state_from_capsule(src_obj);
    if (!src) return nullptr;
    if (dst == src) {
        PyErr_SetString(PyExc_ValueError, "cannot merge native state with itself");
        return nullptr;
    }
    bool mutation_started = false;
    if (!merge_resident_states_preflighted(*dst, *src, mutation_started)) return nullptr;
    return PyLong_FromUnsignedLongLong(dst->retained);
}

inline bool set_u64_attr(PyObject* self, const char* name, std::uint64_t value) {
    PyObject* obj = PyLong_FromUnsignedLongLong(value);
    if (!obj) return false;
    const int rc = PyObject_SetAttrString(self, name, obj);
    Py_DECREF(obj);
    return rc == 0;
}

inline bool set_double_attr(PyObject* self, const char* name, double value) {
    PyObject* obj = PyFloat_FromDouble(value);
    if (!obj) return false;
    const int rc = PyObject_SetAttrString(self, name, obj);
    Py_DECREF(obj);
    return rc == 0;
}

inline bool inherit_min_k_if_needed(PyObject* self, PyObject* other, const SketchState& src) {
    if (!(src.n > src.k || src.levels.size() > 1U)) return true;
    PyObject* dst_min_k = PyObject_GetAttrString(self, "_min_k");
    if (!dst_min_k) return false;
    PyObject* src_min_k = PyObject_GetAttrString(other, "_min_k");
    if (!src_min_k) {
        Py_DECREF(dst_min_k);
        return false;
    }
    const int smaller = PyObject_RichCompareBool(src_min_k, dst_min_k, Py_LT);
    Py_DECREF(dst_min_k);
    if (smaller < 0) {
        Py_DECREF(src_min_k);
        return false;
    }
    if (smaller && PyObject_SetAttrString(self, "_min_k", src_min_k) < 0) {
        Py_DECREF(src_min_k);
        return false;
    }
    Py_DECREF(src_min_k);
    return true;
}

inline bool mirror_merge_visible_state(
    PyObject* self,
    const SketchState& dst,
    bool old_has_values,
    double old_min,
    double old_max
) {
    if (!set_u64_attr(self, "_n", dst.n)) return false;
    if (!set_u64_attr(self, "_num_retained", dst.retained)) return false;
    if (dst.has_values) {
        if (!old_has_values || dst.min_value != old_min) {
            if (!set_double_attr(self, "_min_value", dst.min_value)) return false;
        }
        if (!old_has_values || dst.max_value != old_max) {
            if (!set_double_attr(self, "_max_value", dst.max_value)) return false;
        }
    }
    // `_generation` only invalidates the Python materialization cache. Resident
    // native state uses its own query_valid flag and `_sync_state()` clears the
    // Python cache before any Python-only path, so incrementing a Python bigint
    // on every native merge is unnecessary hot-path work.
    return true;
}

PyObject* fast_merge_method(
    PyObject* self,
    PyObject* const* args,
    Py_ssize_t nargs,
    PyObject* kwnames
) {
    if (!g_type_fastpaths_enabled || nargs != 1 || (kwnames && PyTuple_GET_SIZE(kwnames))) {
        return call_merge_fallback(self, args, nargs, kwnames);
    }
    PyObject* other = args[0];
    if (!g_kll_type || !PyObject_TypeCheck(other, reinterpret_cast<PyTypeObject*>(g_kll_type)) || other == self) {
        return call_merge_fallback(self, args, nargs, kwnames);
    }

    PyObject* dst_capsule = nullptr;
    SketchState* dst = resident_state_from_kll(self, &dst_capsule);
    if (!dst) {
        if (PyErr_Occurred()) return nullptr;
        dst = bootstrap_empty_resident_state(self, &dst_capsule);
        if (!dst) {
            if (PyErr_Occurred()) return nullptr;
            return call_merge_fallback(self, args, nargs, kwnames);
        }
    }
    PyObject* src_capsule = nullptr;
    SketchState* src = resident_state_from_kll(other, &src_capsule);
    if (!src) {
        Py_DECREF(dst_capsule);
        if (PyErr_Occurred()) return nullptr;
        return call_merge_fallback(self, args, nargs, kwnames);
    }

    if (!src->n) {
        Py_DECREF(src_capsule);
        Py_DECREF(dst_capsule);
        Py_RETURN_NONE;
    }
    if (dst == src) {
        Py_DECREF(src_capsule);
        Py_DECREF(dst_capsule);
        return call_merge_fallback(self, args, nargs, kwnames);
    }

    if (!inherit_min_k_if_needed(self, other, *src)) {
        Py_DECREF(src_capsule);
        Py_DECREF(dst_capsule);
        return nullptr;
    }

    const bool old_has_values = dst->has_values;
    const double old_min = dst->min_value;
    const double old_max = dst->max_value;
    bool mutation_started = false;
    if (!merge_resident_states_preflighted(*dst, *src, mutation_started)) {
        Py_DECREF(src_capsule);
        Py_DECREF(dst_capsule);
        // Only a preflight/early failure is replay-safe. Once data mutation has
        // begun, an impossible internal failure is surfaced rather than risking
        // a second merge through the fallback path.
        if (!mutation_started &&
            (PyErr_ExceptionMatches(PyExc_TypeError) ||
             PyErr_ExceptionMatches(PyExc_ValueError) ||
             PyErr_ExceptionMatches(PyExc_OverflowError))) {
            PyErr_Clear();
            return call_merge_fallback(self, args, nargs, kwnames);
        }
        return nullptr;
    }

    const bool mirrored = mirror_merge_visible_state(self, *dst, old_has_values, old_min, old_max);
    Py_DECREF(src_capsule);
    Py_DECREF(dst_capsule);
    if (!mirrored) return nullptr;
    Py_RETURN_NONE;
}

PyObject* fast_quantiles_at_method(
    PyObject* self,
    PyObject* const* args,
    Py_ssize_t nargs,
    PyObject* kwnames
) {
    if (!g_type_fastpaths_enabled || nargs != 1 || (kwnames && PyTuple_GET_SIZE(kwnames))) {
        return call_quantiles_fallback(self, args, nargs, kwnames);
    }
    PyObject* probabilities = args[0];

    PyObject* capsule = nullptr;
    SketchState* state = resident_state_from_kll(self, &capsule);
    if (!state) {
        if (PyErr_Occurred()) return nullptr;
        return call_quantiles_fallback(self, args, nargs, kwnames);
    }

    if (state->n > MAX_EXACT_DOUBLE_INTEGER) {
        Py_DECREF(capsule);
        return call_quantiles_fallback(self, args, nargs, kwnames);
    }

    PyObject* seq = PySequence_Fast(probabilities, "probabilities must be an iterable");
    if (!seq) {
        Py_DECREF(capsule);
        return nullptr;
    }
    const Py_ssize_t count = PySequence_Fast_GET_SIZE(seq);
    if (count == 0) {
        Py_DECREF(seq);
        Py_DECREF(capsule);
        return PyList_New(0);
    }
    if (state->n == 0) {
        Py_DECREF(seq);
        Py_DECREF(capsule);
        PyErr_SetString(PyExc_ValueError, "empty sketch");
        return nullptr;
    }
    if (!build_query(*state)) {
        Py_DECREF(seq);
        Py_DECREF(capsule);
        return nullptr;
    }

    PyObject* out = PyList_New(count);
    if (!out) {
        Py_DECREF(seq);
        Py_DECREF(capsule);
        return nullptr;
    }

    double previous_q = -1.0;
    std::size_t lower_hint = 0;
    for (Py_ssize_t i = 0; i < count; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
        const double q = PyFloat_CheckExact(item) ? PyFloat_AS_DOUBLE(item) : PyFloat_AsDouble(item);
        if (PyErr_Occurred() || !std::isfinite(q) || q < 0.0 || q > 1.0) {
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "probability must be in [0,1]");
            Py_DECREF(out);
            Py_DECREF(seq);
            Py_DECREF(capsule);
            return nullptr;
        }

        double answer = 0.0;
        if (q <= 0.0) {
            answer = state->min_value;
        } else if (q >= 1.0) {
            answer = state->max_value;
            if (q >= previous_q && !state->query.values.empty()) {
                lower_hint = state->query.values.size() - 1U;
            }
        } else {
            const double target = q * static_cast<double>(state->n - 1ULL);
            auto begin = state->query.prefix.begin();
            if (q >= previous_q && lower_hint < state->query.prefix.size()) {
                begin += static_cast<std::ptrdiff_t>(lower_hint);
            } else {
                lower_hint = 0;
            }
            auto it = std::upper_bound(
                begin, state->query.prefix.end(), target,
                [](double lhs, std::uint64_t rhs) { return lhs < static_cast<double>(rhs); });
            std::size_t pos = static_cast<std::size_t>(it - state->query.prefix.begin());
            if (pos >= state->query.values.size()) pos = state->query.values.size() - 1U;
            answer = state->query.values[pos];
            if (q >= previous_q) lower_hint = pos;
        }
        previous_q = q;

        PyObject* value = PyFloat_FromDouble(answer);
        if (!value) {
            Py_DECREF(out);
            Py_DECREF(seq);
            Py_DECREF(capsule);
            return nullptr;
        }
        PyList_SET_ITEM(out, i, value);
    }

    Py_DECREF(seq);
    Py_DECREF(capsule);
    return out;
}

static PyMethodDef fast_quantiles_at_def = {
    "quantiles_at",
    reinterpret_cast<PyCFunction>(fast_quantiles_at_method),
    METH_FASTCALL | METH_KEYWORDS,
    "Return approximate quantiles using the resident native query view when available."
};

static PyMethodDef fast_merge_def = {
    "merge",
    reinterpret_cast<PyCFunction>(fast_merge_method),
    METH_FASTCALL | METH_KEYWORDS,
    "Merge another KLL using resident native state when available."
};

PyObject* install_type_fastpaths(PyObject*, PyObject* args) {
    PyObject* type_obj = nullptr;
    PyObject* quantiles_fallback = nullptr;
    PyObject* merge_fallback = nullptr;
    PyObject* state_handle_type = nullptr;
    if (!PyArg_ParseTuple(args, "OOOO:install_type_fastpaths",
                          &type_obj, &quantiles_fallback, &merge_fallback, &state_handle_type)) {
        return nullptr;
    }
    if (!PyType_Check(type_obj) || !PyCallable_Check(quantiles_fallback) ||
        !PyCallable_Check(merge_fallback) || !PyCallable_Check(state_handle_type)) {
        PyErr_SetString(PyExc_TypeError, "expected KLL type, callable fallbacks, and state handle type");
        return nullptr;
    }

    PyObject* quantiles_descriptor = PyDescr_NewMethod(
        reinterpret_cast<PyTypeObject*>(type_obj), &fast_quantiles_at_def);
    if (!quantiles_descriptor) return nullptr;
    PyObject* merge_descriptor = PyDescr_NewMethod(
        reinterpret_cast<PyTypeObject*>(type_obj), &fast_merge_def);
    if (!merge_descriptor) {
        Py_DECREF(quantiles_descriptor);
        return nullptr;
    }

    if (PyObject_SetAttrString(type_obj, "quantiles_at", quantiles_descriptor) < 0 ||
        PyObject_SetAttrString(type_obj, "merge", merge_descriptor) < 0) {
        Py_DECREF(merge_descriptor);
        Py_DECREF(quantiles_descriptor);
        return nullptr;
    }
    Py_DECREF(merge_descriptor);
    Py_DECREF(quantiles_descriptor);

    Py_XSETREF(g_kll_type, Py_NewRef(type_obj));
    Py_XSETREF(g_fallback_quantiles_at, Py_NewRef(quantiles_fallback));
    Py_XSETREF(g_fallback_merge, Py_NewRef(merge_fallback));
    Py_XSETREF(g_state_handle_type, Py_NewRef(state_handle_type));
    Py_RETURN_NONE;
}

PyObject* set_type_fastpaths_enabled(PyObject*, PyObject* arg) {
    const int enabled = PyObject_IsTrue(arg);
    if (enabled < 0) return nullptr;
    g_type_fastpaths_enabled = enabled;
    Py_RETURN_NONE;
}

static PyMethodDef type_fastpath_methods[] = {
    {"state_merge_retained", state_merge_retained, METH_VARARGS, "Merge resident native KLL states and return retained count."},
    {"install_type_fastpaths", install_type_fastpaths, METH_VARARGS, "Install direct C-level KLL hot methods."},
    {"set_type_fastpaths_enabled", set_type_fastpaths_enabled, METH_O, "Enable or disable direct KLL hot methods."},
    {nullptr, nullptr, 0, nullptr},
};
