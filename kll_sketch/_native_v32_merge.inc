// v3.2: specialized first-merge adoption path.
//
// Merging a resident source into a fresh empty destination with the same k
// cannot trigger compaction: the source hierarchy is already within the same
// capacity geometry.  Instead of bootstrapping an empty native state, running
// the structural planner, and appending every level twice (raw + sorted), copy
// the already-valid hierarchy directly.  The destination keeps its own RNG and
// compaction stream exactly as the Python reference merge does.

bool read_empty_adoption_config_v32(
    PyObject* self,
    const SketchState& src,
    std::uint64_t& rng_state,
    std::uint64_t& compactions
) {
    std::uint64_t n = 0;
    std::uint64_t k = 0;
    if (!read_u64_attr(self, "_n", n) || !read_u64_attr(self, "_k", k)) return false;
    if (n != 0ULL || k != src.k) return false;
    if (!read_u64_attr(self, "_rng_state", rng_state) ||
        !read_u64_attr(self, "_compaction_count", compactions)) return false;
    return src.levels.size() <= static_cast<std::size_t>(src.max_levels) &&
           src.sorted.size() == src.levels.size() &&
           src.capacities.size() == src.levels.size() &&
           src.retained <= src.total_capacity;
}

PyObject* adopt_empty_resident_merge_v32(
    PyObject* self,
    PyObject* other,
    const SketchState& src,
    std::uint64_t rng_state,
    std::uint64_t compactions
) {
    if (!g_state_handle_type) {
        PyErr_SetString(PyExc_RuntimeError, "native state handle type is not installed");
        return nullptr;
    }
    if (!inherit_min_k_if_needed(self, other, src)) return nullptr;

    auto* state = new (std::nothrow) SketchState();
    if (!state) return PyErr_NoMemory();
    try {
        state->levels = src.levels;
        state->sorted = src.sorted;
        state->capacities = src.capacities;
    } catch (const std::bad_alloc&) {
        delete state;
        return PyErr_NoMemory();
    }
    state->total_capacity = src.total_capacity;
    state->n = src.n;
    state->k = src.k;
    state->rng_state = rng_state;
    state->compactions = compactions;
    state->retained = src.retained;
    state->min_level_capacity = src.min_level_capacity;
    state->max_levels = src.max_levels;
    state->has_values = src.has_values;
    state->min_value = src.min_value;
    state->max_value = src.max_value;
    state->level0_sorted = src.level0_sorted;
    state->query_valid = false;

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

    const bool mirrored = mirror_merge_visible_state(self, *state, false, 0.0, 0.0);
    Py_DECREF(capsule);
    if (!mirrored) return nullptr;
    Py_RETURN_NONE;
}

PyObject* fast_merge_method_v32(
    PyObject* self,
    PyObject* const* args,
    Py_ssize_t nargs,
    PyObject* kwnames
) {
    if (!g_type_fastpaths_enabled || nargs != 1 || (kwnames && PyTuple_GET_SIZE(kwnames))) {
        return fast_merge_method(self, args, nargs, kwnames);
    }
    PyObject* other = args[0];
    if (!g_kll_type ||
        !PyObject_TypeCheck(other, reinterpret_cast<PyTypeObject*>(g_kll_type)) ||
        other == self) {
        return fast_merge_method(self, args, nargs, kwnames);
    }

    // The v3.2 optimization is deliberately restricted to a nonresident empty
    // destination and a resident source.  All established resident/nonempty,
    // keyword, disabled, and fallback behavior remains v3.1.
    PyObject* dst_capsule = nullptr;
    SketchState* dst = resident_state_from_kll(self, &dst_capsule);
    if (dst) {
        Py_DECREF(dst_capsule);
        return fast_merge_method(self, args, nargs, kwnames);
    }
    if (PyErr_Occurred()) return nullptr;

    PyObject* src_capsule = nullptr;
    SketchState* src = resident_state_from_kll(other, &src_capsule);
    if (!src) {
        if (PyErr_Occurred()) return nullptr;
        return fast_merge_method(self, args, nargs, kwnames);
    }
    if (!src->n) {
        Py_DECREF(src_capsule);
        Py_RETURN_NONE;
    }

    std::uint64_t rng_state = 0;
    std::uint64_t compactions = 0;
    const bool eligible = read_empty_adoption_config_v32(
        self, *src, rng_state, compactions);
    if (!eligible) {
        Py_DECREF(src_capsule);
        if (PyErr_Occurred()) return nullptr;
        return fast_merge_method(self, args, nargs, kwnames);
    }

    PyObject* out = adopt_empty_resident_merge_v32(
        self, other, *src, rng_state, compactions);
    Py_DECREF(src_capsule);
    return out;
}

static PyMethodDef fast_merge_v32_def = {
    "merge",
    reinterpret_cast<PyCFunction>(fast_merge_method_v32),
    METH_FASTCALL | METH_KEYWORDS,
    "Merge another KLL with an optimized empty-destination native adoption path."
};

PyObject* install_v32_merge_fastpath(PyObject*, PyObject*) {
    if (!g_kll_type || !g_state_handle_type || !g_fallback_merge) {
        PyErr_SetString(PyExc_RuntimeError, "install_type_fastpaths must run before v3.2 merge fastpath");
        return nullptr;
    }
    PyObject* descriptor = PyDescr_NewMethod(
        reinterpret_cast<PyTypeObject*>(g_kll_type), &fast_merge_v32_def);
    if (!descriptor) return nullptr;
    const int rc = PyObject_SetAttrString(g_kll_type, "merge", descriptor);
    Py_DECREF(descriptor);
    if (rc < 0) return nullptr;
    Py_RETURN_NONE;
}

static PyMethodDef v32_merge_methods[] = {
    {"install_v32_merge_fastpath", install_v32_merge_fastpath, METH_NOARGS,
     "Install the v3.2 empty-destination merge optimization."},
    {nullptr, nullptr, 0, nullptr},
};
