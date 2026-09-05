// v3.2 cached-slot merge specialization.
//
// KLL is a slotted Python type. The direct native merge path touches the same
// handful of slots on every call. PyObject_GetAttrString/SetAttrString must
// turn a C string into a Python unicode key repeatedly; at ~40 us for a full
// 8-way merge that framing is measurable. Intern the slot keys once when the
// fast path is installed and use PyObject_GetAttr/SetAttr thereafter.
//
// This layer deliberately changes no sketch mathematics. It delegates
// structural planning/compaction to the exact-sequence v3.2 implementation and
// preserves its fallbacks. Only Python object framing is specialized.

static PyObject* g_slot_cache_prefix_v32 = nullptr;
static PyObject* g_slot_cache_values_v32 = nullptr;
static PyObject* g_slot_cache_generation_v32 = nullptr;
static PyObject* g_slot_n_v32 = nullptr;
static PyObject* g_slot_num_retained_v32 = nullptr;
static PyObject* g_slot_k_v32 = nullptr;
static PyObject* g_slot_rng_state_v32 = nullptr;
static PyObject* g_slot_compaction_count_v32 = nullptr;
static PyObject* g_slot_min_value_v32 = nullptr;
static PyObject* g_slot_max_value_v32 = nullptr;
static PyObject* g_slot_min_k_v32 = nullptr;

bool intern_slot_v32(PyObject*& target, const char* name) {
    PyObject* value = PyUnicode_InternFromString(name);
    if (!value) return false;
    Py_XSETREF(target, value);
    return true;
}

bool init_slots_v32() {
    return intern_slot_v32(g_slot_cache_prefix_v32, "_cache_prefix") &&
           intern_slot_v32(g_slot_cache_values_v32, "_cache_values") &&
           intern_slot_v32(g_slot_cache_generation_v32, "_cache_generation") &&
           intern_slot_v32(g_slot_n_v32, "_n") &&
           intern_slot_v32(g_slot_num_retained_v32, "_num_retained") &&
           intern_slot_v32(g_slot_k_v32, "_k") &&
           intern_slot_v32(g_slot_rng_state_v32, "_rng_state") &&
           intern_slot_v32(g_slot_compaction_count_v32, "_compaction_count") &&
           intern_slot_v32(g_slot_min_value_v32, "_min_value") &&
           intern_slot_v32(g_slot_max_value_v32, "_max_value") &&
           intern_slot_v32(g_slot_min_k_v32, "_min_k");
}

inline SketchState* resident_state_from_kll_slots_v32(
    PyObject* self,
    PyObject** capsule_out
) {
    *capsule_out = nullptr;
    PyObject* capsule = PyObject_GetAttr(self, g_slot_cache_prefix_v32);
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

inline bool read_u64_slot_v32(PyObject* self, PyObject* slot, std::uint64_t& out) {
    PyObject* obj = PyObject_GetAttr(self, slot);
    if (!obj) return false;
    out = PyLong_AsUnsignedLongLong(obj);
    Py_DECREF(obj);
    return !PyErr_Occurred();
}

inline bool set_u64_slot_v32(PyObject* self, PyObject* slot, std::uint64_t value) {
    PyObject* obj = PyLong_FromUnsignedLongLong(value);
    if (!obj) return false;
    const int rc = PyObject_SetAttr(self, slot, obj);
    Py_DECREF(obj);
    return rc == 0;
}

inline bool set_double_slot_v32(PyObject* self, PyObject* slot, double value) {
    PyObject* obj = PyFloat_FromDouble(value);
    if (!obj) return false;
    const int rc = PyObject_SetAttr(self, slot, obj);
    Py_DECREF(obj);
    return rc == 0;
}

inline bool mirror_merge_visible_slots_v32(
    PyObject* self,
    const SketchState& dst,
    bool old_has_values,
    double old_min,
    double old_max
) {
    if (!set_u64_slot_v32(self, g_slot_n_v32, dst.n)) return false;
    if (!set_u64_slot_v32(self, g_slot_num_retained_v32, dst.retained)) return false;
    if (dst.has_values) {
        if (!old_has_values || dst.min_value != old_min) {
            if (!set_double_slot_v32(self, g_slot_min_value_v32, dst.min_value)) return false;
        }
        if (!old_has_values || dst.max_value != old_max) {
            if (!set_double_slot_v32(self, g_slot_max_value_v32, dst.max_value)) return false;
        }
    }
    return true;
}

bool load_min_k_cache_slots_v32(PyObject* owner, SketchState& state) {
    if (state.min_k_cache != 0ULL) return true;
    PyObject* obj = PyObject_GetAttr(owner, g_slot_min_k_v32);
    if (!obj) return false;
    const std::uint64_t value = PyLong_AsUnsignedLongLong(obj);
    Py_DECREF(obj);
    if (PyErr_Occurred()) return false;
    if (value == 0ULL || value > state.k) {
        PyErr_SetString(PyExc_RuntimeError, "native min_k metadata is inconsistent");
        return false;
    }
    state.min_k_cache = value;
    return true;
}

bool inherit_resident_min_k_slots_v32(
    PyObject* self,
    PyObject* other,
    SketchState& dst,
    SketchState& src
) {
    if (!(src.n > src.k || src.levels.size() > 1U)) return true;
    if (!load_min_k_cache_slots_v32(other, src) ||
        !load_min_k_cache_slots_v32(self, dst)) return false;
    if (src.min_k_cache >= dst.min_k_cache) return true;

    // Python `_min_k` may have been tightened by an intentional fallback path.
    // Re-read it only when the source can actually tighten the cached value.
    PyObject* current_obj = PyObject_GetAttr(self, g_slot_min_k_v32);
    if (!current_obj) return false;
    const std::uint64_t current = PyLong_AsUnsignedLongLong(current_obj);
    Py_DECREF(current_obj);
    if (PyErr_Occurred()) return false;
    if (current == 0ULL || current > dst.k) {
        PyErr_SetString(PyExc_RuntimeError, "destination min_k metadata is inconsistent");
        return false;
    }
    dst.min_k_cache = current;
    if (src.min_k_cache >= dst.min_k_cache) return true;

    PyObject* obj = PyLong_FromUnsignedLongLong(src.min_k_cache);
    if (!obj) return false;
    const int rc = PyObject_SetAttr(self, g_slot_min_k_v32, obj);
    Py_DECREF(obj);
    if (rc < 0) return false;
    dst.min_k_cache = src.min_k_cache;
    return true;
}

bool inherit_empty_min_k_slots_v32(PyObject* self, PyObject* other, SketchState& src) {
    if (!(src.n > src.k || src.levels.size() > 1U)) {
        if (src.min_k_cache == 0ULL) src.min_k_cache = src.k;
        return true;
    }
    if (!load_min_k_cache_slots_v32(other, src)) return false;
    if (src.min_k_cache == src.k) return true;
    return set_u64_slot_v32(self, g_slot_min_k_v32, src.min_k_cache);
}

bool read_empty_adoption_config_slots_v32(
    PyObject* self,
    const SketchState& src,
    std::uint64_t& rng_state,
    std::uint64_t& compactions
) {
    std::uint64_t n = 0;
    std::uint64_t k = 0;
    if (!read_u64_slot_v32(self, g_slot_n_v32, n) ||
        !read_u64_slot_v32(self, g_slot_k_v32, k)) return false;
    if (n != 0ULL || k != src.k) return false;
    if (!read_u64_slot_v32(self, g_slot_rng_state_v32, rng_state) ||
        !read_u64_slot_v32(self, g_slot_compaction_count_v32, compactions)) return false;
    return src.levels.size() <= static_cast<std::size_t>(src.max_levels) &&
           src.sorted.size() == src.levels.size() &&
           src.capacities.size() == src.levels.size() &&
           src.retained <= src.total_capacity;
}

PyObject* adopt_empty_resident_merge_slots_v32(
    PyObject* self,
    PyObject* other,
    SketchState& src,
    std::uint64_t rng_state,
    std::uint64_t compactions
) {
    if (!g_state_handle_type) {
        PyErr_SetString(PyExc_RuntimeError, "native state handle type is not installed");
        return nullptr;
    }
    if (!inherit_empty_min_k_slots_v32(self, other, src)) return nullptr;

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
    state->min_k_cache = src.min_k_cache == 0ULL ? src.k : src.min_k_cache;
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

    if (PyObject_SetAttr(self, g_slot_cache_values_v32, handle) < 0 ||
        PyObject_SetAttr(self, g_slot_cache_prefix_v32, capsule) < 0 ||
        PyObject_SetAttr(self, g_slot_cache_generation_v32, sentinel) < 0) {
        Py_DECREF(sentinel);
        Py_DECREF(handle);
        Py_DECREF(capsule);
        return nullptr;
    }
    Py_DECREF(sentinel);
    Py_DECREF(handle);

    const bool mirrored = mirror_merge_visible_slots_v32(
        self, *state, false, 0.0, 0.0);
    Py_DECREF(capsule);
    if (!mirrored) return nullptr;
    Py_RETURN_NONE;
}

PyObject* fast_resident_merge_slots_v32(
    PyObject* self,
    PyObject* other,
    SketchState* dst,
    PyObject* dst_capsule,
    SketchState* src,
    PyObject* src_capsule,
    PyObject* const* args,
    Py_ssize_t nargs,
    PyObject* kwnames
) {
    MergeSequencePlanV32 plan;
    if (!preflight_resident_merge_sequence_v32(*dst, *src, plan)) {
        Py_DECREF(src_capsule);
        Py_DECREF(dst_capsule);
        if (PyErr_ExceptionMatches(PyExc_TypeError) ||
            PyErr_ExceptionMatches(PyExc_ValueError) ||
            PyErr_ExceptionMatches(PyExc_OverflowError)) {
            PyErr_Clear();
            return fast_merge_method_sequence_v32(self, args, nargs, kwnames);
        }
        return nullptr;
    }
    if (plan.sequence_overflow) {
        Py_DECREF(src_capsule);
        Py_DECREF(dst_capsule);
        return fast_merge_method_sequence_v32(self, args, nargs, kwnames);
    }
    if (!inherit_resident_min_k_slots_v32(self, other, *dst, *src)) {
        Py_DECREF(src_capsule);
        Py_DECREF(dst_capsule);
        return nullptr;
    }

    const bool old_has_values = dst->has_values;
    const double old_min = dst->min_value;
    const double old_max = dst->max_value;
    bool mutation_started = false;
    if (!execute_resident_merge_sequence_v32(*dst, *src, plan, mutation_started)) {
        Py_DECREF(src_capsule);
        Py_DECREF(dst_capsule);
        return nullptr;
    }

    const bool mirrored = mirror_merge_visible_slots_v32(
        self, *dst, old_has_values, old_min, old_max);
    Py_DECREF(src_capsule);
    Py_DECREF(dst_capsule);
    if (!mirrored) return nullptr;
    Py_RETURN_NONE;
}

PyObject* fast_merge_method_slots_v32(
    PyObject* self,
    PyObject* const* args,
    Py_ssize_t nargs,
    PyObject* kwnames
) {
    if (!g_type_fastpaths_enabled || nargs != 1 || (kwnames && PyTuple_GET_SIZE(kwnames))) {
        return fast_merge_method_sequence_v32(self, args, nargs, kwnames);
    }
    PyObject* other = args[0];
    if (!g_kll_type ||
        !PyObject_TypeCheck(other, reinterpret_cast<PyTypeObject*>(g_kll_type)) ||
        other == self) {
        return fast_merge_method_sequence_v32(self, args, nargs, kwnames);
    }

    PyObject* dst_capsule = nullptr;
    SketchState* dst = resident_state_from_kll_slots_v32(self, &dst_capsule);
    if (dst) {
        PyObject* src_capsule = nullptr;
        SketchState* src = resident_state_from_kll_slots_v32(other, &src_capsule);
        if (!src) {
            Py_DECREF(dst_capsule);
            if (PyErr_Occurred()) return nullptr;
            return fast_merge_method_sequence_v32(self, args, nargs, kwnames);
        }
        if (!src->n) {
            Py_DECREF(src_capsule);
            Py_DECREF(dst_capsule);
            Py_RETURN_NONE;
        }
        if (dst == src) {
            Py_DECREF(src_capsule);
            Py_DECREF(dst_capsule);
            return fast_merge_method_sequence_v32(self, args, nargs, kwnames);
        }
        return fast_resident_merge_slots_v32(
            self, other, dst, dst_capsule, src, src_capsule, args, nargs, kwnames);
    }
    if (PyErr_Occurred()) return nullptr;

    PyObject* src_capsule = nullptr;
    SketchState* src = resident_state_from_kll_slots_v32(other, &src_capsule);
    if (!src) {
        if (PyErr_Occurred()) return nullptr;
        return fast_merge_method_sequence_v32(self, args, nargs, kwnames);
    }
    if (!src->n) {
        Py_DECREF(src_capsule);
        Py_RETURN_NONE;
    }

    std::uint64_t rng_state = 0;
    std::uint64_t compactions = 0;
    const bool eligible = read_empty_adoption_config_slots_v32(
        self, *src, rng_state, compactions);
    if (!eligible) {
        Py_DECREF(src_capsule);
        if (PyErr_Occurred()) return nullptr;
        return fast_merge_method_sequence_v32(self, args, nargs, kwnames);
    }

    PyObject* out = adopt_empty_resident_merge_slots_v32(
        self, other, *src, rng_state, compactions);
    Py_DECREF(src_capsule);
    return out;
}

static PyMethodDef fast_merge_slots_v32_def = {
    "merge",
    reinterpret_cast<PyCFunction>(fast_merge_method_slots_v32),
    METH_FASTCALL | METH_KEYWORDS,
    "Merge another KLL using cached Python slot keys and exact native compaction planning."
};

PyObject* install_v32_slot_fastpath(PyObject*, PyObject*) {
    if (!g_kll_type || !g_state_handle_type || !g_fallback_merge) {
        PyErr_SetString(PyExc_RuntimeError, "install_type_fastpaths must run before cached-slot fastpath");
        return nullptr;
    }
    if (!init_slots_v32()) return nullptr;
    PyObject* descriptor = PyDescr_NewMethod(
        reinterpret_cast<PyTypeObject*>(g_kll_type), &fast_merge_slots_v32_def);
    if (!descriptor) return nullptr;
    const int rc = PyObject_SetAttrString(g_kll_type, "merge", descriptor);
    Py_DECREF(descriptor);
    if (rc < 0) return nullptr;
    Py_RETURN_NONE;
}

static PyMethodDef v32_slot_methods[] = {
    {"install_v32_slot_fastpath", install_v32_slot_fastpath, METH_NOARGS,
     "Install cached-slot v3.2 resident merge optimization."},
    {nullptr, nullptr, 0, nullptr},
};
