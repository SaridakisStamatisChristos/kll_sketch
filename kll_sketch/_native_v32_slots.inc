// v3.2 cold-path experiment: direct access to KLL's CPython __slots__.
//
// The public KLL object deliberately remains the API boundary.  Because KLL is
// a fixed __slots__ class, its member descriptors expose stable offsets for the
// active CPython build.  Resolve those offsets once at installation and avoid
// repeated name hashing / descriptor dispatch on every resident merge.  This is
// CPython-specific by design, just like the optional native extension itself;
// pure Python and all semantic fallbacks remain unchanged.

struct KLLSlotOffsetsV32 {
    Py_ssize_t n = -1;
    Py_ssize_t k = -1;
    Py_ssize_t min_k = -1;
    Py_ssize_t rng_state = -1;
    Py_ssize_t retained = -1;
    Py_ssize_t min_value = -1;
    Py_ssize_t max_value = -1;
    Py_ssize_t cache_generation = -1;
    Py_ssize_t cache_values = -1;
    Py_ssize_t cache_prefix = -1;
    Py_ssize_t compactions = -1;
    bool ready = false;
};

static KLLSlotOffsetsV32 g_v32_slots;
static PyObject* g_v32_state_sentinel = nullptr;

bool member_slot_offset_v32(PyObject* type_obj, const char* name, Py_ssize_t& out) {
    PyObject* descr = PyObject_GetAttrString(type_obj, name);
    if (!descr) return false;
    if (!Py_IS_TYPE(descr, &PyMemberDescr_Type)) {
        Py_DECREF(descr);
        PyErr_Format(PyExc_RuntimeError, "KLL slot %s is not a CPython member descriptor", name);
        return false;
    }
    auto* member_descr = reinterpret_cast<PyMemberDescrObject*>(descr);
    PyMemberDef* member = member_descr->d_member;
    if (!member || (member->type != T_OBJECT && member->type != T_OBJECT_EX) || member->offset <= 0) {
        Py_DECREF(descr);
        PyErr_Format(PyExc_RuntimeError, "KLL slot %s has an unsupported descriptor layout", name);
        return false;
    }
    out = member->offset;
    Py_DECREF(descr);
    return true;
}

bool resolve_kll_slots_v32(PyObject* type_obj) {
    KLLSlotOffsetsV32 slots;
    if (!member_slot_offset_v32(type_obj, "_n", slots.n) ||
        !member_slot_offset_v32(type_obj, "_k", slots.k) ||
        !member_slot_offset_v32(type_obj, "_min_k", slots.min_k) ||
        !member_slot_offset_v32(type_obj, "_rng_state", slots.rng_state) ||
        !member_slot_offset_v32(type_obj, "_num_retained", slots.retained) ||
        !member_slot_offset_v32(type_obj, "_min_value", slots.min_value) ||
        !member_slot_offset_v32(type_obj, "_max_value", slots.max_value) ||
        !member_slot_offset_v32(type_obj, "_cache_generation", slots.cache_generation) ||
        !member_slot_offset_v32(type_obj, "_cache_values", slots.cache_values) ||
        !member_slot_offset_v32(type_obj, "_cache_prefix", slots.cache_prefix) ||
        !member_slot_offset_v32(type_obj, "_compaction_count", slots.compactions)) {
        return false;
    }
    slots.ready = true;
    g_v32_slots = slots;
    return true;
}

inline PyObject** slot_address_v32(PyObject* obj, Py_ssize_t offset) {
    return reinterpret_cast<PyObject**>(reinterpret_cast<char*>(obj) + offset);
}

inline PyObject* slot_borrowed_v32(PyObject* obj, Py_ssize_t offset) {
    return *slot_address_v32(obj, offset);
}

inline void slot_replace_newref_v32(PyObject* obj, Py_ssize_t offset, PyObject* value) {
    PyObject** address = slot_address_v32(obj, offset);
    PyObject* old = *address;
    *address = value;  // value is an owned reference transferred to the slot
    Py_XDECREF(old);
}

inline void slot_store_borrowed_v32(PyObject* obj, Py_ssize_t offset, PyObject* value) {
    Py_INCREF(value);
    slot_replace_newref_v32(obj, offset, value);
}

bool slot_u64_v32(PyObject* obj, Py_ssize_t offset, std::uint64_t& out) {
    PyObject* value = slot_borrowed_v32(obj, offset);
    if (!value) {
        PyErr_SetString(PyExc_RuntimeError, "uninitialized KLL native slot");
        return false;
    }
    out = PyLong_AsUnsignedLongLong(value);
    return !PyErr_Occurred();
}

SketchState* resident_state_from_kll_slots_v32(PyObject* obj) {
    PyObject* capsule = slot_borrowed_v32(obj, g_v32_slots.cache_prefix);
    if (!capsule || !PyCapsule_IsValid(capsule, CAPSULE_NAME)) return nullptr;
    return static_cast<SketchState*>(PyCapsule_GetPointer(capsule, CAPSULE_NAME));
}

bool inherit_min_k_slots_v32(PyObject* self, PyObject* other, const SketchState& src) {
    if (!(src.n > src.k || src.levels.size() > 1U)) return true;
    PyObject* dst_min_k = slot_borrowed_v32(self, g_v32_slots.min_k);
    PyObject* src_min_k = slot_borrowed_v32(other, g_v32_slots.min_k);
    if (!dst_min_k || !src_min_k) {
        PyErr_SetString(PyExc_RuntimeError, "uninitialized KLL min_k slot");
        return false;
    }
    if (dst_min_k == src_min_k) return true;
    const int smaller = PyObject_RichCompareBool(src_min_k, dst_min_k, Py_LT);
    if (smaller < 0) return false;
    if (smaller) slot_store_borrowed_v32(self, g_v32_slots.min_k, src_min_k);
    return true;
}

bool mirror_merge_visible_slots_v32(
    PyObject* self,
    const SketchState& dst,
    bool old_has_values,
    double old_min,
    double old_max
) {
    PyObject* n_obj = PyLong_FromUnsignedLongLong(dst.n);
    PyObject* retained_obj = PyLong_FromUnsignedLongLong(dst.retained);
    PyObject* min_obj = nullptr;
    PyObject* max_obj = nullptr;
    const bool update_min = dst.has_values && (!old_has_values || dst.min_value != old_min);
    const bool update_max = dst.has_values && (!old_has_values || dst.max_value != old_max);
    if (update_min) min_obj = PyFloat_FromDouble(dst.min_value);
    if (update_max) max_obj = PyFloat_FromDouble(dst.max_value);
    if (!n_obj || !retained_obj || (update_min && !min_obj) || (update_max && !max_obj)) {
        Py_XDECREF(n_obj);
        Py_XDECREF(retained_obj);
        Py_XDECREF(min_obj);
        Py_XDECREF(max_obj);
        return false;
    }
    slot_replace_newref_v32(self, g_v32_slots.n, n_obj);
    slot_replace_newref_v32(self, g_v32_slots.retained, retained_obj);
    if (update_min) slot_replace_newref_v32(self, g_v32_slots.min_value, min_obj);
    if (update_max) slot_replace_newref_v32(self, g_v32_slots.max_value, max_obj);
    return true;
}

bool empty_adoption_config_slots_v32(
    PyObject* self,
    const SketchState& src,
    std::uint64_t& rng_state,
    std::uint64_t& compactions
) {
    std::uint64_t n = 0;
    std::uint64_t k = 0;
    if (!slot_u64_v32(self, g_v32_slots.n, n) ||
        !slot_u64_v32(self, g_v32_slots.k, k)) return false;
    if (n != 0ULL || k != src.k) return false;
    if (!slot_u64_v32(self, g_v32_slots.rng_state, rng_state) ||
        !slot_u64_v32(self, g_v32_slots.compactions, compactions)) return false;
    return src.levels.size() <= static_cast<std::size_t>(src.max_levels) &&
           src.sorted.size() == src.levels.size() &&
           src.capacities.size() == src.levels.size() &&
           src.retained <= src.total_capacity;
}

PyObject* adopt_empty_slots_v32(
    PyObject* self,
    PyObject* other,
    const SketchState& src,
    std::uint64_t rng_state,
    std::uint64_t compactions
) {
    if (!g_state_handle_type || !g_v32_state_sentinel) {
        PyErr_SetString(PyExc_RuntimeError, "v3.2 native slot fastpath is not initialized");
        return nullptr;
    }

    PyObject* desired_min_k = slot_borrowed_v32(self, g_v32_slots.min_k);
    PyObject* src_min_k = slot_borrowed_v32(other, g_v32_slots.min_k);
    if (!desired_min_k || !src_min_k) {
        PyErr_SetString(PyExc_RuntimeError, "uninitialized KLL min_k slot");
        return nullptr;
    }
    if (src.n > src.k || src.levels.size() > 1U) {
        if (desired_min_k != src_min_k) {
            const int smaller = PyObject_RichCompareBool(src_min_k, desired_min_k, Py_LT);
            if (smaller < 0) return nullptr;
            if (smaller) desired_min_k = src_min_k;
        }
    }

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
    PyObject* n_obj = PyLong_FromUnsignedLongLong(state->n);
    PyObject* retained_obj = PyLong_FromUnsignedLongLong(state->retained);
    PyObject* min_obj = state->has_values ? PyFloat_FromDouble(state->min_value) : Py_NewRef(Py_None);
    PyObject* max_obj = state->has_values ? PyFloat_FromDouble(state->max_value) : Py_NewRef(Py_None);
    if (!handle || !n_obj || !retained_obj || !min_obj || !max_obj) {
        Py_XDECREF(handle);
        Py_XDECREF(n_obj);
        Py_XDECREF(retained_obj);
        Py_XDECREF(min_obj);
        Py_XDECREF(max_obj);
        Py_DECREF(capsule);
        return nullptr;
    }

    // All allocations have succeeded. Commit the resident representation with
    // direct slot stores; from this point the capsule owns `state`.
    slot_store_borrowed_v32(self, g_v32_slots.min_k, desired_min_k);
    slot_replace_newref_v32(self, g_v32_slots.n, n_obj);
    slot_replace_newref_v32(self, g_v32_slots.retained, retained_obj);
    slot_replace_newref_v32(self, g_v32_slots.min_value, min_obj);
    slot_replace_newref_v32(self, g_v32_slots.max_value, max_obj);
    slot_store_borrowed_v32(self, g_v32_slots.cache_values, handle);
    slot_store_borrowed_v32(self, g_v32_slots.cache_prefix, capsule);
    slot_store_borrowed_v32(self, g_v32_slots.cache_generation, g_v32_state_sentinel);

    Py_DECREF(handle);
    Py_DECREF(capsule);
    Py_RETURN_NONE;
}

PyObject* fast_merge_method_v32_slots(
    PyObject* self,
    PyObject* const* args,
    Py_ssize_t nargs,
    PyObject* kwnames
) {
    if (!g_type_fastpaths_enabled || !g_v32_slots.ready || nargs != 1 ||
        (kwnames && PyTuple_GET_SIZE(kwnames))) {
        return fast_merge_method_v32(self, args, nargs, kwnames);
    }
    PyObject* other = args[0];
    if (!g_kll_type ||
        !PyObject_TypeCheck(other, reinterpret_cast<PyTypeObject*>(g_kll_type)) ||
        other == self) {
        return fast_merge_method_v32(self, args, nargs, kwnames);
    }

    SketchState* src = resident_state_from_kll_slots_v32(other);
    if (!src) {
        if (PyErr_Occurred()) return nullptr;
        return fast_merge_method_v32(self, args, nargs, kwnames);
    }
    if (!src->n) Py_RETURN_NONE;

    SketchState* dst = resident_state_from_kll_slots_v32(self);
    if (!dst) {
        if (PyErr_Occurred()) return nullptr;
        std::uint64_t rng_state = 0;
        std::uint64_t compactions = 0;
        const bool eligible = empty_adoption_config_slots_v32(
            self, *src, rng_state, compactions);
        if (!eligible) {
            if (PyErr_Occurred()) return nullptr;
            return fast_merge_method_v32(self, args, nargs, kwnames);
        }
        return adopt_empty_slots_v32(self, other, *src, rng_state, compactions);
    }
    if (dst == src) return fast_merge_method_v32(self, args, nargs, kwnames);

    if (!inherit_min_k_slots_v32(self, other, *src)) return nullptr;
    const bool old_has_values = dst->has_values;
    const double old_min = dst->min_value;
    const double old_max = dst->max_value;
    bool mutation_started = false;
    if (!merge_resident_states_preflighted(*dst, *src, mutation_started)) {
        if (!mutation_started &&
            (PyErr_ExceptionMatches(PyExc_TypeError) ||
             PyErr_ExceptionMatches(PyExc_ValueError) ||
             PyErr_ExceptionMatches(PyExc_OverflowError))) {
            PyErr_Clear();
            return fast_merge_method_v32(self, args, nargs, kwnames);
        }
        return nullptr;
    }
    if (!mirror_merge_visible_slots_v32(self, *dst, old_has_values, old_min, old_max)) return nullptr;
    Py_RETURN_NONE;
}

static PyMethodDef fast_merge_v32_slots_def = {
    "merge",
    reinterpret_cast<PyCFunction>(fast_merge_method_v32_slots),
    METH_FASTCALL | METH_KEYWORDS,
    "Merge another KLL using direct CPython slot access for resident metadata."
};

PyObject* install_v32_slot_fastpath(PyObject*, PyObject*) {
    if (!g_kll_type || !g_state_handle_type || !g_fallback_merge) {
        PyErr_SetString(PyExc_RuntimeError, "install_type_fastpaths must run before v3.2 slot fastpath");
        return nullptr;
    }
    if (!resolve_kll_slots_v32(g_kll_type)) return nullptr;
    if (!g_v32_state_sentinel) {
        g_v32_state_sentinel = PyLong_FromLong(-2L);
        if (!g_v32_state_sentinel) return nullptr;
    }
    PyObject* descriptor = PyDescr_NewMethod(
        reinterpret_cast<PyTypeObject*>(g_kll_type), &fast_merge_v32_slots_def);
    if (!descriptor) return nullptr;
    const int rc = PyObject_SetAttrString(g_kll_type, "merge", descriptor);
    Py_DECREF(descriptor);
    if (rc < 0) return nullptr;
    Py_RETURN_NONE;
}

static PyMethodDef v32_slot_methods[] = {
    {"install_v32_slot_fastpath", install_v32_slot_fastpath, METH_NOARGS,
     "Install the v3.2 direct-slot resident merge optimization."},
    {nullptr, nullptr, 0, nullptr},
};
