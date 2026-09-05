// v3.2: merge-engine specializations.
//
// 1. A fresh empty same-k destination can adopt an already-valid resident
//    source hierarchy without running the generic planner/append path.
// 2. For resident-to-resident merges, structural preflight tells us exactly
//    which higher levels will be compacted. Raw values appended to those levels
//    are dead: canonical compaction immediately replaces the whole raw level by
//    at most one leftover. Keep the sorted shadow authoritative transiently,
//    skip those dead raw writes, and restore raw/sorted parity at compaction.
//
// Both paths preserve the destination RNG stream, source immutability, Python
// visibility, KLL2 byte semantics, and the established v3.1 fallback behavior.

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

// For a fresh empty same-k destination, `_min_k` starts at exactly k. Only a
// source that has inherited a tighter bound can change it, so the generic
// destination lookup + rich comparison is unnecessary on the first merge.
bool inherit_empty_min_k_v32(PyObject* self, PyObject* other, const SketchState& src) {
    if (!(src.n > src.k || src.levels.size() > 1U)) return true;

    PyObject* src_min_k_obj = PyObject_GetAttrString(other, "_min_k");
    if (!src_min_k_obj) return false;
    const std::uint64_t src_min_k = PyLong_AsUnsignedLongLong(src_min_k_obj);
    Py_DECREF(src_min_k_obj);
    if (PyErr_Occurred()) return false;

    if (src_min_k > src.k) {
        PyErr_SetString(PyExc_RuntimeError, "native source min_k exceeds configured k");
        return false;
    }
    if (src_min_k == src.k) return true;
    return set_u64_attr(self, "_min_k", src_min_k);
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
    if (!inherit_empty_min_k_v32(self, other, src)) return nullptr;

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

// -------------------------------------------------------------------------
// Compaction-aware resident merge
// -------------------------------------------------------------------------

struct MergePlanV32 {
    std::size_t initial_levels = 0;
    std::size_t final_levels = 0;
    std::uint64_t final_retained = 0;
    std::uint64_t final_compactions = 0;
    std::uint64_t compact_mask = 0;
};

bool preflight_resident_merge_v32(
    const SketchState& dst,
    const SketchState& src,
    MergePlanV32& plan
) {
    if (src.n > U64_MAX - dst.n || src.retained > U64_MAX - dst.retained) {
        PyErr_SetString(PyExc_OverflowError, "native merge exceeds uint64 range");
        return false;
    }

    plan.initial_levels = std::max(dst.levels.size(), src.levels.size());
    if (plan.initial_levels == 0U ||
        plan.initial_levels > static_cast<std::size_t>(dst.max_levels) ||
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
        if (h < 64U) plan.compact_mask |= (1ULL << h);
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

// Compact one level while allowing selected higher levels to have an incomplete
// raw representation transiently. The sorted shadow always contains the full
// logical level. Once this level is compacted, canonical raw/sorted parity is
// restored by writing the boundary leftover to both representations.
bool compact_level_elided_v32(
    SketchState& s,
    std::size_t h,
    std::uint64_t& raw_elided_mask
) {
    if (h >= s.levels.size() || s.sorted[h].size() < 2U) {
        PyErr_SetString(PyExc_RuntimeError, "native state attempted invalid v3.2 compaction");
        return false;
    }
    if (!ensure_levels(s, h + 2U)) return false;

    const bool raw_elided = h > 0U && h < 64U && ((raw_elided_mask >> h) & 1ULL);
    if (h == 0U && !s.level0_sorted) {
        s.sorted[0] = s.levels[0];
        sort_level(s.sorted[0]);
        s.level0_sorted = true;
    }
    if (!raw_elided && s.sorted[h].size() != s.levels[h].size()) {
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

    const std::size_t next_h = h + 1U;
    const bool next_raw_elided = next_h > 0U && next_h < 64U &&
        ((raw_elided_mask >> next_h) & 1ULL);
    auto& next_sorted = s.sorted[next_h];
    const std::size_t next_sorted_old = next_sorted.size();

    if (next_raw_elided) {
        // The next level is itself guaranteed to compact before merge returns.
        // Append the already-sorted promoted run only to its sorted shadow.
        next_sorted.reserve(next_sorted_old + promoted_count);
        for (std::size_t i = start + static_cast<std::size_t>(offset); i < stop; i += 2U) {
            next_sorted.push_back(source[i]);
        }
        if (next_sorted_old) {
            std::inplace_merge(
                next_sorted.begin(),
                next_sorted.begin() + static_cast<std::ptrdiff_t>(next_sorted_old),
                next_sorted.end(),
                [](double a, double b) { return a < b; });
        }
    } else {
        // This level survives the merge, so maintain its exact canonical raw
        // append order as well as the sorted compaction shadow.
        auto& raw_next = s.levels[next_h];
        const std::size_t raw_next_old = raw_next.size();
        raw_next.reserve(raw_next_old + promoted_count);
        for (std::size_t i = start + static_cast<std::size_t>(offset); i < stop; i += 2U) {
            raw_next.push_back(source[i]);
        }
        stable_append_sorted_range(
            next_sorted,
            raw_next.begin() + static_cast<std::ptrdiff_t>(raw_next_old),
            raw_next.end());
    }

    auto& raw = s.levels[h];
    raw.clear();
    source.clear();
    if (has_leftover) {
        raw.push_back(leftover);
        source.push_back(leftover);
    }
    if (h == 0U) s.level0_sorted = true;
    if (raw_elided) raw_elided_mask &= ~(1ULL << h);

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

bool compress_elided_v32(SketchState& s, std::uint64_t& raw_elided_mask) {
    int guard = 0;
    while (s.retained > s.total_capacity) {
        std::size_t h = 0;
        bool found = false;
        for (; h < s.levels.size(); ++h) {
            // Elided raw levels may be smaller than their logical sorted shadow.
            const std::size_t logical_size = (h > 0U && h < 64U &&
                ((raw_elided_mask >> h) & 1ULL)) ? s.sorted[h].size() : s.levels[h].size();
            if (logical_size > s.capacities[h]) {
                found = true;
                break;
            }
        }
        if (!found) {
            PyErr_SetString(PyExc_RuntimeError, "native KLL capacity accounting became inconsistent");
            return false;
        }
        if (!compact_level_elided_v32(s, h, raw_elided_mask)) return false;
        if (++guard > 10000) {
            PyErr_SetString(PyExc_RuntimeError, "native KLL compaction did not converge");
            return false;
        }
    }
    if (raw_elided_mask != 0ULL) {
        PyErr_SetString(PyExc_RuntimeError, "v3.2 merge left an uncompacted elided raw level");
        return false;
    }
    return true;
}

bool merge_resident_states_v32(
    SketchState& dst,
    const SketchState& src,
    bool& mutation_started
) {
    mutation_started = false;
    if (!src.n) return true;

    MergePlanV32 plan;
    if (!preflight_resident_merge_v32(dst, src, plan)) return false;
    if (!ensure_levels(dst, plan.initial_levels)) return false;

    // Level zero is intentionally excluded. Its sorted shadow may be lazy after
    // ingestion, while every higher-level shadow is authoritative. Higher levels
    // proven to compact can safely omit raw appends until their compaction.
    std::uint64_t raw_elided_mask = plan.compact_mask & ~1ULL;

    mutation_started = true;
    for (std::size_t h = 0; h < src.levels.size(); ++h) {
        if (src.levels[h].empty()) continue;
        const bool elide_raw = h > 0U && h < 64U && ((raw_elided_mask >> h) & 1ULL);
        if (!elide_raw) {
            dst.levels[h].insert(
                dst.levels[h].end(), src.levels[h].begin(), src.levels[h].end());
        }
        if (h == 0U) {
            dst.level0_sorted = false;
        } else {
            stable_append_sorted(dst.sorted[h], src.sorted[h]);
        }
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

    if (!compress_elided_v32(dst, raw_elided_mask)) return false;
    if (dst.levels.size() != plan.final_levels ||
        dst.retained != plan.final_retained ||
        dst.compactions != plan.final_compactions) {
        PyErr_SetString(PyExc_RuntimeError, "v3.2 native merge diverged from structural preflight");
        return false;
    }
    dst.query_valid = false;
    return true;
}

PyObject* fast_resident_merge_v32(
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
    if (!merge_resident_states_v32(*dst, *src, mutation_started)) {
        Py_DECREF(src_capsule);
        Py_DECREF(dst_capsule);
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

    PyObject* dst_capsule = nullptr;
    SketchState* dst = resident_state_from_kll(self, &dst_capsule);
    if (dst) {
        PyObject* src_capsule = nullptr;
        SketchState* src = resident_state_from_kll(other, &src_capsule);
        if (!src) {
            Py_DECREF(dst_capsule);
            if (PyErr_Occurred()) return nullptr;
            return fast_merge_method(self, args, nargs, kwnames);
        }
        return fast_resident_merge_v32(
            self, other, dst, dst_capsule, src, src_capsule, args, nargs, kwnames);
    }
    if (PyErr_Occurred()) return nullptr;

    // Nonresident destination: retain the specialized same-k empty adoption.
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
    "Merge another KLL with v3.2 compaction-aware native fast paths."
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
     "Install the v3.2 compaction-aware merge optimization."},
    {nullptr, nullptr, 0, nullptr},
};
