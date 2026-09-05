// v3.2 exact-sequence resident merge specialization.
//
// The v3.2 structural preflight already proves that KLL compaction *sizes* are
// independent of random parity/value choice.  The original executor then scans
// every level again after each compaction to rediscover the same next level.
// For resident-to-resident merges, record that compact-level sequence in a
// fixed stack buffer during preflight and execute it directly.  No public
// semantics change: empty/nonresident/keyword/self/error paths remain routed
// through the proven v3.2 descriptor, and unusually deep plans fall back too.

struct MergeSequencePlanV32 {
    std::size_t initial_levels = 0;
    std::size_t final_levels = 0;
    std::uint64_t final_retained = 0;
    std::uint64_t final_compactions = 0;
    std::uint64_t compact_mask = 0;
    std::array<std::uint8_t, 128> compact_levels;
    std::size_t compact_count = 0;
    bool sequence_overflow = false;
};

bool preflight_resident_merge_sequence_v32(
    const SketchState& dst,
    const SketchState& src,
    MergeSequencePlanV32& plan
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

        if (!plan.sequence_overflow) {
            if (plan.compact_count < plan.compact_levels.size()) {
                plan.compact_levels[plan.compact_count++] = static_cast<std::uint8_t>(h);
            } else {
                plan.sequence_overflow = true;
            }
        }
        plan.compact_mask |= (1ULL << h);

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

bool execute_resident_merge_sequence_v32(
    SketchState& dst,
    const SketchState& src,
    const MergeSequencePlanV32& plan,
    bool& mutation_started
) {
    mutation_started = false;
    if (!src.n) return true;
    if (plan.sequence_overflow) {
        PyErr_SetString(PyExc_RuntimeError, "native merge sequence exceeds fast-path capacity");
        return false;
    }
    if (!ensure_levels(dst, plan.initial_levels)) return false;

    std::uint64_t raw_elided_mask = plan.compact_mask & ~1ULL;
    mutation_started = true;

    for (std::size_t h = 0; h < src.levels.size(); ++h) {
        if (src.levels[h].empty()) continue;
        const bool elide_raw = h > 0U && ((raw_elided_mask >> h) & 1ULL);
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

    for (std::size_t i = 0; i < plan.compact_count; ++i) {
        if (!compact_level_elided_v32(
                dst,
                static_cast<std::size_t>(plan.compact_levels[i]),
                raw_elided_mask)) {
            return false;
        }
    }

    if (raw_elided_mask != 0ULL || dst.retained > dst.total_capacity ||
        dst.levels.size() != plan.final_levels ||
        dst.retained != plan.final_retained ||
        dst.compactions != plan.final_compactions) {
        PyErr_SetString(PyExc_RuntimeError, "v3.2 exact merge sequence diverged from structural preflight");
        return false;
    }
    dst.query_valid = false;
    return true;
}

PyObject* fast_resident_merge_sequence_v32(
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
            return fast_merge_method_v32(self, args, nargs, kwnames);
        }
        return nullptr;
    }
    if (plan.sequence_overflow) {
        Py_DECREF(src_capsule);
        Py_DECREF(dst_capsule);
        return fast_merge_method_v32(self, args, nargs, kwnames);
    }

    if (!inherit_resident_min_k_v32(self, other, *dst, *src)) {
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

    const bool mirrored = mirror_merge_visible_state(self, *dst, old_has_values, old_min, old_max);
    Py_DECREF(src_capsule);
    Py_DECREF(dst_capsule);
    if (!mirrored) return nullptr;
    Py_RETURN_NONE;
}

PyObject* fast_merge_method_sequence_v32(
    PyObject* self,
    PyObject* const* args,
    Py_ssize_t nargs,
    PyObject* kwnames
) {
    if (!g_type_fastpaths_enabled || nargs != 1 || (kwnames && PyTuple_GET_SIZE(kwnames))) {
        return fast_merge_method_v32(self, args, nargs, kwnames);
    }
    PyObject* other = args[0];
    if (!g_kll_type ||
        !PyObject_TypeCheck(other, reinterpret_cast<PyTypeObject*>(g_kll_type)) ||
        other == self) {
        return fast_merge_method_v32(self, args, nargs, kwnames);
    }

    PyObject* dst_capsule = nullptr;
    SketchState* dst = resident_state_from_kll(self, &dst_capsule);
    if (!dst) {
        if (PyErr_Occurred()) return nullptr;
        return fast_merge_method_v32(self, args, nargs, kwnames);
    }

    PyObject* src_capsule = nullptr;
    SketchState* src = resident_state_from_kll(other, &src_capsule);
    if (!src) {
        Py_DECREF(dst_capsule);
        if (PyErr_Occurred()) return nullptr;
        return fast_merge_method_v32(self, args, nargs, kwnames);
    }
    if (!src->n) {
        Py_DECREF(src_capsule);
        Py_DECREF(dst_capsule);
        Py_RETURN_NONE;
    }
    if (dst == src) {
        Py_DECREF(src_capsule);
        Py_DECREF(dst_capsule);
        return fast_merge_method_v32(self, args, nargs, kwnames);
    }

    return fast_resident_merge_sequence_v32(
        self, other, dst, dst_capsule, src, src_capsule, args, nargs, kwnames);
}

static PyMethodDef fast_merge_sequence_v32_def = {
    "merge",
    reinterpret_cast<PyCFunction>(fast_merge_method_sequence_v32),
    METH_FASTCALL | METH_KEYWORDS,
    "Merge another KLL using a preflight-recorded native compaction sequence."
};

PyObject* install_v32_sequence_fastpath(PyObject*, PyObject*) {
    if (!g_kll_type || !g_state_handle_type || !g_fallback_merge) {
        PyErr_SetString(PyExc_RuntimeError, "install_type_fastpaths must run before sequence fastpath");
        return nullptr;
    }
    PyObject* descriptor = PyDescr_NewMethod(
        reinterpret_cast<PyTypeObject*>(g_kll_type), &fast_merge_sequence_v32_def);
    if (!descriptor) return nullptr;
    const int rc = PyObject_SetAttrString(g_kll_type, "merge", descriptor);
    Py_DECREF(descriptor);
    if (rc < 0) return nullptr;
    Py_RETURN_NONE;
}

static PyMethodDef v32_sequence_methods[] = {
    {"install_v32_sequence_fastpath", install_v32_sequence_fastpath, METH_NOARGS,
     "Install exact-sequence v3.2 resident merge optimization."},
    {nullptr, nullptr, 0, nullptr},
};
