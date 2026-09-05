#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>
#undef PyInit__native

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <utility>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define KLL_STATE_X86 1
#else
#define KLL_STATE_X86 0
#endif
#if KLL_STATE_X86 && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define KLL_STATE_AVX2 1
#else
#define KLL_STATE_AVX2 0
#endif

namespace kll_state_addon {
#include "_native_state_primitives.inc"
#include "_native_state_engine.inc"
#include "_native_state_bindings.inc"
#include "_native_type_fastpaths.inc"
#include "_native_v32_merge.inc"
#include "_native_v32_slots.inc"
} // namespace kll_state_addon

extern "C" PyObject* PyInit__native_base(void);

PyMODINIT_FUNC PyInit__native(void) {
    PyObject* module = PyInit__native_base();
    if (!module) return nullptr;
    if (PyModule_AddFunctions(module, kll_state_addon::addon_methods) < 0 ||
        PyModule_AddFunctions(module, kll_state_addon::type_fastpath_methods) < 0 ||
        PyModule_AddFunctions(module, kll_state_addon::v32_merge_methods) < 0 ||
        PyModule_AddFunctions(module, kll_state_addon::v32_slot_methods) < 0) {
        Py_DECREF(module);
        return nullptr;
    }
    return module;
}
