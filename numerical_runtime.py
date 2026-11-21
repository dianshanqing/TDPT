"""Inspect and control already loaded Windows OpenBLAS libraries.

No numerical package is imported or loaded here. Call after the numerical
imports, and repeat after importing Aer: each DLL owns its own thread setting.
An empty snapshot means no supported loaded library was discovered, not that a
thread limit was verified. Other platforms currently have no discovery adapter.
"""
from __future__ import annotations

import ctypes
from numbers import Integral
import os
import re


class BlasConfigurationError(RuntimeError):
    def __init__(self, message, records):
        super().__init__(message)
        self.records = records


def _loaded_modules():
    """Return existing module handles; never load a numerical DLL by path."""
    if os.name != 'nt':
        return []
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    enumerate_modules = kernel.K32EnumProcessModules
    enumerate_modules.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE),
                                 wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    enumerate_modules.restype = wintypes.BOOL
    filename = kernel.GetModuleFileNameW
    filename.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    filename.restype = wintypes.DWORD
    process = kernel.GetCurrentProcess()
    capacity = 256
    while True:
        modules = (wintypes.HMODULE * capacity)()
        needed = wintypes.DWORD()
        if not enumerate_modules(process, modules, ctypes.sizeof(modules), ctypes.byref(needed)):
            raise ctypes.WinError(ctypes.get_last_error())
        if needed.value <= ctypes.sizeof(modules):
            break
        capacity = needed.value // ctypes.sizeof(wintypes.HMODULE) + 32
    rows = []
    for handle in modules[:needed.value // ctypes.sizeof(wintypes.HMODULE)]:
        name = ctypes.create_unicode_buffer(32768)
        length = filename(handle, name, len(name))
        if not length or length >= len(name):
            raise ctypes.WinError(ctypes.get_last_error())
        rows.append((name.value, handle))
    return rows


def _library_handle(handle):
    # Supplying the existing handle prevents LoadLibrary from opening a DLL.
    return ctypes.CDLL(None, handle=handle)


def _function(library, name, result, arguments=()):
    try:
        function = getattr(library, name)
    except AttributeError:
        return None
    function.restype = result
    function.argtypes = list(arguments)
    return function


def _owner(path):
    path = path.replace('\\', '/').lower()
    if '/qiskit_aer/' in path:
        return 'aer_private'
    if '/numpy.libs/' in path or '/numpy/' in path:
        return 'numpy'
    if '/scipy.libs/' in path or '/scipy/' in path:
        return 'scipy'
    return 'unknown'


def _read(function, errors, name):
    if function is None:
        return None
    try:
        value = function()
        return value.decode('utf-8', errors='replace') if isinstance(value, bytes) else value
    except Exception as exc:
        errors.append(name+': '+repr(exc))
        return None


def _libraries():
    rows = []
    for path, handle in _loaded_modules():
        if 'openblas' not in path.replace('\\', '/').rsplit('/', 1)[-1].lower():
            continue
        errors = []
        get_threads = set_threads = get_config = get_parallel = None
        prefix = suffix = None
        try:
            library = _library_handle(handle)
            for candidate_prefix, candidate_suffix in (
                (p, s) for p in ('openblas_', 'scipy_openblas_')
                for s in ('', '64_', '_64_')
            ):
                get_candidate = _function(library, candidate_prefix+'get_num_threads'+candidate_suffix, ctypes.c_int)
                set_candidate = _function(library, candidate_prefix+'set_num_threads'+candidate_suffix,
                                          None, (ctypes.c_int,))
                if get_candidate is not None or set_candidate is not None:
                    prefix, suffix = candidate_prefix, candidate_suffix
                    get_threads, set_threads = get_candidate, set_candidate
                    get_config = _function(library, prefix+'get_config'+suffix, ctypes.c_char_p)
                    get_parallel = _function(library, prefix+'get_parallel'+suffix, ctypes.c_int)
                    break
        except Exception as exc:
            errors.append('inspect_handle: '+repr(exc))
        config = _read(get_config, errors, 'get_config')
        parallel = _read(get_parallel, errors, 'get_parallel')
        version = re.search(r'OpenBLAS\s+(\S+)', config or '')
        record = dict(path=path, owner=_owner(path), internal_api='openblas',
            symbol_prefix=prefix, symbol_suffix=suffix,
            get_num_threads_supported=get_threads is not None,
            set_num_threads_supported=set_threads is not None,
            num_threads=_read(get_threads, errors, 'get_num_threads'),
            config=config, version=version.group(1) if version else None,
            parallel=parallel, threading_layer={0:'sequential', 1:'native_threads', 2:'openmp'}.get(parallel),
            errors=errors)
        rows.append((record, get_threads, set_threads))
    return rows


def numerical_libraries():
    """Read per-library settings without changing any BLAS or OpenMP setting."""
    return [record for record, _, _ in _libraries()]


def configure_blas_threads(count=1):
    """Set and verify every discovered OpenBLAS DLL; never alter OpenMP.

    Explicit parallel requests fail if no library is found or any library cannot
    be controlled. On setter failure, restore known prior settings before raising
    BlasConfigurationError; its ``records`` preserve the failed observations.
    For the default one-thread request, unknown capabilities remain unapplied in
    the returned records. Libraries imported later require another call.
    """
    if isinstance(count, bool) or not isinstance(count, Integral) or not 0 < count < 2**31:
        raise ValueError('BLAS thread count must be a positive 32-bit integer')
    count = int(count)
    libraries = _libraries()
    records = []
    for record, _, _ in libraries:
        record.update(requested_threads=count, num_threads_before=record['num_threads'],
                      num_threads_after=None, applied=False, restored_after_failure=None)
        records.append(record)
    unsupported = [record for record, getter, setter in libraries
                   if getter is None or setter is None or record['num_threads_before'] is None]
    if count > 1 and (not libraries or unsupported):
        raise BlasConfigurationError('Requested parallel BLAS control is unavailable for one or more loaded libraries', records)
    attempted = []
    failed = False
    for record, getter, setter in libraries:
        if record in unsupported:
            record['errors'].append('thread_control_unavailable')
            continue
        attempted.append((record, getter, setter))
        try:
            setter(count)
            record['num_threads_after'] = getter()
            record['num_threads'] = record['num_threads_after']
            record['applied'] = record['num_threads_after'] == count
            if not record['applied']:
                record['errors'].append('thread_setting_not_confirmed')
                failed = True
        except Exception as exc:
            record['errors'].append('set_num_threads: '+repr(exc))
            failed = True
    if failed:
        for record, getter, setter in attempted:
            record['applied'] = False
            try:
                setter(record['num_threads_before'])
                record['num_threads'] = getter()
                record['restored_after_failure'] = record['num_threads'] == record['num_threads_before']
            except Exception as exc:
                record['num_threads'] = None
                record['restored_after_failure'] = False
                record['errors'].append('restore_num_threads: '+repr(exc))
        raise BlasConfigurationError('BLAS thread setting could not be verified; previous settings were restored where possible', records)
    return records
