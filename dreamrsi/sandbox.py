"""Unprivileged Linux filesystem confinement plus syscall filtering.

No user namespaces required. Restrictions are applied inside the worker before
loading generated code. Parent evaluator and saved histories are never mounted
or granted. Unsupported kernels/libraries fail closed.
"""
import ctypes
import ctypes.util
import errno
import os
from pathlib import Path
import resource
import sys


def confine(source, cpu_seconds=30):
    libc = ctypes.CDLL(None, use_errno=True)
    sec = ctypes.CDLL("libseccomp.so.2")
    libc.syscall.restype = ctypes.c_long
    # Landlock uses the same syscall numbers on supported x86_64/aarch64 Linux.
    if os.uname().machine not in ("aarch64", "x86_64"):
        raise RuntimeError("Unsupported sandbox architecture")
    abi = libc.syscall(444, 0, 0, 1)
    if abi < 1:
        raise RuntimeError("Landlock unavailable; refusing generated code")
    handled = (1 << 13) - 1
    if abi >= 2:
        handled |= 1 << 13  # REFER
    if abi >= 3:
        handled |= 1 << 14  # TRUNCATE

    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int)]

    rules = Ruleset(handled)
    fd = libc.syscall(444, ctypes.byref(rules), ctypes.sizeof(rules), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")
    try:
        paths = [Path(sys.prefix), Path(sys.base_prefix) / "lib",
                 Path("/usr/local/lib/python3.12"), Path("/etc/ld.so.cache"),
                 Path("/dev/null"), Path("/dev/urandom"), Path(source)]
        for path in dict.fromkeys(p.resolve() for p in paths if p.exists()):
            pfd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                access = (1 << 2) | ((1 << 3) if path.is_dir() else 0)
                rule = PathRule(access, pfd)  # read file / read directory only
                if libc.syscall(445, fd, 1, ctypes.byref(rule), 0) != 0:
                    raise OSError(ctypes.get_errno(), "landlock_add_rule")
            finally:
                os.close(pfd)
        if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
            raise OSError(ctypes.get_errno(), "no_new_privs")
        if libc.syscall(446, fd, 0) != 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self")
    finally:
        os.close(fd)
    sec.seccomp_init.argtypes = [ctypes.c_uint32]
    sec.seccomp_init.restype = ctypes.c_void_p
    sec.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    sec.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    sec.seccomp_load.argtypes = [ctypes.c_void_p]
    sec.seccomp_release.argtypes = [ctypes.c_void_p]
    context = sec.seccomp_init(0x7FFF0000)  # ALLOW by default
    if not context:
        raise RuntimeError("seccomp_init failed")
    try:
        for name in ("socket", "socketpair", "connect", "socketcall", "ptrace",
                     "process_vm_readv", "process_vm_writev", "kill", "tkill", "tgkill",
                     "execve", "execveat", "fork", "vfork", "clone", "clone3",
                     "mount", "umount2", "unshare", "setns", "bpf", "io_uring_setup"):
            number = sec.seccomp_syscall_resolve_name(name.encode())
            if number >= 0 and sec.seccomp_rule_add(context, 0x00050000 | errno.EPERM, number, 0) != 0:
                raise RuntimeError("seccomp rule failed: " + name)
        if sec.seccomp_load(context) != 0:
            raise RuntimeError("seccomp_load failed")
    finally:
        sec.seccomp_release(context)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024**2, 8 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
