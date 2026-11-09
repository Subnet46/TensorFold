"""Build the CUDA extensions for the GPU present, whatever architecture list the container sets, saying when a build runs or waits on a lock."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import threading
from typing import Any

MIN_CAPABILITY = (8, 9)         # FP8 MMA and e4m3 conversions (Ada); kernels with clusters use them from 9.0
CLUSTERS = (9, 0)               # extensions built only on thread-block clusters (NVFP4) need Hopper or newer
FLASHNEXT_FLOOR = (12, 0)       # Flash Next serves CUDA on sm_120/sm_121 cards only; anything below is refused by name
# stop first: when the lock goes, a waiting start imports whatever module is there without building, even an old one
HINT = "if no other build is running, a killed build left it: stop this start, delete the lock and start again"
LOCK_WAIT_SECONDS = 60.0        # a start still waiting on the same lock this long says so again
PREBUILT = "TENSORFOLD_PREBUILT"                # a directory of modules built ahead (``record``): imported, not compiled
RECORD = "TENSORFOLD_PREBUILT_RECORD"           # copy every module this process loads into such a directory
FLAG_KEYS = ("extra_cflags", "extra_cuda_cflags", "extra_ldflags", "extra_include_paths", "with_cuda",
             "is_python_module", "is_standalone")
HEADERS = (".h", ".hpp", ".cuh", ".inc", ".inl")


def arch_flags(need: tuple[int, int] = MIN_CAPABILITY, arch_specific: bool = False) -> list[str]:
    """nvcc flags for this GPU alone (``arch_specific``: its ``a`` target); a GPU under ``need`` is refused by name."""

    import torch

    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < need:
        why = "thread-block clusters" if need >= CLUSTERS else "FP8 MMA"
        raise RuntimeError(f"TensorFold's CUDA kernels need compute capability {need[0]}.{need[1]} or newer ({why}"
                           f"{' for these weights' if need > MIN_CAPABILITY else ''}); this GPU "
                           f"({torch.cuda.get_device_name()}) is {major}.{minor}")
    a = "a" if arch_specific else ""
    return [f"-gencode=arch=compute_{major}{minor}{a},code=sm_{major}{minor}{a}"]


def refuse_old_gpu(need: tuple[int, int] = MIN_CAPABILITY) -> None:
    """At startup, before any weight loads: a GPU older than ``need`` is refused by name (no GPU: skipped)."""

    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        try:
            arch_flags(need)
        except RuntimeError as exc:
            raise ValueError(str(exc)) from None


def refuse_small_gpu(need: tuple[int, int] = FLASHNEXT_FLOOR) -> None:
    """At startup, before any weight loads: a GPU below Flash Next's CUDA floor is refused by name (no GPU: skipped)."""

    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if (major, minor) < need:
            raise ValueError("TensorFold's Flash Next CUDA kernels run on sm_120 and sm_121 GPUs only "
                             "(DGX Spark GB10, RTX 50 and RTX PRO 6000); this GPU "
                             f"({torch.cuda.get_device_name()}) is {major}.{minor}. Serve Flash Next on CUDA on one "
                             "of those, on the MLX backend, or with another model.")


def load(name: str, sources: str | list[str], need: tuple[int, int] = MIN_CAPABILITY, *, arch_specific: bool = False,
         **kwargs: Any) -> Any:
    """torch's JIT ``load`` for this GPU only (NVIDIA's containers list every architecture back to sm_80), with a line when it compiles or waits on a lock."""

    from torch.utils import cpp_extension

    kwargs["extra_cuda_cflags"] = [*kwargs.get("extra_cuda_cflags", []), *arch_flags(need, arch_specific)]
    # The key before the toolkit's link flags: where a compiler was found says nothing about the module.
    key = prebuilt_key(name, sources, kwargs)
    links = _toolkit()
    if links:
        kwargs["extra_ldflags"] = [*kwargs.get("extra_ldflags", []), *links]
    # A recording start compiles everything it records: what it imports ahead would not be copied.
    ahead = None if os.environ.get(RECORD) else os.environ.get(PREBUILT)
    if ahead:
        module = Path(ahead) / name / key / (name + getattr(cpp_extension, "LIB_EXT", ".so"))
        if module.is_file():
            return _import_prebuilt(name, module, kwargs.get("is_python_module", True))
        _say(f"no prebuilt CUDA extension {name} for this build ({key}); compiling it")
    held = _announce(cpp_extension, name, sources, kwargs.get("build_directory"))
    timer = None
    if held is not None:
        timer = threading.Timer(LOCK_WAIT_SECONDS, _still_waiting, held)
        timer.daemon = True
        timer.start()
    try:
        loaded = cpp_extension.load(name=name, sources=sources, **kwargs)
    finally:
        if timer is not None:
            timer.cancel()
    record = os.environ.get(RECORD)
    if record:
        _record(name, loaded, key, Path(record))
    return loaded


def prebuilt_key(name: str, sources: str | list[str], kwargs: dict[str, Any]) -> str:
    """What a module built ahead must match to stand in for this build: the TensorFold, torch, CUDA and Python (ABI)
    versions, the flags (install paths written as ``<root>``, so a release unpacked elsewhere still matches), and the
    bytes of every source and of every header in the TensorFold package, beside the sources or under the include
    paths (a source may include a header from any of them)."""

    import sysconfig

    import torch
    import tensorfold

    # Every root and every path in a flag canonical (symlinks resolved), so the same files under a link match.
    roots = sorted({os.path.realpath(r) for r in (str(Path(tensorfold.__file__).resolve().parents[1]), sys.prefix,
                                                   os.environ.get("TORCH_EXTENSIONS_DIR", ""),
                                                   os.environ.get("CUDA_HOME", "")) if r},
                   key=len, reverse=True)

    def plain(value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        text = str(value)
        flag, path = (text[:2], text[2:]) if text[:2] in ("-I", "-L") else ("", text)
        if path.startswith("/") and os.path.exists(path):
            text = flag + os.path.realpath(path)
        for root in roots:
            text = text.replace(root, "<root>")
        return text

    h = hashlib.sha256()
    h.update(json.dumps({"name": name, "tensorfold": tensorfold.__version__, "torch": torch.__version__,
                         "cuda": torch.version.cuda, "python": sysconfig.get_config_var("SOABI") or sys.version,
                         "flags": {k: plain(kwargs[k]) for k in FLAG_KEYS if k in kwargs}},
                        sort_keys=True).encode())
    files = [Path(s) for s in ([sources] if isinstance(sources, str) else sources)]
    package = Path(tensorfold.__file__).resolve().parent
    folders = ({package} | {f.resolve().parent for f in files}
               | {Path(p).resolve() for p in kwargs.get("extra_include_paths", [])})
    headers = sorted({f for folder in folders if folder.is_dir() for f in folder.rglob("*") if f.suffix in HEADERS})
    for f in files:
        h.update(b"source " + f.name.encode() + b"\0" + f.read_bytes())
    for f in headers:
        rel = next((str(f.relative_to(d)) for d in sorted(folders) if f.is_relative_to(d)), f.name)
        h.update(b"header " + rel.encode() + b"\0" + f.read_bytes())
    return h.hexdigest()[:32]


def _import_prebuilt(name: str, module: Path, python: bool) -> Any:
    """The module built ahead at ``module``, as torch's ``load`` would return it."""

    if not python:                                  # an op library: torch.ops holds it, load returns its path
        import torch

        torch.ops.load_library(str(module))
        return str(module)
    loader = importlib.machinery.ExtensionFileLoader(name, str(module))
    spec = importlib.util.spec_from_file_location(name, str(module), loader=loader)
    found = importlib.util.module_from_spec(spec)
    loader.exec_module(found)
    return found


def _record(name: str, loaded: Any, key: str, into: Path) -> None:
    """Copy the artifact torch just loaded for ``name`` (a module's file, or an op library's path) to
    ``into/<name>/<key>/<name><ext>``, for a later start to import. A versioned build (``<name>_v1``: the same name
    loaded twice with other sources in one process) initializes under another name and is not recorded."""

    built = Path(getattr(loaded, "__file__", None) or str(loaded))
    if not built.is_file():
        return
    if built.name.split(".", 1)[0] != name:
        _say(f"CUDA extension {name} was built as {built.name}; not recorded for prebuilt starts")
        return
    target = into / name / key / built.name
    if target.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + f".{os.getpid()}")
    shutil.copy2(built, partial)
    os.replace(partial, target)
    _say(f"recorded CUDA extension {name} ({key}) for prebuilt starts")


def _announce(cpp_extension: Any, name: str, sources: str | list[str],
              directory: str | None) -> tuple[str, tuple[int, int]] | None:
    """Say whether ``name`` builds or waits on a lock; return (lock path, (inode, mtime)) when it waits."""

    try:
        directory = directory or cpp_extension._get_build_directory(name, verbose=False)   # private in torch
    except Exception:  # noqa: BLE001 - no lookup in this torch: build without the lines
        return None
    lock = os.path.join(directory, "lock")          # torch's FileBaton for this extension
    try:
        seen = os.stat(lock)
    except OSError:
        seen = None
    if seen is not None:
        _say(f"CUDA extension {name} waits on the build lock {lock}; {HINT}")
        return lock, (seen.st_ino, seen.st_mtime_ns)
    if _needs_build(os.path.join(directory, name + getattr(cpp_extension, "LIB_EXT", ".so")), sources):
        _say(f"building CUDA extension {name} (first start after an install or update; later starts reuse it)")
    return None


def _needs_build(module: str, sources: str | list[str]) -> bool:
    """No built module, or a source newer than it (ninja then compiles again)."""

    try:
        built = os.path.getmtime(module)
    except OSError:
        return True
    for source in [sources] if isinstance(sources, str) else sources:
        try:
            if os.path.getmtime(source) > built:
                return True
        except OSError:
            continue
    return False


def _still_waiting(lock: str, identity: tuple[int, int]) -> None:
    """Repeat the hint while the lock seen before the build is still the one there."""

    try:
        now = os.stat(lock)
    except OSError:
        return
    if (now.st_ino, now.st_mtime_ns) == identity:
        _say(f"still waiting after {LOCK_WAIT_SECONDS:g} s on the build lock {lock}; {HINT}")


def _toolkit() -> list[str]:
    """The pip route's tools: the venv's ninja on PATH, NVIDIA's pip toolkit, and the link flags it needs."""

    venv = Path(sys.executable).parent              # an inactive venv: ninja is beside its python, not on PATH
    tool = "ninja.exe" if os.name == "nt" else "ninja"              # the venv's ninja, named as Windows names it
    if shutil.which("ninja") is None and (venv / tool).is_file():
        os.environ["PATH"] = f"{venv}{os.pathsep}{os.environ.get('PATH', '')}"
    return list(_pip_flags())


@lru_cache(maxsize=1)
def _pip_flags() -> tuple[str, ...]:
    """Once a process: NVIDIA's pip toolkit when torch found none (the module keeps it after the first call)."""

    import torch
    from torch.utils import cpp_extension

    return tuple(pip_toolkit(cpp_extension, torch))


def pip_toolkit(cpp_extension: Any, torch: Any) -> list[str]:
    """No toolkit found (CUDA_HOME, nvcc, /usr/local/cuda): NVIDIA's pip one beside torch, its cudart linked by name."""

    if cpp_extension.CUDA_HOME is not None or not getattr(torch.version, "cuda", None):
        return []
    home = Path(torch.__file__).resolve().parents[1] / "nvidia" / f"cu{torch.version.cuda.split('.')[0]}"
    if os.name == "nt":                       # a Windows wheel: DLLs beside torch and in NVIDIA's cu13 folder
        search = [folder for folder in (home / "bin", home / "lib", Path(torch.__file__).resolve().parent / "lib")
                  if folder.is_dir()]
        for folder in search:                 # a built .pyd loads cudart64_13.dll by name at import
            try:
                os.add_dll_directory(str(folder))
            except (AttributeError, OSError):
                pass                          # old CPython or a refused folder: PATH is the fallback
            os.environ["PATH"] = f"{folder}{os.pathsep}{os.environ.get('PATH', '')}"
        return [f"/LIBPATH:{folder}" for folder in search if (folder / "cudart.lib").is_file()]
    if not (home / "bin" / "nvcc").is_file():
        return []
    cpp_extension.CUDA_HOME = os.environ["CUDA_HOME"] = str(home)    # torch reads its CUDA_HOME at every build
    os.environ["PATH"] = f"{home / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
    _say(f"CUDA compiler: NVIDIA's pip toolkit for CUDA {torch.version.cuda} at {home}")
    versioned = sorted((home / "lib").glob("libcudart.so.*"))
    if (home / "lib" / "libcudart.so").exists() or not versioned:
        return []
    root = os.environ.get("TORCH_EXTENSIONS_DIR") or cpp_extension.get_default_build_root()
    links = Path(root) / "tensorfold_cudart"                         # -lcudart wants the bare name
    links.mkdir(parents=True, exist_ok=True)
    bare = links / "libcudart.so"
    if not bare.is_symlink() or bare.resolve() != versioned[0].resolve():
        bare.unlink(missing_ok=True)
        try:
            bare.symlink_to(versioned[0])
        except FileExistsError:                                      # another start made the same link first
            pass
    return [f"-L{links}"]


def _say(text: str) -> None:
    """One ``[tensorfold]`` line, flushed like the CLI's."""

    print(f"[tensorfold] {text}", flush=True)


__all__ = ["CLUSTERS", "MIN_CAPABILITY", "PREBUILT", "RECORD", "arch_flags", "load", "pip_toolkit", "prebuilt_key"]
