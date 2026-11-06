"""Modules built ahead (TENSORFOLD_PREBUILT) stand in for the JIT build when their key matches, and a recording start
(TENSORFOLD_PREBUILT_RECORD) copies what it loaded for the next one. torch is faked: no compiler, no GPU."""

import sys
import types
from pathlib import Path

import pytest

from tensorfold.cuda import build


@pytest.fixture
def fake_torch(monkeypatch, tmp_path):
    """A torch whose cpp_extension.load 'builds' a module file and imports nothing: the calls are counted."""

    calls = []
    versioned = []
    built = tmp_path / "jit"
    ext = types.ModuleType("torch.utils.cpp_extension")
    ext.LIB_EXT = ".so"
    ext.CUDA_HOME = "/opt/cuda"

    def load(name, sources, **kwargs):
        # An op library's load returns its path; a second build of a name in one process is versioned, as torch does.
        calls.append(name)
        artifact = name if calls.count(name) == 1 or not versioned else f"{name}_v{calls.count(name) - 1}"
        (built / name).mkdir(parents=True, exist_ok=True)
        (built / name / f"{artifact}.so").write_bytes(b"module " + artifact.encode())
        return str(built / name / f"{artifact}.so")

    ext.load = load
    ext._get_build_directory = lambda name, verbose=False: str(built / name)
    torch = types.ModuleType("torch")
    torch.__version__ = "2.13.0+cu130"
    torch.version = types.SimpleNamespace(cuda="13.0")
    torch.cuda = types.SimpleNamespace(get_device_capability=lambda *a: (12, 0), get_device_name=lambda *a: "GPU")
    torch.utils = types.ModuleType("torch.utils")
    torch.utils.cpp_extension = ext
    loaded = []
    torch.ops = types.SimpleNamespace(load_library=loaded.append)
    for name, module in {"torch": torch, "torch.utils": torch.utils, "torch.utils.cpp_extension": ext}.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(build, "_toolkit", lambda: [])
    monkeypatch.delenv(build.PREBUILT, raising=False)
    monkeypatch.delenv(build.RECORD, raising=False)
    return types.SimpleNamespace(calls=calls, built=built, loaded=loaded, versioned=versioned)


def _sources(tmp_path, body="__global__ void k() {}"):
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    (src / "k.cu").write_text(body)
    (src / "k.cuh").write_text("#pragma once")
    return [str(src / "k.cu")]


def test_a_recording_start_copies_the_module_and_a_later_start_imports_it(fake_torch, tmp_path, monkeypatch):
    sources = _sources(tmp_path)
    record = tmp_path / "prebuilt"
    monkeypatch.setenv(build.RECORD, str(record))
    assert build.load("tf_k", sources, is_python_module=False) == str(fake_torch.built / "tf_k" / "tf_k.so")
    [copied] = list(record.glob("tf_k/*/tf_k.so"))
    monkeypatch.delenv(build.RECORD)
    monkeypatch.setenv(build.PREBUILT, str(record))
    assert build.load("tf_k", sources, is_python_module=False) == str(copied)
    assert fake_torch.calls == ["tf_k"], "the second start compiled nothing"
    assert fake_torch.loaded == [str(copied)]


def test_a_changed_header_or_flag_misses_the_prebuilt_module_and_compiles(fake_torch, tmp_path, monkeypatch):
    sources = _sources(tmp_path)
    record = tmp_path / "prebuilt"
    monkeypatch.setenv(build.RECORD, str(record))
    build.load("tf_k", sources, is_python_module=False)
    monkeypatch.delenv(build.RECORD)
    monkeypatch.setenv(build.PREBUILT, str(record))
    jit = str(fake_torch.built / "tf_k" / "tf_k.so")
    (tmp_path / "src" / "k.cuh").write_text("#pragma once // changed")
    assert build.load("tf_k", sources, is_python_module=False) == jit
    (tmp_path / "src" / "k.cuh").write_text("#pragma once")
    assert build.load("tf_k", sources, is_python_module=False, extra_cuda_cflags=["-O1"]) == jit
    assert fake_torch.calls == ["tf_k", "tf_k", "tf_k"]


def test_the_key_does_not_depend_on_where_the_runtime_is_unpacked(fake_torch, tmp_path, monkeypatch):
    sources = _sources(tmp_path)
    root = str(Path(build.__file__).resolve().parents[2])
    one = build.prebuilt_key("tf_k", sources, {"extra_include_paths": [], "extra_cflags": [f"-I{root}/x"]})
    monkeypatch.setattr(sys, "prefix", "/somewhere/else")
    two = build.prebuilt_key("tf_k", sources, {"extra_include_paths": [], "extra_cflags": [f"-I{root}/x"]})
    assert one == two
    other = build.prebuilt_key("tf_k", sources, {"extra_include_paths": [], "extra_cflags": ["-I/other/x"]})
    assert other != one


def test_a_header_anywhere_in_the_package_is_in_the_key(fake_torch, tmp_path, monkeypatch):
    sources = _sources(tmp_path)
    package = Path(build.__file__).resolve().parents[1]
    header = package / "cuda" / "zz_prebuilt_key_probe.cuh"
    try:
        header.write_text("// one")
        one = build.prebuilt_key("tf_k", sources, {})
        header.write_text("// two")
        assert build.prebuilt_key("tf_k", sources, {}) != one
    finally:
        header.unlink(missing_ok=True)


def test_the_python_abi_is_in_the_key(fake_torch, tmp_path, monkeypatch):
    import sysconfig

    sources = _sources(tmp_path)
    one = build.prebuilt_key("tf_k", sources, {})
    real = sysconfig.get_config_var
    monkeypatch.setattr(sysconfig, "get_config_var", lambda k: "cpython-399-test-abi" if k == "SOABI" else real(k))
    assert build.prebuilt_key("tf_k", sources, {}) != one


def test_a_versioned_build_is_not_recorded_under_the_name(fake_torch, tmp_path, monkeypatch):
    sources = _sources(tmp_path)
    record = tmp_path / "prebuilt"
    monkeypatch.setenv(build.RECORD, str(record))
    fake_torch.versioned.append(True)
    build.load("tf_k", sources, is_python_module=False)
    (tmp_path / "src" / "k.cu").write_text("__global__ void k2() {}")
    build.load("tf_k", sources, is_python_module=False)
    recorded = [p.read_bytes() for p in record.glob("tf_k/*/tf_k.so")]
    assert recorded == [b"module tf_k"], "the second build (tf_k_v1) is not recorded as tf_k"


def test_a_recording_start_compiles_even_with_a_prebuilt_directory(fake_torch, tmp_path, monkeypatch):
    sources = _sources(tmp_path)
    ahead, record = tmp_path / "ahead", tmp_path / "record"
    monkeypatch.setenv(build.RECORD, str(ahead))
    build.load("tf_k", sources, is_python_module=False)
    monkeypatch.setenv(build.PREBUILT, str(ahead))
    monkeypatch.setenv(build.RECORD, str(record))
    build.load("tf_k", sources, is_python_module=False)
    assert fake_torch.calls == ["tf_k", "tf_k"] and list(record.glob("tf_k/*/tf_k.so"))


def test_a_path_reached_through_a_symlink_keys_like_the_real_one(fake_torch, tmp_path, monkeypatch):
    sources = _sources(tmp_path)
    real = tmp_path / "inc"
    real.mkdir()
    link = tmp_path / "inc-link"
    link.symlink_to(real)
    one = build.prebuilt_key("tf_k", sources, {"extra_cflags": [f"-I{real}"]})
    assert build.prebuilt_key("tf_k", sources, {"extra_cflags": [f"-I{link}"]}) == one


def test_the_toolkit_link_flags_are_not_in_the_key(fake_torch, tmp_path, monkeypatch):
    sources = _sources(tmp_path)
    record = tmp_path / "prebuilt"
    monkeypatch.setenv(build.RECORD, str(record))
    monkeypatch.setattr(build, "_toolkit", lambda: ["-L/somewhere/tensorfold_cudart"])
    build.load("tf_k", sources, is_python_module=False)
    monkeypatch.delenv(build.RECORD)
    monkeypatch.setenv(build.PREBUILT, str(record))
    monkeypatch.setattr(build, "_toolkit", lambda: [])
    build.load("tf_k", sources, is_python_module=False)
    assert fake_torch.calls == ["tf_k"], "the start without nvcc imports the recorded module"
