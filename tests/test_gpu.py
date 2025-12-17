"""tests for GPU detection and configuration."""
import os
from unittest.mock import patch

import pytest

from bookmarks.models.gpu import GpuConfig, detect_gpu_backend, get_gpu_config


def test_gpu_config_dataclass():
    """test GpuConfig dataclass properties."""
    config = GpuConfig(n_gpu_layers=-1, backend="cuda", available=True)
    assert config.is_accelerated is True
    assert config.n_gpu_layers == -1
    assert config.backend == "cuda"

    cpu_config = GpuConfig(n_gpu_layers=0, backend="cpu", available=False)
    assert cpu_config.is_accelerated is False


def test_detect_gpu_backend_cpu_fallback():
    """test CPU fallback when no GPU is available."""
    with patch("bookmarks.models.gpu._has_nvidia_gpu", return_value=False):
        with patch("bookmarks.models.gpu._has_vulkan", return_value=False):
            with patch("platform.system", return_value="Linux"):
                backend = detect_gpu_backend()
                assert backend == "cpu"


def test_detect_gpu_backend_metal():
    """test Metal detection on Apple Silicon."""
    with patch("platform.system", return_value="Darwin"):
        with patch("platform.machine", return_value="arm64"):
            backend = detect_gpu_backend()
            assert backend == "metal"


def test_detect_gpu_backend_cuda():
    """test CUDA detection on Linux with NVIDIA GPU."""
    with patch("bookmarks.models.gpu._has_nvidia_gpu", return_value=True):
        with patch("platform.system", return_value="Linux"):
            backend = detect_gpu_backend()
            assert backend == "cuda"


def test_detect_gpu_backend_vulkan():
    """test Vulkan fallback on Linux without NVIDIA."""
    with patch("bookmarks.models.gpu._has_nvidia_gpu", return_value=False):
        with patch("bookmarks.models.gpu._has_vulkan", return_value=True):
            with patch("platform.system", return_value="Linux"):
                backend = detect_gpu_backend()
                assert backend == "vulkan"


def test_get_gpu_config_force_cpu():
    """test forced CPU mode."""
    config = get_gpu_config(force_cpu=True)
    assert config.n_gpu_layers == 0
    assert config.backend == "cpu"
    assert config.available is False


def test_get_gpu_config_env_override():
    """test environment variable override."""
    with patch.dict(os.environ, {"BOOKMARKS_FORCE_CPU": "1"}):
        config = get_gpu_config()
        assert config.n_gpu_layers == 0
        assert config.backend == "cpu"


def test_get_gpu_config_custom_layers():
    """test custom GPU layer count from environment."""
    with patch("bookmarks.models.gpu.detect_gpu_backend", return_value="cuda"):
        with patch.dict(os.environ, {"BOOKMARKS_GPU_LAYERS": "20"}):
            config = get_gpu_config()
            assert config.n_gpu_layers == 20
            assert config.backend == "cuda"
            assert config.is_accelerated is True


def test_get_gpu_config_auto_detect():
    """test automatic GPU detection and configuration."""
    with patch("bookmarks.models.gpu.detect_gpu_backend", return_value="metal"):
        config = get_gpu_config()
        assert config.n_gpu_layers == -1  # all layers
        assert config.backend == "metal"
        assert config.available is True
