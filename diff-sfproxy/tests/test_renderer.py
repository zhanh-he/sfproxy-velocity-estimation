import pytest

from renderers.fluidsynth_sf2 import FluidSynthConfig, FluidSynthSF2Renderer
from renderers.sfizz_sfz import SfizzConfig, SfizzSFZRenderer


def test_renderer_requires_existing_sf2(tmp_path):
    missing = tmp_path / "missing.sf2"
    with pytest.raises(FileNotFoundError):
        FluidSynthSF2Renderer(FluidSynthConfig(sf2_path=str(missing)))


def test_renderer_requires_existing_sfz(tmp_path):
    missing = tmp_path / "missing.sfz"
    with pytest.raises(FileNotFoundError):
        SfizzSFZRenderer(SfizzConfig(sfz_path=str(missing)))
