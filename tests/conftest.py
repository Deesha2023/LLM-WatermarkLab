"""Shared test setup: force the offline toy language model so tests need no downloads."""

import os
import sys

os.environ["WATERMARKLAB_BACKEND"] = "toy"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from watermarklab import config
from watermarklab.backend import ToyBackend, set_backend


@pytest.fixture(autouse=True, scope="session")
def toy_backend():
    config.BACKEND = "toy"
    set_backend(ToyBackend())
    yield
    set_backend(None)


PROMPT = "Explain what artificial intelligence is to a beginner."


@pytest.fixture
def prompt():
    return PROMPT
