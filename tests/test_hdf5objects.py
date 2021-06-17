
from hdf5objects.cli import main


def test_main():
    assert main([]) == 0
