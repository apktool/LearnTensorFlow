from unittest import TestCase

from chapter01 import ch03


class TestCh03(TestCase):
    def test_constant(self):
        ch03.constant()

    def test_tensor(self):
        ch03.tensor()
