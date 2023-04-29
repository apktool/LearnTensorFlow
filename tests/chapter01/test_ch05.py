from unittest import TestCase

from chapter01 import ch05


class TestCh05(TestCase):
    def test_load(self):
        ch05.load()

    def test_train(self):
        ch05.train()
