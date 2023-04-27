from unittest import TestCase
from chapter02 import ch01


class TestCh02(TestCase):
    def test_sequence(self):
        res = ch01.sequence();
        res.history['loss'][0]
        self.assertGreater(res.history['loss'][0], res.history['loss'][len(res.history['loss']) - 1])
