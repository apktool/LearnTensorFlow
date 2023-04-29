from unittest import TestCase

from chapter01 import ch04


class TestCh04(TestCase):
    def test_func(self):
        ch04.gradient()
        ch04.enumerate1()
        ch04.onehot()
        ch04.softmax()
        ch04.assignsub()
        ch04.argmax()
