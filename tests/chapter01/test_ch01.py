from unittest import TestCase
from chapter01 import ch01


class TestCh01(TestCase):
    def test_gpus(self):
        gpus = ch01.gpus()
        self.assertEqual(0, len(gpus))

    def test_cpus(self):
        cpus = ch01.cpus()
        self.assertEqual(1, len(cpus))
        for cpu in cpus:
            self.assertEqual("CPU", cpu.device_type)
