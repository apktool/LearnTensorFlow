from tensorflow import config


def gpus():
    all_gpu = config.list_physical_devices("GPU")
    return all_gpu


def cpus():
    all_cpu = config.list_physical_devices("CPU")
    return all_cpu
