from tensorflow.python.client import device_lib


def detect_devices():
    devices = device_lib.list_local_devices()
    devices = [(d.name, d.device_type) for d in devices]
    return {'CPU': [d_name for d_name, d_type in devices if d_type == 'CPU'],
            'GPU': [d_name for d_name, d_type in devices if d_type == 'GPU']}


def get_device():
    devices = detect_devices()
    assert len(devices['CPU']) > 0
    if devices['GPU']:
        return devices['GPU'][0]
    else:
        return devices['CPU'][0]
