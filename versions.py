import platform
import socket
import uuid
import re
import logging
import wmi

import sklearn
import numpy
import pandas
import matplotlib
import seaborn
import psutil
import scikitplot
import psutil

# Define your thresholds
CPU_USAGE_THRESHOLD = 80  # percent
MEMORY_USAGE_THRESHOLD = 80  # percent
DISK_USAGE_THRESHOLD = 80  # percent

def check_system_health():
    # Check CPU usage
    cpu_usage = psutil.cpu_percent(interval=1)
    if cpu_usage > CPU_USAGE_THRESHOLD:
        print(f"WARNING: High CPU usage detected: {cpu_usage}%")

    # Check Memory usage
    memory_usage = psutil.virtual_memory().percent
    if memory_usage > MEMORY_USAGE_THRESHOLD:
        print(f"WARNING: High Memory usage detected: {memory_usage}%")

    # Check Disk usage
    disk_usage = psutil.disk_usage('/').percent  # Change '/' to the disk you want to monitor
    if disk_usage > DISK_USAGE_THRESHOLD:
        print(f"WARNING: High Disk usage detected: {disk_usage}%")

# Example usage
check_system_health()


def print_versions():
    print('scikit-learn v{},'.format(sklearn.__version__))
    print('NumPy v{},'.format(numpy.__version__))
    print('Pandas v{},'.format(pandas.__version__))
    print('Matplotlib v{},'.format(matplotlib.__version__))
    print('Seaborn v{},'.format(seaborn.__version__))
    print('Psutil v{},'.format(psutil.__version__))
    print('Scikitplot, v{}'.format(scikitplot.__version__))


def print_system_info():
    info = {'platform': platform.system(),
            'platform-release': platform.release(),
            'platform-version': platform.version(),
            'architecture': platform.machine(),
            'hostname': socket.gethostname(),
            'ip-address': socket.gethostbyname(socket.gethostname()),
            'mac-address': ':'.join(re.findall('..', '%012x' % uuid.getnode())),
            'processor': platform.processor(),
            'ram': str(round(psutil.virtual_memory().total / (1024.0 ** 3))) + " GB"}

    computer = wmi.WMI()
    os_info = computer.Win32_OperatingSystem()[0]
    os_name = platform.system() + ' ' + platform.release()
    os_version = ' '.join([os_info.Version, os_info.BuildNumber])
    proc_info = computer.Win32_Processor()[0]
    gpu_info = computer.Win32_VideoController()[0]
    system_ram = float(os_info.TotalVisibleMemorySize) / 1048576  # KB to GB

    print('OS Name: {0}'.format(os_name))
    print('OS Version: {0}'.format(os_version))
    print('Architecture: {0}'.format(platform.machine()))
    print('CPU: {0}'.format(proc_info.Name))
    print('RAM: {0} GB'.format(round(system_ram)))
    print('Graphics Card: {0}'.format(gpu_info.Name))

    print(info)


print_versions()
print_system_info()


p = psutil.Process()
info = p.memory_percent()
print(info)

print('The end.')
quit()
