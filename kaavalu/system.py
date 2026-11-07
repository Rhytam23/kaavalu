"""Pi health readings: CPU temperature, throttle flag, peak RAM. Each returns None off-Pi."""

import platform
import shutil
import subprocess


def _vcgencmd(*args):
    exe = shutil.which("vcgencmd")
    if not exe:
        return None
    try:
        return subprocess.run([exe, *args], capture_output=True, text=True, timeout=2).stdout.strip()
    except Exception:
        return None


def cpu_temp_c():
    out = _vcgencmd("measure_temp")  # "temp=54.2'C"
    if out and "=" in out:
        try:
            return float(out.split("=")[1].split("'")[0])
        except ValueError:
            pass
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", encoding="ascii") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return None


def throttled():
    """Raw `vcgencmd get_throttled` value, e.g. '0x0' (0x0 = never throttled)."""
    out = _vcgencmd("get_throttled")  # "throttled=0x0"
    return out.split("=")[1] if out and "=" in out else None


def peak_rss_mb():
    try:
        import resource
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return kb / 1024.0 if platform.system() != "Darwin" else kb / (1024.0 * 1024.0)
    except Exception:
        return None


def machine():
    return {"machine": platform.machine(), "system": platform.system(),
            "python": platform.python_version(), "node": platform.node()}
