import socket
from functools import lru_cache

error_host = set()
orig_getaddrinfo = socket.getaddrinfo

@lru_cache(maxsize=50_000)
def shim_getaddrinfo(host, port, family=0, socktype=0, proto=0, flags=0):
    try:
        if host in error_host:
            raise socket.gaierror
        return orig_getaddrinfo(host, port, socket.AF_INET, socktype, proto, flags)
    except socket.gaierror as e:
        if "Name or service not known" in str(e):
            error_host.add(host)
        raise e

socket.getaddrinfo = shim_getaddrinfo
