import socket
from functools import lru_cache

orig_getaddrinfo = socket.getaddrinfo
@lru_cache(maxsize=50_000)
def shim_getaddrinfo(host, port, family=0, socktype=0, proto=0, flags=0):
    return orig_getaddrinfo(host, port, socket.AF_INET, socktype, proto, flags)

socket.getaddrinfo = shim_getaddrinfo
