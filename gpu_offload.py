# attempt to offload clip inference to GPU machine
# using a poll on the client machine to see when the GPU station returns calculated files in order to continue execution

import time

def wait_until(somepredicate, timeout, period=30, *args, **kwargs):
  mustend = time.time() + timeout
  while time.time() < mustend:
    if somepredicate(*args, **kwargs): return True
    time.sleep(period)
  return False

