"""In-process sliding-window rate limiter for endpoints that call the LLM (they cost money).
Single-process only; behind a multi-worker or multi-host deployment use a shared limiter at the proxy."""

import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, max_requests, window_seconds=60.0, clock=time.monotonic):
        self.max_requests = max_requests
        self.window = window_seconds
        self.clock = clock
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key):
        """Record a hit for `key`; returns seconds to wait if over the limit, else None."""
        if self.max_requests <= 0:
            return None
        now = self.clock()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.max_requests:
                return max(1, int(self.window - (now - hits[0])) + 1)
            hits.append(now)
            return None
