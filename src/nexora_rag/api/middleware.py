"""
Request-ID middleware, timing, error logging.

STATUS: placeholder — implemented in Step P.
"""

# TODO(Step P): implement this module


import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{(time.perf_counter() - start)*1000:.1f}ms"
        return response