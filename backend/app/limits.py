"""Request body size limit (pure ASGI, so it also covers chunked bodies without Content-Length)."""

import os

from starlette.responses import JSONResponse


def max_body_bytes():
    return int(os.environ.get("CREDIT_MAX_BODY_BYTES", str(16 * 1024)))


class BodySizeLimitMiddleware:
    def __init__(self, app, max_bytes):
        self.app = app
        self.max_bytes = max_bytes

    def _too_large(self):
        return JSONResponse({"detail": f"Request body exceeds {self.max_bytes} bytes."}, status_code=413)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            return await self._too_large()(scope, receive, send)

        received = 0
        rejected = False
        response_started = False

        async def limited_receive():
            nonlocal received, rejected
            if rejected:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes and not response_started:
                    # Answer 413 ourselves, then tell the app the client went away.
                    rejected = True
                    await self._too_large()(scope, receive, send)
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message):
            nonlocal response_started
            if rejected:
                return  # a 413 was already sent; drop whatever the app tries to send
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        await self.app(scope, limited_receive, guarded_send)
