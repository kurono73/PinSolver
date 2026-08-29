from typing import Any, Optional


class IcpCancelled(Exception):
    pass


def _windows_escape_pressed() -> bool:
    try:
        import ctypes
        return bool(ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000)
    except Exception:
        return False


def is_cancelled(cancel_event: Optional[Any] = None) -> bool:
    if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
        return True
    return _windows_escape_pressed()


def raise_if_cancelled(cancel_event: Optional[Any] = None) -> None:
    if is_cancelled(cancel_event):
        raise IcpCancelled()
