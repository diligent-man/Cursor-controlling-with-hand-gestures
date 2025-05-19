import time

__all__ = ["FPSCalculator"]


class FPSCalculator(object):
    __prev_time: float = 0

    def __init__(self):
        pass

    def _update(self) -> float:
        cur_time: float = time.time()
        fps = 1 / (cur_time - self.__prev_time)

        self.__prev_time = cur_time
        return fps

    def __call__(self, *args, **kwargs):
        fps = self._update()
        return fps
