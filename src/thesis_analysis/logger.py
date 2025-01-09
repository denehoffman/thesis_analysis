import sys as _sys

from loguru import _defaults, logger

__all__ = ['logger']


def formatter(record) -> str:  # noqa: PLR0911
    level = record['level'].name
    if level == 'TRACE':
        return '\x1b[30;47;1m TRACE    \x1b[0m {time:YYYY-MM-DDTHH::mm:ss.SSSS!UTC} - {message}\n{exception}'
    if level == 'DEBUG':
        return '\x1b[30;44;1m DEBUG    \x1b[0m {time:YYYY-MM-DDTHH::mm:ss.SSSS!UTC} - {message}\n{exception}'
    if level == 'INFO':
        return '\x1b[37;40;1m INFO     \x1b[0m {time:YYYY-MM-DDTHH::mm:ss.SSSS!UTC} - {message}\n{exception}'
    if level == 'SUCCESS':
        return '\x1b[30;42;1m SUCCESS  \x1b[0m {time:YYYY-MM-DDTHH::mm:ss.SSSS!UTC} - {message}\n{exception}'
    if level == 'WARNING':
        return '\x1b[30;43;1m WARNING  \x1b[0m {time:YYYY-MM-DDTHH::mm:ss.SSSS!UTC} - {message}\n{exception}'
    if level == 'ERROR':
        return '\x1b[30;45;1m ERROR    \x1b[0m {time:YYYY-MM-DDTHH::mm:ss.SSSS!UTC} - {message}\n{exception}'
    if level == 'CRITICAL':
        return '\x1b[30;41;1m CRITICAL \x1b[0m {time:YYYY-MM-DDTHH::mm:ss.SSSS!UTC} - {message}\n{exception}'
    return str(_defaults.LOGURU_FORMAT)


logger.remove()
logger.add(_sys.stderr, format=formatter, level=0)
