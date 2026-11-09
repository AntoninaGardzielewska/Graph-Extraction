import logging
import sys

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging() -> None:
    """Configure simple console logging for command-line pipelines."""
    logging.basicConfig(
        level=logging.INFO,
        format=_FORMAT,
        stream=sys.stdout,
        force=True,
    )
