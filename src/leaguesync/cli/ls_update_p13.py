"""
Update the Pike13 instance into a mongo database.
"""

import argparse
import logging
import sys

from leaguesync import __version__

__author__ = "Eric Busboom"
__copyright__ = "Eric Busboom"
__license__ = "MIT"

_logger = logging.getLogger(__name__)



def parse_args(args):
    """Parse command line parameters
    """
    parser = argparse.ArgumentParser(description="Update the Pike13 instance into a mongo database.")
    parser.add_argument(
        "--version",
        action="version",
        version=f"leaguesync {__version__}",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )

    # add argument to specify config file
    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        help="path to config file",
        action="store",
        nargs='?',
        default=None,
    )

    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def main(args):
    """Wrapper allowing :func:`fib` to be called with string arguments in a CLI fashion

    Instead of returning the value from :func:`fib`, it prints the result to the
    ``stdout`` in a nicely formatted message.

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--verbose", "42"]``).
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.debug("Starting crazy calculations...")
    print("HERE")
    _logger.info("Script ends here")


def run():

    main(sys.argv[1:])


if __name__ == "__main__":

    run()
