"""
Update the Pike13 instance into a mongo database.
"""

import argparse
import logging
import sys
from leaguesync.pike13 import logger as p13_logger
from leaguesync import *

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
    parser.set_defaults(run_command=None)

    subparsers = parser.add_subparsers(help='sub-command help')

    sync_parser = subparsers.add_parser('sync', help='Sync mongo database')
    sync_parser.set_defaults(run_command='sync')

    return parser.parse_args(args)

def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig( stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S")
    p13_logger.setLevel(loglevel)

def main(argv):
    """Wrapper allowing :func:`fib` to be called with string arguments in a CLI fashion

    Instead of returning the value from :func:`fib`, it prints the result to the
    ``stdout`` in a nicely formatted message.

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--verbose", "42"]``).
    """
    args = parse_args(argv)
    setup_logging(args.loglevel)

    if args.run_command is not None:
        # Lookup the string in run_coimmand to find the function in the module
        cmd = getattr(sys.modules[__name__], args.run_command)

        if cmd:
            cmd(args)


def sync(args):
    p13 = Pike13(args.config)
    p13.update()

def run():

    main(sys.argv[1:])


if __name__ == "__main__":

    run()
