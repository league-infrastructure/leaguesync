"""
Update the Pike13 instance into a mongo database.
"""

import argparse
import logging
from pathlib import Path

from leaguesync import *
from leaguesync import __version__
from leaguesync.pike13 import logger as logger

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

    config_parser = subparsers.add_parser('config', help='Show configuration')
    config_parser.set_defaults(run_command='config')

    extract_parser = subparsers.add_parser('extract', help='Extract data')
    extract_parser.set_defaults(run_command='extract')

    extract_parser.add_argument("-d", "--dir", help="Where to write extracted data",
                                default=Path.cwd())

    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """

    if loglevel:
        logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
        logging.basicConfig(stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S")
        logger.setLevel(loglevel)


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
    config = get_config(args.config)

    if config is None:
        logger.error("No configuration file found")
        sys.exit(1)

    p13 = Pike13(config)
    p13.update()


def extract(args):
    import pandas as pd

    dd = (Path(args.dir) if not isinstance(args.dir, Path) else args.dir).resolve()

    # Make dir and parents if it does not exist
    dd.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting data to {dd}")

    config = get_config(args.config)

    if config is None:
        logger.error("No configuration file found")
        sys.exit(1)

    p13 = Pike13(config)
    pdf = Pike13DataFrames(p13)

    s = p13.students
    s = expand_custom(s)
    s = pd.DataFrame(s)
    s.to_csv(dd / 'students_expanded.csv', index=False)

    pdf.visits.to_parquet(dd / 'visits.parquet')

    pdf.students.to_csv(dd / 'students.csv', index=False)
    pdf.parents.to_csv(dd / 'parents.csv', index=False)
    p13.make_people_df().to_csv(dd / 'people.csv', index=False)
    pdf.events.to_csv(dd / 'events.csv', index=False)
    pdf.event_occurrences.to_csv(dd / 'event_occ.csv', index=False)
    pdf.services.to_csv(dd / 'services.csv', index=False)
    pdf.locations.to_csv(dd / 'locations.csv', index=False)

def config(args):
    from leaguesync.pike13 import PIKE13_BUSINESS_DOMAIN

    print(f"leaguesync {__version__}")

    config = get_config(args.config)

    print("Config:")
    print(f"    PIKE13_CLIENT_ID:       {config.get('PIKE13_CLIENT_ID')}")
    print(f"    PIKE13_CLIENT_SECRET:   {config.get('PIKE13_CLIENT_SECRET')}")
    print(f"    PIKE13_BUSINESS_DOMAIN: {config.get('PIKE13_BUSINESS_DOMAIN', PIKE13_BUSINESS_DOMAIN)}")
    print(f"    LEAGUESYNC_MONGO_URI:   {config.get('LEAGUESYNC_MONGO_URI')}")


def run():
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
