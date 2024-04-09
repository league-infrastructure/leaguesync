import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple, List

from dateutil.relativedelta import relativedelta
from dotenv import dotenv_values

try:
    EARLIEST_DATE = datetime.fromisoformat(os.environ.get('LEAGUE_BEFORETIMES', '2015-01-01T00:00:00Z'))
except:
    # for 3.10 and earlier.
    EARLIEST_DATE = datetime(2015, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc)

def get_config(file: str | Path = None) -> Dict[str, Any]:

    if file:
        fp = Path(file)
    else:

        fp = Path.home().joinpath('.league.env')

    config = {
        **os.environ,
        **dotenv_values(fp),
    }

    return config


def path_interp(path: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
    """
    Interpolates the parameters into the endpoint URL. So if you have a path
    like '/api/v1/leagues/:league_id/teams/:team_id' and you call

            path_interp(path, league_id=1, team_id=2, foobar=3)

    it will return '/api/v1/leagues/1/teams/2', along with a dictionary of
    the remaining parameters {'foobar': 3}.

    :param path: The endpoint URL template with placeholders.
    :param kwargs: The keyword arguments where the key is the placeholder (without ':') and the value is the actual value to interpolate.

    :return: A string with the placeholders in the path replaced with actual values from kwargs.
    """

    params = {}
    for key, value in kwargs.items():
        placeholder = f":{key}"  # Placeholder format in the path
        if placeholder in path:
            path = path.replace(placeholder, str(value))
        else:
            # Remove the trailing underscore from the key, so we can use params
            # like 'from' that are python keywords.
            params[key.rstrip('_')] = value

    return path, params


def end_of_today():
    """Return the end of today as a string in isoformat in the Z timezone"""
    from datetime import datetime, timedelta

    # Get current time
    now = datetime.now()

    # Get end of the day
    return datetime(now.year, now.month, now.day) + timedelta(days=1, seconds=-1)


def one_month_ago():
    return (datetime.now() - relativedelta(months=1)) \
        .replace(hour=0, minute=0, second=0, microsecond=0)


def one_month_before(dt: datetime):
    return (dt - relativedelta(months=1)).replace(hour=0, minute=0, second=0, microsecond=0)


def one_month_after(dt: datetime):
    return (dt + relativedelta(months=1)).replace(hour=23, minute=59, second=59, microsecond=0)


def month_range(start_date, end_date):
    """Generate Month ranges from the start date to the end date """
    while start_date <= end_date:
        yield start_date.replace(hour=0, minute=0, second=0, microsecond=0), \
            start_date + relativedelta(months=1)
        start_date += relativedelta(months=1)


def convert_naive(dt: datetime):
    """Convert a naive datetime to a timezone aware datetime, but only if it is naive"""

    if dt.tzinfo is None:
        from pytz import utc
        return utc.localize(dt)
    else:
        return dt


def last_event_date(events):
    from datetime import datetime
    """Find the end time of the last event in the set"""
    events = list(sorted(events, key=lambda e: e['end_at']))
    if not events:
        return EARLIEST_DATE
    else:
        return datetime.fromisoformat(events[-1]['end_at'])


def expand_custom(p: dict | List) -> dict:
    """Expand the custom fields into the top level of the dictionary as normal
    key/value pairs"""

    if isinstance(p, list):
        return [expand_custom(x) for x in p]

    import re
    d = dict(**p)

    for attr in d.get('custom_fields', []):
        attr_name = re.sub(r'[^A-Za-z0-9 ]+', '', attr['name']).lower().replace(' ', '_')
        if attr_name not in d:
            d[attr_name] = attr['value']

    if 'custom_fields' in d:
        del d['custom_fields']
    return d
