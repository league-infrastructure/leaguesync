
from .util import *
from leaguesync.util import convert_naive

def pike13_events_df(p13: Pike13, location=None, select_f=None, incl_str=None):
    """Get the events from Pike13 as a DataFrame, form on month ago to now."""

    pdf = Pike13DataFrames(p13)

    eo = pdf.event_occurrences
    loc = pdf.locations
    srv = pdf.services

    t = eo.merge(loc).merge(srv)
    t = t[t.start_at >= str(one_month_ago().date())]

    if location:
        t = Calendar.filter_location(t, location)

    if select_f:
        t = t[t.apply(select_f, axis=1)]

    if incl_str:
        t = includes_str(t, incl_str)

    t = in_date_range(t)

    # Ensure the datetime is timezone-aware, set to UTC
    for c in ['start_at', 'end_at']:
        t[c] = t[c].apply(lambda x: convert_naive(x).tz_convert('America/Los_Angeles'))

    return t

def render_events_html(p13: Pike13, location=None, select_f=None, incl_str=None):
    events = pike13_events_df(p13, location, select_f, incl_str)

    # Add the day of week and hours strings, such as
    # Sundays at 4:00PM, Saturdays at 4:00PM
    events = make_when_df(events)

    return render('calendar.html',
                    events = events.to_dict(orient='records'))
