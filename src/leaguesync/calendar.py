"""Read and write events to Google Calendars"""

import logging
from datetime import datetime
from datetime import timedelta
from functools import cached_property
from pathlib import Path
from tqdm.auto import tqdm

import pytz
from google.oauth2 import service_account
from googleapiclient.discovery import build

from leaguesync import Pike13
from .util import get_config, convert_naive

logger = logging.getLogger(__name__)

# Google Apps color ids to colors and names, for events.
colors = {
    'Light Blue': '1',
    'Turquoise': '2',
    'Lavender': '3',
    'Coral': '4',
    'Mustard': '5',
    'Orange': '6',
    'Sky Blue': '7',
    'Grey': '8',
    'Royal Blue': '9',
    'Green': '10',
    'Red': '11'}


class Calendar:

    def __init__(self, config_file: str | Path = None,
                 location=None,
                 cal_id=None,
                 event_f=None,):
        """Initialize the Pike13 class. This will get the access token and
        initialize the API client.

        :param config_file: The path to the config file.
        :param location: If null, all locations. May be the location code, a number or the name
        :param cal_id: The calendar id to use. If null, it will use the one in the config file.
        :param event_f: A function to filter the events. It should take an event and return True if it should be included.



        If the location is a code, like 'CV' or 'MX', you can prefix it with a '!'
        to invert the selection, so '!CV' will select all locations except 'CV'.
        """

        self.location = location
        self.event_f = event_f

        self.config = get_config(config_file)

        self.cred_file = self.config['GA_CREDENTIALS']
        self.cal_id = cal_id if cal_id else self.config[
            'GA_CLASSES_CAL']  # The calendar ID you shared with the service account

        SCOPES = ['https://www.googleapis.com/auth/calendar']

        credentials = service_account.Credentials.from_service_account_file(self.cred_file, scopes=SCOPES)

        self.service = build('calendar', 'v3', credentials=credentials)

    @cached_property
    def pike13(self):
        """Get the Pike13 service object from the config file for this object"""

        return Pike13(self.config)

    @staticmethod
    def get_private(event):
        """Get the private data from the event"""
        return event.get('extendedProperties', {}).get('private', {})

    def get_events(self, start_time=None, end_time=None):
        """Get events between the start and end times. If no start time is
        provided, it will default to 30 days ago. If no end time is provided,
        it will default to the end of the calendar"""

        if start_time is None:
            start_time = datetime.now(pytz.utc) - timedelta(days=30)
        start_time = convert_naive(start_time).isoformat()
        end_time = convert_naive(end_time).isoformat() if end_time else None

        page_token = None
        while True:

            events_result = self.service.events().list(calendarId=self.cal_id, timeMin=start_time, timeMax=end_time,
                                                       pageToken=page_token).execute()

            events = events_result.get('items', [])

            for event in events:
                event_id = self.get_private(event).get('event_occ_id')

                if event_id:
                    yield event, self.get_private(event)

            page_token = events_result.get('nextPageToken')
            if not page_token:
                break

    @property
    def events(self):
        return list(self.get_events())

    def _make_event_map(self, events):
        d = {}
        for event, prv in events:
            d[str(prv['event_occ_id'])] = event

        return d

    @property
    def events_map(self):
        return self._make_event_map(self.events)

    def get_event(self, event_id):
        """Get an event from the calendar, given its calendar id ( not the pike 13 id )"""
        return self.service.events().get(calendarId=self.cal_id, eventId=event_id).execute()

    def delete_event(self, event_id):
        """Delete an event from the calendar, given its calendar id ( not the pike 13 id )"""
        return self.service.events().delete(calendarId=self.cal_id, eventId=event_id).execute()

    def delete_events(self, events=None, progress=False):
        """Delete events from the calendar. If no events are provided, it will
        delete all of the events returned by self.events"""

        events = events if events else self.events

        if progress:
            events = tqdm(events)

        for event, prv in events:
            self.delete_event(event['id'])

        return len(events)

    @staticmethod
    def make_event_data(event_id: int, event_occ_id: int,
                        start_dt: datetime, end_dt: datetime,
                        summary: str, desc: str,
                        color: int, state: bool):
        event = {
            'summary': summary,
            'description': desc,
            'start': {
                'dateTime': start_dt.isoformat(),
                'timeZone': 'America/Los_Angeles',
            },
            'end': {
                'dateTime': end_dt.isoformat(),
                'timeZone': 'America/Los_Angeles',
            },
            'extendedProperties': {
                'private': {
                    'event_id': event_id,
                    'event_occ_id': event_occ_id,  # Your custom metadata
                    'state': state
                }
            },
            'colorId': color
        }

        return event

    @staticmethod
    def select_color(e):
        """Select a color for the event based on the event type"""

        if e['state'] == 'canceled':
            return colors['Red']
        if 'python' in e['event_name'].lower():
            return colors['Royal Blue']
        elif 'java' in e['event_name'].lower():
            return colors['Light Blue']
        elif 'make-up' in e['event_name'].lower():
            return colors['Grey']
        else:
            return colors['Green']

    @staticmethod
    def meeting_html(e):

        ds = [f"<p>s</p>" for s in (e['description_short'], e['description']) if
              s and s.strip()]  # remove blank and empty descriptions
        desc = '\n\n'.join(ds)

        return f"""{"<b>CANCELED</b><br>" if e['state'] == 'canceled' else ""}
<a href="https://jtl.pike13.com/group_classes/{e['service_id']}">{e['service_name']}</a> at {e['location_name']}<br>
{e['category_name']}<br>
{desc}
    """

    @staticmethod
    def make_event_data_from_event(e):

        title = e['event_name'] or 'No Title'

        if e['state'] == 'canceled':
            title = f'**CANCELLED** {title}'

        return Calendar.make_event_data(
            e['event_id'],
            e['event_occurrence_id'],
            e['start_at'],
            e['end_at'],
            title,
            Calendar.meeting_html(e),
            Calendar.select_color(e),
            e['state']
        )

    def create_event(self, event):
        """Create an event on the calendar from a pike13 event occurrance"""

        event_data = self.make_event_data_from_event(event)

        event = self.service.events().insert(calendarId=self.cal_id, body=event_data).execute()

        return event

    def update_event(self, event_id, event):
        """Update an event on the calendar

        event: the Pike13 event occurrance to update
        event_id: the calendar event id

        """

        event_data = self.make_event_data_from_event(event)

        event = self.service.events().update(calendarId=self.cal_id, eventId=event_id, body=event_data).execute()

        return event

    def match_event_ids(self, p13_e, cal_e=None):
        """Match the calendar events with the Pike13 event, returning just the ids"""

        cal_e = cal_e if cal_e else self.events

        cal_ids = set([str(prv['event_occ_id']) for event, prv in cal_e])
        p13_ids = set([str(e['event_occurrence_id']) for e in p13_e])

        cal_only_ids = cal_ids - p13_ids
        p13_only_ids = p13_ids - cal_ids
        # ids that are in both
        both_ids = cal_ids & p13_ids

        return cal_ids, p13_ids, both_ids, p13_only_ids, cal_only_ids

    def match_events(self, p13_e, cal_e=None):

        cal_ids, p13_ids, both_ids, p13_only_ids, cal_only_ids = self.match_event_ids(p13_e, cal_e)

        cal_e = cal_e if cal_e else self.events
        ev_map = self._make_event_map(cal_e)

        cal_only_events = [ev_map[str(ev_id)] for ev_id in cal_only_ids]
        p13_only_events = [e for e in p13_e if str(e['event_occurrence_id']) in p13_only_ids]
        both_events = [(e, ev_map[str(e['event_occurrence_id'])]) for e in p13_e if
                       str(e['event_occurrence_id']) in both_ids]

        return both_events, p13_only_events, cal_only_events

    @staticmethod
    def is_location(r, l):
        """Return true if the location spec matches something in the record"""

        if isinstance(l, str) and l[0] == '!': # Negation
            l = l[1:]
            return (r.location_id != l) and (r.location_code != l) and (r.location_name != l)
        elif isinstance(l, list): # AND a list of terms
            t = [Calendar.is_location(r, x) for x in l]
            return all(t)
        elif isinstance(l, tuple): # OR a list of terms
            t = [Calendar.is_location(r, x) for x in l]
            return any(t)
        else:
            return (l is None) or (r.location_id == l) or (r.location_code == l) or (r.location_name == l)

    @staticmethod
    def filter_location(df, l):
        """Filter the Pike13 events DataFrame to only include events at the location"""

        return df[df.apply(Calendar.is_location, args=(l,), axis=1)]


    def pike13_events_df(self, p13: Pike13 = None):
        """Get the events from Pike13 as a DataFrame, form on month ago to now."""

        from leaguesync import Pike13DataFrames
        from .util import one_month_ago

        p13 = p13 if p13 is not None else self.pike13

        pdf = Pike13DataFrames(p13)

        eo = pdf.event_occurrences
        loc = pdf.locations
        srv = pdf.services

        t = eo.merge(loc).merge(srv)
        t = t[t.start_at >= str(one_month_ago().date())]

        if self.location:
            t = t[t.apply(Calendar.is_location, args=(self.location,), axis=1)]
        elif self.event_f:
            t = t[t.apply(self.event_f, axis=1)]

        # Ensure the datetime is timezone-aware, set to UTC
        for c in ['start_at', 'end_at']:
            t[c] = t[c].apply(lambda x: convert_naive(x).tz_convert('America/Los_Angeles'))


        return t

    def get_event_sets(self, p13: Pike13 = None):
        """Get the events from the calendar and Pike13 for the same
        time range, one month ago on"""

        p13 = p13 if p13 is not None else self.pike13

        p13df = self.pike13_events_df(p13)
        p13_events = p13df.to_dict('records')

        logger.debug(f'Got {len(p13_events)} events from Pike13')
        cal_events = list(self.get_events(p13df.start_at.min()))
        logger.debug(f'Got {len(cal_events)} events from Google Calendar')

        return p13_events, cal_events

    def update_events(self, p13: Pike13 = None, progress=False):
        """Recompute the  contents of the events, for instance if you've changed
         the text templates, or colors. """
        from tqdm.auto import tqdm

        p13_events, cal_events = self.get_event_sets(p13)
        both, p13_only, cal_only = self.match_events(p13_events, cal_events)

        if progress:
            both = tqdm(both)

        for p13_event, cal_event in both:
            self.update_event(cal_event['id'], p13_event)

    def update(self, p13: Pike13 = None):
        """Update calendar from Pike13"""

        p13_events, cal_events = self.get_event_sets(p13)

        both, p13_only, cal_only = self.match_events(p13_events, cal_events)

        logger.debug(f'Got {len(both)} events in both Pike13 and Google Calendar')
        logger.debug(f'Got {len(p13_only)} events in Pike13 only')
        logger.debug(f'Got {len(cal_only)} events in Google Calendar only')

        for p13_event in p13_only:
            self.create_event(p13_event)

        for p13_event, cal_event in both:
            self.update_event(cal_event['id'], p13_event)

        for cal_event in cal_only:
            self.delete_event(cal_event['id'])
