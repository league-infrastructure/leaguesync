import logging
from functools import cached_property

import pandas as pd
import pytz
import requests
from leaguesync.mongokv import MongoKV
from more_itertools import chunked
from pymongo import MongoClient
from pymongo import UpdateOne
from pymongo.database import Database
from ratelimit import limits, sleep_and_retry
from tenacity import retry, wait_fixed, stop_after_attempt, before_sleep_log, retry_if_exception_type
from .util import *
from .util import EARLIEST_DATE
import warnings
from datetime import timedelta

logger = logging.getLogger(__name__)

PIKE13_BUSINESS_DOMAIN='jtl.pike13.com' # Default business domain

def get_token(config) -> str:
    """Use the OAUTH 2 flow to get an access token. Reads variables from the
    environment and the ~/.league.env file."""

    client_id = config['PIKE13_CLIENT_ID']
    client_secret = config['PIKE13_CLIENT_SECRET']

    payload = {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret,
    }

    # Where to get the access token
    business_domain = config.get('PIKE13_BUSINESS_DOMAIN', PIKE13_BUSINESS_DOMAIN)
    token_url = f'https://{business_domain}/oauth/token'

    response = requests.post(token_url, data=payload)
    token = response.json()  # The token and other details are in the response

    return token['access_token']


class Pike13:
    """Interface class to Pike13 API."""

    token: str  # The oauth access token
    db: Database  # The mongodb database

    def __init__(self, config_file: str | Path | dict = None):
        """Initialize the Pike13 class. This will get the access token and
        initialize the API client."""


        self.config = get_config(config_file) if isinstance(config_file, (str, Path)) else config_file

        self.token = get_token(self.config)
        self.db = self._get_mongodb_db()

        self.kv = MongoKV(self.db['kv'])

    def _get_mongodb_db(self) -> Database:
        client = MongoClient(self.config['LEAGUESYNC_MONGO_URI'])
        db = client.pike13
        db.people.create_index("id", unique=True)
        db.event_occs.create_index("id", unique=True)
        db.events.create_index("id", unique=True)
        db.visits.create_index("id", unique=True)

        return db

    @retry(wait=wait_fixed(2),  # wait 2 seconds between each retry
           stop=stop_after_attempt(5),  # stop after 5 attempts
           before_sleep=before_sleep_log(logger, logging.DEBUG),  # log before sleep
           retry=retry_if_exception_type(requests.exceptions.ReadTimeout))  # only retry on ReadTimeout
    @sleep_and_retry
    @limits(calls=180, period=60)
    def get(self, url, headers, params):
        # logger.debug(f"GET {url}, params={params}")
        return requests.get(url, headers=headers, params=params)

    def pike13_get_pages(self, endpoint: str, per_page=100, **kwargs):
        """
        A generator function to get all pages of results from a pike13 API endpoint.

        :param token: The OAuth token for authorization.
        :param endpoint: The API endpoint for the pike13 data.
        :param per_page: Number of results per page (max and default is 100).
        :return: Yields the result of each page.
        """
        headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }

        # Initialize pagination parameters
        params = {'per_page': per_page}
        ipath, params = path_interp(endpoint, **kwargs)
        params['per_page'] = per_page
        url = 'https://jtl.pike13.com' + ipath  # Start with the initial URL
        while url:
            # Make the API call
            try:
                response = self.get(url, headers=headers, params=params)

                # Check if the response is successful
                if response.status_code == 200:
                    data = response.json()
                    yield data  # Yield the current page of results

                    # Prepare for the next iteration
                    url = data.get('next')  # Get the URL for the next page

                else:
                    raise Exception(
                        f"Failed to retrieve data. Status Code: {response.status_code}, Detail: {response.text}")
            except requests.exceptions.ReadTimeout as e:
                logger.info(f"ReadTimeout: {e}")
                raise  # Reraise the exception to trigger tenacity retry

    def delete_duplicates(self):
        collection = self.db.people

        cursor = collection.aggregate([
            {"$group": {"_id": "$id", "unique_ids": {"$addToSet": "$_id"}, "count": {"$sum": 1}}},
            {"$match": {"count": {"$gte": 2}}}
        ])

        response = []
        for doc in cursor:
            del doc["unique_ids"][0]
            for id in doc["unique_ids"]:
                response.append(collection.delete_one({"_id": id}))

        print(f'Deleted {len(response)} duplicate documents')

    @property
    def last_people_update_time(self):
        """Find the largest last updated time, which is the time we should
        start searching for newly updated records"""

        collection = self.db.people
        largest_updated_at = None

        # Find the document with the largest 'updated_at' value
        largest_updated_at_document = collection.find_one(
            {},  # No filter, checking all documents
            sort=[('updated_at', -1)]  # Sorting documents by 'updated_at' in descending order
        )

        # Extracting the 'updated_at' value
        if largest_updated_at_document:
            largest_updated_at = largest_updated_at_document.get('updated_at')

        if not largest_updated_at:
            largest_updated_at = str(EARLIEST_DATE)

        # Add one second, because reutnring the largest value will match with that value
        # when using updated_since with the api
        date = datetime.fromisoformat(largest_updated_at)
        return date + timedelta(seconds=1)

    def get_updated_people(self, since: datetime = None) -> List:

        api_endpoint = '/api/v2/desk/people'  # Endpoint for clients' information

        people = []

        if not since:
            since = self.last_people_update_time
        elif isinstance(since, str):
            since = datetime.fromisoformat(since)

        since = since.astimezone(pytz.utc).isoformat() + 'Z'

        for page in self.pike13_get_pages(api_endpoint,
                                          include_relationships=True,
                                          updated_since=since,
                                          sort='-updated_at'):
            for p in page['people']:
                people.append(p)

        return list(sorted(people, key=lambda p: p['updated_at']))

    def make_people_df(self) -> pd.DataFrame:
        """Return all people as a dataframe"""

        keys = 'id email birthdate membership is_member guardian_name guardian_email joined_at updated_at github_acct_name'.split()
        from operator import itemgetter
        ig = itemgetter(*keys)

        def sub_dict(keys, p):
            p = expand_custom(p)
            return {k: p.get(k) for k in keys}

        pdf = pd.DataFrame(sub_dict(keys, p) for p in self.people)

        pdf['github_acct_name'] = pdf.github_acct_name.str.strip().replace('', None)
        pdf['birthdate'] = pd.to_datetime(pdf.birthdate)
        pdf['joined_at'] = pd.to_datetime(pdf.joined_at)
        pdf['updated_at'] = pd.to_datetime(pdf.updated_at)
        pdf['membership_duration_days'] = (pdf.updated_at - pdf.joined_at).dt.days

        return pdf

    def last_event_occ_end_time(self):
        last_event = self.db.event_occs.find_one(
            {},  # No filter, checking all documents
            sort=[('end_at', -1)]  # Sorting documents by 'updated_at' in descending order
        )

        if not last_event:
            return EARLIEST_DATE.astimezone(pytz.utc)
        else:
            return datetime.fromisoformat(last_event['end_at']).astimezone(pytz.utc)

    @property
    def last_visit_time(self):

        last_visit = self.db.visits.find_one(
            {"completed_at": {"$ne": None}},  # Filter for documents where completed_at is not None
            sort=[("created_at", -1)]
        )

        if not last_visit:
            return EARLIEST_DATE.astimezone(pytz.utc)
        else:
            eocc = self.db.event_occs.find_one({'id': last_visit['event_occurrence_id']})
            return datetime.fromisoformat(eocc['end_at']).astimezone(pytz.utc)

    def get_event_occs_range(self, start: datetime = None, end: datetime = None) -> List:
        """Return up to one month of event occurances. """
        api_endpoint = '/api/v2/desk/event_occurrences'  # Endpoint for clients' information

        events = []

        if isinstance(start, str):
            start_dt = datetime.fromisoformat(start)
        elif isinstance(start, datetime):
            start_dt = start
        else:
            start_dt = one_month_ago()

        if isinstance(end, str):
            end_dt = datetime.fromisoformat(end)
        elif isinstance(end, datetime):
            end_dt = end
        else:
            end_dt = one_month_after(start_dt)

        logger.info('Getting event occurances from {} to {}'.format(start_dt, end_dt))

        for page in self.pike13_get_pages(api_endpoint,
                                          from_=start_dt.isoformat() + 'Z',
                                          to=end_dt.isoformat() + 'Z'):
            for p in page['event_occurrences']:
                events.append(p)

        return list(sorted(events, key=lambda p: p['start_at']))

    def get_event_occs(self, start: datetime = None):
        """Yield all events after a start date"""

        events = []

        if start is None:
            start = self.last_event_occ_end_time()

        range_start = start.replace(tzinfo=pytz.utc)
        # Get event occurrances that are scheduled up to a few months into the future.
        range_end = datetime.now(pytz.utc) + timedelta(days=90)

        for start_dt, end_dt in month_range(range_start, range_end):

            _events = self.get_event_occs_range(start_dt, end_dt)

            if not _events:
                continue

            yield _events

            start_dt = last_event_date(events)

            if start_dt > datetime.now().astimezone(start_dt.tzinfo):
                break

        return events

    def get_events_by_id(self, e_ids) -> List:
        """Return events for a list of event ids """
        api_endpoint = '/api/v2/desk/events'  # Endpoint for clients' information

        events = []

        logger.info(f'Getting events for {len(e_ids)} event_ids')

        eids_str = ','.join([str(e) for e in e_ids])

        for page in self.pike13_get_pages(api_endpoint, ids=eids_str):

            for p in page['events']:
                events.append(p)

        return events

    def get_new_events(self):
        """Get all events"""

        api_endpoint = '/api/v2/desk/events'

        eo_event_ids = set([e['event_id'] for e in self.event_occs])
        extant_event_ids = set([e['id'] for e in self.events])

        remain_ids = list(eo_event_ids - extant_event_ids)

        logger.debug(f"{len(eo_event_ids)} event occs, {len(extant_event_ids)} events, {len(remain_ids)} remain")

        for chunked_ids in chunked(remain_ids, 100):
            id_list = ','.join(str(e) for e in chunked_ids)
            for page in self.pike13_get_pages(api_endpoint, ids=id_list):
                for e in page['events']:
                    yield e

    def get_recent_events(self):
        """Get the recent, because the API doesn't allow
        filtering by updated time"""

        api_endpoint = '/api/v2/desk/events'

        start = (datetime.now() - relativedelta(months=1)) \
            .replace(hour=0, minute=0, second=0, microsecond=0)

        end = datetime.now().replace(hour=23, minute=59, second=59, microsecond=0)

        for page in self.pike13_get_pages(api_endpoint, from_=start.isoformat() + 'Z', to_=end.isoformat() + 'Z'):
            for e in page['events']:
                yield e

    def get_visits_for_occurrances(self, occs: List = None):
        """ Get the visits for recent but completed events.

        """
        from datetime import datetime

        lvt = self.last_visit_time

        if occs is None:
            # Find all of the events that have visits, and ended after the last visit time
            # and ended before now
            occs = list(self.db.event_occs.find({
                "end_at": {
                    "$gt": lvt.isoformat(),
                    "$lte": datetime.utcnow().isoformat()
                },
                "visits_count": {"$gt": 0}  # Additional condition for visits_count to be greater than 0
            }))
            logger.info(f'Loaded {len(occs)} recent event occurances to look for new visits')

        api_endpoint = '/api/v2/desk/event_occurrences/:event_occurrence_id/visits'

        visits = []

        for occ in occs:
            for page in self.pike13_get_pages(api_endpoint, event_occurrence_id=occ['id'],
                                              from_=lvt.isoformat(), to_=datetime.utcnow().isoformat()):

                for v in page['visits']:
                    if v['completed_at'] is not None:
                        # It is included in other requests, but not this one, because we specified the occ as a parameter
                        v['event_occurrence'] = occ
                        yield v

    def get_locations(self):
        """Get all locations"""

        api_endpoint = '/api/v2/desk/locations'

        locations = []

        for page in self.pike13_get_pages(api_endpoint):
            for l in page['locations']:
                locations.append(l)

        self.kv.set('locations_fetch_time', datetime.utcnow().isoformat())

        return locations

    def get_services(self):
        """Get all services"""

        api_endpoint = '/api/v2/desk/services'

        services = []

        for page in self.pike13_get_pages(api_endpoint):
            for s in page['services']:
                services.append(s)

        return services

    ##
    ## Record Acessors
    ##

    @property
    def people(self):
        """Get the people from the database"""

        return list(self.db.people.find())

    @property
    def students(self):
        """return the students, people records which have a provider
        listed"""

        return list(self.db.people.find({'providers': {'$not': {'$size': 0}}}))

    @property
    def active_students(self):
        """Return all active students"""

        return list(self.db.people.find(
            {
                'providers': {'$not': {'$size': 0}},
                'is_member': {'$eq': True}
            }
        ))

    @property
    def parents(self):
        """return the parents, people records which have dependents listed"""
        return list(self.db.people.find({'dependents': {'$not': {'$size': 0}}}))

    @property
    def active_parents(self):
        """Return all active students"""

        return list(self.db.people.find(
            {
                'dependents': {'$not': {'$size': 0}},
                'is_member': {'$eq': True}
            }
        ))

    @property
    def parent_mailing_list(self):
        """Return a dataframe of parents with email addresses, with the
        number of children who are members, and the last update time, suitable for
        creating a mailing list"""

        parents = expand_custom(self.parents)

        # Number of students with memberships. The students seem to be updated properly,
        # while the parents are not
        for p in parents:
            p['n_members'] = sum([int(e['is_member']) for e in p.get('dependents', [])])

        p_df = pd.DataFrame(parents)
        loc_df = pd.DataFrame(self.locations).rename(
            columns={'id': 'location_id', 'name': 'location_name', 'zip': 'location_zip'})
        t = p_df.merge(loc_df[['location_id', 'location_name']], on='location_id', how='left')
        t['updated_at'] = pd.to_datetime(t['updated_at'])
        cols = ['first_name', 'middle_name', 'last_name', 'email', 'address', 'n_members', 'location_id',
                'location_name', 'updated_at']
        return t[cols].copy()

    @property
    def others(self):
        """return the others, people records which are neither students nor parents"""
        query = {
            '$and': [
                {
                    '$or': [
                        {'providers': {'$exists': False}},
                        {'providers': None},
                        {'providers': {'$size': 0}}
                    ]
                },
                {
                    '$or': [
                        {'dependents': {'$exists': False}},
                        {'dependents': None},
                        {'dependents': {'$size': 0}}
                    ]
                }
            ]
        }
        return list(self.db.people.find(query))

    @property
    def event_occs(self):
        """Get all events occuraances from the database"""

        return list(self.db.event_occs.find())

    @property
    def events(self):
        """Get all events from the database"""

        return list(self.db.events.find())

    @property
    def visits(self):
        """Get all events occuraances from the database"""

        return list(self.db.visits.find())

    @property
    def locations(self):
        """Return all of the locations, or fetch them if they are older than 4 hours"""

        lft_str = self.kv.get('locations_fetch_time')
        if lft_str is None:
            lft_str = '2000-01-01T00:00:00Z'

        lft = datetime.fromisoformat(lft_str).astimezone(pytz.utc)
        now = datetime.utcnow().astimezone(pytz.utc)

        if (now - lft).total_seconds() > 60 * 60 * 4:
            logger.debug("Old locations, refetching")
            loc = self.get_locations()
            logger.debug(f"Found {len(loc)} locations")

            bulk_operations = [
                UpdateOne(
                    {'id': doc['id']},  # Match the document by 'id', not '_id'
                    {'$set': doc},  # Update the document with the data of 'doc'
                    upsert=True  # Upsert option allows insertion if the document doesn't exist
                )
                for doc in loc
            ]

            # Execute the bulk operations
            self.db.locations.bulk_write(bulk_operations)

            self.kv.set('locations_fetch_time', now.isoformat())
            return loc
        else:
            return list(self.db.locations.find())

    @property
    def services(self):
        return self.get_services()

    ##
    ## Update methods
    ##

    def update_people(self):
        """Update the mongo database with the recently updated people"""

        logger.debug(f"Updating people, since {self.last_people_update_time}")

        people = self.get_updated_people()

        logger.debug(f"Found {len(people)} updated people")

        if people:

            if len(self.people) == 0:
                # For cold start, there are no people in the databse
                logger.info("People collection is empty, inserting")
                self.db.people.insert_many(people)
            else:
                # There are people in the database, so we have to upsert,
                # replacing the existing records with the new ones
                logger.info("People collection is not empty, upserting")
                for p in people:

                    if '_id' in p:  # Why is there an _id in the pike13 data?
                        del p['_id']

                    result = self.db.people.replace_one({'id': p['id']}, p, upsert=True)

        return people

    def write_many_or_upsert(self, collection, docs):

        def _f(v):
            """Remove the _id and event_occurance fields from the visit, if they exist"""
            v = v.copy()
            v.pop('_id', None)
            v.pop('event_occurrence', None)
            return v

        try:
            docs = [_f(doc) for doc in docs]  # Yes, must be inside and duplicated. maybe insert_many modifies?
            result = collection.insert_many(docs, ordered=False)
            logger.debug(f"Inserted {len(result.inserted_ids)} new docs")
        except Exception as e:
            try:
                docs = [_f(doc) for doc in docs]  # Yes, must be inside and duplicated. maybe insert_man modifies?
                result = collection.bulk_write(
                    [UpdateOne({'id': doc['id']}, {'$set': doc}, upsert=True) for doc in docs]
                )
                logger.debug(f"Upserted {result.upserted_count}  docs")
            except:
                print(docs)
                raise

    def update_visits(self):

        logger.debug(f"Updating visits, since {self.last_visit_time}")
        new_visits = []

        for visit_chunk in chunked(self.get_visits_for_occurrances(), 500):
            self.write_many_or_upsert(self.db.visits, visit_chunk)

        return new_visits

    def update_event_occs(self):

        logger.debug(f"Updating event occurrances since {self.last_event_occ_end_time()}")

        events_occs = []
        for ranged_events in self.get_event_occs():

            logger.info(f"Found {len(ranged_events)} new event occurrancess")
            events_occs.extend(ranged_events)
            if ranged_events:
                self.db.event_occs.bulk_write(
                    [UpdateOne({'id': doc['id']}, {'$set': doc}, upsert=True) for doc in ranged_events]
                )

        return events_occs

    def update_events(self):

        for events_chunk in chunked(self.get_new_events(), 250):
            self.write_many_or_upsert(self.db.events, events_chunk)

        for events_chunk in chunked(self.get_recent_events(), 250):
            self.write_many_or_upsert(self.db.events, events_chunk)

    def update(self):
        """Update the database with new data from Pike13"""
        logger.info("Updating Pike13 data")
        self.update_people()
        self.update_event_occs()
        self.update_visits()
        self.update_events()


class Pike13DataFrames:

    def __init__(self, pike13: Pike13):
        self.p13 = pike13

    @cached_property
    def people(self):
        return pd.DataFrame(self.p13.people).rename(columns={'id': 'person_id'})

    @cached_property
    def parents(self):
        return pd.DataFrame(self.p13.parents).rename(columns={'id': 'person_id'})

    @cached_property
    def students(self):
        return pd.DataFrame(self.p13.students).rename(columns={'id': 'person_id'})

    @cached_property
    def active_students(self):
        return pd.DataFrame(self.p13.active_students)

    @cached_property
    def services(self):
        return pd.DataFrame(self.p13.get_services()) \
            [['id', 'name', 'type', 'duration_in_minutes', 'maximum_clients', 'category_id', 'category_name',
              'description','description_short','instructions']] \
            .rename(columns={'id': 'service_id', 'name': 'service_name'})

    @cached_property
    def locations(self):
        t =  pd.DataFrame(self.p13.locations)[['id', 'name', 'latitude', 'longitude']].rename(
            columns={'id': 'location_id', 'name': 'location_name'})

        # Extract the code from the location_name, the code in the parens,
        # so, "(CV)" -> "CV"
        t['location_code'] = t.location_name.str.extract(r'\((\w+)\)')
        return t

    @cached_property
    def event_occurrences(self):
        evento = pd.DataFrame(self.p13.event_occs)[
            ['id', 'event_id', 'name', 'service_id', 'location_id', 'state', 'visits_count',
             'start_at', 'end_at']] \
            .rename(columns={'id': 'event_occurrence_id', 'name': 'event_name'})

        for c in ['start_at', 'end_at']:
            evento[c] = pd.to_datetime(evento[c])

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Converting to PeriodArray/Index")
            evento['occ_month'] = evento.start_at.dt.to_period('M')
            evento['occ_week'] = evento.start_at.dt.to_period('W')
        evento['dow'] = evento.start_at.dt.dayofweek
        return evento

    @cached_property
    def visits(self):
        visits = pd.DataFrame(self.p13.visits)[
            ['id', 'state', 'status', 'person_id', 'event_occurrence_id',
             'cancelled_at', 'noshow_at', 'registered_at',
             'completed_at']].rename(columns={'id': 'visit_id'})

        for c in ['cancelled_at', 'noshow_at', 'registered_at', 'completed_at']:
            visits[c] = pd.to_datetime(visits[c], errors='ignore')

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Converting to PeriodArray/Index")
            visits['visit_month'] = visits.completed_at.dt.to_period('M')
            visits['visit_week'] = visits.completed_at.dt.to_period('W')
        visits['dow'] = visits.completed_at.dt.dayofweek
        return visits

    @cached_property
    def events(self):
        events = pd.DataFrame(self.p13.events).drop(columns=['icals', '_id']).rename(
            columns={'id': 'event_id', 'name': 'event_name'})

        for c in ['start_time', 'end_time', 'created_at', 'updated_at']:
            try:
                events[c] = pd.to_datetime(events[c])
            except:
                raise

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Converting to PeriodArray/Index")
            events['event_start_month'] = events.start_time.dt.to_period('M')

        return events
