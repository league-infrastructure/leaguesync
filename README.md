
# League Sync

Data Synchronization between various League services. 

* Reads from Pike13 and updates records in a Mongo database
* Reads from the Mongo database to update Google Calendars


## Docker

The docker configuration in the `docker` directory will: 

* Sync with the mondg data base every 30 minutes
* Write out a set of data files
* Serve the files over nginx

The files are:

* event_occ.csv
* events.csv
* locations.csv
* parents.csv
* people.csv
* services.csv
* students.csv
* students_expanded.csv
* visits.parquet


The files are available from the root of the web server. 

To run the container, you will have to copy `leaguesync.env.template` to `leaguesync.env` 
and set the secrets and urls