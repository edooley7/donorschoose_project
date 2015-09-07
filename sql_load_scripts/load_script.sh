#!/bin/sh
# Uses the -U $DBUSER -h $DBHOST -d $DBNAME environment variables.

(
  # \COPY opendata_projects FROM PSTDIN WITH CSV HEADER
  cat opendata_projects.csv
  echo '\.'

  # \COPY new_essays FROM PSTDIN WITH CSV HEADER
  cat new_essays.csv
  echo '\.'
) \
| psql -U erindooley -h 127.0.0.1 -d erindooley -f load_postgres.sql
