DROP TABLE essays;
DROP TABLE projects;


\qecho
\qecho LOAD PROJECTS TABLE ...

CREATE TABLE projects
(
  -- IDs
  _projectid text NOT NULL,
  _teacher_acctid text NOT NULL,
  _schoolid text NOT NULL,
  school_ncesid text,

  -- School Location
  school_latitude numeric(11,6),
  school_longitude numeric(11,6),
  school_city text,
  school_state character(2),
  school_zip text,
  school_metro text,
  school_district text,
  school_county text,

  -- School Types
  school_charter boolean,
  school_magnet boolean,
  school_year_round boolean,
  school_nlns boolean,
  school_kipp boolean,
  school_charter_ready_promise boolean,

  -- Teacher Attributes
  teacher_prefix text,
  teacher_teach_for_america boolean,
  teacher_ny_teaching_fellow boolean,

  -- Project Categories
  primary_focus_subject text,
  primary_focus_area text,
  secondary_focus_subject text,
  secondary_focus_area text,
  resource_type text,
  poverty_level text,
  grade_level text,

  -- Project Pricing and Impact
  vendor_shipping_charges text,
  sales_tax text,
  payment_processing_charges text,
  fulfillment_labor_materials text,
  total_price_excluding_optional_support text,
  total_price_including_optional_support text,
  students_reached text,

  -- Project Donations
  total_donations numeric(10,2),
  num_donors integer,
  eligible_double_your_impact_match boolean,
  eligible_almost_home_match boolean,

  -- Project Status
  funding_status text,
  date_posted date,
  date_completed text,
  date_thank_you_packet_mailed text,
  date_expiration text
)
WITHOUT OIDS;


\COPY projects FROM PSTDIN WITH CSV HEADER
\qecho ... DONE


-------------------------------

\qecho
\qecho LOAD ESSAYS TABLE ...

CREATE TABLE essays
(
  _projectid text NOT NULL,
  _teacher_acctid text NOT NULL,

  title text,
  short_description text,
  need_statement text,
  essay text,
  thankyou_note text, 
  impact_letter text
)
WITHOUT OIDS;


\COPY essays FROM PSTDIN WITH CSV HEADER
\qecho ... DONE

-------------------------------


\qecho
\qecho ALTER PROJECTS TABLE ...
ALTER TABLE projects
      ADD CONSTRAINT pk_projects PRIMARY KEY(_projectid);

CREATE INDEX projects_teacher_acctid
  ON projects
  USING btree
  (_teacher_acctid);

CREATE INDEX projects_schoolid
  ON projects
  USING btree
  (_schoolid);


VACUUM ANALYZE projects;
\qecho ... DONE



\qecho
\qecho ALTER ESSAYS TABLE ...
ALTER TABLE essays
      ADD CONSTRAINT pk_essays PRIMARY KEY(_projectid);

CREATE INDEX essays_teacher_acctid
  ON essays
  USING btree
  (_teacher_acctid);


VACUUM ANALYZE essays;
\qecho ... DONE
