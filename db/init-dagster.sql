-- The evaluator's Dagster instance keeps its run/event/schedule storage in a
-- separate database in the same Postgres container. Created once on first init.
CREATE DATABASE dagster;
