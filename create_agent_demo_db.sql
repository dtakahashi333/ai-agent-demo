-- SELECT
-- 	grantee,
-- 	table_schema,
-- 	table_name,
-- 	privilege_type
-- FROM information_schema.table_privileges
-- WHERE table_schema NOT IN ('pg_catalog', 'information_schema')

-- GRANT SELECT, INSERT, UPDATE, DELETE
-- ON ALL TABLES IN SCHEMA agent
-- TO agent_demo_user;

-- GRANT USAGE, SELECT
-- ON ALL SEQUENCES IN SCHEMA agent
-- TO agent_demo_user;

-- ALTER DEFAULT PRIVILEGES
-- FOR ROLE agent_demo_owner
-- IN SCHEMA agent
-- GRANT SELECT, INSERT, UPDATE, DELETE
-- ON TABLES
-- TO agent_demo_user;

-- ALTER DEFAULT PRIVILEGES
-- FOR ROLE agent_demo_owner
-- IN SCHEMA agent
-- GRANT USAGE, SELECT
-- ON SEQUENCES
-- TO agent_demo_user;

REVOKE CONNECT ON DATABASE agent_demo FROM PUBLIC;

GRANT CONNECT ON DATABASE agent_demo
TO agent_demo_user;

