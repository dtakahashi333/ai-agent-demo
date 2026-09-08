INSERT INTO agent.customers (name, email, plan, idempotency_key)
VALUES
	('Alice','alice@example.com','pro','3f8a2c71-6d45-4b93-ae18-527c9f04d836'),
	('Bob','bob@example.com','basic','b7e1d509-3a62-48f4-9c27-6158e0ab4d93'),
	('Carol','carol@example.com','enterprise','d5b3e7f9-4a16-48c2-91d8-6275e0a3f849'),
	('Alice','alice01@example.com','basic','1a6d9c47-8e23-45b1-b7f4-5362c9a0d815'),
	('Alice','alice02@example.com','pro','c8f1e5a9-3d72-4b60-92c4-6178e3f5a029'),
	('Alice','alice03@example.com','enterprise','76d2b8e4-9c15-4a63-a7f1-5283e6d0b947'),
	('Alice','alice04@example.com','basic','e3c7a9f5-2b41-46d8-91e6-5730c4a8f219'),
	('Alice','alice05@example.com','pro','5a1f8c63-d947-4e20-b5a7-3892c6d1f804'),
	('Alice','alice06@example.com','pro','b9e4c7d2-1f68-43a5-8c91-7246d3e0f815'),
	('John','john01@example.com','pro','82c5e9f1-4a73-48b6-b2d8-6153e7a9c420'),
	('Jane','jane01@example.com','enterprise','3e9a6c21-7f45-4b83-a5d2-184c7e9f6306'),
	('Alice','alice07@example.com','basic','f4b8d2a7-6c19-45e3-9f80-2a5d7c1e8463'),
	('Alice','alice08@example.com','basic','a7d1e5c9-3b68-42f0-8c14-5962e7a4b831'),
	('Tom','tom01@example.com','pro','68f3c9a1-5b27-4d84-91e6-7a2c8f4b3509'),
	('Alice Smith','alice.smith@example.com','enterprise','d2a6f8b4-1c73-4e95-a827-6391b5d0f482');

INSERT INTO agent.orders (customer_id, status, total, idempotency_key)
VALUES
	(1,'shipped',149.99,'550e8400-e29b-41d4-a716-446655440000'),
	(1,'pending',49.99,'7f3c2a91-8b54-4e67-9d12-5a8b3c1e7426'),
	(2,'delivered',299.0,'c1d4e8a2-6f39-47b1-83d5-9a27c4e6f810'),
	(15,'shipped',199.98,'2b7a9c15-4d83-4f26-a1e9-6385b2c7d904'),
	(15,'pending',24.99,'e8f1426b-3c97-4a50-b8d1-7259e4c3a681'),
	(15,'pending',36.99,'91b6d3f8-2e45-4c79-9a13-57e8b4d620ca'),
	(15,'pending',118.99,'4c8e1a73-9f25-46d8-b3c1-8257e4a9d610');

