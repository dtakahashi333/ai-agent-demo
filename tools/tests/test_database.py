# tools/tests/test_database.py
from unittest import TestCase
from unittest.mock import MagicMock, patch

import psycopg

from tools.database import (
    count_customers,
    create_order,
    find_order_by_idempotency_key,
    get_customer,
    get_customer_orders,
    get_order,
    get_order_status,
    insert_order,
    search_customers,
)


class DatabaseTestCase(TestCase):

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def mock_database(self):
        connection = MagicMock()
        cursor = MagicMock()

        connection.cursor.return_value.__enter__.return_value = cursor

        return connection, cursor

    # ------------------------------------------------------------------
    # get_customer
    # ------------------------------------------------------------------

    @patch("tools.database.psycopg.connect")
    def test_get_customer_returns_customer(self, mock_connect):
        connection, cursor = self.mock_database()

        mock_connect.return_value.__enter__.return_value = connection

        cursor.fetchone.return_value = (
            1,
            "Alice",
            "alice@example.com",
            "premium",
        )

        result = get_customer(1)

        self.assertEqual(
            result,
            {
                "success": True,
                "data": {
                    "id": 1,
                    "name": "Alice",
                    "email": "alice@example.com",
                    "plan": "premium",
                },
                "error": None,
            },
        )

        cursor.execute.assert_called_once()

    @patch("tools.database.psycopg.connect")
    def test_get_customer_returns_not_found(self, mock_connect):
        connection, cursor = self.mock_database()

        mock_connect.return_value.__enter__.return_value = connection
        cursor.fetchone.return_value = None

        result = get_customer(999)

        self.assertEqual(
            result,
            {
                "success": False,
                "data": None,
                "error": {
                    "type": "not_found",
                    "message": "Customer was not found",
                },
            },
        )

    @patch("tools.database.psycopg.connect")
    def test_get_customer_returns_database_error(self, mock_connect):
        mock_connect.side_effect = psycopg.Error("database unavailable")

        result = get_customer(1)

        self.assertEqual(
            result,
            {
                "success": False,
                "data": None,
                "error": {
                    "type": "database_error",
                    "message": "Unable to retrieve customer",
                },
            },
        )

    # ------------------------------------------------------------------
    # get_order
    # ------------------------------------------------------------------

    @patch("tools.database.psycopg.connect")
    def test_get_order_returns_order(self, mock_connect):
        connection, cursor = self.mock_database()

        mock_connect.return_value.__enter__.return_value = connection

        cursor.fetchone.return_value = (
            10,
            1,
            "created",
            99.50,
        )

        result = get_order(10)

        self.assertEqual(
            result,
            {
                "success": True,
                "data": {
                    "id": 10,
                    "customer_id": 1,
                    "status": "created",
                    "total": 99.50,
                },
                "error": None,
            },
        )

    @patch("tools.database.psycopg.connect")
    def test_get_order_returns_not_found(self, mock_connect):
        connection, cursor = self.mock_database()

        mock_connect.return_value.__enter__.return_value = connection
        cursor.fetchone.return_value = None

        result = get_order(999)

        self.assertEqual(result["success"], False)
        self.assertEqual(result["data"], None)
        self.assertEqual(result["error"]["type"], "not_found")

    # ------------------------------------------------------------------
    # get_order_status
    # ------------------------------------------------------------------

    @patch("tools.database.psycopg.connect")
    def test_get_order_status_returns_status(self, mock_connect):
        connection, cursor = self.mock_database()

        mock_connect.return_value.__enter__.return_value = connection

        cursor.fetchone.return_value = ("shipped",)

        result = get_order_status(10)

        self.assertEqual(
            result,
            {
                "success": True,
                "data": {
                    "order_id": 10,
                    "status": "shipped",
                },
                "error": None,
            },
        )

    @patch("tools.database.psycopg.connect")
    def test_get_order_status_returns_not_found(self, mock_connect):
        connection, cursor = self.mock_database()

        mock_connect.return_value.__enter__.return_value = connection
        cursor.fetchone.return_value = None

        result = get_order_status(999)

        self.assertEqual(result["success"], False)
        self.assertEqual(result["error"]["type"], "not_found")

    # ------------------------------------------------------------------
    # search_customers
    # ------------------------------------------------------------------

    @patch("tools.database.psycopg.connect")
    @patch("tools.database.config", {"page_size": 2})
    def test_search_customers_returns_page(self, mock_connect):
        connection, cursor = self.mock_database()

        mock_connect.return_value.__enter__.return_value = connection

        cursor.fetchall.return_value = [
            (1, "Alice", "alice@example.com", "premium"),
            (2, "Alicia", "alicia@example.com", "basic"),
        ]

        result = search_customers("Ali")

        self.assertEqual(
            result,
            {
                "success": True,
                "data": {
                    "customers": [
                        {
                            "id": 1,
                            "name": "Alice",
                            "email": "alice@example.com",
                            "plan": "premium",
                        },
                        {
                            "id": 2,
                            "name": "Alicia",
                            "email": "alicia@example.com",
                            "plan": "basic",
                        },
                    ],
                    "has_more": False,
                    "next_cursor": None,
                },
                "error": None,
            },
        )

    @patch("tools.database.psycopg.connect")
    @patch("tools.database.config", {"page_size": 2})
    def test_search_customers_returns_next_cursor(self, mock_connect):
        connection, cursor = self.mock_database()

        mock_connect.return_value.__enter__.return_value = connection

        cursor.fetchall.return_value = [
            (1, "Alice", "alice@example.com", "premium"),
            (2, "Alicia", "alicia@example.com", "basic"),
            (3, "Alina", "alina@example.com", "basic"),
        ]

        result = search_customers("Ali")

        self.assertTrue(result["success"])
        self.assertTrue(result["data"]["has_more"])
        self.assertEqual(result["data"]["next_cursor"], 2)
        self.assertEqual(len(result["data"]["customers"]), 2)

    @patch("tools.database.psycopg.connect")
    @patch("tools.database.config", {"page_size": 2})
    def test_search_customers_returns_not_found(self, mock_connect):
        connection, cursor = self.mock_database()

        mock_connect.return_value.__enter__.return_value = connection

        cursor.fetchall.return_value = []

        result = search_customers("Nobody")

        self.assertEqual(result["success"], False)
        self.assertEqual(result["error"]["type"], "not_found")

    # ------------------------------------------------------------------
    # get_customer_orders
    # ------------------------------------------------------------------

    @patch("tools.database.psycopg.connect")
    def test_get_customer_orders_returns_orders(self, mock_connect):
        connection, cursor = self.mock_database()

        mock_connect.return_value.__enter__.return_value = connection

        cursor.fetchall.return_value = [
            (10, 1, "created", 100.00),
            (11, 1, "shipped", 50.00),
        ]

        result = get_customer_orders(1)

        self.assertEqual(
            result,
            {
                "success": True,
                "data": [
                    {
                        "id": 10,
                        "customer_id": 1,
                        "status": "created",
                        "total": 100.00,
                    },
                    {
                        "id": 11,
                        "customer_id": 1,
                        "status": "shipped",
                        "total": 50.00,
                    },
                ],
                "error": None,
            },
        )

    @patch("tools.database.psycopg.connect")
    def test_get_customer_orders_returns_not_found(self, mock_connect):
        connection, cursor = self.mock_database()

        mock_connect.return_value.__enter__.return_value = connection
        cursor.fetchall.return_value = []

        result = get_customer_orders(999)

        self.assertEqual(result["success"], False)
        self.assertEqual(result["error"]["type"], "not_found")

    # ------------------------------------------------------------------
    # count_customers
    # ------------------------------------------------------------------

    @patch("tools.database.psycopg.connect")
    def test_count_customers_returns_count(self, mock_connect):
        connection, cursor = self.mock_database()

        mock_connect.return_value.__enter__.return_value = connection

        cursor.fetchone.return_value = (3,)

        result = count_customers("Alice")

        self.assertEqual(
            result,
            {
                "success": True,
                "data": {
                    "count": 3,
                },
                "error": None,
            },
        )

    @patch("tools.database.psycopg.connect")
    def test_count_customers_returns_database_error(self, mock_connect):
        mock_connect.side_effect = psycopg.Error("database unavailable")

        result = count_customers("Alice")

        self.assertEqual(
            result,
            {
                "success": False,
                "data": None,
                "error": {
                    "type": "database_error",
                    "message": "Unable to count customers",
                },
            },
        )

    # ------------------------------------------------------------------
    # find_order_by_idempotency_key
    # ------------------------------------------------------------------

    @patch("tools.database.psycopg.connect")
    def test_find_order_by_idempotency_key_returns_order(self, mock_connect):
        connection, cursor = self.mock_database()

        mock_connect.return_value.__enter__.return_value = connection

        cursor.fetchone.return_value = (
            100,
            1,
            "created",
            99.99,
            "abc-123",
        )

        result = find_order_by_idempotency_key("abc-123")

        self.assertEqual(
            result,
            {
                "id": 100,
                "customer_id": 1,
                "status": "created",
                "total": 99.99,
                "idempotency_key": "abc-123",
            },
        )

    @patch("tools.database.psycopg.connect")
    def test_find_order_by_idempotency_key_returns_none(self, mock_connect):
        connection, cursor = self.mock_database()

        mock_connect.return_value.__enter__.return_value = connection
        cursor.fetchone.return_value = None

        result = find_order_by_idempotency_key("missing-key")

        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # create_order
    # ------------------------------------------------------------------

    @patch("tools.database.insert_order")
    @patch("tools.database.find_order_by_idempotency_key")
    def test_create_order_returns_existing_order_without_insert(
        self,
        mock_find_order,
        mock_insert_order,
    ):
        existing_order = {
            "id": 100,
            "customer_id": 1,
            "status": "created",
            "total": 99.99,
            "idempotency_key": "abc-123",
        }

        mock_find_order.return_value = existing_order

        result = create_order(
            customer_id=1,
            status="created",
            total=99.99,
            idempotency_key="abc-123",
        )

        self.assertEqual(
            result,
            {
                "success": True,
                "data": existing_order,
                "error": None,
            },
        )

        mock_insert_order.assert_not_called()

    @patch("tools.database.insert_order")
    @patch("tools.database.find_order_by_idempotency_key")
    def test_create_order_inserts_new_order(
        self,
        mock_find_order,
        mock_insert_order,
    ):
        mock_find_order.return_value = None

        new_order = {
            "id": 101,
            "customer_id": 1,
            "status": "created",
            "total": 99.99,
            "idempotency_key": "abc-123",
        }

        mock_insert_order.return_value = new_order

        result = create_order(
            customer_id=1,
            status="created",
            total=99.99,
            idempotency_key="abc-123",
        )

        self.assertEqual(
            result,
            {
                "success": True,
                "data": new_order,
                "error": None,
            },
        )

        mock_insert_order.assert_called_once_with(
            customer_id=1,
            status="created",
            total=99.99,
            idempotency_key="abc-123",
        )

    # ------------------------------------------------------------------
    # insert_order
    # ------------------------------------------------------------------

    @patch("tools.database.psycopg.connect")
    def test_insert_order_returns_inserted_order(self, mock_connect):
        connection, cursor = self.mock_database()

        mock_connect.return_value.__enter__.return_value = connection

        cursor.fetchone.return_value = (
            101,
            1,
            "created",
            99.99,
            "abc-123",
        )

        result = insert_order(
            customer_id=1,
            status="created",
            total=99.99,
            idempotency_key="abc-123",
        )

        self.assertEqual(
            result,
            {
                "id": 101,
                "customer_id": 1,
                "status": "created",
                "total": 99.99,
                "idempotency_key": "abc-123",
            },
        )

        cursor.execute.assert_called_once()
