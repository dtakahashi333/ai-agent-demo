# tools/database.py
import psycopg

from config.settings import config

# there are more results than the allowed maximum
too_many_results_error = {
    "success": False,
    "data": None,
    "error": {
        "type": "too_many_results",
        "message": "The search matched too many customers. Please provide more specific information.",
    },
}


def get_customer(customer_id: int) -> dict:
    try:
        with psycopg.connect(
            "postgresql://agent_demo_user:agent_demo_password@localhost:5432/agent_demo"
        ) as connection:

            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, name, email, plan
                    FROM agent.customers
                    WHERE id = %s
                    """,
                    (customer_id,),
                )

                result = cursor.fetchone()

                if result is None:
                    return {
                        "success": False,
                        "data": None,
                        "error": {
                            "type": "not_found",
                            "message": "Customer was not found",
                        },
                    }

                return {
                    "success": True,
                    "data": {
                        "id": result[0],
                        "name": result[1],
                        "email": result[2],
                        "plan": result[3],
                    },
                    "error": None,
                }
    except psycopg.Error:
        return {
            "success": False,
            "data": None,
            "error": {
                "type": "database_error",
                "message": "Unable to retrieve customer",
            },
        }


def get_order(order_id: int) -> dict:
    """
    Return the specified order.
    """
    try:
        with psycopg.connect(
            "postgresql://agent_demo_user:agent_demo_password@localhost:5432/agent_demo"
        ) as connection:

            with connection.cursor() as cursor:

                cursor.execute(
                    """
                    SELECT id, customer_id, status, total
                    FROM agent.orders
                    WHERE id = %s
                    """,
                    (order_id,),
                )

                result = cursor.fetchone()

                if result is None:
                    return {
                        "success": False,
                        "data": None,
                        "error": {
                            "type": "not_found",
                            "message": "Order was not found",
                        },
                    }

                return {
                    "success": True,
                    "data": {
                        "id": result[0],
                        "customer_id": result[1],
                        "status": result[2],
                        "total": result[3],
                    },
                    "error": None,
                }
    except psycopg.Error:
        return {
            "success": False,
            "data": None,
            "error": {
                "type": "database_error",
                "message": "Unable to retrieve order",
            },
        }


def get_order_status(order_id: int) -> dict:
    """
    Look up an order and return its current status.
    """
    try:
        with psycopg.connect(
            "postgresql://agent_demo_user:agent_demo_password@localhost:5432/agent_demo"
        ) as connection:

            with connection.cursor() as cursor:

                cursor.execute(
                    """
                    SELECT status
                    FROM agent.orders
                    WHERE id = %s
                    """,
                    (order_id,),
                )

                result = cursor.fetchone()

                if result is None:
                    return {
                        "success": False,
                        "data": None,
                        "error": {
                            "type": "not_found",
                            "message": "Order was not found",
                        },
                    }

                return {
                    "success": True,
                    "data": {
                        "order_id": order_id,
                        "status": result[0],
                    },
                    "error": None,
                }
    except psycopg.Error:
        return {
            "success": False,
            "data": None,
            "error": {
                "type": "database_error",
                "message": "Unable to retrieve order",
            },
        }


"""
| Tool result                | Meaning                    | Typical agent behavior                        |
|----------------------------|----------------------------|-----------------------------------------------|
| not_found                  | Nothing matched            | Inform user / possibly try a different search |
| success + 1 result         | Unambiguous                | Continue                                      |
| success + multiple results | Potentially ambiguous      | Clarify                                       |
| too_many_results           | Search exceeded tool limit | Ask for more specific criteria                |
| database_error             | Tool infrastructure failed | Apply error/retry policy                      |
"""


def search_customers(name: str, pagination_cursor: int | None = None) -> dict:
    try:
        with psycopg.connect(
            "postgresql://agent_demo_user:agent_demo_password@localhost:5432/agent_demo"
        ) as connection:

            with connection.cursor() as db_cursor:

                where_clause = "WHERE name ILIKE %s"
                params = [f"%{name}%"]

                if pagination_cursor is not None:
                    where_clause += " AND id > %s"
                    params.append(pagination_cursor)

                params.append(config["page_size"] + 1)

                db_cursor.execute(
                    """
                    SELECT id, name, email, plan
                    FROM agent.customers
                    """
                    + where_clause
                    + """
                    ORDER BY id
                    LIMIT %s
                    """,
                    params,
                )

                result = db_cursor.fetchall()

                if not result:
                    return {
                        "success": False,
                        "data": None,
                        "error": {
                            "type": "not_found",
                            "message": "Customer was not found",
                        },
                    }

                has_more = len(result) > config["page_size"]
                page = result[:-1] if has_more else result

                return {
                    "success": True,
                    "data": {
                        "customers": [
                            {
                                "id": customer[0],
                                "name": customer[1],
                                "email": customer[2],
                                "plan": customer[3],
                            }
                            for customer in page
                        ],
                        "has_more": has_more,
                        "next_cursor": page[-1][0] if has_more else None,
                    },
                    "error": None,
                }
    except psycopg.Error:
        return {
            "success": False,
            "data": None,
            "error": {
                "type": "database_error",
                "message": "Unable to retrieve customer",
            },
        }


def get_customer_orders(customer_id: int) -> dict:
    """
    Return all orders belonging to a customer.
    """
    try:
        with psycopg.connect(
            "postgresql://agent_demo_user:agent_demo_password@localhost:5432/agent_demo"
        ) as connection:

            with connection.cursor() as cursor:

                cursor.execute(
                    """
                    SELECT id, customer_id, status, total
                    FROM agent.orders
                    WHERE customer_id = %s
                    """,
                    (customer_id,),
                )

                result = cursor.fetchall()

                if not result:
                    return {
                        "success": False,
                        "data": None,
                        "error": {
                            "type": "not_found",
                            "message": "No orders were found for customer",
                        },
                    }

                return {
                    "success": True,
                    "data": [
                        {
                            "id": order[0],
                            "customer_id": order[1],
                            "status": order[2],
                            "total": order[3],
                        }
                        for order in result
                    ],
                    "error": None,
                }
    except psycopg.Error:
        return {
            "success": False,
            "data": None,
            "error": {
                "type": "database_error",
                "message": "Unable to retrieve order",
            },
        }


def count_customers(name: str) -> dict:
    try:
        with psycopg.connect(
            "postgresql://agent_demo_user:agent_demo_password@localhost:5432/agent_demo"
        ) as connection:

            with connection.cursor() as db_cursor:

                db_cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM agent.customers
                    WHERE name = %s 
                    """,
                    (name,),
                )

                result = db_cursor.fetchone()

                return {
                    "success": True,
                    "data": {
                        "count": result[0],
                    },
                    "error": None,
                }

    except psycopg.Error:
        return {
            "success": False,
            "data": None,
            "error": {
                "type": "database_error",
                "message": "Unable to count customers",
            },
        }


def create_order(
    customer_id: int,
    status: str,
    total: float,
    idempotency_key: str,
) -> dict:
    """
    Create an order for a customer.

    The idempotency key guarantees that retrying the same request
    does not create a duplicate order.
    """

    try:
        # For now, this represents the database/service boundary.
        existing_order = find_order_by_idempotency_key(idempotency_key)

        if existing_order:
            return {
                "success": True,
                "data": existing_order,
                "error": None,
            }

        order = insert_order(
            customer_id=customer_id,
            status=status,
            total=total,
            idempotency_key=idempotency_key,
        )

        return {
            "success": True,
            "data": order,
            "error": None,
        }

    except psycopg.Error:
        return {
            "success": False,
            "data": None,
            "error": {
                "type": "database_error",
                "message": "Unable to create order",
            },
        }


def insert_order(
    customer_id: int,
    status: str,
    total: float,
    idempotency_key: str,
) -> dict:
    with psycopg.connect(
        "postgresql://agent_demo_user:agent_demo_password@localhost:5432/agent_demo"
    ) as connection:

        with connection.cursor() as db_cursor:

            db_cursor.execute(
                """
                INSERT INTO agent.orders (customer_id, status, total, idempotency_key)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    customer_id,
                    status,
                    total,
                    idempotency_key,
                ),
            )

            result = db_cursor.fetchone()

            return {
                "id": result[0],
                "customer_id": result[1],
                "status": result[2],
                "total": result[3],
                "idempotency_key": result[4],
            }


def find_order_by_idempotency_key(idempotency_key: str) -> dict | None:
    try:
        with psycopg.connect(
            "postgresql://agent_demo_user:agent_demo_password@localhost:5432/agent_demo"
        ) as connection:

            with connection.cursor() as cursor:

                cursor.execute(
                    """
                    SELECT id, customer_id, status, total, idempotency_key
                    FROM agent.orders
                    WHERE idempotency_key = %s
                    """,
                    (idempotency_key,),
                )

                result = cursor.fetchone()

                if result is None:
                    return None

                return {
                    "id": result[0],
                    "customer_id": result[1],
                    "status": result[2],
                    "total": result[3],
                    "idempotency_key": result[4],
                }

    except psycopg.Error:
        return None
