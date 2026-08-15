# tools/database.py
from typing import Optional

import psycopg


def get_order_status(order_id: str):
    """
    Look up an order and return its current status.
    """

    # database query here

    return {"order_id": order_id, "status": "Shipped"}


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
                    WHERE id=%s
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
                else:
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
