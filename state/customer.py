# state/customer.py
from dataclasses import dataclass, field


@dataclass
class Customer:
    """
    "data": {
        "id": result[0],
        "name": result[1],
        "email": result[2],
        "plan": result[3],
    },
    """

    id: int
    name: str
    email: str
    plan: str

    def __post_init__(self):
        if self.id <= 0:
            raise ValueError("customer id must be positive")
