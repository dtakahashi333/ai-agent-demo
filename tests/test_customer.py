# tests/test_customer.py

from unittest import TestCase

from state.customer import Customer


class TestCustomer(TestCase):
    def test_rejects_non_positive_id(self):
        with self.assertRaises(ValueError):
            Customer(
                id=0,
                name="Alice Smith",
                email="alice@example.com",
                plan="premium",
            )
