# tests/test_validate_arguments.py

from unittest import TestCase
from agent import validate_arguments


class TestValidateArguments(TestCase):
    def setUp(self):
        return super().setUp()

    def test_validate_arguments1(self):
        result = validate_arguments("get_customer", {"customer_id": 1})
        self.assertTrue(result["success"])

    def test_validate_arguments2(self):
        result = validate_arguments("get_customer", {"customer_id": "abc"})
        self.assertFalse(result["success"])

    def test_validate_arguments3(self):
        result = validate_arguments("get_customer", {})
        self.assertFalse(result["success"])

    def test_validate_arguments4(self):
        result = validate_arguments("get_customer", {"customer_id": 1, "foo": "bar"})
        self.assertFalse(result["success"])

    def test_validate_arguments5(self):
        result = validate_arguments("get_weather", {"city": "Dallas"})
        self.assertTrue(result["success"])
