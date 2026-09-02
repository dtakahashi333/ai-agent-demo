#!/usr/bin/env bash

set -e

pytest tests/
pytest executor/tests/
pytest llm/tests/
pytest planner/tests/
pytest state/tests/
# pytest tools/tests/