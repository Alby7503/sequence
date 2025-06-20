#!/bin/bash
echo "Running basic_test.sh"
tests/basic_test.sh
echo "Running race_condition_1.sh"
tests/race_condition_1.sh
echo "Running race_condition_2.sh"
tests/race_condition_2.sh
echo "Running shorter_than_int.sh"
tests/shorter_than_int.sh