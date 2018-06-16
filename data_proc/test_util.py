"""
test for util.py
"""

import unittest
import util

class TestLimitVarRange(unittest.TestCase):
    """
    check if limit_var_range() works correctly
    """
    
    def test_limit_var_range_between_max_and_min(self):
        """
        check the case input number is between max and min
        """

        # settings
        input_num = 10
        max_num = 20
        min_num = 0
        expected_num = 10
        result = util.limit_var_range(input_num, min_num, max_num)
        
        # assert
        self.assertEqual(expected_num, result)


    def test_limit_var_range_more_than_max(self):
        """
        check the case input number is more than max
        """

        # settings
        input_num = 50
        max_num = 20
        min_num = 0
        expected_num = 20
        result = util.limit_var_range(input_num, min_num, max_num)

        # assert
        self.assertEqual(expected_num, result)

    
    def test_limit_var_range_less_than_min(self):
        """
        check the case input number is less than min
        """

        # settings
        input_num = -10
        max_num = 20
        min_num = 0
        expected_num = 0
        result = util.limit_var_range(input_num, min_num, max_num)

        # assert
        self.assertEqual(expected_num, result)


class TestReplaceNum(unittest.TestCase):
    """
    check if replace_num() works correctly
    """

    def test_replace_num_replace_case(self):
        """
        check the case input number is replaced
        """
        # settings
        input_num = 0
        condition_num = 0
        expected_num = 1
        replaced_num = 1
        result = util.replace_num(input_num, condition_num, replaced_num)

        # assert
        self.assertEqual(expected_num, result)

    def test_replace_num_not_replace_case(self):
        """
        check the case input number is NOT replaced
        """
        # settings
        input_num = 100
        condition_num = 0
        expected_num = 100
        replaced_num = 1
        result = util.replace_num(input_num, condition_num, replaced_num)

        # assert
        self.assertEqual(expected_num, result)


if __name__ == '__main__':
    unittest.main()
