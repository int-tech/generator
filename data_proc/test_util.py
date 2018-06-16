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


if __name__ == '__main__':
    unittest.main()
