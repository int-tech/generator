"""
utility functions for make_procimg.py
"""

def limit_var_range(input_num, min, max):
    """
    limit input_num range (input_num ->  (min <= output_num <= max))

        :param input_num : input number
        :param min       : min number of range
        :param max       : max number of range
        :return          : limited number
    """

    if (input_num < min):
        output_num = min
    elif (input_num > max):
        output_num = max
    else:
        output_num = input_num
    
    return output_num
    

def replace_num(input_num, condition_num, replaced_num):
    """
    if input_num is condition_num, return replaced_num

        :param  : input_num
        :param  : condition_num
        :return : input_num or replaced num
    """

    if (input_num == condition_num):
        return replaced_num
    else:
        return input_num
