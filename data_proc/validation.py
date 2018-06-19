"""
validation functions for make_procimg.py
"""

def validate_corner_size_ratio_range(corner_size_ratio):
    """
    check if ratio of corner size is between 0 and 1 or not
    if the number is out of range, return exception message

        :param input_num : ratio of corner size
    """

    if (corner_size_ratio <= 0):
        print("corner_size_ratio is equal to or less than 0.")
        print("Please set corner_size_ratio between 0 and 1.")
    elif (corner_size_ratio > 1):
        print("corner_size_ratio is more than 1.")
        print("Please set corner_size_ratio between 0 and 1.")


def validate_option_resize_keeping_aspect_ratio(OPT):
    """
    validate option is set correctly in the function "resize_keeping_aspect_ratio"

        :param OPT : str, "LONG" or "SHORT" 
    """

    assert (OPT == 'LONG' or OPT == 'SHORT'), (
        "OPT must be 'LONG' or 'SHORT'."
    )
