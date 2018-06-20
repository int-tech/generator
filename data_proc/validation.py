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


def validate_resized_size(resized_size, ratio_min, OPT):
    """
    validate if resized size is integer or not and positive number or not
    and also validate if shorter length of resized image is equal to or less than 0 or not

        :param resized_size : 
        :param ratio_min    :
        :param OPT          :
    """

    # validate positive number
    assert (resized_size >= 0), (
        "Please set a positive number."
    )

    # validate integer
    flag_integer = isinstance(resized_size, int)
    assert (flag_integer == True), (
        "Please set integers instead of decimals."
    )    
    
    if (OPT == "LONG"):
        # validate shorter length of resized image
        shorter_length = int(resized_size * ratio_min)
        assert (shorter_length > 0 ), (
            "Output image size is zero. "
            "Please set more larger resized_size."
        )
