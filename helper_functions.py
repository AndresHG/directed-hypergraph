#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    helper_functions.py
# @Author:      andres
# @Time:        7/5/24 12:08 AM

from parameters_validation import parameter_validation
from typing import List

import re


@parameter_validation
def valid_concepts_list(string_list: List[str]):
    for string in string_list:
        if not bool(string and string.strip()):
            raise Exception("Strings in the list can not be blank nor empty")
    return


def clean_string(string: str) -> str:
    string = string.strip()  # Remove leading and ending spaces
    string = re.sub(r"[^\w\s]", "", string)  # Remove punctuation
    string = string.lower()  # Move all string to lower_case
    # TODO: remove consecutive duplicated strings

    return string
