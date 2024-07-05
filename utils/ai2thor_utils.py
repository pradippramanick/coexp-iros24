import re

def id2name(ID):
    return ID.split('|')[0]
    


def camel_case_split(str):
    return re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', str)
