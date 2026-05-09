'''
Custom funtions for Jinja
'''
def all_lst(lst):
    return all(lst)

def neg_all_lst(lst):
    return all([not v for v in lst])

