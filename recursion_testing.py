#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 07:14:21 2018

@author: Jake
"""

menu_1 = ['apple', 'orange', 'banana']
menu_2 = ['cat', 'dog', 'bird', 'fish']
menu_3 = ['red', 'yellow', 'green', 'blue','violet']
menu_names = ['menu 1', 'menu 2', 'menu 3']
menus = [menu_1, menu_2, menu_3]

def recursion_testing(menus):
    """Returns a list of solutions
    each solution is a list of strings"""
    if len(menus) == 1:
        return [menus[0]]

    subtree_solutions = recursion_testing(menus[1:])
    return [[item] + subtree_solution for item in menus[0] for subtree_solution in subtree_solutions]

print('\nRecursion Testing\n')
print(recursion_testing(menus))

print('\nDictionary Testing\n')

combos = {}

for i in menu_1:
    combos[i] = []
    for j in menu_2:
        combos[i].append({j:menu_3})

for key in combos.keys():
    print(key,combos[key])
