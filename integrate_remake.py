from fractions import Fraction
from collections import deque
import copy
import itertools
class TreeNode:
    def __init__(self, name, children=None):
        self.name = name
        self.children = children or []
    def __str__(self):
        return str_form(self)+"\n"
def tree_form(tabbed_strings):
    lines = tabbed_strings.split("\n")
    root = TreeNode("Root")
    current_level_nodes = {0: root}
    stack = [root]
    for line in lines:
        level = line.count(' ')
        node_name = line.strip()
        node = TreeNode(node_name)
        while len(stack) > level + 1:
            stack.pop()
        parent_node = stack[-1]
        parent_node.children.append(node)
        current_level_nodes[level] = node
        stack.append(node)
    return root.children[0]
def str_form(node):
    def recursive_str(node, depth=0):
        result = "{}{}".format(' ' * depth, node.name)
        for child in node.children:
            result += "\n" + recursive_str(child, depth + 1)
        return result
    return recursive_str(node)
import re
def apply_individual_formula(equation, formula_input, formula_output, chance=False): 
    variable_list = {}
    def node_type(s):
        if s[:2] == "f_":
            return s
        else:
            return s[:2]
    def formula_given(equation, formula):
        nonlocal variable_list
        if node_type(formula.name) in {"u_", "p_"}:
            if formula.name in variable_list.keys():
                return str_form(variable_list[formula.name]) == str_form(equation)
            else:
                if node_type(formula.name) == "p_" and "v_" in str_form(equation):
                    return False
                variable_list[formula.name] = copy.deepcopy(equation)
                return True
        if equation.name != formula.name or len(equation.children) != len(formula.children):
            return False
        for i in range(len(equation.children)):
            if formula_given(equation.children[i], formula.children[i]) is False:
                return False
        return True
    def formula_apply(formula):
        nonlocal variable_list
        if formula.name in variable_list.keys():
            return variable_list[formula.name]
        data_to_return = TreeNode(formula.name, None)
        for child in formula.children:
            data_to_return.children.append(formula_apply(copy.deepcopy(child)))
        return data_to_return
    count_spot = 1
    def formula_recur(equation, formula_input, formula_output, chance):
        nonlocal variable_list
        nonlocal count_spot
        data_to_return = TreeNode(equation.name, children=[])
        variable_list = {}
        if chance == False:
            if formula_given(equation, copy.deepcopy(formula_input)) is True:
                count_spot -= 1
                if count_spot == 0:
                    return formula_apply(copy.deepcopy(formula_output))
        else:
            if len(equation.children)==2 and all(node_type(item.name) == "d_" for item in equation.children):
                x = []
                for item in equation.children:
                    x.append(int(item.name[2:]))
                if equation.name == "f_add":
                    count_spot -= 1
                    if count_spot == 0:
                        return TreeNode("d_"+str(sum(x)))
                elif equation.name == "f_mul":
                    count_spot -= 1
                    if count_spot == 0:
                        p = 1
                        for item in x:
                            p *= item
                        return TreeNode("d_"+str(p))
                elif equation.name == "f_div" and x[1] != 0 and (x[0]/x[1]).is_integer():
                    count_spot -= 1
                    if count_spot == 0:
                        return TreeNode("d_"+str(int(x[0]/x[1])))
                elif equation.name == "f_pow" and x[1]>=2:
                    count_spot -= 1
                    if count_spot == 0:
                        return TreeNode("d_"+str(int(x[0]**x[1])))
        if node_type(equation.name) in {"d_", "v_", "s_"}:
            return equation
        for child in equation.children:
            data_to_return.children.append(formula_recur(copy.deepcopy(child), formula_input, formula_output, chance))
        return data_to_return
    cn = 0
    def count_nodes(equation):
        nonlocal cn
        cn += 1
        for child in equation.children:
            count_nodes(child)
    outputted_val = []
    count_nodes(equation)
    for i in range(1, cn+1):
        count_spot = i
        orig_len = len(outputted_val)
        tmp = formula_recur(equation, formula_input, formula_output, chance)        
        if str_form(tmp) != str_form(equation):
            outputted_val.append(tmp)
    return outputted_val
def return_axiom_file(file_name):
    content = None
    with open(file_name, 'r') as file:
        content = file.read()
    x = content.split("\n\n")
    input_f = [x[i] for i in range(0, len(x), 2)]
    output_f = [x[i] for i in range(1, len(x), 2)]
    input_f = [tree_form(item) for item in input_f]
    output_f = [tree_form(item) for item in output_f]
    unique_pairs = {}
    for inp, out in zip(input_f, output_f):
        key = str(inp) + str(out)  # Assuming TreeNode objects are hashable
        unique_pairs[key] = (inp, out)
    unique_input_f, unique_output_f = zip(*unique_pairs.values())
    return [list(unique_input_f), list(unique_output_f)]
def bfs2(function_list, start_node):
    queue = deque()
    visited = set()
    smallest = "-"*1000
    queue.append(start_node)
    count = 0
    while queue:
        current_node = queue.popleft()
        if current_node not in visited:
            visited.add(current_node)
            neighbors = function_list[0](current_node)
            for item in neighbors:
                if "f_eq" in item: 
                    print(item)
                    print()
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)
def integrate(node):
    input_f, output_f = return_axiom_file("axiom_3.txt")
    output_list = []
    output_list += apply_individual_formula(tree_form(node), None, None, True)
    for i in range(len(input_f)):
        if i == 0 and node.count("f_dif") > 2:
            continue
        output_list += apply_individual_formula(tree_form(node), copy.deepcopy(input_f[i]), copy.deepcopy(output_f[i]))
    if node.count("f_eq") < 2:
        output_list += [new_equation(tree_form(node), item) for item in part(tree_form(node)) if new_equation(tree_form(node), item) is not None]
    output_list += subs(tree_form(node))
    output_list = [str_form(item) for item in output_list]
    return list(set(output_list))
def replace(node, to_find, to_replace):
    coll = TreeNode(node.name, [])
    if str_form(to_find) == str_form(node):
        return to_replace
    for child in node.children:
        coll.children.append(replace(child, to_find, to_replace))
    return coll
sub_taken = []
def subs(equation):
    output = []
    global sub_taken
    for eq in sub_taken:
        for item in part_2(tree_form(eq)):
            if item.name == "f_eq" and "f_int" not in str_form(item):
                output += [replace(equation, item.children[0], item.children[1])]
                output += [replace(equation, item.children[1], item.children[0])]
    return output
def fix_neg_var(node):
    node = str_form(node)
    v_list = re.findall(r'v_[^\n]*\n', node)
    v_list = [int(item[2:]) for item in v_list]
    v_list = sorted(v_list)
    if v_list[0]<0 and node is not None:
        for i in range(-v_list[0]):
            node = node.replace("v_"+str(-i-1), "v_"+str(v_list[-1]+i+1))
    return tree_form(node)
def part_2(equation):
    output = [equation]
    for child in equation.children:
        output += part_2(child)
    return output
def part(equation):
    output = part_2(equation)
    return [item for item in output if "f_eq" not in str_form(item) and "f_int" not in str_form(item) and "f_obj" not in str_form(item)]
def new_equation(equation, node):
    if node.name != "f_add":
        return None
    global sub_taken
    output = fix_neg_var(TreeNode("f_obj", [TreeNode("f_eq", [TreeNode("v_-1", []), node]), equation]))
    sub_taken.append(str_form(output))
    return output
find = """f_int
 f_mul
  f_dif
   v_0
  f_pow
   f_add
    v_0
    d_1
   d_2"""
while True:
    output = integrate(find)
    for i,item in enumerate(output):
        for sub_item in part_2(tree_form(item)):
            if sub_item.name == "f_eq":
                sub_taken.append(str_form(sub_item))
        sub_taken = list(set(sub_taken))
        print(i)
        print(item)
        print()
    sel = input("select: ")
    sel = int(sel)
    find = output[sel]
