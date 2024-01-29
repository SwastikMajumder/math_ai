import copy
from collections import deque

class TreeNode:
    break_ch = "\n"
    sep_ch = " "
    def __init__(self, data="", left=None, right=None):   
        self.data = data
        self.left = left
        self.right = right
    # converts given binary tree into a string
    def __str__(self):
        def print_tree(self, depth=0):
            output = depth * TreeNode.sep_ch + self.data + TreeNode.break_ch
            if self.left:
                output += print_tree(self.left, depth + 1)
            if self.right:
                output += print_tree(self.right, depth + 1)
            return output
        return TreeNode.break_ch + print_tree(self)[:-1]

# converts a binary tree into a string
def parse_tree(tree_string):
    def parse_tree_r(tree_string, depth=0, data="f_root"):
        tree = TreeNode(data, None, None)
        prc = copy.copy(tree_string)
        a = prc[0].lstrip()
        tree.left = TreeNode(a, None, None)
        prc = prc[1:]
        if not prc:
            tree.left = TreeNode(a, None, None)
            return tree
        if len(prc[0]) != len(prc[0].lstrip()):
            tmp = [word[1:] for word in prc]
            for i in range(len(tmp)):
                if tmp[i][0]=="_":
                    tmp = tmp[:i]
                    break
            tree.left = parse_tree_r(tmp, depth+1, a)
        for i in range(1,len(prc)):
            if len(prc[i]) == len(prc[i].lstrip()):
                prc = prc[i:]
                break
        if prc[0][0] == TreeNode.sep_ch:
            return tree
        b = prc[0].lstrip()
        prc = prc[1:]
        if not prc:
            tree.right = TreeNode(b, None, None)
            return tree
        if len(prc[0]) != len(prc[0].lstrip()):
            tmp = [word[1:] for word in prc]
            for i in range(len(tmp)):
                if tmp[i][0]=="_":
                    tmp = tmp[:i]
                    break
            tree.right = parse_tree_r(tmp, depth+1, b)
        return tree
    t = tree_string.split(TreeNode.break_ch)[1:]
    return parse_tree_r(t).left

def fx_nesting(terminal, fx, depth):
    all_possible = []
    def neighbour_node(curr_tree):
        nonlocal depth
        possibly = terminal + list(fx.keys())
        letter = None
        def is_nnn(s):
            return not (s in fx.keys() and fx[s] > 0)
        def append_highest_depth(curr_tree, depth):
            nonlocal letter
            if (is_nnn(letter) and depth == 0) or (not is_nnn(letter) and depth == 1):
                return None
            if not is_nnn(curr_tree.data):
                if curr_tree.left is None:
                    
                    curr_tree.left = TreeNode(letter, None, None)
                    return curr_tree
                elif curr_tree.right is None and fx[curr_tree.data] == 2:
                    
                    curr_tree.right = TreeNode(letter, None, None)
                    return curr_tree
                else:                
                    output = append_highest_depth(curr_tree.left, depth-1)
                    if output is not None:
                        curr_tree.left = output
                        return curr_tree
                    if curr_tree.right is not None:
                        output = append_highest_depth(curr_tree.right, depth-1)
                        if output is not None:
                            curr_tree.right = output
                            return curr_tree
            return None
        output = []
        for item in possibly:
            letter = item
            result = append_highest_depth(copy.deepcopy(curr_tree), copy.copy(depth))
            if result is not None:
                output.append(result)
        return output
    def bfs(start_node, neighbor_node):
        nonlocal depth
        queue = deque()
        visited = set()
        queue.append(start_node)
        while queue:
            current_node = queue.popleft()
            if current_node not in visited:
                visited.add(current_node)
                neighbors = neighbour_node(current_node)
                if neighbors == []:
                    all_possible.append(current_node.__str__())
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append(neighbor)
    for item in fx.keys():
        node = TreeNode(item, None, None)
        bfs(node, neighbour_node(node))
    for item in fx.keys():
        if fx[item]==0:
            all_possible.append(TreeNode(item, None, None).__str__())
    for item in terminal:
        all_possible.append(TreeNode(item, None, None).__str__())
    return set(all_possible)

def term_generation():
    function_list = {"f_sum": 2}
    max_digits = 1
    max_var = 1
    return fx_nesting(["c_" + str(i) for i in range(max_digits)]+["v_" + str(i) for i in range(max_var)], function_list, 2)

def node_type(s):
    function_list = {"f_sum": 2}
    if s in function_list.keys():
        return s
    elif s[:2] == "v_":
        return s[:2]
    else:
        return "digits"
    
def child_count(curr_tree):
    if curr_tree.left is None:
        return 0
    if curr_tree.right is None:
        return 1
    return 2

def apply_individual_formula(equation, formula_input, formula_output):
    variable_list = {}
    def formula_given(equation, formula):
        nonlocal variable_list
        if node_type(formula.data) == "v_":
            if formula.data in variable_list.keys(): # already encountered variable, check if same variable represent the same thing only
                return variable_list[formula.data].__str__() == equation.__str__()
            else:
                variable_list[formula.data] = copy.deepcopy(equation) # new variable in the formula
                return True
        if node_type(equation.data) != node_type(formula.data) or child_count(equation) != child_count(formula): # different structure of formula or different mathematical operations
            return False
        if equation.left:
            if formula_given(equation.left, formula.left) is False:
                return False
            if equation.right:
                if formula_given(equation.right, formula.right) is False:
                    return False
        return True
    def formula_apply(formula):
        nonlocal variable_list
        if formula.data in variable_list.keys():
            return variable_list[formula.data] # the variable list already generated, replace the variables in the formula
        data_to_return = TreeNode(formula.data, None, None)
        if formula.left:
            data_to_return.left = formula_apply(formula.left)
            if formula.right:
                data_to_return.right = formula_apply(formula.right)
        return data_to_return

    count_spot = 1
    def formula_recur(equation, formula_input, formula_output):
        nonlocal variable_list
        nonlocal count_spot
        
        data_to_return = TreeNode(equation.data, [])
        variable_list = {}
        if formula_given(equation, copy.deepcopy(formula_input)) is True:
            count_spot -= 1
            if count_spot == 0: # try different locations
                return formula_apply(copy.deepcopy(formula_output))
        if node_type(equation.data) in {"digits", "v_"}:
            return equation
        if equation.left:
            data_to_return.left = formula_recur(equation.left, formula_input, formula_output)
            if equation.right:
                data_to_return.right = formula_recur(equation.right, formula_input, formula_output)
        return data_to_return
    cn = 0
    def count_nodes(equation):
        nonlocal cn
        cn += 1
        if equation.left:
            count_nodes(equation.left)
            if equation.right:
                count_nodes(equation.right)
    outputted_val = []
    count_nodes(equation)
    for i in range(1, cn+1): # try different locations where a formula could be applied
        count_spot = i
        orig_len = len(outputted_val)
        tmp = formula_recur(equation, formula_input, formula_output)
        if tmp.__str__() != equation.__str__():
            outputted_val.append(tmp)
    return outputted_val

test_string_2 ="""
v_1"""
test_string_3 ="""
f_sum
 v_1
 c_0"""

term_list = list(term_generation())

equal_category = [[item] for item in term_list]

axiom_list = [[parse_tree(test_string_2), parse_tree(test_string_3)], [parse_tree(test_string_3), parse_tree(test_string_2)]]

for term in term_list:
    for axiom in axiom_list:
        output_list = apply_individual_formula(parse_tree(term), copy.deepcopy(axiom[0]), copy.deepcopy(axiom[1]))
        for output in output_list:
            output = output.__str__()
            output_loc = -1
            term_loc = -1
            for i in range(len(equal_category)):
                if term in equal_category[i]:
                    term_loc = i
                if output in equal_category[i]:
                    output_loc = i
            if output_loc != -1:
                equal_category.append(equal_category[output_loc]+equal_category[term_loc])
                equal_category.pop(max(output_loc, term_loc))
                if output_loc != term_loc:
                    equal_category.pop(min(output_loc, term_loc))
for item in equal_category:
    item = list(set(item))
    for sub_item in item:
        print(sub_item)
    print("----")
