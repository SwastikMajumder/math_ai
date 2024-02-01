from collections import deque
from anytree import Node, RenderTree, PreOrderIter
import copy

def build_tree_from_tabbed_strings(tabbed_strings):
    root = Node("Root")
    current_level_nodes = {0: root}
    stack = [root]
    for tabbed_string in tabbed_strings:
        level = tabbed_string.count(' ')
        node_name = tabbed_string.strip()
        node = Node(node_name)
        while len(stack) > level + 1:
            stack.pop()
        parent_node = stack[-1]
        parent_node.children = parent_node.children + (node,)
        current_level_nodes[level] = node
        stack.append(node)
    return root

def convert_tree_2_tabbed_string(tree):
    lines = ["{}{}".format(' ' * (node.depth-1), node.name) for node in PreOrderIter(tree)]
    return "\n".join(lines[1:])

def eq_str(tree):
    return convert_tree_2_tabbed_string(tree)

def convert_tree_2_string(tree):
    lines = []
    for pre, _, node in RenderTree(tree):
        lines.append(f"{pre}{node.name}")
    return "\n".join(lines)

def fx_nest(terminal, fx, depth):
    def nn(curr_tree, depth=depth):
        def is_terminal(name):
            return not (name in fx.keys())
        element = None
        def append_at_last(curr_node, depth):
            if (is_terminal(element) and depth == 0) or (not is_terminal(element) and depth == 1):
                return None
            if not is_terminal(curr_node.name):
                if (not curr_node.parent and len(curr_node.children)==0 and element in fx.keys()) or curr_node.parent:
                    if len(curr_node.children) < fx[curr_node.name]:
                        new_children = curr_node.children + (Node(element, parent=curr_node),)
                        curr_node.children = new_children
                        return curr_node

                for i in range(len(curr_node.children)):
                    output = append_at_last(copy.deepcopy(curr_node.children[i]), depth - 1)
                    if output is not None:
                        new_children = list(copy.deepcopy(curr_node.children))
                        new_children[i] = output
                        return Node(curr_node.name, children=new_children, parent=curr_node.parent)
            return None
        output = []
        for item in terminal + list(fx.keys()):
            element = item
            tmp = copy.deepcopy(curr_tree)
            result = append_at_last(tmp, depth)
            if result is not None:
                output.append(result)
        return output
    all_poss = []
    def bfs(start_node):
        nonlocal all_poss
        queue = deque()
        visited = set()
        queue.append(start_node)
        while queue:
            current_node = queue.popleft()
            if current_node not in visited:
                visited.add(current_node)
                neighbors = nn(current_node)
                if neighbors == []:
                    all_poss.append(convert_tree_2_tabbed_string(current_node))
                    all_poss = list(set(all_poss))
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append(neighbor)
    for item in fx.keys():
        bfs(Node(item))
    return all_poss

con_variable = ["v_" + str(i) for i in range(3,4)]
con_digits = ["d_" + str(i) for i in range(1)] + ["d_-1"]

con_dic_fx = {"f_add": 2, "f_mul": 2, "f_pow": 2, "f_log": 1, "f_sin": 1, "f_cos": 1}
con_term = fx_nest(con_variable+con_digits, con_dic_fx, 2) + con_variable + con_digits

def apply_individual_formula(equation, formula_input, formula_output, chance=False):
    global con_dic_fx
    variable_list = {}
    def node_type(s):
        if s in con_dic_fx.keys():
            return s
        else:
            return s[:2]
    def formula_given(equation, formula):
        nonlocal variable_list
        if node_type(formula.name) == "v_":
            if formula.name in variable_list.keys(): # already encountered variable, check if same variable represent the same thing only
                return eq_str(variable_list[formula.name]) == eq_str(equation)
            else:
                variable_list[formula.name] = copy.deepcopy(equation) # new variable in the formula
                return True
        if node_type(equation.name) != node_type(formula.name) or len(equation.children) != len(formula.children): # different structure of formula or different mathematical operations
            return False
        for i in range(len(equation.children)):
            if formula_given(equation.children[i], formula.children[i]) is False:
                return False
        return True
    
    def formula_apply(formula):
        nonlocal variable_list
        if formula.name in variable_list.keys():
            return variable_list[formula.name] # the variable list already generated, replace the variables in the formula
        data_to_return = Node(formula.name, None, None)
        for child in formula.children:
            new_children = data_to_return.children + (formula_apply(copy.deepcopy(child)),)
            data_to_return.children = new_children
            #data_to_return.children.append(formula_apply(child))
        return data_to_return

    count_spot = 1
    def formula_recur(equation, formula_input, formula_output, chance):
        nonlocal variable_list
        nonlocal count_spot
        
        data_to_return = Node(equation.name, children=[])
        variable_list = {}
        if chance == False:
            if formula_given(equation, copy.deepcopy(formula_input)) is True:
                count_spot -= 1
                if count_spot == 0: # try different locations
                    return formula_apply(copy.deepcopy(formula_output))
        else:
            if len(equation.children)==2 and all(node_type(item.name) == "d_" for item in equation.children):
                x = []
                for item in equation.children:
                    x.append(int(item.name[2:]))
                if equation.name == "f_add":
                    count_spot -= 1
                    if count_spot == 0:
                        return Node("d_"+str(sum(x)))
                elif equation.name == "f_mul":
                    count_spot -= 1
                    if count_spot == 0:
                        return Node("d_"+str(x[0]*x[1]))
                elif equation.name == "f_pow" and not ((x[0]==0 and x[1]<=0) or (x[0]<=0 and x[1]==0)):
                    count_spot -= 1
                    if count_spot == 0:
                        return Node("d_"+str(x[0]**x[1]))
        if node_type(equation.name) in {"d_", "v_"}:
            return equation
        for child in equation.children:
            new_children = data_to_return.children + (formula_recur(copy.deepcopy(child), formula_input, formula_output, chance),)
            data_to_return.children = new_children
        return data_to_return
    cn = 0
    def count_nodes(equation):
        nonlocal cn
        cn += 1
        for child in equation.children:
            count_nodes(child)
    outputted_val = []
    count_nodes(equation)
    for i in range(1, cn+1): # try different locations where a formula could be applied
        count_spot = i
        orig_len = len(outputted_val)
        
        tmp = formula_recur(equation, formula_input, formula_output, chance)
        
        if eq_str(tmp) != eq_str(equation):
            outputted_val.append(tmp)
    return outputted_val

content = None
with open("axiom.txt", 'r') as file:
    content = file.read()
x = content.split("\n\n")
input_f = [x[i] for i in range(0, len(x), 2)]
output_f = [x[i] for i in range(1, len(x), 2)]

input_f = [build_tree_from_tabbed_strings(item.split("\n")) for item in input_f]
output_f = [build_tree_from_tabbed_strings(item.split("\n")) for item in output_f]

equal_category = [[item] for item in con_term]

for term in con_term:
    for i in range(len(input_f)+1):
        output_list = None
        if i == len(input_f):
            output_list = apply_individual_formula(build_tree_from_tabbed_strings(term.split("\n")).children[0], None, None, True)
        else:
            output_list = apply_individual_formula(build_tree_from_tabbed_strings(term.split("\n")).children[0],\
                                                   copy.deepcopy(input_f[i].children[0]), copy.deepcopy(output_f[i].children[0]))
        for output in output_list:
            output.parent = Node("Root")
            output = eq_str(output.parent)
            output_loc = -1
            term_loc = -1
            for i in range(len(equal_category)):
                if term in equal_category[i]:
                    term_loc = i
                if output in equal_category[i]:
                    output_loc = i
            if term_loc != -1 and output_loc != 1 and term_loc != output_loc:
                equal_category.append(equal_category[output_loc]+equal_category[term_loc])
                equal_category.pop(max(output_loc, term_loc))
                equal_category.pop(min(output_loc, term_loc))

for item in equal_category:
    item = list(set(item))
    for sub_item in item:
        print(convert_tree_2_string(build_tree_from_tabbed_strings(sub_item.split("\n"))))
    print("----")
