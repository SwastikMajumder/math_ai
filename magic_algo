from fractions import Fraction
from collections import deque
import copy
import itertools
from collections import Counter

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

def apply_individual_formula(equation, formula_input, formula_output, chance=False, direct=False):
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
                elif equation.name == "f_pow" and x[1]>=2:
                    count_spot -= 1
                    if count_spot == 0:
                        return TreeNode("d_"+str(int(x[0]**x[1])))
                elif equation.name == "f_div" and x[1] != 0 and (x[0]/x[1]).is_integer():
                    count_spot -= 1
                    if count_spot == 0:
                        return TreeNode("d_"+str(int(x[0]/x[1])))
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
    if direct:
        if formula_given(equation, copy.deepcopy(formula_input)) is True:
            return [formula_apply(copy.deepcopy(formula_output))]
        return []
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
    start_node = form(start_node)
    
    queue.append(start_node)
    count = 0
    while queue:
        current_node = queue.popleft()
        if current_node not in visited:
            visited.add(current_node)
            neighbors = fx(current_node)
            neighbors = [form(item) for item in neighbors]
            for item in neighbors:
                print(item)
                print()
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)

def flatten_tree(node):
    if not node.children:
        return node
    if node.name in ("f_add", "f_mul"):
        merged_children = []
        for child in node.children:
            flattened_child = flatten_tree(child)
            if flattened_child.name == node.name:
                merged_children.extend(flattened_child.children)
            else:
                merged_children.append(flattened_child)
        return TreeNode(node.name, merged_children)
    else:
        node.children = [flatten_tree(child) for child in node.children]
        return node

def calc(input_f, output_f, term):
    term = tree_form(term)
    while True:
        output_list = []
        output_list += apply_individual_formula(term, None, None, True)
        for i in range(len(input_f)):
            output_list += apply_individual_formula(term, input_f[i], output_f[i])
        if output_list==[]:
            return str_form(term)
        if flatten:
            term = flatten_tree(output_list[0])
        else:
            term = output_list[0]

def integrate(node):
    c, d = return_axiom_file("back/integrate.txt")
    return [calc(c, d, node)]

def fx_nest(terminal, fx, depth):
    def nn(curr_tree, depth=depth):
        def is_terminal(name):
            return not (name in fx.keys())
        element = None
        def append_at_last(curr_node, depth):
            if (is_terminal(element) and depth == 0) or (not is_terminal(element) and depth == 1):
                return None
            if not is_terminal(curr_node.name):
                if len(curr_node.children) < fx[curr_node.name]:
                    curr_node.children.append(TreeNode(element))
                    return curr_node
                for i in range(len(curr_node.children)):
                    output = append_at_last(copy.deepcopy(curr_node.children[i]), depth - 1)
                    if output is not None:
                        curr_node.children[i] = copy.deepcopy(output)
                        return curr_node
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
                    all_poss.append(str_form(current_node))
                    all_poss = list(set(all_poss))
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append(neighbor)
    for item in fx.keys():
        bfs(TreeNode(item))
    return all_poss

def part(term):
    sub_term_list = [term]
    term = tree_form(term)
    for child in term.children:
        sub_term_list += part(str_form(child))
    return sub_term_list

def illegal_eq(term):
    term = tree_form(term)
    if term.name in {"f_pow", "f_root"}:
        return term.children[1].name[:2] == "d_" and int(term.children[1].name[2:]) >= 2
    return True

def flatten_tree(node):
    if not node.children:
        return node
    if node.name in ("f_add", "f_mul"):
        merged_children = []
        for child in node.children:
            flattened_child = flatten_tree(child)
            if flattened_child.name == node.name:
                merged_children.extend(flattened_child.children)
            else:
                merged_children.append(flattened_child)
        return TreeNode(node.name, merged_children)
    else:
        node.children = [flatten_tree(child) for child in node.children]
        return node

def simplify_2(term):
    while True:
        output_list = apply_individual_formula(term, None, None, True)
        if output_list==[]:
            return term
        term = output_list[0]

def simplify(term):
    output = simplify_2(term)
    if output.name[:2]=="d_":
        return Fraction(int(output.name[2:]), 1)
    elif output.name == "f_div" and len(output.children) == 2 and all(child.name[:2] == "d_" for child in output.children):
        return Fraction(int(output.children[0][2:]), int(output.children[0][2:]))
def flip_less_than(inter):
    return [not item if isinstance(item, bool) else item for item in inter]
def factor_form(node):
    node = flatten_tree(node)
    a, b = return_axiom_file("factor.txt")
    critical = []
    if "d_0" in str_form(node):
        return None
    if "v_0" not in str_form(node):
        return [True]
    flip = False
    if node.name == "f_mul":
        count = 0
        for term in node.children:
            if term.name[:2] == "d_":
                count += 1
                if int(term.name[2:]) < 0:
                    flip = not flip
            for i in range(len(a)):
                output = apply_individual_formula(term, a[i], b[i], False, True)
                if output != []:
                    power = 1
                    if term.name == "f_pow":
                        power = int(term.children[1].name[2:])
                    for item in part(str_form(term)):
                        item = tree_form(item)
                        if item.name == "f_mul" and item.children[0].name == "v_0" and int(item.children[1].name[2:])<0 and power % 2 == 1:
                            flip = not flip
                            break
                    critical.append(simplify(output[0]))
        if count > 1:
            return None
        if len(node.children) - count != len(critical):
            return None
    else:
        for i in range(len(a)):
            output = apply_individual_formula(node, a[i], b[i], False, True)
            if output != []:
                power = 1
                if node.name == "f_pow":
                    power = int(node.children[1].name[2:])
                for item in part(str_form(node)):
                    item = tree_form(item)
                    if item.name == "f_mul" and item.children[0].name == "v_0" and int(item.children[1].name[2:])<0 and power % 2 == 1:
                        flip = not flip
                        break
                critical.append(simplify(output[0]))
        if critical == []:
            return None
    critical = Counter(critical)
    critical = sorted(critical.items(), key=lambda x: x[0])
    i = len(critical)
    element = True
    while i>=0:
        critical.insert(i, element)
        if i>0 and critical[i-1][1] % 2 != 0:
            element = not element
        i = i - 1
    for i in range(1, len(critical), 2):
        critical[i] = critical[i][0]
    if flip:
        critical = flip_less_than(critical)
    return critical

def intersection(domain_1, domain_2):
    if domain_1 == [True]:
        return domain_2
    if domain_2 == [True]:
        return domain_1
    if domain_1 == [False] or domain_2 == [False]:
        return [False]
    def simplify_ranges(ranges):
        simplified_ranges = []
        i = 0
        while i < len(ranges):
            if i + 2 < len(ranges) and ranges[i] is True and ranges[i + 2] is True:
                simplified_ranges.append(True)
                i += 3
            elif i + 2 < len(ranges) and ranges[i] is False and ranges[i + 2] is False:
                simplified_ranges.append(False)
                i += 3
            else:
                simplified_ranges.append(ranges[i])
                i += 1
        return simplified_ranges
    result = domain_1 + domain_2
    result = [item for item in result if not isinstance(item, bool)]
    result = list(set(result))
    result = sorted(result, key=Fraction)
    i = len(result)
    while i>=0:
        result.insert(i, True)
        i = i - 1
    result[0] = domain_1[0] and domain_2[0]
    result[-1] = domain_1[-1] and domain_2[-1]
    def find_fraction_in_list(fraction_list, target_fraction):
        for i in range(1, len(fraction_list)-1, 2):
            if fraction_list[i].numerator == target_fraction.numerator and fraction_list[i].denominator == target_fraction.denominator:
                return i
        return -1
    for i in range(2, len(result)-1, 2):
        if result[i+1] in domain_1:
            result[i] = result[i] and domain_1[find_fraction_in_list(domain_1, result[i+1])-1]
        if result[i+1] in domain_2:
            result[i] = result[i] and domain_2[find_fraction_in_list(domain_2, result[i+1])-1]
        if result[i-1] in domain_1:
            result[i] = result[i] and domain_1[find_fraction_in_list(domain_1, result[i-1])+1]
        if result[i-1] in domain_2:
            result[i] = result[i] and domain_2[find_fraction_in_list(domain_2, result[i-1])+1]
    result = simplify_ranges(result)
    return result
find = """f_mul
 f_add
  v_0
  d_-1
 v_0"""

find_2 = """f_add
 v_0
 d_1"""

#x = factor_form(tree_form(find))
#y = factor_form(tree_form(find_2))
    

con_term = fx_nest(["d_2", "d_-1", "d_0", "d_1", "v_0"], {"f_add": 2, "f_mul": 2, "f_pow": 2}, 1)

con_term = [term for term in con_term if all(illegal_eq(item) for item in part(term))] + ["d_2", "d_-1", "v_0", "d_0", "d_1"]

input_f, output_f = return_axiom_file("axiom.txt")

equal_category = [[item] for item in con_term]
con_term = list(set(con_term))
for term in con_term:
    for i in range(len(input_f)+1):
        output_list = []
        if i == len(input_f):
            output_list += apply_individual_formula(tree_form(term), None, None, True)
        else:
            output_list += apply_individual_formula(tree_form(term), copy.deepcopy(input_f[i]), copy.deepcopy(output_f[i]))
        for output in output_list:
            output = str_form(output)
            output_loc = -1
            term_loc = -1
            for j in range(len(equal_category)):
                if term in equal_category[j]:
                    term_loc = j
                if output in equal_category[j]:
                    output_loc = j
            if term_loc != -1 and output_loc == -1:
                equal_category.pop(term_loc)
                break
            if term_loc != -1 and output_loc != -1 and term_loc != output_loc:
                equal_category.append(equal_category[output_loc]+equal_category[term_loc])
                equal_category.pop(max(output_loc, term_loc))
                equal_category.pop(min(output_loc, term_loc))

for i in range(len(equal_category)-1,-1,-1):
    domain = None
    for eq in equal_category[i]:
        domain = factor_form(tree_form(eq))
        if domain is not None:
            break
    if domain is not None:
        equal_category[i] = [domain, equal_category[i]]
    else:
        equal_category.pop(i)

for item in equal_category:
    print(item[0])
    for sub_item in item[1]:
        print(sub_item)
    print("----------")

def category_check(term):
    global equal_category
    for i in range(len(equal_category)):
        if term in equal_category[1]:
            return equal_category[0]
    return None

con_term_2 = fx_nest(["v_0", "d_2"], {"f_add": 2, "f_mul": 2, "f_pow": 2, "f_root": 2, "f_div": 2, "f_lwn": 1, "f_sin": 1}, 1)

con_term_2 = [term for term in con_term_2 if all(illegal_eq(item) for item in part(term))] + ["v_0", "d_2"]

def zero_equal(inter):
    return [item for item in inter if not isinstance(item, bool)]

def find_domain(term):
    parts = part(term)
    inter = [] # allowed
    eq_inter = [] # not allowed
    invalid = [[False], []]
    val = None
    for item in parts:
        item = tree_form(item)
        if item.name == "f_div":
            if item.children[1].name == "d_0":
                return invalid
            
            output = factor_form(item.children[1])
            if output is None:
                return invalid
            eq_inter += zero_equal(output)
        elif item.name == "f_lwn":
            if item.children[0].name[:2] == "d_" and int(item.children[0].name[2:]) <= 0:
                return invalid
            output = factor_form(item.children[0])
            if output is None:
                return invalid
            inter.append(output)
            eq_inter += zero_equal(output)
        elif item.name == "f_root":
            if item.children[0].name[:2] == "d_" and int(item.children[0].name[2:]) < 0:
                return invalid
            output = factor_form(item.children[0])
            if output is None:
                return invalid
            inter.append(output)
    if inter == []:
        val = [True]
    elif len(inter) == 1:
        val = inter[0]
    else:
        val = intersection(inter[0], inter[1])
        inter.pop(0)
        inter.pop(0)
        for i in range(len(inter)-3,-1,-1):
            val = intersection(val, inter[i])
            inter.pop(-1)
    return [val, eq_inter]

def split_interval(inter, zero_inter):
    output = []
    if inter == [False]:
        None
    if inter[0] == True:
        output.append(inter[2:])
    if inter[-1] == True:
        output.append(inter[:-2])
    for i in range(2, len(inter)-1, 2):
        if inter[i]:
            output.append([False, inter[i-1], True, inter[i+1], False])
    tmp = zero_equal(inter) + zero_inter
    tmp = list(set(tmp))
    tmp = sorted(tmp, key=Fraction)
    for item in zero_inter:
        index = tmp.index(item)
        if index == 0:
            output.append([True, item, False])
        if index == len(tmp)-1:
            output.append([False, item, True])
        if index >0 and index < len(tmp)-1:
            output.append([False, tmp[index-1], True, tmp[index], False])
            output.append([False, tmp[index], True, tmp[index+1], False])
    if output == []:
        return [True]
    output = [item for item in output if item != []]
    return output

def bfs2(start_node, end_node, input_f, output_f):
    queue = deque()
    visited = set()
    start_node = str_form(start_node)
    
    queue.append(start_node)
    
    while queue:
        current_node = queue.popleft()
        if current_node not in visited:
            visited.add(current_node)
            neighbors = []
            neighbors += apply_individual_formula(current_node, None, None, True)
            for i in range(len(input_f)):
                neighbors += apply_individual_formula(current_node, input_f[i], output_f[i])
            if end_node in neighbors:
                return True
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)
    return None

find_2 = """f_div
 d_1
 v_0"""

question_domain = split_interval(*find_domain(question))

def piece(ques, domain):
    answer = []
    der_i, der_o = return_axiom_file("derivative.txt")
    for term in con_term_2:
        term = tree_form(term)
        term = TreeNode("f_dif", [term])
        term = str_form(term)
        if find_domain(term)[0] != domain:
            continue
        if not bfs2(ques, der_i, der_o, term):
            continue
        answer.append(term)
    return answer
print(
#bfs2([diverge, integrate], find)
