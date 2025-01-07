from nltk import CFG, ChartParser, Tree
import nltk.tree as sTree
import copy
import itertools

# tree data structure, which will be, how equations will be represented with
class TreeNode:
    def __init__(self, label, children=None):
        self.label = label
        self.children = children or []

    def __str__(self):
        tree = self.to_nltk_tree()
        return sTree.TreePrettyPrinter(tree).text()

    # convert a number in hierarchical form to a string
    @staticmethod
    def integer_resolve(tree):
        output_string = ""
        for child in tree.children:
            if child.label == "-":
                output_string += "-"
                continue
            if child.label == "Digit":
                output_string += child.children[0].label
            else:
                output_string += TreeNode.integer_resolve(child)
        return output_string

    # the function which convert hierarchy equation object into a string which looks like math equation
    @staticmethod
    def custom_print_tree(tree):
        output_string = ""
        sym = None # for storing the operator symbol
        start_index = 0
        if tree.label == "Multiply":
            sym = "*"
        elif tree.label == "Add":
            sym = "+"
        elif tree.label == "Subtract":
            sym = "-"
        elif tree.label == "Divide":
            sym = "/"
        elif tree.label == "Power":
            sym = "^"
        elif tree.label in ["Sin", "Cos", "Tan", "Cosec", "Sec", "Cot", "Integrate", "Differentiate", "Arctan", "Arcsin", "Arccos", "Lawn", "Exp"]:
            output_string += tree.label.lower() # we need function name in lower class letters, because in the tree created by cfg, we have first letter capital
            start_index = 1
        elif tree.label in ["Letter", "Digit"]:
            return tree.children[0].label
        elif tree.label == "Integer":
            return TreeNode.integer_resolve(tree)
        if tree.label == "Pi":
            return output_string + "pi"
        elif tree.label == "Equal":
            return TreeNode.custom_print_tree(tree.children[0]) + "=" + TreeNode.custom_print_tree(tree.children[1])
        output_string += "("
        for child in tree.children[start_index:-1]: # last children has no symbol after it
            output_string += TreeNode.custom_print_tree(child)
            if tree.label == "Add" and tree.children[-1].label == "Integer" and TreeNode.integer_resolve(tree.children[-1]) == "-1":
                continue # problems with minus sign handling
            output_string += sym
        if tree.label == "Power" and tree.children[-1].label == "Integer" and TreeNode.integer_resolve(tree.children[-1]) == "-1":
                output_string += "(-1)"
        else:
            output_string += TreeNode.custom_print_tree(tree.children[-1])
        output_string += ")"
        return output_string

    # copying each sub equation from the equation hierarchy
    def part_generation(self):
        component_list = [self]
        for child in self.children:
            if not child.children or self.label in ["Letter", "Integer", "Digit"]:
                continue
            component_list += child.part_generation()
        return remove_duplicate(component_list, compare_equation)

    # wrapping function so that more fancy
    def print_algebra(self):
        return TreeNode.custom_print_tree(self)

    # be default a equation hierarchy, no addition parent has a child which is addition. commutativity
    def merge_add_multiply(self):
        new_children = []
        for child in self.children:
            if isinstance(child, TreeNode):
                child.merge_add_multiply()
                if child.label in {'Add', 'Multiply'} and child.label == self.label:
                    new_children.extend(child.children)
                else:
                    new_children.append(child)
            else:
                new_children.append(child)
        self.children = new_children
        return self

    # convert local equation hierarchy format to cfg nltk tree
    def to_nltk_tree(self):
        if not self.children:
            return self.label
        else:
            return Tree(self.label, [child.to_nltk_tree() if isinstance(child, TreeNode) else child for child in self.children])

# each math string is unique
def compare_equation(eq_1, eq_2):
    return eq_1.print_algebra() == eq_2.print_algebra()

# detect if a math string has an unmatched bracket
def has_unmatched_bracket(input_string):
    stack = []
    for char in input_string:
        if char in '(':
            stack.append(char)
        elif char in ')':
            if not stack:
                return True  # unmatched closing bracket
            top = stack.pop()
            if (top == '(' and char == ')'):
                continue  # matching pair found
            else:
                return True  # unmatched closing bracket
    return bool(stack)

# the function which converts math string into equation hierarchy
def math_parser(sentence, to_be_merged=True):

    # preserve the last parent if there are many parents with a single child in a row
    def trim_tree_type_2(tree):
        if isinstance(tree, Tree):
            if len(tree) == 1 and isinstance(tree[0], Tree):
                return trim_tree_type_2(tree[0])
            else:
                return Tree(tree.label(), [trim_tree_type_2(child) for child in tree])
        else:
            return tree

    # this is for making the cfg tree structure get removed which we don't want. the removal would actually be done by the above function
    def remove_parentheses(node):
        if isinstance(node, Tree):
            children = [remove_parentheses(child) for child in node]
            return Tree(node.label(), [child for child in children if child is not None])
        else:
            return None if node in ('(', ')', ',') else node

    # nltk cfg tree to our format
    def nltk_tree_to_custom_tree(nltk_tree):
        if isinstance(nltk_tree, Tree):
            label = nltk_tree.label()
            children = [nltk_tree_to_custom_tree(child) for child in nltk_tree]
            return TreeNode(label, children)
        else:
            return TreeNode(nltk_tree)

    # removing single children because they are extraneous node from cfg. this one is in the local format
    def trim_tree_type_3(tree):
        data_to_return = TreeNode(tree.label, [])
        if tree.label == "Integer":
            return TreeNode(tree.label, copy.deepcopy(tree.children))
        for child in tree.children:
            if not child.children and (len(child.label)>1 or (not child.label.isalpha() and not child.label.isdigit())):
                continue
            data_to_return.children += [trim_tree_type_3(child)]
        return data_to_return

    # handle equal to signs because the cfg can't
    if "=" in sentence:
        a = sentence.split("=")[0]
        b = sentence.split("=")[1]
        if has_unmatched_bracket(a):
            a = a[1:]
        if has_unmatched_bracket(b):
            b = b[:-1]
        return TreeNode("Equal", [math_parser(a), math_parser(b)])

    # handling if the math string is having a single character, because that time, cfg seems to fail
    if len(sentence) == 1:
        if sentence.isalpha():
            return TreeNode("Letter", [TreeNode(sentence, [])])
        elif sentence.isdigit():
            return TreeNode("Digit", [TreeNode(sentence, [])])

    # context free grammar to convert math string to tree format
    grammar = CFG.fromstring("""
    Expression    -> Add | Term
    Add           -> Expression '+' Term
    Term          -> Multiply | Factor
    Multiply      -> Factor '*' Term
    Factor        -> Power | '(' Expression ')' | Integer | Letter | Function
    Power         -> Atom '^' Factor | Atom
    Atom          -> '(' Expression ')' | Integer | Letter | Function
    Integer       -> Digit | Digit Integer | '-' Integer
    Digit         -> '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'
    Letter        -> 'a' | 'b' | 'c' | 'd' | 'e' | 'f' | 'g' | 'h' | 'i' | 'j' | 'k' | 'l' | 'm' | 'n' | 'o' | 'p' | 'q' | 'r' | 's' | 't' | 'u' | 'v' | 'w' | 'x' | 'y' | 'z' | 'A' | 'B' | 'C' | 'D' | 'E' | 'F' | 'G' | 'H' | 'I' | 'J' | 'K' | 'L' | 'M' | 'N' | 'O' | 'P' | 'Q' | 'R' | 'S' | 'T' | 'U' | 'V' | 'W' | 'X' | 'Y' | 'Z'
    Function      -> Sin | Cos | Tan | Cosec | Sec | Cot | Integrate | Differentiate | Arctan | Arcsin | Arccos | Lawn | Exp | Pi
    Sin           -> 'sin' '(' Expression ')' | 'sin' '(' Function ')'
    Cos           -> 'cos' '(' Expression ')' | 'cos' '(' Function ')'
    Tan           -> 'tan' '(' Expression ')' | 'tan' '(' Function ')'
    Cosec         -> 'cosec' '(' Expression ')' | 'cosec' '(' Function ')'
    Sec           -> 'sec' '(' Expression ')' | 'sec' '(' Function ')'
    Cot           -> 'cot' '(' Expression ')' | 'cot' '(' Function ')'
    Integrate     -> 'integrate' '(' Expression ')' | 'integrate' '(' Function ')'
    Arctan        -> 'arctan' '(' Expression ')' | 'arctan' '(' Function ')'
    Arcsin        -> 'arcsin' '(' Expression ')' | 'arcsin' '(' Function ')'
    Arccos        -> 'arccos' '(' Expression ')' | 'arccos' '(' Function ')'
    Lawn          -> 'lawn' '(' Expression ')' | 'lawn' '(' Function ')'
    Differentiate -> 'differentiate' '(' Expression ')' | 'differentiate' '(' Function ')'
    Exp           -> 'exp' '(' Expression ')' | 'exp' '(' Function ')'
    Pi            -> 'pi'
    """)
    parser = ChartParser(grammar)
    # some operations to handle minus sign because cfg has no substraction
    sentence = sentence.replace(" ", "")
    sentence = sentence.replace("-", "+-1*")
    sentence = sentence.replace("(+", "(")
    sentence = sentence.replace("*+", "*")
    sentence = sentence.replace("-1*1", "-1")
    if sentence[0] == "+":
        sentence = sentence[1:]
    # handling spaces, because, spaces are what decides what are tokens for cfg to process
    sentence = ' '.join([i for i in sentence])
    sentence = ''.join([sentence[0]]+[sentence[i] for i in range(1, len(sentence)-1) if not(sentence[i-1].isalpha() and sentence[i]==" " and sentence[i+1].isalpha())]+[sentence[-1]])
    tokens = sentence.split()
    nltk_tree = None
    for tree in parser.parse(tokens):
        # remove unnecessary nodes
        tree = remove_parentheses(tree)
        tree = trim_tree_type_2(tree)
        nltk_tree = tree
        break
    nltk_tree = nltk_tree_to_custom_tree(nltk_tree)
    nltk_tree = trim_tree_type_3(nltk_tree)
    if to_be_merged: # param
        nltk_tree.merge_add_multiply()
    return nltk_tree

# single child trimmer duplicate
def trim_tree_type_1(tree):
    if tree.children:
        if len(tree.children) == 1 and tree.children[0].children:
            return trim_tree_type_1(tree.children[0])
        else:
            return TreeNode(tree.label, [trim_tree_type_1(child) for child in tree.children])
    else:
        return tree

# generate all possible hierarchies given leaf node count
def commutative_struct(count, label):
    def generate_tree_bfs(child_counts): # attach children to parents when given child count in bfs order
        root = TreeNode(label, [])
        queue = [root]
        index = 0
        while index < len(child_counts):
            if not queue:
                return None
            current = queue.pop(0)
            total_nodes = len(current.children) + 1
            for i in range(child_counts[index]):
                child = TreeNode(label, [])
                current.children.append(child)
                queue.append(child)
            index += 1
        return root
    def count_leaf_nodes(tree):
        if not tree.children:
            return 1
        else:
            return sum(count_leaf_nodes(child) for child in tree.children)
    def serialize_tree(node):
        if not node:
            return ""
        serialized_children = ",".join(serialize_tree(child) for child in node.children)
        return f"{node.label}({serialized_children})"
    collect = {}
    for i in range(1,count+1):
        for x in itertools.product([0]+list(range(2,count+1)), repeat=i): # all possible children counts
            x  = list(x)
            if x[0]==0: # the root node can't have 0 children
                continue
            tree = generate_tree_bfs(x)
            if tree is None or count_leaf_nodes(tree) != count: # but whatever, the leaf node count should be as required
                continue
            collect[serialize_tree(tree)] = tree
    outputted_val = []
    for item in collect.keys():
        outputted_val.append(collect[item])
    return outputted_val

# given an equation, explore commutivity of the root node
def commutative_property(equation):
    commute_equation = None
    # rearranging the equation's leaf node according to the structure given
    def commute(structure):
        nonlocal commute_equation
        if not structure.children:
            tmp = commute_equation.children.pop(0)
            return tmp
        data_to_return = TreeNode(structure.label, [])
        for child in structure.children:
            tmp = commute(child)
            data_to_return.children.append(tmp)
        return data_to_return
    eq_list = []
    if equation.label == "Multiply" or equation.label == "Add":
        if len(equation.children)>2: # multiply and add can have 2 or more children, commutativity difficult when more than two
            structure_list = copy.deepcopy(commutative_struct(len(equation.children), equation.label))
            for structure in structure_list:
                for jumbled_equation in itertools.permutations(equation.children):
                    commute_equation = copy.deepcopy(TreeNode(equation.label, list(jumbled_equation)))
                    eq_list.append(commute(structure))
        else:
            tmp = copy.deepcopy(equation)
            tmp.children[0], tmp.children[1]  = tmp.children[1], tmp.children[0] # when two children, swapping is enough, for commutativity
            eq_list.append(tmp)
    outputted_val = []
    # remove duplicates
    for i in range(len(eq_list)-1):
        success = True
        for j in range(i+1, len(eq_list)):
            if eq_list[i].print_algebra() == eq_list[j].print_algebra():
                success=False
                break
        if success:
            outputted_val.append(eq_list[i])
    outputted_val.append(eq_list[-1])
    return outputted_val

# generate all possible commutations of the multiplication and addition, bracketting in all the possible ways
equation_restructure_buffer = []
def restructure_equation_commutativity(equation, depth=1): # the depth first search is because, re-bracketing could be done in many possible locations of the tree. 
    global equation_restructure_buffer
    equation_restructure_buffer.append(equation)
    commute_hierarchy_record = {}
    hierarchical_location = 0
    if depth == 0:
        return None
    # check where are the multiplication and addition in the equation heirarchy
    def find_all_commute_instances(equation):
        nonlocal hierarchical_location
        nonlocal commute_hierarchy_record
        hierarchical_location += 1 # track the location where is that bracket in a linear iterative order, ordering in depth first way
        if equation.label == "Multiply" or equation.label == "Add":
            commute_hierarchy_record[str(hierarchical_location)] = commutative_property(equation)
        for child in equation.children:
            find_all_commute_instances(child)
    # replace with replace_object given location
    def change(equation, target_location, replace_object):
        nonlocal hierarchical_location
        data_to_return = TreeNode(equation.label, [])
        hierarchical_location += 1
        if hierarchical_location == target_location:
            return replace_object
        for child in equation.children:
            data_to_return.children.append(change(child, target_location, replace_object))
        return data_to_return
    find_all_commute_instances(equation)
    for key in commute_hierarchy_record.keys():
        for i in range(len(commute_hierarchy_record[key])):
            hierarchical_location = 0
            orig = copy.deepcopy(equation)
            equation = change(equation, int(key), commute_hierarchy_record[key][i]) # do the rebracketting
            restructure_equation_commutativity(equation, depth-1)
            equation = orig # undo the rebracketting, so that, other possible rebracketting could be done
    remove_duplicate(equation_restructure_buffer, compare_equation_direct)
    return None

# given an equation, try applying a formula, in possible location of the equation
def apply_individual_formula(equation, formula_input, formula_output):
    variable_list = {}
    # check if a formula holds for the root node
    def formula_given(equation, formula):
        nonlocal variable_list
        if formula.label == "Letter":
            if formula.children[0].label in variable_list.keys(): # already encountered variable, check if same variable represent the same thing only
                return variable_list[formula.children[0].label].print_algebra() == equation.print_algebra()
            else:
                if formula.children[0].label == "k" and last_letter(equation) != -999:
                    return False
                variable_list[formula.children[0].label] = equation # new variable in the formula
                return True
        if equation.label != formula.label or len(equation.children) != len(formula.children): # different structure of formula or different mathematical operations
            return False
        for i in range(len(equation.children)):
            if formula_given(equation.children[i], formula.children[i]) is False: # if a formula is to be applied, it should fail no where in the hierarchy
                return False
        return True
    # apply a formula in root
    def formula_apply(formula):
        nonlocal variable_list
        if formula.label == "Letter":
            return variable_list[formula.children[0].label] # the variable list already generated, replace the variables in the formula
        data_to_return = TreeNode(formula.label, [])
        for child in formula.children:
            data_to_return.children.append(formula_apply(child))
        return data_to_return
    count_spot = 1
    # recursively check if a formula could be applied in various location of the hierarchy
    def formula_recur(equation, formula_input, formula_output):
        nonlocal variable_list
        nonlocal count_spot
        data_to_return = TreeNode(equation.label, [])
        variable_list = {}
        if formula_given(equation, copy.deepcopy(formula_input)) is True:
            count_spot -= 1
            if count_spot == 0: # try different locations
                return formula_apply(copy.deepcopy(formula_output))
        if equation.label in ["Integer", "Digit", "Letter"]:
            return equation
        for child in equation.children:
            data_to_return.children.append(formula_recur(child, formula_input, formula_output))
        return data_to_return
    outputted_val = []
    for i in range(1, 10): # try different locations where a formula could be applied
        count_spot = i
        orig_len = len(outputted_val)
        tmp = formula_recur(equation, formula_input, formula_output)
        if tmp.print_algebra() != equation.print_algebra():
            outputted_val.append(tmp)
    if len(outputted_val) == 0:
        return None
    outputted_val = remove_duplicate(outputted_val, compare_equation_direct)
    return outputted_val

# compare equation, taking commutativity into account
def compare_equation(eq_1, eq_2):
    if eq_1.print_algebra() == eq_2.print_algebra():
        return True
    if len(eq_1.children) != len(eq_2.children) or eq_1.label != eq_2.label:
        return False
    if eq_1.label in ["Multiply", "Add", "Equal"]:
        for permuted_eq_1 in itertools.permutations(eq_1.children):
            if all(compare_equation(permuted_eq_1[i], eq_2.children[i]) for i in range(len(eq_1.children))): # if any one of the permutation holds
                return True
    else:
        if eq_1.label not in ["Letter", "Digit", "Integer"]: # no permutation allowed other than multiply add or equal
            if all(compare_equation(eq_1.children[i], eq_2.children[i]) for i in range(len(eq_1.children))):
                return True
    return False

# compare equation, not taking commutativity into account
def compare_equation_direct(eq_1, eq_2):
    return eq_1.print_algebra() == eq_2.print_algebra()

# remove duplicates, given a function fx which can define what constitues comparing if two things are the same thing
def remove_duplicate(processing_list, fx):
    outputted_val = []
    if len(processing_list) == 0:
        return []
    for i in range(len(processing_list)-1):
        if any(fx(processing_list[i], processing_list[j]) for j in range(i+1, len(processing_list))):
            continue
        outputted_val.append(processing_list[i])
    outputted_val.append(processing_list[-1]) # atleast one element is not a duplicate
    return outputted_val

# apply a list of formulas on a given equation
def test_formula_list(equation, formula_input, formula_output):
    global restructured_possibilities    
    hoard = []
    parsed_formula_in = []
    parsed_formula_out = []
    for eq in formula_input:
        parsed_formula_in.append(math_parser(eq))
    for eq in formula_output:
        parsed_formula_out.append(math_parser(eq))
    for shape in restructured_possibilities[equation]:
        for i in range(len(parsed_formula_in)):
            if last_letter(parsed_formula_in[i]) == -999: # don't apply a formula when a numerical calculation comes
                continue
            tmp = apply_individual_formula(shape, parsed_formula_in[i], parsed_formula_out[i])
            if tmp is not None:
                hoard += tmp
    if len(hoard) == 0:
        return []
    hoard = [eq.merge_add_multiply() for eq in hoard]
    outputted_val = remove_duplicate(hoard, compare_equation)
    outputted_val =[eq.print_algebra() for eq in outputted_val]
    return outputted_val
# simplify all numerical calculations in a given equation
def solve_numerical(equation):
    global restructured_possibilities
    collect = []
    for shape in copy.deepcopy(restructured_possibilities[equation]): # to handle instances like (x + 1 + 2)=(x + 3)
        collect.append(solve_number_recursive(shape))
    if len(collect) == 0:
        return []
    collect = [eq.merge_add_multiply() for eq in collect]
    outputted_val = remove_duplicate(collect, compare_equation)
    outputted_val =[eq.print_algebra() for eq in outputted_val]
    return outputted_val
# try to substitute the first equation with other equations in the system of equations
def substitute_by_friend(equation, sub_eq):
    global restructured_possibilities
    hoard = []
    if last_letter(math_parser(equation)) == -999: # don't substitute when the equation is a calculation
        return []
    sub_eq = math_parser(sub_eq)
    for permuted in restructured_possibilities[equation]: # commutativity matters when trying to substitute
        hoard.append(replace_instance(permuted, sub_eq.children[1], sub_eq.children[0]))
    if len(hoard) == 0:
        return []
    hoard = [eq.merge_add_multiply() for eq in hoard]
    
    outputted_val = remove_duplicate(hoard, compare_equation)
    outputted_val =[eq.print_algebra() for eq in outputted_val]
    return outputted_val

# replacing find_obj with replace_obj in an equation
def replace_instance(equation, find_obj, replace_object):
    if compare_equation_direct(find_obj, equation):
        return copy.deepcopy(replace_object)
    if equation.label in ["Letter", "Digit", "Integer"]:
        return equation
    data_to_return = TreeNode(equation.label, [])
    for child in equation.children:
        data_to_return.children.append(replace_instance(child, find_obj, replace_object))
    return data_to_return

# make a new equation. a good heuristics is needed to solve math problems
def generate_magic_equation(equation, var_should, mode_normal_variable=True):
    global restructured_possibilities
    outputted_val = []
    lhs_obj = math_parser(var_should)
    extra = {}
    for item in restructured_possibilities[equation.print_algebra()]:
        tmp = item.part_generation() # making a new equation often is inspired by the elements in the already present equation
        for x in tmp:
            if mode_normal_variable == False and x.print_algebra().find("differentiate") == -1: # keep the lhs either x = ... or dx = ...
                continue
            if x.print_algebra().find("integrate") != -1:
                continue
            if mode_normal_variable == True and x.print_algebra().find("differentiate") != -1:
                continue
            tmp_2 = replace_instance(item, x, lhs_obj)
            outputted_val.append(tmp_2)
            extra[copy.deepcopy(tmp_2).merge_add_multiply().print_algebra()] = copy.deepcopy(x).merge_add_multiply().print_algebra()
    outputted_val = [item.merge_add_multiply() for item in outputted_val]
    outputted_val = remove_duplicate(outputted_val, compare_equation)
    outputted_val = [item for item in outputted_val if var_should != item.print_algebra()]
    outputted_val = [[item, [lhs_obj, math_parser(extra[item.print_algebra()])]] for item in outputted_val]
    return outputted_val

# m+n+o+p, in here the last letter is p. 1+1, in here there is no last letter
def last_letter(equation):
    best_letter_till_now = -999 # 
    if equation.label == "Letter":
        return ord(equation.children[0].label)
    for child in equation.children:
        val = last_letter(child)
        if val > best_letter_till_now:
            best_letter_till_now = val
    return best_letter_till_now

# simple calculator
def calculator(equation):
    answer = None
    if equation.label not in ["Multiply", "Add", "Power", "Differentiate"]:
        return None
    if equation.label == "Multiply":
        for child in equation.children:
            if child.label == "Digit" and child.children[0].label == "0": # a single zero multiplied to anything makes the whole thing zero
                return 0
        answer = 1
    elif equation.label == "Add":
        answer = 0
    if equation.label == "Differentiate": # differentiation of constant is zero
        if last_letter(equation.children[0]) == -999:
            return 0
        else:
            return None
    if equation.label == "Power":
        base, expo_power = calculator(equation.children[0]), calculator(equation.children[1])
        for i in range(2):
            if [base, expo_power][i] is None and equation.children[i].label in ["Integer", "Digit"]:
                num = int(equation.children[i].print_algebra())
                if i==0:
                    base = num
                else:
                    expo_power = num
        if base is None or expo_power is None:
            return None
        if base**expo_power == int(base**expo_power):
            return base**expo_power
        else:
            return None
    for child in equation.children:
        tmp = calculator(child)
        if tmp is None:
            if child.label in ["Integer", "Digit"]:
                tmp = int(child.print_algebra())
            else:
                return None
        if equation.label == "Multiply":
            answer *= tmp
        else:
            answer += tmp
    return answer

# solve numericals in all locations in the equation hierarchy
def solve_number_recursive(equation):
    data_to_return = TreeNode(equation.label, [])
    tmp = calculator(equation)
    if tmp is not None:
        return math_parser(str(int(tmp)))
    for child in equation.children:
        data_to_return.children.append(solve_number_recursive(child))
    return data_to_return

# formulas apply anywhere
formula_set_1 = \
"""
a*b+a*c a*(b+c)
a*1 a
a+0 a
a^1 a
a*(a^(-1)) 1
(a*b)^c a^c*b^c
(a^b)^c a^(b*c)
a^(b*c) (a^b)^c
a*a a^2
differentiate(a+b) differentiate(a)+differentiate(b)
differentiate(a^b) b*(a^(b-1))*differentiate(a)+(a^b)*lawn(a)*differentiate(b)
differentiate(a*b) b*differentiate(a)+differentiate(b)*a
integrate(differentiate(a)) a
integrate(a*differentiate(a)) a^2*2^(-1)
integrate(k*a) k*integrate(a)
"""
# formulas apply everywhere expect the first equation
formula_set_2 = \
"""
a=b+c a-b=c
a=b*c a*c^(-1)=b
a=b differentiate(a)=differentiate(b)
a=b b=a
"""
par1 = [(item.split())[0] for item in formula_set_1.splitlines()[1:]]
par2 = [(item.split())[1] for item in formula_set_1.splitlines()[1:]]
clean_set_2_1 = [(item.split())[0] for item in formula_set_2.splitlines()[1:]]
clean_set_2_2 = [(item.split())[1] for item in formula_set_2.splitlines()[1:]]
store_formula = [] # process the formula list string
for a,b in zip(par1, par2):
    store_formula.append(a+"="+b)
    store_formula = list(set(store_formula))

# print equation list no matter if its in string format or hierarchy format  
def print_equation_list_comma(eq):
    output_string = ""
    if isinstance(eq[0], str):
        output_string += eq[0]
        output_string += ", "
        for item in eq[1:]:
            if isinstance(item, list):
                output_string += item[0] + "=" + item[1]
            else:
                output_string += item
            output_string += ", "
    else:
        output_string += eq[0].print_algebra() + ", "
        for item in eq[1:]:
            if isinstance(item, list):
                output_string += item[0].print_algebra() + "=" + item[1].print_algebra()
            else:
                output_string += item.print_algebra()
            output_string += ", "
    return output_string[:-2]

# mathematics has a branching factor. generate many equations given a equation as input, those generations, contain exactly the same information, as the inputted
restructured_possibilities = {}
def branch_mathematics(equation_list):
    outputted_val = []
    global equation_restructure_buffer
    
    global restructured_possibilities
    for item in equation_list:
        equation_restructure_buffer = []
        restructure_equation_commutativity(math_parser(item))
        restructured_possibilities[math_parser(item).print_algebra()] = copy.deepcopy(equation_restructure_buffer)
    first_eq = last_letter(math_parser(equation_list[0])) != -999
    # solve numericals
    for i in range(len(equation_list)):
        tmp = solve_numerical(equation_list[i])
        for x in tmp: 
            st = copy.deepcopy(equation_list)
            st[i] = x
            outputted_val.append(st)
    nl = max([last_letter(math_parser(eq)) for eq in equation_list])
    # make new equations
    if nl != -999 and first_eq:
        tmp = generate_magic_equation(math_parser(equation_list[0]), chr(nl+1))
        for item in tmp:
            st = copy.deepcopy(equation_list)
            st.append(chr(nl+1)+"="+item[1][1].print_algebra())
            outputted_val.append(st)
        tmp = generate_magic_equation(math_parser(equation_list[0]), "differentiate("+chr(nl+1)+")", False)
        for item in tmp:
            st = copy.deepcopy(equation_list)
            st.append("differentiate("+chr(nl+1)+")"+"="+item[1][1].print_algebra())
            outputted_val.append(st)
    global store_formula
    a = [math_parser(item).children[0].print_algebra() for item in store_formula]
    b = [math_parser(item).children[1].print_algebra() for item in store_formula]
    global formula_set_1
    global clean_set_2_2
    # apply formulas
    for i in range(len(equation_list)):
        if last_letter(math_parser(equation_list[i])) == -999:
            continue
        tmp = []
        tmp += test_formula_list(equation_list[i], a, b)
        if i>0:
            tmp += test_formula_list(equation_list[i], clean_set_2_1, clean_set_2_2)
        for item in tmp:
            eq_tmp = copy.deepcopy(equation_list)
            eq_tmp[i] = item
            outputted_val.append(eq_tmp)
    # substitute equations into each other, in a system of equations
    for i in range(1,len(equation_list)):
        tmp = substitute_by_friend(equation_list[0], equation_list[i])
        for item in tmp:
            eq_tmp = copy.deepcopy(equation_list)
            eq_tmp[0] = item
            outputted_val.append(eq_tmp)
    # duplicate or remove existing equations
    for i in range(1, len(equation_list)):
        eq_tmp = copy.deepcopy(equation_list)
        eq_tmp.pop(i)
        outputted_val.append(eq_tmp)
        eq_tmp = copy.deepcopy(equation_list)
        eq_tmp.append(equation_list[i])
        outputted_val.append(eq_tmp)
    outputted_val = [x.split("#") for x in list(set(["#".join(item) for item in outputted_val]))] # remove duplicates
    for i in range(len(outputted_val)):
        print(str(i+1)+". ", sep='', end='')
        print(print_equation_list_comma(outputted_val[i]))
    choice = input("enter your choice: ")
    return outputted_val[int(choice)-1]

equation_in_memory = ["integrate((2^(-1))*m*differentiate(m))"] # simulatenous equations
for i in range(len(equation_in_memory)):
    equation_in_memory[i] = math_parser(equation_in_memory[i]).print_algebra()
while True:
    equation_in_memory = branch_mathematics(equation_in_memory)
    
