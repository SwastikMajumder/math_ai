from nltk import CFG, Tree
from nltk.parse import EarleyChartParser
import nltk.tree as sTree
import copy
import itertools

MULTI_VARIABLE_FUNCTION = ["Add", "Mul", "Pow", "Lim", "Vec"]
SINGLE_VARIABLE_FUNCTION = ["Dif", "Int", "Sin", "Cos", "Exp", "Lwn", "Tan", "Det", "Arctan", "Arcsin", "Arccos", "Neg", "Cpx", "Even"]
COMMUTATIVE_FUNCTION = ["Add", "Mul"]

question_list =\
"""
int(mul(a,pow(add(a,mul(5,neg(0))),mul(2,neg(0))),dif(a)))
int(mul(dif(a),pow(sin(a),mul(2,neg(0))),pow(cos(a),mul(2,neg(0)))))
int(mul(pow(a,mul(4,neg(0))),add(pow(A,2),mul(neg(0),pow(a,2))),dif(a)))
det(vec(vec(1,1,1),vec(1,add(1,a),1),vec(1,1,add(1,b))))
det(vec(vec(pow(a,neg(0)),a,mul(b,c)),vec(pow(b,neg(0)),b,mul(a,c)),vec(pow(c,neg(0)),c,mul(a,b))))
lim(a,0,mul(pow(a,neg(0)),add(pow(add(2,a),mul(2,neg(0))),pow(2,neg(0)))))
lim(a,1,mul(add(mul(2,a),mul(3,neg(0))),add(pow(a,mul(2,neg(0))),neg(0)),add(a,mul(3,neg(0)),mul(2,pow(a,2)))))
lim(a,4,mul(add(3,mul(neg(0),pow(add(a,5),mul(2,neg(0))))),add(1,mul(neg(0),pow(add(5,mul(neg(0),a)),pow(2,neg(0)))))))
int(mul(dif(a),pow(add(1,exp(a)),neg(0))))
int(mul(dif(a),arctan(a)))
int(mul(dif(a),lwn(a),pow(a,3)))
int(mul(pow(add(pow(a,2),5),3),dif(a)))
det(vec(vec(a,b,c),vec(add(a,mul(2,d)),add(b,mul(2,e)),add(c,mul(2,f))),vec(d,e,f)))
det(vec(a,2,4),vec(4,8,0),vec(1,1,0))
add(pow(cpx(0),2015),pow(cpx(0),2016),pow(cpx(0),2017),pow(cpx(0),2018))
"""

# tree data structure, which will be, how equations will be represented with
class TreeNode:
    def __init__(self, label, children=None):
        self.label = label
        self.children = children or []

    def __str__(self):
        nltk_tree = self.to_nltk_tree()
        return sTree.TreePrettyPrinter(nltk_tree).text()

    # convert a number in hierarchical form to a string
    @staticmethod
    def integer_resolve(tree):
        output_string = ""
        if tree.label == "Digit":
            return tree.children[0].label
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
        if tree.label in MULTI_VARIABLE_FUNCTION:
            sym = ","
            output_string += tree.label.lower()
        elif tree.label in SINGLE_VARIABLE_FUNCTION:
            output_string += tree.label.lower() # we need function name in lower class letters, because in the tree created by cfg, we have first letter capital
            start_index = 1
        elif tree.label in {"Letter", "Digit"}:
            return tree.children[0].label
        elif tree.label == "Integer":
            return TreeNode.integer_resolve(tree)
        output_string += "("
        for child in tree.children[start_index:-1]: # last children has no symbol after it
            output_string += TreeNode.custom_print_tree(child)
            output_string += sym
        output_string += TreeNode.custom_print_tree(tree.children[-1])
        output_string += ")"
        return output_string

    # copying each sub equation from the equation hierarchy
    def part_generation(self):
        component_list = [self]
        for child in self.children:
            if not child.children or self.label in {"Letter", "Integer", "Digit"}:
                continue
            component_list += child.part_generation()
        return remove_duplicate(component_list, compare_equation)

    # wrapping function so that more fancy
    def print_algebra(self):
        return TreeNode.custom_print_tree(self)

    # be default a equation hierarchy, no addition parent has a child which is addition. commutativity
    def merge_commutative(self):
        new_children = []
        for child in self.children:
            if isinstance(child, TreeNode):
                child.merge_commutative()
                if child.label in COMMUTATIVE_FUNCTION and child.label == self.label:
                    new_children.extend(child.children)
                else:
                    new_children.append(child)
            else:
                new_children.append(child)
        self.children = new_children
        return self

    def to_nltk_tree(self):
        if self.children == []:
            return self.label
        return Tree(self.label, [item.to_nltk_tree() for item in self.children])
  
# the function which converts math string into equation hierarchy
def math_parser(sentence):
    # preserve the last parent if there are many parents with a single child in a row
    def trim_tree(tree):
        while tree.label not in (["Letter", "Digit", "Integer"]+SINGLE_VARIABLE_FUNCTION) and len(tree.children) == 1:
            tree = tree.children[0]
        coll = TreeNode(tree.label, [])
        for item in tree.children:
            coll.children.append(trim_tree(item))
        return coll

    # this is for making the cfg tree structure get removed which we don't want. the removal would actually be done by the above function
    def remove_parentheses(node):
        coll = TreeNode(node.label, [])
        for item in node.children:
            if item.label in {"Letter", "Digit", "Integer"}:
                coll.children.append(item)
            elif item.children == []:
                pass
            else:
                coll.children.append(remove_parentheses(item))
        return coll

    def nltk_tree_to_custom_tree(nltk_tree):
        if not isinstance(nltk_tree, str):
            coll = TreeNode(nltk_tree.label(), [])
        else:
            return TreeNode(nltk_tree, [])
        for item in nltk_tree:
            coll.children.append(nltk_tree_to_custom_tree(item))
        return coll
        
    # context free grammar to convert math string to tree format
    def pad_grammar_str(req):
        return req + " "*(len("Expression    ")-len(req)) + "->"
    grammar_str = "Expression    -> Integer | Letter | Function | Digit\n"
    grammar_str += pad_grammar_str("Integer") + " |".join([" Digit"*i for i in range(2,5)]) + " | '-'" + " | '-'".join([" Digit"*i for i in range(1,5)]) + "\n"
    grammar_str += pad_grammar_str("Letter")
    letter_list = [chr(ord("A")+j) for j in range(26)]+[chr(ord("a")+j) for j in range(26)]
    
    grammar_str += " '" + "' | '".join(letter_list) + "'\n"
    grammar_str += pad_grammar_str("Digit") + " |".join([" '"+str(i)+"'" for i in range(10)]) + "\n"

    fx_list = {"Add": 10, "Mul": 10, "Pow": 2, "Lim": 3, "Vec": 3}
    for item in SINGLE_VARIABLE_FUNCTION:
        fx_list[item] = 1
        
    grammar_str += pad_grammar_str("Function") + " |".join([" "+item for item in fx_list.keys()]) + "\n"
    for item in fx_list.keys():
        grammar_str += pad_grammar_str(item)
        for i in range(fx_list[item]):
            if i != 0:
                grammar_str += " |"
            grammar_str += " '" + item.lower() + "' '('" + " Expression ','"*i + " Expression ')'"
        grammar_str += "\n"
        
    grammar = CFG.fromstring(grammar_str)
    parser = EarleyChartParser(grammar)
    
    constant_fx = ["neg", "cpx"]
    if any(sentence.count(item + "(") != sentence.count(item + "(0)") for item in constant_fx):
        return None
    
    # handling spaces, because, spaces are what decides what are tokens for cfg to process
    
    if len(sentence) > 1:
        sentence = ' '.join([i for i in sentence])
        result = sentence[0]
        for i in range(1,len(sentence)-1):
            if sentence[i-1].isalpha() and sentence[i] == " " and sentence[i+1].isalpha():
                pass
            else:
                result += sentence[i]
        sentence = result + sentence[-1]
    
    tokens = sentence.split()
    
    nltk_tree = None
    try:
        for tree in parser.parse(tokens):
            tree = nltk_tree_to_custom_tree(tree)
            tree = remove_parentheses(tree)
            tree = trim_tree(tree)
            nltk_tree = tree
            break
        
    except ValueError as e:
        if "Grammar does not cover some of the input words" in str(e):
            # If desired, you can log the error or take other actions
            pass
        else:
            # Handle other parsing errors
            pass
        return None

    if nltk_tree is None:
        return None
    
    part = copy.deepcopy(nltk_tree).merge_commutative().part_generation()
    
    for item in part:
        if item.label == "Int":
            if any("Int" in child.print_algebra() for child in item.children) or all("Dif" not in child.print_algebra() for child in item.children):
                return None
            if item.children[0].label != "Mul":
                return None
            else:
                if sum(1 for sub_item in item.children[0].children if sub_item.label == "Dif") != 1:
                    return None
        elif item.label == "Dif":
            if any("Dif" in child.print_algebra() for child in item.children):
                return None
        elif item.label == "Lim":
            if item.children[0].label != "Letter" or item.children[0].children[0].label.isupper():
                return None
            if item.children[1].label not in {"Digit", "Integer"}:
                return None
        elif item.label == "Even" and int(item.children[1].print_algebra()) < 0:
                return None
        elif item.label == "Pow" and len(item.children) != 2:
            return None
    return nltk_tree

def for_all_equation(a_small, p_small, a_capital, p_capital):
    def condition(equation):
        if equation.label == "Letter":
            if equation.children[0].label == equation.children[0].label.upper():
                if p_capital is None:
                    return equation.children[0].label in a_capital
                if a_capital is None:
                    return equation.children[0].label not in p_capital
                return equation.children[0].label in a_capital and equation.children[0].label not in p_capital
            else:
                if p_small is None:
                    return equation.children[0].label in a_small
                if a_small is None:
                    return equation.children[0].label not in p_small
                return equation.children[0].label in a_small and equation.children[0].label not in p_small
        else:
            return True
    return_equation_set = []
    all_symbol = [ord("a"+i) for i in range(0,26)] + [ord("A"+i) for i in range(0,26)] + [ord("0"+i) for i in range(0,10)] + ["(", ")", "-"]
    for i in range(9999):
        for eq in itertools.product(all_symbol, repeat=i):
            output = math_parser(eq.join(""))
            if output is not None and all(condition(item) for item in output.part_generation()):
                return_equation_set.append(output.print_equation())
    return return_equation_set

def extract_variable_list(equation):
    output_list = []
    part= equation.part_generation()
    for item in part:
        if item.label == "Letter":
            output_list.append(item.children[0].label)
    return output_list

def formula_lhs_rhs_extra(fa, fb):
    a = extract_variable_list(fa)
    b = extract_variable_list(fb)
    if all(item in b for item in a):
        return True
    return True

def connect_c(category_database, equation):
    
    for i in range(len(category_database)-1):
        for j in range(len(category_database)-1, -1, i):
            if any(item in category_database[j] for item in category_database[i]):
                category_database[i] += category_database[j]
                category_database[i] = remove_duplicate(category_database[i], compare_equation_direct)
                category_database.pop(j)
                
    return category_database
            

def category(c):
    for eq in for_all_equation(["".join(item) for item in itertools.product([ord("a"+i) for i in range(0,26)], repeat=9999)]\
                               ,[],["".join(item) for item in itertools.product([ord("A"+i) for i in range(0,26)], repeat=9999)],[]):
        fa = []
        fb = []
        for item in c:
            for x in itertools.permutations(item, 2):
                if not formula_lhs_rhs_extra(x[0], x[1]):
                    continue
                fa.append(x[0])
                fb.append(x[1])
        data = [equation]
        i=0
        while i<len(data):
            t = []
            t += restructure_equation_commutativity(data[i])
            t += test_formula_list(data[i], f_a, f_b)
            t += solve_number_recursive(data[i])
            t = remove_duplicate(t, compare_equation_direct)
            for j in range(len(t),-1,-1):
                if t[j] in data[:i+1]:
                    t.pop(j)
            i += 1
        c.append(data)
        c = connect_c(c)
    return c
# generate all possible commutations of the multiplication and addition, bracketting in all the possible ways
def restructure_equation_commutativity(equation):
    return_equation_set = []
    equation = equation.merge_commutative()
    all_symbol = [ord("a"+i) for i in range(0,26)] + [ord("A"+i) for i in range(0,26)] + [ord("0"+i) for i in range(0,10)] + ["(", ")", "-"]
    for i in range(9999):
        for eq in itertools.product(all_symbol, repeat=i):
            output = math_parser("".join(eq))
            if output is not None and output.merge_commutative().print_equation() == equation.print_equation():
                return_equation_set.append(output.print_equation())
    return_equation_set = remove_duplicate(return_equation_set, compare_equation_direct)
    return return_equation_set

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
                if formula.children[0].label.isupper() and not only_constant(equation):
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
        if equation.label in {"Integer", "Digit", "Letter"}:
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
    if eq_1.label in COMMUTATIVE_FUNCTION:
        for permuted_eq_1 in itertools.permutations(eq_1.children):
            if all(compare_equation(permuted_eq_1[i], eq_2.children[i]) for i in range(len(eq_1.children))): # if any one of the permutation holds
                return True
    else:
        if eq_1.label not in {"Letter", "Digit", "Integer"}: # no permutation allowed other than multiply add or equal
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
        
    for i in range(len(parsed_formula_in)):
        tmp = apply_individual_formula(equation, parsed_formula_in[i], parsed_formula_out[i])
        if tmp is not None:
            hoard += tmp
            
    if len(hoard) == 0:
        return []
    hoard = [eq.merge_commutative() for eq in hoard]
    outputted_val = remove_duplicate(hoard, compare_equation)
    outputted_val =[eq.print_algebra() for eq in outputted_val]
    return outputted_val

def only_constant(equation):
    if equation.label == "Letter" and equation.children[0].label.islower():
        return False
    for child in equation.children:
        if only_constant(child) is False:
            return False
    return True

# simple calculator
def calculator(equation):
    answer = None
    if equation.label not in {"Mul", "Add", "Pow", "Dif", "Even"}:
        return None
    if equation.label == "Mul":
        for child in equation.children:
            if child.label == "Digit" and child.children[0].label == "0": # a single zero multiplied to anything makes the whole thing zero
                return 0
        answer = 1
    elif equation.label == "Add":
        answer = 0
    if equation.label == "Dif": # differentiation of constant is zero
        if only_constant(equation.children[0]):
            return 0
        else:
            return None
    if equation.label == "Even":
        return 1-(int(equation.children[0].label) % 2)
    if equation.label == "Pow":
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
        if equation.label == "Mul":
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

formula_set = \
"""
add(multiply(a,neg(0)),a) 0
add(a,0) a
mul(a,1) a
"""

formula_set_lhs = [(item.split())[0] for item in formula_set.splitlines()[1:]]
formula_set_rhs = [(item.split())[1] for item in formula_set.splitlines()[1:]]
category_database = []
for i in range(len(formula_set_lhs)):
    category_database.append([formula_set_lhs[i],formula_set_rhs[i]])
    
category_database = category(category_database)

question_list = question_list.splitlines()[1:]
for i in range(len(question_list)):
    for category in category_database:
        if question_list[i] in category:
            should_not_be = ["int", "int", "int", "det", "det", "lim", "lim", "lim", "int", "int", "int", "int", "det", "det", "impossible_string"][i]
            print(sorted([equation not in should_not_be for equation in category], key=lambda x: len(x))[0])
