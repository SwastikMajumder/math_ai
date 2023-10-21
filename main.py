import copy
from functools import cmp_to_key

FUNCTION_COUNT = 16
STR_MAX = 75
MAX_BRACKET_LENGTH = 25
ABRITRARY_MAX = 300
MAX_TT_SIZE = 1000

TYPE_CONSTANT = -6
TYPE_VARIABLE = -5
ADD = -4
MULTIPLY = -3
POWER = -2
DIVISION = -1
SINE = 0
COSINE = 1
TANGENT = 2
COSECANT = 3
SQUARE_ROOT = 6
LAWN = 7
EXPONENTIAL = 8
INTEGRATION = 9
DIF = 10
INF = 11
function_full_name_list = ["add", "mul", "pow", "div", "sin", "cos", "tan", "csc", "sec", "cot", "sqt", "lwn", "exp", "int", "dif", "inf"]
def is_char_small_letter(x):
    if ord("a") <= ord(x) and ord(x) <= ord("z"):
        return True
    return False
def is_char_cap_letter(x):
    if ord("A") <= ord(x) and ord(x) <= ord("Z"):
        return True
    return False
def is_char_number(x):
    if ord("0") <= ord(x) and ord(x) <= ord("9"):
        return True
    return False
def is_char_calculus(x):
    if x == "m" or x == "n" or x == "o":
        return True
    return False
class equation_node:
    def __init__(self, function_type, children, constant_value=None, variable_name=None):
        self.function_type = function_type
        self.children = children
        self.constant_value = constant_value
        self.variable_name = variable_name
def insert_char_at_index(input_string, character, index):
    return input_string[0:index]+character+input_string[index:]
def remove_char_at_index(input_string, index):
    return input_string[0:index]+input_string[index+1:]
def math_parser_remove_extra_bracket(given_string):
    i=0
    while i != len(given_string):
        if given_string[i] == "(" and (not(is_char_cap_letter(given_string[i-1]) if i != 0 else True) ):
            if find_corresponding_bracket(given_string, i) < search_for_basic_operation_symbol(given_string, i+1):
                given_string = remove_char_at_index(given_string, find_corresponding_bracket(given_string, i))
                given_string = remove_char_at_index(given_string, i)
                i -= 1
        i += 1
    i=0
    while i != len(given_string):
        if given_string[i] == "(" and given_string[i+1] == "(" and (find_corresponding_bracket(given_string, i)-find_corresponding_bracket(given_string, i+1))==1:
            given_string = remove_char_at_index(given_string, find_corresponding_bracket(given_string, i))
            given_string = remove_char_at_index(given_string, i)
            i -=1
        i += 1
    return given_string
def duplicate_equation(equation):
    '''
    new_equation = equation_node(equation.function_type, equation.children, equation.constant_value, equation.variable_name)
    for i in range(len(equation.children)):
        new_equation.children[i] = duplicate_equation(equation.children[i])
    return copy.deepcopy(new_equation)
    '''
    return copy.deepcopy(equation)
def merge_nested_bracket(equation):
    if equation.function_type == ADD or equation.function_type == MULTIPLY:
        i=0
        while i < len(equation.children):
            if equation.function_type == equation.children[i].function_type:
                tmp = duplicate_equation(equation.children[i])
                equation.children += tmp.children
                equation.children.pop(i)
                i -= 1
            i += 1
    if equation.function_type == TYPE_VARIABLE or equation.function_type == TYPE_CONSTANT:
        return equation
    i=0
    while i < len(equation.children):
        equation.children[i] = merge_nested_bracket(equation.children[i])
        i += 1
    return equation
def math_parser_no_subtraction(given_string):
    i=0
    while (i+1)!=len(given_string):
        if given_string[i] != "(" and given_string[i] != "*" and given_string[i] != "^" and given_string[i] != "+" and given_string[i+1] == "-":
            given_string = insert_char_at_index(given_string, "+", i+1)
            i += 1
        i += 1
    i=0
    while (i+1)!=len(given_string):
        if given_string[i] == "-" and (given_string[i+1] == "(" or is_char_cap_letter(given_string[i+1]) or is_char_small_letter(given_string[i+1])):
            given_string = insert_char_at_index(given_string, "*", i+1)
            given_string = insert_char_at_index(given_string, "1", i+1)
            i += 2
        i += 1
    return given_string
def merge_loop(equation):
    orig_equation = None
    while True:
        orig_equation = equation_node(equation.function_type, equation.children, equation.constant_value, equation.variable_name)
        orig_equation = duplicate_equation(equation)
        equation = merge_nested_bracket(equation)
        if compare_equation(orig_equation, equation) == True:
            break
    return equation
def math_parser_explicit_multiply_sign(given_string):
    i=0
    while (i+1) != len(given_string):
        if (given_string[i] == ")" or is_char_small_letter(given_string[i]) or is_char_number(given_string[i])) and (given_string[i+1] == "(" or is_char_small_letter(given_string[i+1]) or is_char_cap_letter(given_string[i+1])):
            given_string = insert_char_at_index(given_string, "*", i+1)
            i += 1
        i += 1
    return given_string
def math_parser_remove_first_bracket(given_string):
    if given_string[0] == "(" and find_corresponding_bracket(given_string, 0) == len(given_string)-1:
        given_string = remove_char_at_index(given_string, 0)
        given_string = remove_char_at_index(given_string, len(given_string)-1)
    return given_string

math_parser_explicit_bracket_bodmas__end = None
def math_parser_explicit_bracket_bodmas(given_string, start, operation):
    global math_parser_explicit_bracket_bodmas__end
    index = start
    list_operators = []
    while True:
        if given_string[index] == "+" or given_string[index] == "*" or given_string[index] == "/" or given_string[index] == "^":
            index = search_for_basic_operation_symbol(given_string, index+1)
        else:
            index = search_for_basic_operation_symbol(given_string, index)
        if index >= math_parser_explicit_bracket_bodmas__end:
            break
        else:
            list_operators.append(index)
    if len(list_operators) > 1:
        if operation == "^":
            i=len(list_operators)-1
            while i>=0:
                if given_string[list_operators[i]] == "^":
                    if i == len(list_operators)-1:
                        given_string = insert_char_at_index(given_string, ")", math_parser_explicit_bracket_bodmas__end)
                    else:
                        given_string = insert_char_at_index(given_string, ")", list_operators[i+1])
                    if i == 0:
                        given_string = insert_char_at_index(given_string, "(", start)
                    else:
                        given_string = insert_char_at_index(given_string, "(", list_operators[i-1]+1)
                    math_parser_explicit_bracket_bodmas__end += 2
                    return given_string
                i -=1
            return given_string
        if operation == "/":
            i=0
            while i<len(list_operators):
                if given_string[list_operators[i]] == "/":
                    if i == len(list_operators)-1:
                        given_string = insert_char_at_index(given_string, ")", math_parser_explicit_bracket_bodmas__end)
                    else:
                        given_string = insert_char_at_index(given_string, ")", list_operators[i+1])
                    if i==0:
                        given_string = insert_char_at_index(given_string, "(", start)
                    else:
                        given_string = insert_char_at_index(given_string, "(", list_operators[i-1]+1)
                    math_parser_explicit_bracket_bodmas__end += 2
                    return given_string
                i += 1
            return given_string
        if operation == "*":
            i=0
            plus_occur=0
            while i<len(list_operators):
                if given_string[list_operators[i]] == "+":
                    plus_occur = 1
                    break
                i += 1
            if plus_occur ==1:
                i=0
                while i<len(list_operators):
                    if given_string[list_operators[i]] == "*":
                        orig = i
                        while i<len(list_operators) and given_string[list_operators[i]]== "*":
                            i += 1
                        if i == len(list_operators):
                            given_string = insert_char_at_index(given_string, ")", math_parser_explicit_bracket_bodmas__end)
                        else:
                            given_string = insert_char_at_index(given_string, ")", list_operators[i])
                        if orig  == 0:
                            given_string = insert_char_at_index(given_string, "(", start)
                        else:
                            given_string = insert_char_at_index(given_string, "(", list_operators[orig-1]+1)
                        math_parser_explicit_bracket_bodmas__end += 2
                        break
                    i += 1
    return given_string
def math_parser_explicit_bracket_bodmas_apply(given_string, start):
    global math_parser_explicit_bracket_bodmas__end
    prev_string = given_string[start:math_parser_explicit_bracket_bodmas__end-start+1]
    bodmas_order = "^/*"
    i=0
    while i<len(bodmas_order):
        given_string = math_parser_explicit_bracket_bodmas(given_string, start, bodmas_order[i])
        while prev_string != given_string[start:math_parser_explicit_bracket_bodmas__end-start+1]:
            prev_string = given_string[start:math_parser_explicit_bracket_bodmas__end-start+1]
            given_string = math_parser_explicit_bracket_bodmas(given_string, start, bodmas_order[i])
        i += 1
    return given_string

def math_parser_string_to_equation(given_string, start, end):
    equation = equation_node(None, [])
    i = start
    while i<end:
        if not(is_char_small_letter(given_string[i])) and not(is_char_number(given_string[i])) and given_string[i] != "-":
            break
        if i == (end-1):
            if is_char_small_letter(given_string[start]):
                equation.function_type = TYPE_VARIABLE
                equation.variable_name = given_string[start]
            else:
                string_number = given_string[start:end]
                equation.function_type = TYPE_CONSTANT
                equation.constant_value = int(string_number)
            equation.children = []
            return duplicate_equation(equation)
        i += 1
    
    equation.children = []
    prev_index = start -1
    function_handle =0
    if search_for_basic_operation_symbol(given_string, start) >= end:
        function_handle = 1
    else:
        operation_extract = given_string[search_for_basic_operation_symbol(given_string, start)]
        if operation_extract == "+":
            equation.function_type = ADD
        elif operation_extract == "*":
            equation.function_type = MULTIPLY
        elif operation_extract == "^":
            equation.function_type = POWER
        elif operation_extract == "/":
            equation.function_type = DIVISION
        else:
            function_handle =1
    if function_handle==1 and is_char_cap_letter(given_string[start]):
        child = None
        equation.function_type = ord(given_string[start])-ord("A")
        child = math_parser_string_to_equation(given_string, start+2, end-1)
        equation.children.append(child)
        return duplicate_equation(equation)
    curr_index = None
    while True:
        
        curr_index = search_for_basic_operation_symbol(given_string, prev_index+1)
        child = None
        if curr_index >= end:
            if given_string[prev_index+1] == "(":
                child = math_parser_string_to_equation(given_string, prev_index+2, end-1)
            else:
                child = math_parser_string_to_equation(given_string, prev_index+1, end)
            equation.children.append(child)
            return duplicate_equation(equation)
        else:
            if given_string[prev_index+1] == "(":
                child = math_parser_string_to_equation(given_string, prev_index+2, curr_index-1)
            else:
                child = math_parser_string_to_equation(given_string, prev_index+1, curr_index)
        equation.children.append(child)
        prev_index = curr_index
    return duplicate_equation(equation)
def math_parser_recursive_bodmas(given_string, start, first_time):
    global math_parser_explicit_bracket_bodmas__end
    end = len(given_string) if first_time else find_corresponding_bracket(given_string, start-1)
    math_parser_explicit_bracket_bodmas__end = end
    given_string = math_parser_explicit_bracket_bodmas_apply(given_string, start)
    i=start
    while i<(len(given_string) if first_time else find_corresponding_bracket(given_string, start-1)):
        if given_string[i] == "(":
            given_string = math_parser_recursive_bodmas(given_string, i+1, False)
            i=  find_corresponding_bracket(given_string, i)
        i += 1
    return given_string
def math_parser_capital_letter(given_string):
    for i in range(4, len(function_full_name_list)):
        given_string = given_string.replace(function_full_name_list[i], chr(i-4+ord("A")))
    return given_string
def string_to_equation_short_helper(equation_string):
    given_string = ""
    given_string += equation_string
    given_string = math_parser_capital_letter(given_string)
    given_string = math_parser_explicit_multiply_sign(given_string)
    given_string = math_parser_no_subtraction(given_string)
    given_string = math_parser_remove_extra_bracket(given_string)
    given_string = math_parser_remove_first_bracket(given_string)
    given_string = math_parser_recursive_bodmas(given_string, 0, True)
    equation = math_parser_string_to_equation(given_string, 0, len(given_string))
    return equation

def pre_apply_formula(equation, equation_input, equation_output):
    global equation_iterate__index
    formula_input = string_to_equation_short_helper(equation_input)
    formula_output = string_to_equation_short_helper(equation_output)
    tmp_result = generate_equation_summit(equation, formula_input, formula_output)
    if tmp_result is not None:
        equation = tmp_result
    i=1
    while True:
        equation_iterate__index = i
        tmp = equation_iterate(equation, formula_input, formula_output)
        if tmp[0] == False:
            break
        else:
            equation = tmp[1]
        i += 1
    return equation
def convert_equation_to_string(equation, output_string):
    if equation.function_type == TYPE_VARIABLE:
        output_string += equation.variable_name
    elif equation.function_type == TYPE_CONSTANT:
        output_string += str(equation.constant_value)
    else:
        is_operation_basic = 0
        if equation.function_type < 0:
            is_operation_basic = 1
        if is_operation_basic == 0:
            output_string += function_full_name_list[equation.function_type+4]
        output_string += "("
        i=0
        while i<len(equation.children):
            output_string = convert_equation_to_string(equation.children[i], output_string)
            if i != len(equation.children)-1:
                if is_operation_basic == 0:
                    output_string += ","
                else:
                    if equation.function_type == ADD:
                        output_string += "+"
                    elif equation.function_type == MULTIPLY:
                        output_string += "*"
                    elif equation.function_type == DIVISION:
                        output_string += "/"
                    elif equation.function_type == POWER:
                        output_string += "^"
            i += 1
        output_string += ")"
    return output_string
def search_for_basic_operation_symbol(given_string, index):
    values = []
    values.append(999 if given_string.find("+", index)==-1 else given_string.find("+", index))
    values.append(999 if given_string.find("*", index)==-1 else given_string.find("*", index))
    values.append(999 if given_string.find("^", index)==-1 else given_string.find("^", index))
    values.append(999 if given_string.find("/", index)==-1 else given_string.find("/", index))
    min_index = values[0]
    i=1
    while i<4:
        if values[i] < min_index:
            min_index = values[i]
        i += 1
    if given_string.find("(", index) == -1:
        return min_index
    bracket_index = given_string.find("(", index)
    if bracket_index  < min_index and min_index < find_corresponding_bracket(given_string, bracket_index):
        return search_for_basic_operation_symbol(given_string, find_corresponding_bracket(given_string, bracket_index)+1)
    return min_index
def find_corresponding_bracket(given_string, index):
    count = 1
    index += 1
    while True:
        a = given_string.find("(", index)
        b = given_string.find(")", index)
        if a != -1 and b != -1:
            index = min(a, b)
        elif  a != -1:
            index = a
        elif b != -1:
            index = b
        if given_string[index] == "(":
            count += 1
        elif given_string[index] == ")":
            count -= 1
        if count == 0:
            return index
        index += 1
def compare_equation(equation_1, equation_2):
    if equation_1.function_type != equation_2.function_type:
        return False
    if equation_1.function_type <= TYPE_VARIABLE:
        equation_1.children = []
        equation_2.children = []
    if len(equation_1.children) != len(equation_2.children):
        return False
    if equation_1.function_type == TYPE_VARIABLE and equation_1.variable_name != equation_2.variable_name:
        return False
    if equation_1.function_type == TYPE_CONSTANT and equation_1.constant_value != equation_2.constant_value:
        return False
    for i in range(len(equation_1.children)):
        if compare_equation(equation_1.children[i], equation_2.children[i]) == False:
            return False
    return True
def string_to_equation_short(given_string):
    equation = string_to_equation_short_helper(given_string)
    '''
    equation = pre_apply_formula(equation, "a/b", "ab^-1")
    equation = pre_apply_formula(equation, "a*1", "a")
    equation = pre_apply_formula(equation, "1*a", "a")
    equation = pre_apply_formula(equation, "a^1", "a")
    '''
    equation = merge_loop(equation)
    return equation
generate_equation_summit__variable_list = []
def apply_formula(equation, formula):
    global generate_equation_summit__variable_list
    if formula.function_type == TYPE_VARIABLE:
        if is_char_calculus(formula.variable_name):
            if equation.function_type == TYPE_VARIABLE and is_char_calculus(equation.variable_name):
                if generate_equation_summit__variable_list[ord(formula.variable_name)-ord("a")] == None:
                    generate_equation_summit__variable_list[ord(formula.variable_name)-ord("a")] = duplicate_equation(equation)
                elif generate_equation_summit__variable_list[ord(formula.variable_name)-ord("a")].function_type == TYPE_VARIABLE and generate_equation_summit__variable_list[ord(formula.variable_name)-ord("a")].variable_name == equation.variable_name:
                    return True
                else:
                    return False
            else:
                return False
        if formula.variable_name == "k" and is_contain_m(equation)==True:
            return False
        if formula.variable_name == "t" and equation.function_type != TYPE_CONSTANT:
            return False
        if generate_equation_summit__variable_list[ord(formula.variable_name)-ord("a")] is None:
            generate_equation_summit__variable_list[ord(formula.variable_name)-ord("a")] = duplicate_equation(equation)
        elif compare_equation(generate_equation_summit__variable_list[ord(formula.variable_name)-ord("a")], equation)==False:
            return False
        return True
    
    if equation.function_type != formula.function_type:
        return False
    if len(equation.children) != len(formula.children):
        return False
    if formula.function_type == TYPE_CONSTANT and formula.constant_value != equation.constant_value:
        return False
    for i in range(len(equation.children)):
        if apply_formula(equation.children[i], formula.children[i]) == False:
            return False
    return True
def replace_variables(equation):
    global generate_equation_summit__variable_list
    for i in range(len(equation.children)):
        child = None
        if equation.children[i].function_type == TYPE_VARIABLE:
            child = generate_equation_summit__variable_list[ord(equation.children[i].variable_name) - ord('a')]
            equation.children[i] = child
            continue
        equation.children[i] = replace_variables(equation.children[i])
    return equation
def print_equation(equation):
    print(convert_equation_to_string(equation, ""))
def generate_equation_summit(equation, formula_input, formula_output):
    global generate_equation_summit__variable_list
    if equation.function_type == TYPE_CONSTANT or equation.function_type == TYPE_VARIABLE:
        return None
    generate_equation_summit__variable_list = []
    for i in range(26):
        generate_equation_summit__variable_list.append(None)
    if len(equation.children) == len(formula_input.children):
        if apply_formula(equation, formula_input)==True:
            new_instance = None
            if formula_output.function_type == TYPE_VARIABLE:
                new_instance = duplicate_equation(generate_equation_summit__variable_list[ord(formula_output.variable_name) - ord("a")])
            else:
                new_instance = duplicate_equation(formula_output)
                new_instance = replace_variables(new_instance)
            return duplicate_equation(new_instance)
    elif len(equation.children) > len(formula_input.children) and len(formula_input.children) >= 2:
        new_instance = duplicate_equation(equation)
        new_instance_2 = duplicate_equation(equation)
        new_instance.children = new_instance.children[:len(formula_input.children)]
        new_instance_2.children = new_instance_2.children[len(formula_input.children):]
        if apply_formula(new_instance, formula_input) == True:
            #print("allowed")
            sub_node = None
            if formula_output.function_type == TYPE_VARIABLE:
                sub_node = duplicate_equation(generate_equation_summit__variable_list[ord(formula_output.variable_name) - ord("a")])
            else:
                sub_node = duplicate_equation(formula_output)
                sub_node = replace_variables(sub_node)
                #print("below")
                #print_equation(sub_node)
            new_instance_2.children.append(sub_node)
            return duplicate_equation(new_instance_2)
    return None
def is_contain_m(equation):
    if equation.function_type == TYPE_VARIABLE and is_char_calculus(equation.variable_name):
        return True
    if equation.function_type == TYPE_VARIABLE or equation.function_type == TYPE_CONSTANT:
        return False
    for i in range(len(equation.children)):
        if is_contain_m(equation.children[i]):
            return True
    return False
def is_contain_inf(equation):
    if equation.function_type == TYPE_VARIABLE or equation.function_type == TYPE_CONSTANT:
        return False
    if equation.function_type == POWER and equation.children[0].function_type == INF:
        return True
    for i in range(len(equation.children)):
        if is_contain_inf(equation.children[i]):
            return True
    return False
def is_contain_dif(equation):
    if equation.function_type == DIF:
        return True
    if equation.function_type == TYPE_VARIABLE or equation.function_type == TYPE_CONSTANT:
        return False
    for i in range(len(equation.children)):
        if is_contain_dif(equation.children[i]):
            return True
    return False
def compute_constant(equation):
    if equation.function_type == TYPE_CONSTANT or equation.function_type == TYPE_VARIABLE:
        return None
    if len(equation.children) < 2 or equation.children[0].function_type != TYPE_CONSTANT or equation.children[1].function_type != TYPE_CONSTANT:
        return None
    term_1 = equation.children[0].constant_value
    term_2 = equation.children[1].constant_value
    result = 0
    if equation.function_type == ADD:
        result = term_1 + term_2
    elif equation.function_type == MULTIPLY:
        result = term_1 * term_2
    else:
        return None
    final_equation = duplicate_equation(equation)
    if len(equation.children) == 2:
        final_equation.function_type = TYPE_CONSTANT
        final_equation.constant_value = result
    elif len(equation.children) > 2:
        final_equation.children.pop(0)
        final_equation.children.pop(0)
        final_equation.children.append(equation_node(TYPE_CONSTANT, [], result, None))
    return duplicate_equation(final_equation)
equation_iterate__index = None
def equation_iterate(equation, formula_input, formula_output):
    global equation_iterate__index
    if equation.function_type == TYPE_CONSTANT or equation.function_type == TYPE_VARIABLE:
        return [False, duplicate_equation(equation)]
    for i in range(len(equation.children)):
        equation_iterate__index -=1
        if equation_iterate__index == 0:
            if formula_input is None:
                curr = compute_constant(equation.children[i])
                if curr is not None:
                    equation.children[i] = curr
            else:
                curr = generate_equation_summit(equation.children[i], duplicate_equation(formula_input), duplicate_equation(formula_output))
                if curr is not None:
                    equation.children[i] = curr
            return [True, duplicate_equation(equation)]
        tmp_result = equation_iterate(equation.children[i], duplicate_equation(formula_input), duplicate_equation(formula_output))
        if tmp_result[0] == True:
            equation.children[i] = tmp_result[1]
            return [True, duplicate_equation(equation)]
    return [False, duplicate_equation(equation)]
def replace_instances(equation, instance_find, instance_replace):
    
    if compare_equation(equation, instance_find)==True:
        return duplicate_equation(instance_replace)
    
    if equation.function_type == TYPE_VARIABLE or equation.function_type == TYPE_CONSTANT:
        return equation
    for i in range(len(equation.children)):
        equation.children[i] = replace_instances(equation.children[i], instance_find, instance_replace)
    return equation
possible_equation_integration_substitute__attempt_list = []
def check_inf_structure(equation):
    if equation.function_type == INF:
        return True
    elif equation.function_type == MULTIPLY:
        if len(equation.children) != 2:
            return False
        for i in range(len(equation.children)):
            if equation.children[i].function_type != INF:
                return False
        return True
    elif equation.function_type == ADD:
        if equation.children[0].function_type != INF:
            return False
        for i in range(1, len(equation.children)):
            if equation.children[i].function_type != MULTIPLY or check_inf_structure(equation.children[i])==False:
                return False
        return True
    return False
def possible_equation_integration_substitute_2(equation):
    global possible_equation_integration_substitute__attempt_list
    if equation.function_type == TYPE_CONSTANT:
        return None
    possible_equation_integration_substitute__attempt_list.append(duplicate_equation(equation))
    if equation.function_type == TYPE_VARIABLE:
        return None
    for i in range(len(equation.children)):
        possible_equation_integration_substitute_2(equation.children[i])
    return None
def possible_equation_integration_substitute(equation, is_inside_int):
    global possible_equation_integration_substitute__attempt_list
    if equation.function_type == TYPE_VARIABLE or equation.function_type == TYPE_CONSTANT:
        return None
    if is_inside_int:
        possible_equation_integration_substitute__attempt_list.append(duplicate_equation(equation))
    for i in range(len(equation.children)):
        possible_equation_integration_substitute(equation.children[i], True if is_inside_int else (equation.function_type == INTEGRATION))
    return None
equation_iterate_swap__index = None
def equation_iterate_swap(equation):
    global equation_iterate_swap__index
    if equation.function_type == TYPE_CONSTANT or equation.function_type == TYPE_VARIABLE:
        return [False, duplicate_equation(equation)]
    if equation.function_type != POWER:
        
        for i in range(1, len(equation.children)):
            equation_iterate_swap__index -=1
            if equation_iterate_swap__index ==0:
                tmp = duplicate_equation(equation)
                tmp_var = tmp.children[0]
                tmp.children[0] = tmp.children[i]
                tmp.children[i] = tmp_var
                return [True, duplicate_equation(tmp)]
    for i in range(len(equation.children)):
        result = equation_iterate_swap(equation.children[i])
        tmp = duplicate_equation(equation)
        if result[1] is not None:
            tmp.children[i] = result[1]
        if result[0] == True:
            return [True, tmp]
    return [False, duplicate_equation(equation)]
next_letter__best = "m"
def next_letter(equation):
    global next_letter__best
    if equation.function_type == TYPE_VARIABLE:
        if ord(next_letter__best)<ord(equation.variable_name):
            next_letter__best = equation.variable_name
        return None
    if equation.function_type == TYPE_CONSTANT:
        return None
    for i in range(len(equation.children)):
        next_letter(equation.children[i])
    return None
class math_step:
    def __init__(self, parent, data, index):
        self.parent = parent
        self.data = data
        self.index = index
generate_equation__equation_list = []
def apply_formula_2(equation, input_string):
    global generate_equation_summit__variable_list
    generate_equation_summit__variable_list = []
    return apply_formula(duplicate_equation(equation), string_to_equation_short(input_string))
def generate_equation(equation, formula_input_list, formula_output_list):
    global equation_iterate__index
    global generate_equation__equation_list
    global possible_equation_integration_substitute__attempt_list
    global next_letter__best
    global generate_equation_summit__variable_list
    next_letter(equation)
    for i in range(len(formula_input_list)):
        
        curr_formula_input = string_to_equation_short(formula_input_list[i])
        curr_formula_output = string_to_equation_short(formula_output_list[i])
        curr = duplicate_equation(equation)
        curr = generate_equation_summit(curr, duplicate_equation(curr_formula_input), duplicate_equation(curr_formula_output))
        if curr is not None and compare_equation(curr, equation)==False:
            generate_equation__equation_list.append(curr)
        j=1
        while True:
            equation_iterate__index = j
            curr = duplicate_equation(equation)
            result = equation_iterate(duplicate_equation(curr), duplicate_equation(curr_formula_input), duplicate_equation(curr_formula_output))
            if result[0] == False:
                break
            else:
                curr = result[1]
            if compare_equation(curr, equation)==False:
                generate_equation__equation_list.append(curr)
            j += 1
    curr = duplicate_equation(equation)
    curr = compute_constant(curr)
    if curr is not None and compare_equation(curr, equation)==False:
        generate_equation__equation_list.append(curr)
    j=1
    while True:
        equation_iterate__index = j
        curr = duplicate_equation(equation)
        result = equation_iterate(curr, None, None)
        if result[0] == False:
            break
        else:
            curr = result[1]
        if compare_equation(curr, equation)==False:
            generate_equation__equation_list.append(curr)
        j += 1
    '''
    if len(equation.children)>2 and equation.function_type == ADD:
        new_equation = duplicate_equation(equation)
        tmp = duplicate_equation(equation)
        new_equation.children = new_equation.children[0:2]
        result = replace_inf(new_equation)
        tmp.children.pop(0)
        tmp.children.pop(0)
        tmp.children = result.children + tmp.children
        generate_equation__equation_list.append(tmp)
    '''
    #substitution
    if len(equation.children) > 2 and equation.function_type == ADD:
        new_equation = duplicate_equation(equation)
        new_equation.children.pop(0)
        new_equation.children[0].children.pop(0)
        new_equation.children = new_equation.children[0:2]
        result = replace_inf(new_equation)
        tmp = duplicate_equation(equation)
        tmp.children[1].children[1] = result.children[0]
        #tmp = merge_loop(tmp)
        generate_equation__equation_list.append(tmp)
    if len(equation.children) == 2 and equation.function_type == ADD:
        result = replace_inf(duplicate_equation(equation))
        if compare_equation(result, equation)==False:
            generate_equation__equation_list.append(result)
    elif len(equation.children)>2 and equation.function_type == ADD:
        new_equation = duplicate_equation(equation)
        new_equation.children = new_equation.children[0:2]
        result = replace_inf(new_equation)
        tmp = duplicate_equation(equation)
        tmp.children[0] = result.children[0]
        if compare_equation(tmp, equation)==False:
            generate_equation__equation_list.append(tmp)
    if len(equation.children) == 1 and equation.function_type == INF:
        some_list = substitute_equation(equation, chr(ord(next_letter__best)+1), False)
        for i in range(len(some_list)):
            tmp = duplicate_equation(some_list[i])
            if compare_equation(equation, tmp)==False:
                generate_equation__equation_list.append(tmp)
    if len(equation.children) >= 2 and equation.function_type == ADD:
        new_equation = duplicate_equation(equation.children[0])
        some_list = substitute_equation(new_equation, chr(ord(next_letter__best)+1), False)
        for i in range(len(some_list)):
            tmp = duplicate_equation(some_list[i])
            tmp_2 = duplicate_equation(equation)
            tmp_2.children.append(tmp.children[1])
            if compare_equation(equation, tmp_2)==False:
                generate_equation__equation_list.append(tmp_2)
    if len(equation.children) == 1 and equation.function_type == INF and equation.children[0].function_type == INTEGRATION:
        some_list = multiply_up_and_down(equation.children[0].children[0])
        for i in range(len(some_list)):
            tmp = duplicate_equation(equation)
            tmp.children[0].children[0] = some_list[i]
            generate_equation__equation_list.append(tmp)
    if len(equation.children) >=2 and equation.children[0].children[0].function_type == INTEGRATION:
        some_list = multiply_up_and_down(equation.children[0].children[0].children[0])
        for i in range(len(some_list)):
            tmp = duplicate_equation(equation)
            tmp.children[0].children[0].children[0] = some_list[i]
            generate_equation__equation_list.append(tmp)

    '''
    if len(equation.children) >= 2 and equation.children[1].children[0].children[0].function_type == INTEGRATION:
        some_list = multiply_up_and_down(equation.children[1].children[0].children[0])
        for i in range(len(some_list)):
            tmp = duplicate_equation(equation)
            tmp.children[1].children[0].children[0] = some_list[i]
            generate_equation__equation_list.append(tmp)
    '''
    remove_duplicate()
def equation_length_cmp(item1, item2):
    x1 = len(convert_equation_to_string(item1, ""))
    x2 = len(convert_equation_to_string(item2, ""))
    if x1 < x2:
        return -1
    elif x1 > x2:
        return 1
    else:
        return 0
def multiply_up_and_down(equation):
    global possible_equation_integration_substitute__attempt_list
    possible_equation_integration_substitute__attempt_list = []
    possible_equation_integration_substitute_2(equation)
    i=0
    while i<len(possible_equation_integration_substitute__attempt_list):
        if is_contain_dif(possible_equation_integration_substitute__attempt_list[i]):
            possible_equation_integration_substitute__attempt_list.pop(i)
            continue
        i += 1
    sorted(possible_equation_integration_substitute__attempt_list, key=cmp_to_key(equation_length_cmp))
    possible_equation_integration_substitute__attempt_list.pop(0)
    some_list = []
    for i in range(len(possible_equation_integration_substitute__attempt_list)):
        tmp = duplicate_equation(possible_equation_integration_substitute__attempt_list[i])
        tmp_2 = equation_node(MULTIPLY, [duplicate_equation(tmp), equation_node(POWER, [duplicate_equation(tmp), equation_node(TYPE_CONSTANT, [], -1)])])
        possible_equation_integration_substitute__attempt_list[i] = tmp_2
        tmp_3 = duplicate_equation(equation)
        tmp_3 = equation_node(MULTIPLY, [tmp_2, tmp_3])
        tmp_3 = merge_loop(tmp_3)
        some_list.append(tmp_3)
        i += 1
    return some_list
    
def substitute_equation(equation, letter, is_inside_int):
    global possible_equation_integration_substitute__attempt_list
    global next_letter__best
    possible_equation_integration_substitute__attempt_list = []
    possible_equation_integration_substitute(equation, is_inside_int)
    orig = next_letter__best
    next_letter__best = "m"
    next_letter(equation)
    #possible_equation_integration_substitute__attempt_list.append(equation_node(POWER, [equation_node(TYPE_VARIABLE, [], None, next_letter__best), equation_node(TYPE_CONSTANT, [], -1, None)]))
    next_letter__best = orig
    some_list = []
    for i in range(len(possible_equation_integration_substitute__attempt_list)):
        edit_1 = equation_node(ADD, [])
        edit_1.children.append(duplicate_equation(equation))
        edit_2 = equation_node(TYPE_VARIABLE, [], None, letter)
        edit_3 = equation_node(INF, [])
        edit_3.children.append(edit_2)
        edit_4 = equation_node(MULTIPLY, [])
        edit_4.children.append(edit_3)
        edit_5 = equation_node(INF, [])
        edit_5.children.append(duplicate_equation(possible_equation_integration_substitute__attempt_list[i]))
        edit_4.children.append(edit_5)
        edit_1.children.append(edit_4)
        tmp = duplicate_equation(edit_1)
        some_list.append(tmp)
    return some_list     
def replace_inf(equation):
    if equation.function_type == ADD:
        tmp = duplicate_equation(equation.children[0])
        tmp = replace_instances(tmp, equation.children[1].children[0].children[0], equation.children[1].children[1].children[0])
        tmp_2 = duplicate_equation(equation)
        tmp_2.children[0]= tmp
        return tmp_2

add_swap__equation_list = []
def add_swap(extra_eq, equation):
    global add_swap__equation_list
    global equation_iterate_swap__index
    global generate_equation__equation_list
    j=1
    while True:
        curr = duplicate_equation(equation)
        equation_iterate_swap__index = j
        result = equation_iterate_swap(curr)
        if result[0] == False:
            break
        else:
            curr = result[1]
        if compare_equation(curr, extra_eq)==False and compare_equation(curr, equation)==False  and find_in_equation_list(curr)==False:
            #print_equation(curr)
            generate_equation__equation_list.append(curr)
        j += 1
def find_in_equation_list(equation):
    global add_swap__equation_list
    for i in range(len(add_swap__equation_list)):
        if compare_equation(add_swap__equation_list[i], equation)==True:
            return True
    return False
def compare_equation_commutative(equation, equation_2):
    global add_swap__equation_list
    add_swap__equation_list = []
    add_swap(equation, equation)
    i=0
    while i<len(add_swap__equation_list):
        add_swap(equation, add_swap__equation_list[i])
        i += 1
    add_swap__equation_list.append(duplicate_equation(equation))
    return find_in_equation_list(equation_2)
def remove_duplicate():
    global generate_equation__equation_list
    new_size = len(generate_equation__equation_list)
    i=0
    while i<new_size:
        j=i+1
        while j<new_size:
            if compare_equation(generate_equation__equation_list[i], generate_equation__equation_list[j]):
                k=j
                while k<new_size-1:
                    generate_equation__equation_list[k]=generate_equation__equation_list[k+1]
                    k += 1
                new_size -=1
            else:
                j += 1
        i += 1
    generate_equation__equation_list = generate_equation__equation_list[:new_size]
def caller_add_swap(equation, formula_input_list, formula_output_list):
    global generate_equation__equation_list
    global add_swap__equation_list
    add_swap__equation_list = []
    add_swap(equation, equation)
    i=0
    while i<len(add_swap__equation_list):
        add_swap(equation, add_swap__equation_list[i])
        i += 1
    add_swap__equation_list.append(equation)
    for i in range(len(add_swap__equation_list)):
        generate_equation(add_swap__equation_list[i], formula_input_list, formula_output_list)
    remove_duplicate()
def formula_function():
    fl ="1^a=1, inf(a)+b+c+d=inf(a)+c+b+d, inf(a)+b+c+d=inf(a)+d+c+b, inf(a)+b+c+d=inf(a)+b+d+c, inf(a*b^-1)+c=inf(a*b^-1)+inf(a)inf(a)+inf(b)inf(b)+c, inf(a*b^-1)+inf(a)inf(0)+inf(b)inf(0)+c=inf(dif(a)*dif(b)^-1)+c, int(k*a*b*dif(m))=k*int(a*b*dif(m)), int(k*a*dif(m))=k*int(a*dif(m)), inf(a)+b+c+d=inf(a)+d+c, inf(d)+inf(bghija)inf(c)+e+f=inf(d)+inf(a)inf(c*b^-1*g^-1*h^-1*i^-1*j^-1)+e+f, dif(k*a)=k*dif(a), a^((2^-1)-1)=(a^(2^-1))^-1, dif(a^k)=k*a^(k-1)*dif(a), -1^-1=-1, a^0=1, aa^b=a^(b+1), a^b=(a^-1)^(-1*b), ((a)^(2^-1)*b)=(a*(b^2))^(2^-1), inf(a)+inf(b)inf(c)+inf(d)inf(e)=inf(a)+inf(b)inf(c), inf(a)+inf(b)inf(c)+d=inf(a)+inf(b^-1)inf(c^-1)+d, inf(a)+b+inf(c)inf(d)=inf(a)+inf(c)inf(d)+b, inf(c)+inf(a)inf(b)+d=inf(c)+inf(a^2)inf(b^2)+d, inf(d)+inf(abe)inf(c)+f=inf(d)+inf(ae)inf(cb^(-1))+f, 1*a=a, (a^b)^c = a^(bc), inf(d)+inf(ab)inf(c)+e=inf(d)+inf(a)inf(cb^(-1))+e, inf(d)+inf(a+b)inf(c)+f=inf(d)+inf(a)inf(c-b)+f, inf(c)+inf(a)inf(b)=inf(c)+inf(a)inf(b)+inf(a)inf(b), inf(c)+inf(a)inf(b)+d=inf(c)+inf(a)inf(b)+inf(a)inf(b)+d, aa=a^2, 1^-1=1, inf(a)inf(b)=inf(dif(a))inf(dif(b)), abb=ab^2, a^b+a^c=a^(b+c), (a^b)*a^(-1*b)=a, dif(a^t)=t*a^(t-1)*dif(a), dif(t)=0, dif(a+b)=dif(a)+dif(b), dif(cos(a))=-sin(a)dif(a), dif(sin(a))=cos(a)dif(a), int(sin(m)dif(m))=-cos(m), a+0=a, ab+abc=ab(c+1), 0*a=0, int(k dif(m))=km, int(m*dif(m))=(m^2)(2^-1), int((a+b)dif(m))=int(a dif(m))+int(b dif(m)), int(ka dif(m))=kint(a dif(m)), a+a=2a, int(m^a dif(m))=(m^(a+1))*((a+1)^(-1)), a^1=a, a^b*a^c=a^(b+c), (ab)^c=a^c*b^c, aa^(-1)=1, 1-a^2=(1+a)(1-a), sin(a)^2=1-cos(a)^2, ab+ac=a(b+c), a+cba=a(bc+1), ab+ac=a(b+c), (a+b)(c+d)=ac+ad+bc+bd, a(b+c)=ab+ac, sin(a)^2+cos(a)^2=1, (a-b)^2=a^2-2ab+b^2, csc(a)=sin(a)^(-1), cot(a)=cos(a)sin(a)^(-1)"
    #fl = "aa=a^2, a+a=2a"
    fl = fl.replace(" ", "")
    fl = fl.replace(",", "=")
    arr = fl.split("=")
    f_i=[]
    f_o=[]
    for i in range(0, len(arr), 2):
        f_i.append(arr[i])
        f_o.append(arr[i+1])
    return [f_i, f_o]
def find_in_tt(equation):
    global search__transposition_table
    for i in range(len(search__transposition_table)):
        if compare_equation(search__transposition_table[i], equation)==True:
            return True
    return False
def make_list(f_input, f_output, equation):
    global generate_equation__equation_list
    generate_equation__equation_list = []
    '''
    equation = pre_apply_formula(equation, "a/b", "ab^-1")
    equation = pre_apply_formula(equation, "a*1", "a")
    equation = pre_apply_formula(equation, "1*a", "a")
    equation = pre_apply_formula(equation, "a^1", "a")
    equation = merge_loop(equation)
    '''
    generate_equation(equation, f_input, f_output)
    add_swap(equation, equation)
    i=0
    count = 0
    while i < len(generate_equation__equation_list):
        generate_equation__equation_list[i] = merge_loop(generate_equation__equation_list[i])
        if check_inf_structure(generate_equation__equation_list[i])==False:
            generate_equation__equation_list.pop(i)
            continue
        
        print(str(count)+". ", sep="", end="")
        print_equation(generate_equation__equation_list[i])
        count += 1
        i += 1
def is_contain_int(equation):
    if equation.function_type == INTEGRATION:
        return True
    if equation.function_type == TYPE_VARIABLE or equation.function_type == TYPE_CONSTANT:
        return False
    for i in range(len(equation.children)):
        if is_contain_int(equation.children[i]) == True:
            return True
    return False
step_table = []
search__transposition_table = []
'''
def search(f_input, f_output, equation):
    global step_table
    global generate_equation__equation_list
    global search__transposition_table
    step = math_step(-1, duplicate_equation(equation), 0)
    step_table.append(step)
    q = []
    q.append(step)
    while len(q)>0:
        v = q.pop(0)
        #print_equation(v.data)
        if is_contain_int(v.data)==False:
            return v
        make_list(f_input, f_output, duplicate_equation(v.data))
        return None
        for i in range(len(generate_equation__equation_list)):
            w = duplicate_equation(generate_equation__equation_list[i])
            if find_in_tt(w) == False:
                search__transposition_table.append(duplicate_equation(w))
                new_step = math_step(v.index, duplicate_equation(w), len(step_table))
                step_table.append(new_step)
                #q.insert(0, new_step)
                q.append(new_step)
    return None
'''

fl = formula_function()
eq = string_to_equation_short("inf(((m+1)^2-1)*m^-1)+inf(m)inf(0)")
#eq = string_to_equation_short("inf(m-22*7^-1)+inf(m)inf(22*7^-1)")
#eq = string_to_equation_short("(inf((-1*(a^-2)*(o^3)*(3^-1)))+(inf(m)*inf((n^-1)))+(inf(((((n^2)*(a^2))+-1)^(2^-1)))*inf(o)))")
while True:
    print(check_inf_structure(eq))
    print_equation(eq)
    make_list(fl[0], fl[1], eq)
    eq = generate_equation__equation_list[int(input("select: "))]

'''
output = search(fl[0], fl[1], eq)
print("solution:")
index = output.index
while index != -1:
    print_equation(step_table[index].data)
    index = step_table[index].parent
'''
