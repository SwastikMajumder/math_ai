import math
import itertools
class EqList:
    def __init__(self):
        self.val = {}
        self.pair_eq = {}
        self.equation_list = []
    @staticmethod
    def fix_key(a, b):
        for key in a.keys():
            if key not in b.keys():
                b[key] = 0
        for key in b.keys():
            if key not in a.keys():
                a[key] = 0
        return a, b

    @staticmethod
    def valid(equation):
        return not all(equation[key] == 0 for key in equation.keys())
    
    @staticmethod
    def eq(a, b):
        a, b = EqList.fix_key(a, b)
        return set(a.keys())==set(b.keys()) and all(a[key] == b[key] for key in a.keys())
    
    @staticmethod
    def add(a, b):
        a, b = EqList.fix_key(a, b)
        equation2 = {}
        for key in a.keys():
            equation2[key] = a[key]+b[key]
        return equation2
    
    @staticmethod
    def sub(a, b):
        a, b = EqList.fix_key(a, b)
        equation2 = {}
        for key in a.keys():
            equation2[key] = a[key]-b[key]
        return equation2
    
    @staticmethod
    def cancel(equation):
        common = math.gcd(*[equation[x] for x in equation.keys()])
        equation2 = {}
        for item in equation.keys():
            equation2[item] = int(equation[item]/common)
        return equation2
    
    def add_equation2(self, equation, depth):
        if any(x != "=" and abs(equation[x])>2 for x in equation.keys()):
            return
        if not EqList.valid(equation) or depth == 0:
            return
        equation = EqList.cancel(equation)
        if any(EqList.eq(x, equation) for x in self.equation_list+self.newlist):
            return
        for item in self.equation_list:
            self.add_equation2(EqList.add(item, equation), depth-1)
            self.add_equation2(EqList.sub(item, equation), depth-1)
            self.add_equation2(EqList.sub(equation, item), depth-1)
        if sum([y for x,y in equation.items() if x != "="]) == 1 and [y for x,y in equation.items() if x != "="].count(0) == len(equation.keys())-2:
            if all(not EqList.eq(x, equation) for x in self.equation_list):
                self.equation_list.append(equation)
            self.val[[x for x,y in equation.items() if x != "=" and y == 1][0]] = equation["="]
        elif sum([y for x,y in equation.items() if x != "="]) == 0 and [y for x,y in equation.items() if x != "="].count(0) == len(equation.keys())-3 and equation["="]==0:
            if all(not EqList.eq(x, equation) for x in self.equation_list):
                self.equation_list.append(equation)
            self.pair_eq[tuple(sorted([x for x,y in equation.items() if x != "=" and abs(y) == 1]))] = equation["="]
        self.newlist.append(equation)

    def add_equation(self, equation):
        self.newlist = []
        self.add_equation2(equation, 3)
        if all(not EqList.eq(x, equation) for x in self.equation_list):
            self.equation_list.append(equation)
        
