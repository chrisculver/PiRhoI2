import sympy as sp

spP1 = sp.MatrixSymbol('p1',3,1)
spP2 = sp.MatrixSymbol('p2',3,1)
spP3 = sp.MatrixSymbol('p3',3,1)

def twomomenta_subs(mom):
    return {spP1: mom[0], spP2: mom[1]}

def threemomenta_subs(mom):
    return {spP1: mom[0], spP2: mom[1], spP3: mom[2]}

# creates a list of unique functions that occur in a sympy expression
def unique_functions(expr):
    unique_functions=[]
    for a in sp.preorder_traversal(expr):
        if type(type(a))==sp.core.function.UndefinedFunction and a not in unique_functions:
            unique_functions.append(a)
    return unique_functions

# creates a list of unique arguments that occur in a functions of a sympy expression
def unique_arguments(expr):
    unique_arguments = []
    for a in unique_functions(expr):
        if(a.args[0] not in unique_arguments):
            unique_arguments.append(a.args[0])
    return unique_arguments


def unique_append(lst, elem):
    if(elem not in lst):
        lst.append(elem)

def unique_mesons(ops):
    res=[]
    for o in ops:
        new_mesons=o.atoms(sp.Function)
        for elem in new_mesons:
            unique_append(res,elem)

    return res

def get_elementals(ops):
    res=[]
    mesons=unique_mesons(ops)
    #print(mesons)
    for o in ops:
        polyOp=sp.Poly(o, *mesons)
        newElems = [sp.prod(x**k for x,k in zip(polyOp.gens,mon)) for mon in polyOp.monoms()]
        for elem in newElems:
            unique_append(res,elem)

    return res

vecRotation = sp.MatrixSymbol('R',3,3)
detR = sp.Symbol('det(R)')

def rotation_group_subs(g):
    return {vecRotation: sp.Matrix(g.rotation),
            detR: g.identifier['parity']}


def prep_threepi_rotations(expr):
    args=unique_arguments(expr)
    momentaRotations = {p: vecRotation*p for p in args}
    return detR*expr.subs(momentaRotations)

def prep_sigmapi_rotations(expr):
    args=unique_arguments(expr)
    momentaRotations = {p: vecRotation*p for p in args}
    return detR*expr.subs(momentaRotations)

def prep_rhopi_rotation1(expr):
    return detR*vecRotation*expr

def prep_rhopi_rotation2(expr):
    args=unique_arguments(expr)
    momentaRotations = {p: vecRotation*p for p in args}
    return expr.subs(momentaRotations)

# creates a list to replace a vector of three momenta with symbols for the momenta.  
def momentumSubs(ops):
    momentumSubs={}
    idx=0
    for o in ops:
        for p in unique_arguments(o):
            if p not in momentumSubs:
                momentumSubs[p]=sp.Symbol(str(int(p[0]))+str(int(p[1]))+str(int(p[2])))
                idx+=1
    return momentumSubs

