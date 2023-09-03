import sympy as sp
from sympy.physics.quantum.cg import CG

u,d,baru,bard = sp.symbols('u d baru bard', commutative=False)
pp=sp.symbols('p')
pseudoScalars = {bard*u: sp.Function('pi^+')(pp), baru*d: -sp.Function('pi^-')(pp),
        -baru*u+bard*d: sp.sqrt(2)*sp.Function('pi^0')(pp),
        baru*u+bard*d: sp.sqrt(2)*sp.Function('eta')(pp)}

scalars = {bard*u: sp.Function('X')(pp), baru*d: -sp.Function('X')(pp),
        -baru*u+bard*d: sp.sqrt(2)*sp.Function('X')(pp),
        baru*u+bard*d: sp.sqrt(2)*sp.Function('sigma')(pp)}

vectors = {bard*u: sp.Function('rho_i^+')(pp), baru*d: -sp.Function('rho_i^-')(pp),
        -baru*u+bard*d: sp.sqrt(2)*sp.Function('rho_i^0')(pp),
        baru*u+bard*d: sp.sqrt(2)*sp.Function('X')(pp)}


class MesonData: 
  def __init__(self, Type, p, i, i3="all"):
    self.type = Type
    self.i = i
    self.i3 = i3
    self.p = p

def quark(j,m):
  if (j==0.5) and (m==0.5):
    return sp.Symbol('u',commutative=False)
  elif (j==0.5) and (m==-0.5):
    return sp.Symbol('d',commutative=False)
  else:
    print('Fatal Error')


def antiquark(j,m):
  if(j==0.5) and (m==0.5):
    return sp.Symbol('bard',commutative=False)
  elif (j==0.5) and (m==-0.5):
    return -sp.Symbol('baru',commutative=False)
  else:
    print('Fatal Error')


def Meson(meson):
  iQ=0.5 #isospin of a quark
  total = 0
  for i1 in [-iQ,iQ]:
    for i2 in [-iQ,iQ]:
      if((i1+i2)==meson.i3):
        coef = CG(iQ,i1,iQ,i2,meson.i,meson.i3).doit()
        total+=coef*antiquark(iQ,i1)*quark(iQ,i2)
  return sp.simplify(total).subs(meson.type).subs({pp: meson.p})


def twoMesons(IS,I3,m1,m2):
  m1i3init = m1.i3
  m2i3init = m2.i3

  if( m1.i3!="all" or m2.i3!="all"):
    print("Warning: Ignoring specification of i3 in twoMesons")
  total = 0
  for i1 in [i for i in range(-m1.i,m1.i+1,1)]:
    for i2 in [i for i in range(-m2.i,m2.i+1,1)]:
      if(i1+i2==I3):
        m1.i3=i1
        m2.i3=i2
        coef = CG(m1.i,i1,m2.i,i2,IS,I3).doit()
        total+=coef*Meson(m1)*Meson(m2)
  
  m1.i3 = m1i3init
  m2.i3 = m2i3init

  return total



def threeMesons(IS,I3,O2IT,m1,m2,m3):
  if( m1.i3!="all" or m2.i3!="all" or m3.i3!="all"):
    print("Warning: Ignoring specification of i3 in threeMesons")
  total = 0
  for i1 in [i for i in range(-O2IT,O2IT+1,1)]:
    for i2 in [i for i in range(-m3.i,m3.i+1,1)]:
      if(i1+i2==I3):
        coef = CG(O2IT,i1,m3.i,i2,IS,I3).doit()
        m3.i3=i2
        total+=coef*twoMesons(O2IT,i1,m1,m2)*Meson(m3)

  return total

