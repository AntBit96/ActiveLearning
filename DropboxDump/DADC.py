 # type: ignore
import numpy as np
from numpy.linalg import norm
from algormeter.tools import counter, dbx
from algormeter import  Param

Param.GAMMAMAX = 2. # 1/gamma Max proximity 
Param.GAMMAMIN = .002 # 1/gamma Min proximity
Param.PROXINC = .5 # Proximity tuning 
Param.MU = .1  # Descent ratio
Param.ETA = 1.E-4 #   Stopping tolerance 
Param.EPS = 1E-5 # stop tolerance 
Param.DELTA = 1E-4
Param.ATOL = 1.E-3 # tollerance isclose e allclose
Param.RHO = 1.
Param.AGG_GRA = False # enable gradiente aggregato

qpstat = {} # Quadratic Problem Stat. key = enum G columns, val=# 

def DADC(p,**kwargs):
    
    def rhoSeriousStep(Xprev, Xcurr, oldRho):
        if (Xcurr == Xprev).all():
            return Param.RHO
        d = (2*(f1Xprev-p.f1(Xcurr)+np.dot(p.gf1(Xcurr).T,Xcurr-Xprev)))
        if d > 1E-10:
            rho = norm(Xcurr-Xprev)**2/d
        else:
            rho = oldRho
        # dbx.print('rho SeriousStep calcolato:', rho, 'v:',v)
        if rho < 1/Param.GAMMAMAX:
            rho = 1/Param.GAMMAMAX
            counter.up('min', cls='rho') 
        if rho > 1/Param.GAMMAMIN:
            rho = 1/Param.GAMMAMIN
            counter.up('max',cls='rho') 
        return rho

    def rhoNullStep(oldRho): 
        rho = oldRho/Param.PROXINC
        # dbx.print('rho nullStep calcolato:', rho, 'v:',v)
        rho = min(1/Param.GAMMAMIN, rho)
        return rho

    # def myStop():
    #     halt =  (status == 'Stop') # or (abs(p.optimumValue - p.fXk) < Param.ETA) 
    #     dbx.print(halt)
    #     return halt
    # p.stop = myStop # substitute default stop stop criterion with custom one

    qpstat.clear()
    p.absTol = Param.ATOL
    rho = Param.RHO
    Xprev = p.XStart
    f1Xprev = p.f1(Xprev)
    if str(p) == 'JB10':
        Param.GAMMAMAX = 0.3

    isMainWay = True
    status = 'StartUp' 
    Direction.resetAgg()

    for k in p.loop():
        dbx.print('status:', status, 'isMainWay:', isMainWay, 'rho:',rho) 

        if isMainWay:
            if status in ('SeriousStep', 'StartUp') :
                mcc = Direction()
                mcc.addGradient(-p.gf2Xk,0,isMainWay) # init bundle
                mcc.addGradient(p.gf1Xk,0,isMainWay)
        
        gg, bb = mcc.direction(rho)
        d = -rho*gg 
        v = -rho*norm(gg)**2 - bb
        dbx.print('d:',d, 'v:',v,'gg:',gg, 'bb:', bb) 
        
        if v > -Param.ETA: #  stationary DA point
            if isMainWay: 
                status = 'Stationary DA'
                isMainWay = False # start alt way until a SeriuosStep
                Direction.resetAgg() #agg
                counter.up('alt', cls='way')
                mcc.buildAltMatrix()
                continue
            else:
                dbx.print('stop v:',v)
                status = 'Stop'
                break ### looppa senza scendere sino a che il bundle dice stop senza xkp1

        if isMainWay and (p.f1(p.Xk + d) < p.f1Xk + Param.MU*v) \
            or not isMainWay and (p.f1(p.Xk + d) < p.f1Xk + Param.MU*v + p.gf2Xk @ d):
            status = 'SeriousStep' 
            counter.up('Serious', cls='Step')
            p.Xkp1 = p.Xk + d
            rho = rhoSeriousStep(Xprev,p.Xkp1, rho)
            Xprev = p.Xk
            f1Xprev = p.f1(Xprev)
            f1d = p.f1(p.Xk + d)
            
            if status == 'SeriousStep':
                mcc.calc_agg(f1d, p.f1Xk, p.gf2Xk, d, isMainWay)

            isMainWay = True # reset to MainWay
            deltaf = f1d - p.f1Xk - p.gf2Xk @ d
            dbx.print('deltaf:',deltaf,'f1d:', f1d,'f1:',p.f1Xk,'ps:',p.gf2Xk @ d)
            if abs(deltaf) < Param.DELTA:
                dbx.print('stop deltaf:',deltaf)
                status = 'Stop'
                break
            # if norm(d) < Param.EPS:
            #     status = 'Stop'
            #     break
        else:
            status = 'NullStep'
            counter.up('Null', cls='Step')
            nX = p.Xk + d
            nf1 = p.f1(nX)
            ngf1 = p.gf1(nX)
            b = p.f1Xk - (nf1 - ngf1.T @  d)
            
            if isMainWay:
                rho = rhoNullStep(rho) 
                mcc.addGradient(p.gf1(nX),b, isMainWay) 
            else:
                rho = Param.RHO
                mcc.addGradient(p.gf1(nX) - p.gf2Xk,b,isMainWay) 
    
    if len(qpstat) > 0:
        s, m, e = 0, 0, 0
        for k,v in qpstat.items():
            s += k*v
            e += v
            if k > m:
                m = k

        counter.log(round(s/e,1), 'avgRow', cls='qp')
        counter.log(m,'maxRow', cls='qp')


import warnings
warnings.filterwarnings(action='error')
warnings.filterwarnings(action='ignore', category = UserWarning, message='.*Converted.*')
warnings.filterwarnings(action='once', category = UserWarning, message='.*maximum iterations reached.*')
warnings.filterwarnings(action='once', category = UserWarning, message='.*solved inaccurate.*')
warnings.filterwarnings(action='once', category = UserWarning, message='.*box cone proj hit maximum.*')
warnings.filterwarnings(action='once', category = UserWarning, message='.*INFEASIBLE_INACCURATE.*')
warnings.filterwarnings(action='once', category = UserWarning, message='.*SOLVED_INACCURATE.*')

from qpsolvers import solve_qp

class Direction :
    agg = None #  aggregate gradient
    ab = None

    def calc_agg(self, f1d, f1, gf2, d, main) -> None:
        if not Param.AGG_GRA:
            return 
        if not hasattr(self,'G'):
            self.resetAgg()
            return

        if main:
            GA = np.delete(self.G,-1,1) 
            BA = np.delete(self.B,-1,0) 
        else:
            GA = self.G + gf2[:,np.newaxis]
            BA = self.B
        dbx.print('Matrix GA:\n',GA,'\nBA:',BA)

        _,l = GA.shape
        if l == 1:
            Direction.agg = GA 
            Direction.ab = BA 

        P = np.dot(GA.T, GA) #  a positive semi definite matrix
        q = BA
        A = np.ones(l)
        b = np.array([1.])
        lb = np.zeros(l) 
        ub = np.ones(l)

        lam = solve_qp(P, q, None, None, A, b ,lb,ub, solver='scs') 
        counter.up('solv', cls='qpa')
 
        if lam is None:
            counter.up('Fail', cls='qpa')
            lam = np.ones(c)/c # kick off
        if (lam < 0.).any():
            counter.up('lb<0', cls='qpa')
            # lam[lam < 0] = 0
            dbx.print('lam agg < 0 :\n',lam)
        Direction.agg = GA @ lam
        Direction.ab = BA @ lam
        Direction.ab = Direction.ab + f1d - f1 - d @ Direction.agg 
        assert Direction.ab >= 0, 'ab negativo'

    def resetAgg():
        Direction.agg = None #  aggregate gradient
        Direction.ab = None

    def addGradient(self,g, b=0., main : bool  = True):  
        g = np.array(g)
        if not hasattr(self,'G'):
            self.G = g
            self.G.shape=(-1,1) # 1 column
            self.B = np.array(b,dtype=float)

            # commenta questo blocco per disattivare logica agg
            if Direction.ab is not None:
                self.G = np.insert(self.G,0,np.ravel(Direction.agg),axis=1) 
                self.B = np.insert(self.B,0,Direction.ab) 
                Direction.resetAgg()
        else:
            self.G = np.insert(self.G,0,g,axis=1) 
            self.B = np.insert(self.B,0,b) 
        return

    def _buildMatrix(self,rho):
        self.P = np.dot(self.G.T, self.G) #  a positive semi definite matrix
        self.P = self.P  * rho
        _,l = self.G.shape
        self.q = self.B
        self.A = np.ones(l)
        self.b = np.array([1.])
        self.lb = np.zeros(l) 
        self.ub = np.ones(l)

    def direction(self,rho):
        if not hasattr(self,'G'):
            raise ValueError('Empty bundle')

        r,c = self.G.shape
        # if c == 1: 
        #     counter.up('1C', cls='qp')
        #     return - self.G[:,0], 0.

        if c == 2: 
            counter.up('2C', cls='qp')

        qpstat[c] = qpstat[c]+1 if c in qpstat else 1

        counter.up('solv', cls='qp')
        self._buildMatrix(rho) 
        #QP Solver
        # https://scaron.info/doc/qpsolvers/
        # https://github.com/stephane-caron/qpsolvers
        # https://osqp.org/docs/index.html
        # https://www.cvxgrp.org/scs/index.html

        # 'ecos', 'osqp', 'proxqp', 'quadprog', 'scs'
        lam = solve_qp(self.P, self.q, None, None, self.A, self.b ,self.lb,self.ub, solver='scs') 
        # lam = solve_qp(self.P, self.q, None, None, self.A, self.b ,self.lb,self.ub, solver='gurobi') 
        dbx.print('Matrix G:\n',self.G,'\nB:',self.B,'rho:',rho) 
        if lam is None:
            counter.up('Fail', cls='qp')
            lam = np.ones(c)/c # kick off
        if (lam < 0.).any():
            # lam[lam < 0] = 0
            counter.up('lb<0', cls='qp')
            dbx.print('lam < 0')
        gg = self.G @ lam
        bb = self.B @ lam
        dbx.print('lam:',lam, 'gg:',gg, 'bb:',bb)

        return gg,bb

    def buildAltMatrix(self):
        lc = self.G[:,-1] # copy last column in lc
        self.G = np.delete(self.G,-1,1) # delete lc from G
        self.B = np.delete(self.B,-1,0) # also from Beta 
        self.G = self.G + lc[:,np.newaxis] # sum lc to all G column