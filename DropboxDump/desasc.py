import numpy as np
import math
from numpy.linalg import norm
from algometer.tools import counter, dbx
from algometer import  TunePar

TunePar.MU = .1  
TunePar.ETA = 1.E-4 #   condizione di arresto 
TunePar.ATOL = 1.E-3 # tolleranza di isclose e allclose
TunePar.RHO = 1.
TunePar.RHOMIN = .01
TunePar.RHOMAX = 450.
TunePar.RHOMINNS = 1.5
TunePar.RHOINC = 1.9 
TunePar.RHOALT = 1.5

qpstat = {} # Quadratic Problem Stat. key = G columns, val=# 


def desasc(p,**kwargs):

    def rhoCalc(Xprev, Xcurr, oldRho):
        if (Xcurr == Xprev).all():
            return 1.
        d = (2*(p.f1(Xprev)-p.f1(Xcurr)+np.dot(p.gf1(Xcurr).T,Xcurr-Xprev)))
        if d > 1E-10:
            rho = norm(Xcurr-Xprev)**2/d
        else:
            rho = oldRho
        dbx.print('rho SeriousStep calcolato:', rho, 'v:',v)
        if rho < TunePar.RHOMIN:
            rho = TunePar.RHOMIN
            counter.up('min', cls='rho') 
        if rho > TunePar.RHOMAX:
            rho = TunePar.RHOMAX
            counter.up('max',cls='rho') 
        return rho

    def rhoPlusNullStep(oldRho, v , f2xd, f2x,ps): 
        fraz = (f2xd-f2x)/(abs(v)+1.)
        if fraz < 0:
            fraz = abs(fraz)
        rhop = oldRho * (fraz +1.)
        dbx.print('rho nullStep calcolato:', rhop, 'v:',v)
        rhopmin = oldRho* TunePar.RHOMINNS
        rhop = max(rhop,rhopmin)
        if rhop == rhopmin:
            counter.up('min', cls='rhop')
        rhop = min(TunePar.RHOMAX, rhop)
        return rhop

    def myHalt():
        halt =  (status == 'Found') or bool(np.isclose (p.optimumValue,p.fXk,atol=TunePar.ETA)) 
        dbx.print(halt)
        return halt

    p.isHalt = myHalt # substitute default isHalt stop criterion with custom one
    qpstat.clear()
    # v = math.inf
    p.absTol = TunePar.ATOL
    rho = TunePar.RHO
    Xprev = p.XStart
    isMainWay = True
    status = 'StartUp' 

    for k in p.loop():
        dbx.print('status:', status, 'isMainWay:', isMainWay) 

        if isMainWay:
            if status in ('SeriousStep', 'StartUp') :
                mcc = Direction()
                mcc.addGradient(-p.gf2Xk,0) # init bundle
                mcc.addGradient(p.gf1Xk,0)
        
        gg, bb = mcc.direction(rho)
        d = -rho*gg 
        v = -rho*norm(gg)**2 - bb
        dbx.print('d:',d, 'v:',v,'gg:',gg, 'bb:', bb) 
        
        if v > -TunePar.ETA: #  stationary DA point
            if isMainWay: 
                isMainWay = False # start alt way until a SeriuosStep
                counter.up('alt', cls='way')
                dbx.print('Inizio alternativa')
                mcc.buildAltMatrix()
            else:
                status = 'Found'
                if p.f(np.add(p.Xk,d)) < p.fXk + 2*TunePar.MU*v:
                    p.Xkp1 = p.Xk + d
            continue

        if p.f(np.add(p.Xk,d)) < p.fXk + 2*TunePar.MU*v:
            status = 'SeriousStep' 
            isMainWay = True # reset flag
            p.Xkp1 = p.Xk + d
            rho = rhoCalc(Xprev,p.Xkp1, rho)
            Xprev = p.Xk
        else:
            status = 'NullStep' 
            nX = p.Xk + d
            nf1 = p.f1(nX)
            ngf1 = p.gf1(nX)
            b = p.f1Xk - (nf1 - ngf1.T @  d)
            av = float(p.f2(nX) - p.f2(p.Xk) - p.gf2Xk.T @ d)
            assert float(p.f2(nX) - p.f2(p.Xk) - p.gf2Xk.T @ d) >= -1.E8 , f'Violata condizione convessità f2: {av}'
            
            if isMainWay:
                rho = rhoPlusNullStep(rho,v,p.f2(nX),p.f2(p.Xk),p.gf2Xk.T @ d) #  fail
                mcc.addGradient(p.gf1(nX),b) # aggiunge gradiente gf1 al bundle
            else:
                rho = TunePar.RHOALT
                dbx.print('Added gradient. rho:',rho)
                mcc.addGradient(p.gf1(nX) - p.gf2Xk,b) # aggiunge variante gradiente gf al bundle
    
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
    def addGradient(self,g,b=0.):  
        g = np.array(g)
        if not hasattr(self,'G'):
            self.G = g
            self.G.shape=(-1,1)
            self.B = np.array(b,dtype=float)
        else:
            self.G = np.insert(self.G,0,g,axis=1) 
            self.B = np.insert(self.B,0,b) 

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
        if c == 1: 
            counter.up('1C', cls='qp')
            return - self.G[:,0], 0.

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

        lam = solve_qp(self.P, self.q, None, None, self.A, self.b ,self.lb,self.ub, solver='scs') #,eps_abs=1E-3,eps_rel=1E-3)
        dbx.print('Matrix G:\n',self.G,'\nBeta:',self.B,'rho:',rho) 
        if lam is None:
            counter.up('Fail', cls='qp')
            lam = np.ones(c)/c # kick off
        if (lam < 0.).any():
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
