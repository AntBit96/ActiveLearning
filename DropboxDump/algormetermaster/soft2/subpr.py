import numpy as np
from algormeter.tools import counter, dbx
from algormeter.kernel import *
from DADC import *

import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_svmlight_file

from algormeter import *
import sys


def solve_prob( w , kni , rho , tol , sigma , beta  , f ):
    
    delta = 0

    p_dim = w.size

    # define my problem parameter K dependent 
    class MyProb (Kernel):
       # '''
       #   _f1: MyProb
       #   _f2: zero
       # '''
        def __inizialize__(self, dimension):
         self.XStart = np.ones(self.dimension)
        def _f1(self, x):

         f1 = 0
         
         argmax = np.argmax(abs(x))
         f1 += abs( x[ argmax ] )  # infinity norm
      
         for i in range(w.size):
          f1 += sigma * abs( x[ i ] )  # sigma * norm 1
    
         #cnstr = 0
         #for i in range(w.size):
         # cnstr -= x[ i ] * w[i]
         #cnstr += rho
         #f1 += beta * pow( cnstr , 2 )

         cnstr = 0
         for i in range(w.size):
          cnstr -= x[ i ] * w[i]
         cnstr += rho

         if cnstr > 0:
          f1 += beta * cnstr

         #print ( 'iter = ', self.K , ' ~ f1 is ' , f1 , ' at ' , x  , file=f )
         return f1 
        def _gf1(self, x):  

         #- - - - - - - - - - - - - - - - - - - - - - - - - -
         g1 = np.zeros(p_dim) # - - - - - - - - - - - - - - - 

         argmax = np.argmax(abs(x))
         if x[argmax] >= 0.0:
          g1[argmax] += 1
         else:
          g1[argmax] -= 1

         for i in range( w.size ):
          if x[i] >= 0.0:
           g1[i] += sigma
          else:
           g1[i] -= sigma

         #cnstr = 0
         #for i in range(w.size):
         # cnstr -= x[ i ] * w[i]
         #cnstr += rho

         #for i in range( w.size ):
         # g1[i] -= 2.0 * beta * cnstr * w[i]

         cnstr = 0
         for i in range(w.size):
          cnstr -= x[ i ] * w[i]
         cnstr += rho

         if cnstr > 0:
          for i in range( w.size ):
           g1[i] -= beta * w[i]
       
         #print ( 'iter = ', self.K , ' ~ g1 is ' , g1 , ' at ' , x  , file=f)
        
         return g1
        def _f2(self,x) :

         f2 = 0

         max_indeces = np.argsort(-abs(x))[:kni]
         for i in range( kni ):
          kindex = max_indeces[i]
          f2 += sigma * abs( x[ kindex ] )
   
         #print ( 'iter = ', self.K , ' ~ f2 is ' , f2 , ' at ' , x  , file=f)
         return f2

        def _gf2(self, x):

         #- - - - - - - - - - - - - - - - - - - - - - - - - -
         g2 = np.zeros(p_dim) # - - - - - - - - - - - - - - - 

         max_indeces = np.argsort(-abs(x))[:kni]
         for i in range( kni ):
          kindex = max_indeces[i]
          if x[kindex] >= 0.0:
           g2[kindex] += sigma
          else:
           g2[kindex] -= sigma

         #print ( 'iter = ', self.K , ' ~ g2 is ' , g2 , ' at ' , x , file=f)
         return g2

    p = MyProb(p_dim) # K again
    status, x, y = p.minimize(DADC, iterations = 50 , dbprint = False )    

    print(f'With Status:{status}, y is {y} at {x}')

    return x

if __name__ == "__main__":
    solve_prob()
