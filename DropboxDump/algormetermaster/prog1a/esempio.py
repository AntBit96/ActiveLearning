'''Base Problems library
'''
 

#import pdb
import sys
sys.path.append("..") # Adds higher directory to python modules path
import numpy as np
import matplotlib.pyplot as plt
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
from math import sqrt, log

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from numpy.linalg import solve, norm
from numpy.random import rand

# ...apply my algorithm to my problem
for K in [1]:

    rho = 0.1
    epsilon = 60.0
    C1 = 80.0
    C2 = 30.0
    C3 = 0.1

    # import the dataSet 
    Xpts,ylbs = load_svmlight_file("dataset", zero_based=False, dtype = np.float64) 

    f_out= open("stampa.log", "w")
    print( "Name of the file: ", f_out.name )

    print( "", file=f_out)
    print( "X: ", Xpts , file=f_out)
    print( "y [-1=A, 1=B]: ", ylbs , file=f_out)
    print( "", file=f_out)

    # split in training and test sets 
    X_train, X_test, y_train, y_test = train_test_split( Xpts, ylbs, test_size = 0.4 , random_state=0)
 
    # training phase
    linear_svc = svm.SVC(kernel='linear', C=10)
    linear_svc.fit(Xpts, ylbs)


    # classifiers ( w , b )
    w = linear_svc.coef_[0,:]
    gamma = -linear_svc.intercept_
    print( "Classifier (f) w^Tx - gamma : w = ", w , " ~ gamma = ", gamma , file=f_out)
    print( "f(X) = ", linear_svc.decision_function(Xpts) , file=f_out) 

    print ( 'w = ', w ,  file=f_out)
    print ( 'gamma  = ' , gamma ,  file=f_out)

    # support vectors info:
    print( "Number of Support Vectors = ", linear_svc.n_support_ , file=f_out)
    print( "Indices of Support Vectors = ", linear_svc.support_ , file=f_out)
    print( "Support Vectors = ", linear_svc.support_vectors_ , file=f_out)

    print( "X_train: ", X_train , file=f_out)
    print( "y_train [-1=A, 1=B]: ", y_train , file=f_out)
    print( "X_test: ", X_test , file=f_out)
    print( "y_test [-1=A, 1=B]: ", y_test , file=f_out)

    # Predict the result by giving Data to the model
    accuracy = linear_svc.score(Xpts, ylbs)
    print( "", file=f_out)
    print( "Testing, accuray = ", accuracy , file=f_out)

    #scores = cross_val_score(linear_svc, Xpts, ylbs, cv=5)
    #mean_score = np.mean( scores )
    #print( "", file=f_out)
    #print( "TenCrossValidation, accuray = ", mean_score , file=f_out)

    #X_m, X_notm, y_m, y_notm = train_test_split( Xpts, ylbs, test_size = 0.8 , random_state=3)

    X_notm = lil_matrix(( np.size(Xpts,0)-1 , np.size(Xpts,1) ))
    X_m = lil_matrix(( 1, np.size(Xpts,1) ))

    y_notm = np.empty( np.size(Xpts,0)-1 )
    y_m = np.empty( 1 )

    for i in range(np.size(Xpts,0) ):
     imid = 1
     if i < imid :
      for j in range(np.size(Xpts,1)):
       X_notm[i,j] = Xpts[i,j]
      y_notm[i]=ylbs[i]
     elif i > imid :
      for j in range(np.size(Xpts,1)):
       X_notm[i-1,j] = Xpts[i,j]
      y_notm[i-1]=ylbs[i]
     else:
      for j in range(np.size(Xpts,1)):
       X_m[0,j] = Xpts[i,j]
      y_m[0]=ylbs[i]

    print( "X_m: ", X_m , file=f_out)
    print( "y_m [-1=A, 1=B]: ", y_m , file=f_out)
    print( "X_notm: ", X_notm , file=f_out)
    print( "y_notm [-1=A, 1=B]: ", y_notm , file=f_out)
 

    # create a mesh to plot
    x_min, x_max = Xpts[:, 0].min() - 2, Xpts[:, 0].max() + 2
    y_min, y_max = Xpts[:, 1].min() - 2, Xpts[:, 1].max() + 2
    h = ( abs(x_max) / abs(x_min))/100

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

    # Predict the result by giving Data to the model
    # Plot the data for Proper Visual Representation
    plt.subplot(1, 1, 1)
    Z=np.c_[xx.ravel(), yy.ravel()];
    #L=np.c_[ylbs.ravel()];
   
    # Predict the result by giving Data to the model
    y_pred = linear_svc.predict(Z)
    accuracy2 = linear_svc.score(Z,y_pred)
    print( "", file=f_out)
    print( "Testing, accuray sim = ", accuracy , file=f_out)
    print( "", file=f_out)
    print( "Testing, accuray = ", accuracy , file=f_out)
   
    y_pred_shaped = y_pred.reshape(xx.shape)

    #plt.contourf(xx, yy, y_pred_shaped, levels=1, cmap = plt.cm.Paired, alpha = 0.2)
    plt.contourf(xx, yy, y_pred_shaped, levels=1, alpha = 0.2, cmap='Greys_r')

    X98 = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0] ,[1.0, 1.0], [2.0,0.0], [2.0,1.0] ])
    L1 = np.array([-1, -1 , -1 , 1 , 1 , 1])


    #plt.scatter(X98[:, 0], X98[:, 1], c = L1, cmap = plt.cm.Paired)
    
    plt.scatter(X98[L1 == -1, 0], X98[L1 == -1, 1], s = 70, c = 'Grey', marker='_')
    plt.scatter(X98[L1 == 1, 0], X98[L1 == 1, 1], s = 70, c = 'Grey', marker='+')


    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim(xx.min(), xx.max())
    #plt.title('SVM before manipulation')
 
    #plt.show()
    plt.savefig('figure_1.pdf', bbox_inches='tight')

    di_dim = 0
    dl_dim = 0
    for i in range(np.size(X_m,0) ):
     if( int( y_m[ i ] ) == -1 ):
      di_dim += w.size
     else:
      dl_dim += w.size
    
    p_dim = w.size +  di_dim + dl_dim + gamma.size  # w + d_i + d_l + gamma 

    d_offset = w.size
    gamma_offset = d_offset + dl_dim + di_dim

    print( "w dim ", w.size , file=f_out)
    print( "d_i dim ", di_dim , file=f_out)
    print( "d_l dim ", dl_dim , file=f_out)
    print( "gamma dim ", gamma.size , file=f_out)
    print( "p_dim ", p_dim , file=f_out)

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
         for i in range(np.size(X_notm,0) ):
          maxfact = 0
          if( int( y_notm[ i ] ) == -1 ):
           for j in range(np.size(X_notm,1)):
            maxfact += X_notm[i,j] * x[ j ]
           maxfact += - x[gamma_offset] + 1.0
           if( maxfact < 0 ):
            maxfact = 0
           f1 += C1 * maxfact
          else:
           for j in range(np.size(X_notm,1)):
            maxfact += - X_notm[i,j] * x[ j ]
           maxfact += x[gamma_offset] + 1.0
           if( maxfact < 0 ):
            maxfact = 0
           f1 += C1 * maxfact

         for i in range(np.size(X_m,0) ):
          fact1 = 0
          fact2 = 0
          maxfact = 0
          offset = w.size + i * w.size
          if( int( y_m[ i ] ) == -1 ):
           for j in range(np.size(X_m,1)):
            fact1 +=  X_m[i,j] * x[ j ] + 0.25 * pow( ( x[ offset + j ] + x[ j ] ) , 2 ) 
            fact2 +=  0.25 * pow( ( x[ offset + j ] - x[ j ] ) , 2 ) 
           fact1 += - x[gamma_offset] + 1.0
           if( fact1 > fact2 ):
            maxfact = fact1
           else:
            maxfact = fact2
           f1 += C1 * maxfact
          else:
           for j in range(np.size(X_m,1)):
            fact1 +=  - X_m[i,j] * x[ j ] + 0.25 * pow( ( x[ offset + j ] - x[ j ] ) , 2 ) 
            fact2 +=  0.25 * pow( ( x[ offset + j ] + x[ j ] ) , 2 ) 
           fact1 += x[gamma_offset] + 1.0
           if( fact1 > fact2 ):
            maxfact = fact1
           else:
            maxfact = fact2
           f1 += C1 * maxfact

         absfact = 0
         for j in range(w.size):
          absfact +=  x[ j ] * w[ 0, j ] 
         f1 += C2 * abs( absfact )

         for j in range(w.size):
          f1 += C3 * 0.5 * x[ j ] * x[ j ]

         for i in range(np.size(X_m,0) ):
          termq = - rho
          offset = w.size + i * w.size
          for j in range(np.size(X_m,1)):
           termq += pow( ( x[ offset + j ] ) , 2 ) 
          if termq > 0 :
           f1 += epsilon * ( termq )
  
         #print ( 'iter = ', self.K , ' ~ f1 is ' , f1 , ' at ' , x  , file=f_out )
         return f1 
        def _gf1(self, x):  

         #- - - - - - - - - - - - - - - - - - - - - - - - - -
         g1 = np.zeros(p_dim) # - - - - - - - - - - - - - - - 

         # subgradient relative manipulated points  - - - - - 

         for i in range(np.size(X_notm,0) ):
          if( int( y_notm[ i ] ) == -1 ):
           fact = 0
           for j in range(np.size(X_notm,1)):
            fact += X_notm[i,j] * x[ j ]
           fact += - x[gamma_offset] + 1.0
           if( fact >= 0 ):
            for j in range(np.size(X_notm,1)):
             g1[j] += C1 * X_notm[i,j]
            g1[gamma_offset] -= C1
          else:
           fact = 0
           for j in range(np.size(X_notm,1)):
            fact += - X_notm[i,j] * x[ j ]
           fact += x[gamma_offset] + 1.0
           if( fact >= 0 ):
            for j in range(np.size(X_notm,1)):
             g1[j] -= C1 * X_notm[i,j]
            g1[gamma_offset] += C1

        # subgradient relative not manipulated points  - - - - - 

         for i in range(np.size(X_m,0) ):
          fact1 = 0
          fact2 = 0
          offset = w.size + i * w.size
          if( int( y_m[ i ] ) == -1 ):  
 
           for j in range(np.size(X_m,1)):
            fact1 +=  X_m[i,j] * x[ j ] + 0.25 * pow( ( x[ offset + j ] + x[ j ] ) , 2 ) 
            fact2 +=  0.25 * pow( ( x[ offset + j ] - x[ j ] ) , 2 ) 
           fact1 += - x[gamma_offset] + 1.0

           if( fact1 >= fact2 ):
            for j in range(np.size(X_m,1)): 
             g1[j] += C1 * ( X_m[i,j] + 0.5 * ( x[ offset + j ] + x[ j ] ) )
             g1[offset + j] = C1 * 0.5 * ( x[offset + j] + x[ j ] )
            g1[gamma_offset] -= C1
           else:
            for j in range(np.size(X_m,1)):
             g1[j] += C1 * ( - 0.5 * ( x[ offset + j ] - x[ j ] ) )
             g1[offset + j] = C1 * 0.5 * ( x[offset + j] - x[ j ] )
           
          else:
      
           for j in range(np.size(X_m,1)):
            fact1 +=  - X_m[i,j] * x[ j ] + 0.25 * pow( ( x[ offset + j ] - x[ j ] ) , 2 ) 
            fact2 +=  0.25 * pow( ( x[ offset + j ] + x[ j ] ) , 2 ) 
           fact1 += x[gamma_offset] + 1.0

           if( fact1 >= fact2 ):
            for j in range(np.size(X_m,1)):
             g1[j] += C1 * ( - X_m[i,j] - 0.5 * ( x[ offset + j ] - x[ j ] ) )
             g1[offset + j] = C1 * 0.5 * ( x[offset + j] - x[ j ] )
            g1[gamma_offset] += C1
           else:
            for j in range(np.size(X_m,1)):
             g1[j] += C1 * ( 0.5 * ( x[ offset + j ] + x[ j ] ) )
             g1[offset + j] = C1 * 0.5 * ( x[offset + j] + x[ j ] )

         for i in range(np.size(X_m,0) ):
          termq = - rho
          offset = w.size + i * w.size
          for j in range(np.size(X_m,1)):
           termq += pow( ( x[ offset + j ] ) , 2 ) 
          if termq > 0 :
           for j in range(np.size(X_m,1)):
            g1[offset + j] += 2 * epsilon * x[offset + j]

         # subgradient of scalar product on w  - - - - - - - - - - - 

         fact = 0
         for j in range(w.size):
          fact +=  x[ j ] * w[ 0, j ] 
         if( fact >= 0 ):
          for j in range(w.size):        
           g1[j] += C2 * w[ 0, j ] + C3 * x[ j ]
         else:
          for j in range(w.size):        
           g1[j] += - C2 * w[ 0, j ] + C3 * x[ j ]

         #print ( 'iter = ', self.K , ' ~ g1 is ' , g1 , ' at ' , x  , file=f_out)
        
         return g1
        def _f2(self,x) :

         f2 = 0
         for i in range(np.size(X_m,0) ):
          termq = 0
          offset = w.size + i * w.size
          if( int( y_m[ i ] ) == -1 ):
           for j in range(np.size(X_m,1)):
            termq +=  0.25 * pow( ( x[ offset + j ] - x[ j ] ) , 2 ) 
           f2 += C1 * termq
          else:
           for j in range(np.size(X_m,1)):
            termq +=  0.25 * pow( ( x[ offset + j ] + x[ j ] ) , 2 ) 
           f2 += C1 * termq

         #print ( 'iter = ', self.K , ' ~ f2 is ' , f2 , ' at ' , x  , file=f_out)
         return f2

        def _gf2(self, x):

         #- - - - - - - - - - - - - - - - - - - - - - - - - -
         g2 = np.zeros(p_dim) # - - - - - - - - - - - - - - - 

         # subgradient relative to w and d_i, d_l  - - - - - - 

         for i in range(np.size(X_m,0) ):
          offset = w.size + i * w.size
          if( int( y_m[ i ] ) == -1 ):
           for j in range(np.size(X_m,1)):
            g2[j] += - 0.5 * C1 * ( x[ offset + j ] - x[ j ] )
            g2[offset + j] = 0.5 * C1 * ( x[offset + j] - x[ j ] )
          else:
           for j in range(np.size(X_m,1)):
            g2[j] += 0.5 * C1 * ( x[ offset + j ] + x[ j ] )
            g2[offset + j] = 0.5 * C1 * ( x[offset + j] + x[ j ] )

         # subgradient relative to gamma

         g2[gamma_offset] = np.zeros(1) # set to zero

         #print ( 'iter = ', self.K , ' ~ g2 is ' , g2 , ' at ' , x , file=f_out)
         return g2

    #pdb.set_trace()
    p = MyProb(p_dim) # K again
    status, x, y = p.minimize(DADC, iterations = 50 , dbprint = False )    

    print(f'With K:{K} Status:{status}, y is {y} at {x}')

    sclarprod = 0
    for j in range(w.size):
     sclarprod +=  x[ j ] * w[ 0, j ] 
    sclarprod += abs( sclarprod )

    f1 = 0
    for i in range(np.size(X_notm,0) ):
     maxfact = 0
     if( int( y_notm[ i ] ) == -1 ):
      for j in range(np.size(X_notm,1)):
       maxfact += X_notm[i,j] * x[ j ]
       X98[i,j] = X_notm[i,j]
      maxfact += - x[gamma_offset] 
      L1[i]=-1
      if( maxfact < 0 ):
       maxfact = 0
      f1 += maxfact
     else:
      for j in range(np.size(X_notm,1)):
       maxfact += - X_notm[i,j] * x[ j ]
       X98[i,j] = X_notm[i,j]
      maxfact += x[gamma_offset]
      L1[i]=1
      if( maxfact < 0 ):
       maxfact = 0
      f1 += maxfact

    f2 = 0
    for i in range(np.size(X_m,0) ):
     maxfact = 0
     offset = w.size + i * w.size
     if( int( y_m[ i ] ) == -1 ):
      for j in range(np.size(X_m,1)):
       maxfact +=  ( X_m[i,j] + x[ offset + j ] ) *  x[ j ]  
       X98[i + np.size(X_notm,0) ,j] = X_m[i,j] + x[ offset + j ]
      L1[i+ np.size(X_notm,0)]=-1
      maxfact += - x[gamma_offset] 
      if( maxfact < 0 ):
       maxfact = 0
      f2 += maxfact
     else:
      for j in range(np.size(X_m,1)):
       maxfact +=  - (X_m[i,j] + x[ offset + j ] ) * x[ j ]
       X98[i + np.size(X_notm,0),j] = X_m[i,j] + x[ offset + j ]
      L1[i+ np.size(X_notm,0) ]=1
      maxfact += x[gamma_offset] 
      if( maxfact < 0 ):
       maxfact = 0
      f2 += maxfact

    new_w = w
    for j in range(w.size):        
     new_w[ 0, j ] = x[ j ]

    new_gamma = gamma
    new_gamma[ 0 ] = x[gamma_offset]
 
    print("f_notM = " , f1 , file=f_out)
    print("f_M = " , f2 , file=f_out)
    print("class err = " , f1+f2 , file=f_out)
    print("|w'w| = " , sclarprod , file=f_out)
    print ( 'new w = ', new_w , file=f_out)
    print ( 'new gamma  = ' , new_gamma , file=f_out)

    print ( 'new x  = ' , X98 , file=f_out)
    print ( 'new y  = ' , L1 , file=f_out)

    for i in range(np.size(Z,0) ):
     val = 0
     for j in range(w.size):  
      val += new_w[ 0, j ] * Z[i][j]
     val -= new_gamma
     if val < 0:
      y_pred[ i ] = -1
     else:
      y_pred[ i ] = 1
   
    y_pred_shaped = y_pred.reshape(xx.shape)


    plt.contourf(xx, yy, y_pred_shaped, levels=4, cmap = 'Greys_r', alpha = 0.99)
   

    #plt.scatter(X98[:, 0], X98[:, 1], c = L1, cmap = plt.cm.Paired)
    plt.scatter(X98[L1 == -1, 0], X98[L1 == -1, 1], s = 70, c = 'Grey', marker='_')
    plt.scatter(X98[L1 == 1, 0], X98[L1 == 1, 1], s = 70, c = 'Grey', marker='+')


    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim(xx.min(), xx.max())
    #plt.title('SVM after manipulation')
 
    #plt.show()
    plt.savefig('figure_2.pdf', bbox_inches='tight')

    print( "optimal solution = "  , x , file=f_out)

    f_out.close()



