'''Base Problems library
'''
 

#import pdb
import sys
sys.path.append("..") # Adds higher directory to python modules path
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_svmlight_file

from math import sqrt, log
from subpr import solve_prob

tol = 1e-6 # precision

sigma = 4
beta = 1.1
kni = 1

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
linear_svc = svm.SVC(kernel='linear', C=50)
linear_svc.fit(Xpts, ylbs)


# classifiers ( w , b )

w = np.array([])
for j in range(np.size(Xpts,1)):
 w = np.append( w , [linear_svc.coef_[0,j]] , axis=0)
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

XDuplex = np.empty(( 0 , np.size(Xpts,1) ))
L1 = []

for i in range(np.size(Xpts,0) ):
 new_e = np.empty(( 1 , 2 ))
 for j in range(np.size(Xpts,1)):
  new_e[0,j] = Xpts[i,j] 
 XDuplex = np.vstack( (XDuplex, new_e) )
 L1 = np.append(L1, [ylbs[i]], axis=0)

print( "X: ", XDuplex , file=f_out)
print( "y_m [-1=A, 1=B]: ", L1 , file=f_out)

# create a mesh to plot
x_min, x_max = XDuplex[:, 0].min() - 4, XDuplex[:, 0].max() + 4
y_min, y_max = XDuplex[:, 1].min() - 4, XDuplex[:, 1].max() + 4
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
#plt.scatter(XDuplex[:, 0], XDuplex[:, 1], c = L1, cmap = plt.cm.Paired)
plt.scatter(XDuplex[L1 == -1, 0], XDuplex[L1 == -1, 1], s = 70, c = 'Grey', marker='_')
plt.scatter(XDuplex[L1 == 1, 0], XDuplex[L1 == 1, 1], s = 70 , c = 'Grey' , marker='+')


plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(xx.min(), xx.max())
#plt.title('SVC with Linear Kernel before ')
 
#plt.show()
plt.savefig('figure_1.pdf', bbox_inches='tight')

print( "Delta computation:", file=f_out )

# compute the minimum delta
 
NewX = np.empty(( 0 , np.size(Xpts,1) ))
Newy = np.array([])

NewXCounter = 0
for Xind in range( Xpts.shape[0] ):

 val = linear_svc.decision_function(Xpts)[Xind]
 if val <= -tol and L1[ Xind ] == -1:
  print( "1val ", val, " ~ y = ", ylbs[Xind] )
  print( "", file=f_out)
  print( XDuplex[Xind] , "in I_A", file=f_out)
  rho = -np.inner(w,XDuplex[Xind]) + gamma + 1.0
  delta = solve_prob( w , kni , rho , tol , sigma , beta , f_out )
  print( "", file=f_out)
  NewXCounter += 1
  NewX = np.append( NewX, XDuplex[Xind] + delta )
  Newy = np.append( Newy, L1[ Xind ] )
  print( "x + delta = ", XDuplex[Xind] + delta , file=f_out)
 elif val >= tol and L1[ Xind ] == 1:
  print( "2val ", val, " ~ y = ", ylbs[Xind])
  print( "", file=f_out)
  print( XDuplex[Xind] , "in I_B", file=f_out)
  rho = np.inner(w,XDuplex[Xind]) - gamma + 1.0
  delta = solve_prob( -w , kni , rho , tol , sigma , beta , f_out )
  print( "", file=f_out)
  NewXCounter += 1
  NewX = np.append( NewX, XDuplex[Xind] + delta )
  Newy = np.append( Newy, L1[ Xind ] )
  print( "x + delta = ", Xpts[Xind] + delta , file=f_out)
 else:
  NewX = np.append( NewX, XDuplex[Xind] )
  Newy = np.append( Newy, L1[ Xind ] )

NewX = np.reshape(NewX, (-1, 2) )
print( "", file=f_out)
print( "New X" , NewX , file=f_out)
print( "New Y" , Newy , file=f_out)
  
plt.clf()

#plt.contourf(xx, yy, y_pred_shaped, levels=1, cmap = plt.cm.Paired, alpha = 0.2)
plt.contourf(xx, yy, y_pred_shaped, levels=1, alpha = 0.2, cmap='Greys_r')
#plt.scatter(NewX[:, 0], NewX[:, 1], c = Newy, cmap = plt.cm.Paired)
plt.scatter(NewX[Newy == -1, 0], NewX[Newy == -1, 1], s = 70, c = 'Grey' , marker='_')
plt.scatter(NewX[Newy == 1, 0], NewX[Newy == 1, 1], s = 70, c = 'Grey' ,marker='+')


plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(xx.min(), xx.max())
#plt.title('SVC with Linear Kernel after')
 
#plt.show()
plt.savefig('figure_2.pdf', bbox_inches='tight')

f_out.close()



