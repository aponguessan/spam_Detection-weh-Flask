# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 17:20:23 2023

@author: hp
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 12:20:42 2023

@author: hp
"""

import numpy as np
import numpy.linalg as alg
from numpy.linalg import norm
import matplotlib.pyplot as plt
from math import sin,exp,pi
###defition de la matrice###
N=100
L=1
deltax=L/(N+1)
nu=0.01
f=-10
lamb=np.random.random(N)
lamb=1-2*lamb
c=0
def Matri(n):
    A=np.zeros((N,N))
    for i in range(n-1):
        A[0,0],A[i,i+1],A[i+1,i],A[i,i],A[i+1,i+1]=(2,-1,-1,2,2)
    return A
G=Matri(N)
def MAT(A,deltax,nu):
    Aa=(nu/((deltax)*(deltax)))*A
    return Aa

def calclU(A,b):
    U=np.linalg.solve(A,b)
    return U
b=np.ones((N))*f
#print(b)
U=calclU(MAT(Matri(N),deltax,nu), b)
print(U)
t=np.linspace(0,L,N)
plt.plot(t,U)
"""def h(x):
    h=5*(x-1)
    return h 
H=[]
Hk=[]
for x in t:
    h_i=h(x)
    #print(h(x))
    H.append(h_i)
print(H)"""

Ik=np.zeros((N))
Ak=np.zeros((N))
Id=np.identity(N)
print("guuiiiohyvuivui")
#print(Id)
"""for j in range(N):
  B=lamb[j]-c*(U[j]-H[j])
  if B>=0:
     Ik[j]=j+1
  else:
      Ak[j]=j+1
 # print(B)"""
print(Ik)
#print(len(Ik))
#print(Ak) 
#print("\nnnnn")
P=np.zeros((N,N))
W=np.zeros((N,N))
def NouMatrice1(A,Ik):
    l=0
    for k in range((N)):
        #Id_i[k]=1
        P[:,k]=-Id[:,k]
        #P[:,k]=A[:,k]
        if Ik[k]!=0:
         #else:
         l=l+1
    for y in range(l):
        P[:,y]=A[:,y]
    return P,l,#W

M,l=NouMatrice1(G,Ik)
#print(M,l)
#print(W) 

print("TRErre")
"""Bk=np.zeros((N))
def Mem2(A,Ik,l,H):
   Ak=np.zeros((N,l))
   Hk=np.zeros((l))
   for d in range(N): 
    #for a in range(l):
        if Ik[d]==0:
            Ak[:,d]=A[:,d]
            Hk[d]=H[d]
   Bk=np.dot(Ak,Hk)
   return Ak,Hk,Bk   
#Z=Mem2(M, Ik, l, H)
#print(Z)
V=calclU(MAT(M,deltax,nu), b-Bk)
#print(V)
plt.plot(t,V)
R=np.zeros((N,N))
Bk=np.zeros((N))
AK1=np.zeros((N,N))
#def AK1(A,Ik):
q=0
for d in range (N):
    if Ik[d]==0:
     AK1[:,d]=G[:,d]
     q=q+1
 #return AK1
#AK1=AK1(G,Ik)
print(AK1) 
print(q)
#for  n in range(N):
# print([row[n] for row in AK1])
 #print(AK1[2,2])

AK2=np.zeros((N,q))
for q in range(q):
 for n in range((N)):
  W=[row[n] for row in AK1]
  #print(W) 
  #print(W[n])
  if W[n]!=0:
   AK2[:,q]=np.transpose(W)
print(AK2)"""     
def h(x):
    h=5*(1-5*x)
    return h
H=[]
Hk=[]
for x in t:
    h_i=h(x)
    #print(h(x))
    H.append(h_i)
print(H)
R=np.zeros((N,N))
Bk=np.zeros((N))
def AK(A,Ik,l):
    AK=np.zeros((N,l))
    Hk=np.zeros((l))
    Rk=np.zeros((N,N))
    
    for d in range(0,N):
      #AK[:,d]=A[:,d]
      if Ik[d]==0:
        #AK[:,d]=A[:,d]
        #Hk[d]=H[d]
       # else:
        #R[:,d]=A[:,d]
        #Rk[d]=H[d]
        AK[:,l]=np.transpose([row[l] for row in A])
    Bk=np.dot(AK,Hk)    
    return AK,Hk,Bk
AK=AK(W,Ik,l) 
print(AK)
V=calclU(MAT(M,deltax,nu), b-Bk)
print(V)
plt.plot(t,V)

    
    
    



                
    

    
    