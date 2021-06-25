import gurobipy as gp

from gurobipy import GRB

from scipy.io import loadmat
import numpy as np
from csv import reader
from itertools import chain

import pandas as pd
import sys
# import pickle
import os
from datetime import datetime






homedir = 'MV/'


size = 400
sizefile = 'size'+str(size)
filedir = homedir+sizefile

# chi = 40






l=os.listdir(filedir)
li=[x.split('.')[0] for x in l]
graphlist =sorted(set(li))


def gurobiMIQP(graphname,chi,sizefile):

	dataname = 'MV/'+sizefile+'/'+graphname

	Qdatafile = dataname+'.mat'
	rhodatafile = dataname+'.rho'
	mudatafile = dataname+'.txt'
	bdsdatafile = dataname+'.bds'



	Qfin = open(Qdatafile,'r')
	rhofin = open(rhodatafile,'r')
	mufin = open(mudatafile,'r')
	bdsfin = open(bdsdatafile,'r')

	a =[]
	# for line in fin.readlines():
	# 	a.append([int(x)for x in line.split(' ')])


	rd = reader(Qfin, delimiter=' ', skipinitialspace=True)
	for row in rd:
	   a.append([int(x)for x in row])

	n = a[0][0]
	Q=a[1::]
	Qnpa = np.array(Q)
	# print(Qnpa.shape)
	# print(np.tril(Qnpa))

	logfile = 'n='+str(size)+'_'+graphname+'_aleph=' + str(chi)+'.txt'

	sys.stdout = open(logfile, "a+")

	print('----------------------------')
	print('Instance name:%s'%(graphname))
	print('----------------------------')

	print('=======================Size of intances==================')
	print('n=%s'%(size))
	print('Cardinaltiy number is %s'%(chi))
	print('=========================================================')




	rho = []
	rd = reader(rhofin, delimiter=' ', skipinitialspace=True)
	for row in rd:
	   rho.append([float(x)for x in row])

	# print(rho)
	rhonpa = np.array(rho)

	# print(rhonpa.shape)
	bds = []
	rd = reader(bdsfin, delimiter=' ', skipinitialspace=True)
	for row in rd:
	   bds.append([float(x)for x in row])

	bdsnpa = np.array(bds)

	mu = []
	rd = reader(mufin, delimiter=' ')
	for row in rd:
	   mu.append([float(x)for x in row])

	mu = mu[1::]
	munpa = np.array(mu)
	munpa = munpa[:,0]

	m1 = gp.Model('ComplementaryCon QP')

	x = m1.addVars(range(n), name="x",lb=[0]*n,ub =bdsnpa[:,1])

	y = m1.addVars(range(n),name="y",vtype =GRB.BINARY)



	m1.addConstr((sum(munpa[i]*x[i] for i in range(n)) >=rho[0][0]), "mu*x>=rho")

	m1.addConstr((sum(x[i] for i in range(n)) <=1), "e*x<=1")

	m1.addConstr((sum(y[i] for i in range(n)) >=n-chi), "e*y>=n-chi")


	m1.addConstrs((x[i]*y[i] == 0 for i in range(n)),name ='c')


	obj = sum(sum(Qnpa[i][j]*x[i]*x[j] for i in range(n)) for j in range(n))



	m1.setObjective(obj, GRB.MINIMIZE) # minimize profit


	m1.setParam("TimeLimit", 90.0)
	# m1.setParam("Threads", 1)

	m1.printStats()
	m1.optimize()

	bestObj = m1.getObjective().getValue()

	m1.printQuality()
	# m1.printAttr('x')
	xsol = m1.getAttr('x')[:n]

	x = np.array(xsol)
	print('sum(x): ',sum(x))
	print('mu*x: ',np.inner(x,munpa))


	TimeLimit = m1.getParamInfo('TimeLimit')

	gap = m1.MIPGap

	sys.stdout.close()

	return chi,xsol,bestObj,TimeLimit,gap


for chi in [5,10,20]:
	results = {}
	instanceslist = []
	objlist = []
	activesetlist = []
	chivalue= []
	timelist = []
	optGAP = []

	for graphname in graphlist:
		chiV,xsol,bestObj,t,gap = gurobiMIQP(graphname,chi,sizefile)
		instanceslist.append(graphname)
		chivalue.append(chiV)
		optGAP.append(gap)
		objlist.append(bestObj)
		activesetlist.append(np.nonzero(xsol))
		timelist.append(t)

	results['graphname'] = instanceslist
	results['Chi'] = chivalue
	results['bestObj'] = objlist
	results['Optimality Gap'] = optGAP
	results['Xact'] = activesetlist
	results['Computation Time'] = timelist



	csv_file = datetime.now().strftime("%Y%m%d-%H%M%S")+"_gurobi_Cardinality{}_{}_{}.csv".format(size,chi,instanceslist[0])


	df = pd.DataFrame.from_dict(results)

	df.to_csv(csv_file)
