from scipy.io import loadmat
import numpy as np
import csv
from itertools import chain

import pandas as pd
import sys
# import pickle
import os
import mosek
import time
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from datetime import datetime

homedir = 'MV/'


size = 400
sizefile = 'size'+str(size)
filedir = homedir+sizefile

# chi = 40




l=os.listdir(filedir)
li=[x.split('.')[0] for x in l]
graphlist =sorted(set(li))


def streamprinter(text):
	    sys.stdout.write(text)
	    sys.stdout.flush()

def plot_matrix(M,f):
	"""
	helper function to plot two colormaps
	"""
	n= len(M)
	data = np.zeros([n,n])
	for i in range(n):
	    for j in range(n):
	        data[i,j] = M[j,i]


	fig= plt.figure(f)
	viridis = plt.cm.get_cmap('viridis', 256)
	newcolors = viridis(np.linspace(0, 1, 256))
	# pink = np.array([1])
	# newcolors[:25, :] = pink
	newcmp = ListedColormap(newcolors)
	# psm = plt.pcolormesh(data, cmap=newcmp, vmin=0, vmax=1)
	psm=plt.imshow(data.T,origin='upper',cmap=newcmp, vmin=-0.05, vmax=1)
	fig.colorbar(psm,)
	plt.show()


def list2matrix(L1,Dim):
	M = np.zeros((Dim,Dim))
	# M = [[0] * Dim for i in range(Dim)]
	s = 0
	for j in range(Dim):
		t = Dim-j
		for i in range(t):
			M[Dim-t+i,j] = L1[s+i]
			M[j,Dim-t+i] = L1[s+i]
		s = s +t

	xx =np.outer(M[:,0],M[0,:])
	Sigma = M- xx

	eigs = np.sort_complex(np.linalg.eigvals(M))[:-6:-1]
	print('eigen values:%s'%(eigs,))
	return M,eigs,np.linalg.norm(Sigma)

def realObj(M,n,Qnpa):
	x = M[0][1:(n+1)]

	var = np.dot(np.dot(x,Qnpa),x.T)
	return var

def ComplementaryVio(M,n):
	x = M[0][1:(n+1)]
	y = M[0][(n+1):(2*n+1)]
	# print("x is:%s\n"%(x,))
	# print("y is:%s\n"%(y,))
	vio = []
	for i in range(n):
		vio = [] + [abs(x[i]*y[i])]
	vio_max = max(vio)
	return vio_max

def activeset(x,y,chi):
	n = len(x)
	# discount = 0.01
	# count = 0
	x_temp = np.zeros(n)
	y_temp = np.ones(n)
	xact = (-x).argsort()[:chi]
	for idx in xact:
		x_temp[idx]= x[idx]
		y_temp[idx] = 1
	# while num_act >=chi+1 and count<5000:
	# 	x_temp = x.copy()
	# 	y_temp = y.copy()
	# 	for i in range(n):
	# 		if y_temp[i]>discount:
	# 			y_temp[i]=1
	# 			x_temp[i]=0
	# 		else:
	# 			y_temp[i]=0
	# 	if num_act<=chi:
	# 		discount = discount+0.01
	# 	else:
	# 		discount = discount-0.01
	# 	count = count +1
	# 	num_act = len(x_temp[x_temp>0])
	# 	# print(num_act)

	# xact =np.where(x_temp>0)

	xact.sort()

	print("x is:%s\n"%(x_temp[xact],))
	print("x active set is:%s\n"%(xact,))
	# print(num_act)
	# print("y is:%s\n"%(y_temp,))
	return x_temp,y_temp,xact


def OrginalSol(M,n,chi,bdsnpa,munpa,rho,Qnpa):
	x = M[0][1:(n+1)]
	y = M[0][(n+1):(2*n+1)]

	x1,y1,xactset=activeset(x,y,chi)

	vio1 = min(0,np.dot(x1,munpa)-rho[0][0])
	vio2 = min(1-sum(x1),0)
	vio3 = min(sum(y1)-n+chi,0)
	viobox = np.where(x1>bdsnpa[:,1])
	obj = np.dot(np.dot(x1,Qnpa),x1.T)
	print('After post process:')
	print('x box constraint is violated in %s as %s with upperbound %s\n'%(viobox,x[viobox],bdsnpa[viobox,1]))
	print("mu*x >= rho is violated:%s\n"%(vio1,))
	print('mu*x is%s\n'%(np.dot(x,munpa),))
	print('rho is%s\n'%(rho[0][0],))
	print("sum(x)<=1 is violated:%s\n"%(vio2,))
	print("sum(y)>=n-chi is violated:%s\n"%(vio3,))
	return viobox,vio1,vio2,vio3,xactset


def rSDP(graphname,chi,sizefile):

	dataname = 'MV/'+sizefile+'/'+graphname
	Qdatafile = dataname+'.mat'
	rhodatafile = dataname+'.rho'
	mudatafile = dataname+'.txt'
	bdsdatafile = dataname+'.bds'

	# with open(datafile) as f:
	#     a = f.read().splitlines()

	Qfin = open(Qdatafile,'r')
	rhofin = open(rhodatafile,'r')
	mufin = open(mudatafile,'r')
	bdsfin = open(bdsdatafile,'r')

	a =[]
	# for line in fin.readlines():
	# 	a.append([int(x)for x in line.split(' ')])


	rd = csv.reader(Qfin, delimiter=' ', skipinitialspace=True)
	for row in rd:
	   a.append([int(x)for x in row])

	n = a[0][0]
	Q=a[1::]
	Qnpa = np.array(Q)
	# print(Qnpa.shape)
	# print(np.tril(Qnpa))

	logfile = 'rSDP_n='+str(size)+'_'+graphname+'_aleph=' + str(chi)+'.txt'
	sys.stdout = open(logfile, "a+")

	print('----------------------------')
	print('Instance name:%s'%(graphname))
	print('----------------------------')



	print('=======================Size of intances==================')
	print('n=%s'%(size))
	print('Cardinaltiy number is %s'%(chi))
	print('=========================================================')





	rho = []
	rd = csv.reader(rhofin, delimiter=' ', skipinitialspace=True)
	for row in rd:
	   rho.append([float(x)for x in row])

	# print(rho)
	rhonpa = np.array(rho)

	# print(rhonpa.shape)
	bds = []
	rd = csv.reader(bdsfin, delimiter=' ', skipinitialspace=True)
	for row in rd:
	   bds.append([float(x)for x in row])

	bdsnpa = np.array(bds)


	# print(bdsnpa.shape)



	mu = []
	rd = csv.reader(mufin, delimiter=' ')
	for row in rd:
	   mu.append([float(x)for x in row])

	mu = mu[1::]
	munpa = np.array(mu)
	munpa = munpa[:,0]
	# print(munpa.shape)


	# def main():
	with mosek.Env() as env:
		env.set_Stream(mosek.streamtype.log, streamprinter)
		with env.Task(0, 0) as task:
			task.set_Stream(mosek.streamtype.log,streamprinter)

			bkc = []
			blc = []
			buc = []


			# numCON= 4+2*n + (1+2*n)*n
			numCON= 4+3*n
			barD = n*2+1
			BarVarDim = [barD]

			task.appendcons(numCON)
			task.appendbarvars(BarVarDim)


	  		# SDP constraints
			# lower tri

			inf = 0.0
			# mu*x >= rho
			barai1 = range(1,n+1,1)
			baraj1 = [0]*n
			baraval1 = 0.5*munpa

			sym1 = task.appendsparsesymmat(barD,barai1,baraj1,baraval1)
			bkc = bkc+ [mosek.boundkey.lo]
			blc = blc+ [rho[0][0]]
			buc = buc+ [+inf]

			# e*x <= 1
			barai2 = range(1,n+1,1)
			baraj2 = [0]*n
			baraval2 = [0.5]*n
			sym2 = task.appendsparsesymmat(barD,barai2,baraj2,baraval2)
			bkc = bkc+ [mosek.boundkey.up]
			blc = blc+ [-inf]
			buc = buc+ [1]


			# e*y ==n-\chi
			barai3 = range(n+1,2*n+1,1)
			baraj3 = [0]*n
			baraval3 = [1/2]*n
			sym3 = task.appendsparsesymmat(barD,barai3,baraj3,baraval3)
			bkc = bkc+ [mosek.boundkey.fx]
			blc = blc+ [n-chi]
			buc = buc+ [n-chi]





			# x00 =1
			barai4 = [0]
			baraj4 = [0]
			baraval4 = [1]
			sym4 = task.appendsparsesymmat(barD,barai4,baraj4,baraval4)
			bkc = bkc+ [mosek.boundkey.fx]
			blc = blc+ [1]
			buc = buc+ [1]





			task.putbaraij(0,0,[sym1],[1])
			task.putbaraij(1,0,[sym2],[1])
			task.putbaraij(2,0,[sym3],[1])
			task.putbaraij(3,0,[sym4],[1])

			#xiyi =0
			for i in range(n):
				barai = [i+1+n]
				baraj = [i+1]
				baraval = [1/2]
				sym = task.appendsparsesymmat(barD,barai,baraj,baraval)
				bkc = bkc+ [mosek.boundkey.fx]
				blc = blc+ [0]
				buc = buc+ [0]
				task.putbaraij(4+i,0,[sym],[1])


			#  yi*1 ==yii
			for i in range(n,2*n):
				barai = [1+i]*2
				baraj = [0,1+i]

				baraval = [0.5*1,-1]
				sym = task.appendsparsesymmat(barD,barai,baraj,baraval)

				bkc = bkc+ [mosek.boundkey.fx]
				blc = blc+ [0]
				buc = buc+ [+inf]
				task.putbaraij(4+i,0,[sym],[1])


			# bound for x
			for i in range(n):
				barai = [1+i]
				baraj = [0]
				baraval = [0.5]
				sym = task.appendsparsesymmat(barD,barai,baraj,baraval)
				bkc = bkc+ [mosek.boundkey.ra]
				blc = blc+ [0]
				buc = buc+ [bdsnpa[i,1]]
				task.putbaraij(4+2*n+i,0,[sym],[1])

			# # doubly nonnegative constrains
			# count = 0
			# for i in range(1,2*n+1,1):
			# 	for j in range(i):
			# 		barai = [i]
			# 		baraj = [j]
			# 		baraval = [0.5]
			# 		sym = task.appendsparsesymmat(barD,barai,baraj,baraval)
			# 		if (j==0) and (i<n+1):
			# 			bkc = bkc+ [mosek.boundkey.ra]
			# 			buc = buc+ [bdsnpa[i-1,1]]
			# 		else:
			# 			bkc = bkc+ [mosek.boundkey.lo]
			# 		blc = blc+ [0]
			# 		buc = buc+ [+inf]

			# 		try:
			# 			task.putbaraij(4+2*n+count,0,[sym],[1])
			# 		except mosek.Error:
			# 			print(count)
			# 		count = count +1



			# objective function

			(I,J) = np.nonzero(np.tril(Qnpa))

			barci1 = I+1
			barcj1 = J+1

			# barci1_t = [list(range(i,n+1,1)) for i in range(1,n+1,1)]
			# barci1 = [val for sublist in barci1_t for val in sublist]
			# barcj1_t = [[i]*(n+1-i) for i in range(1,n+1,1)]
			# barcj1 = [val for sublist in barcj1_t for val in sublist]

			barcval1 = Qnpa[np.nonzero(np.tril(Qnpa))]


			symc1 = task.appendsparsesymmat(barD,barci1,barcj1,barcval1)
			task.putbarcj(0,[symc1],[1])



			for i in range(numCON):
				task.putconbound(i,bkc[i],blc[i],buc[i])

			# Input the objective sense (minimize/maximize)
			task.putobjsense(mosek.objsense.minimize)

			time_start = time.clock() # measure cpu time in python

			task.putdouparam(mosek.dparam.intpnt_co_tol_rel_gap, 1.0e-10)

			# Solve the problem and print summary
			task.optimize()
			# task.solutionsummary(mosek.streamtype.msg)
			#run your code
			time_elapsed = (time.clock() - time_start)

			# Get status information about the solution
			prosta = task.getprosta(mosek.soltype.itr)
			solsta = task.getsolsta(mosek.soltype.itr)
			tm = task.getdouinf(mosek.dinfitem.optimizer_time)


			if (solsta == mosek.solsta.optimal):
				lenbarvar = BarVarDim[0]*(BarVarDim[0] + 1) / 2
				barx = [0.] * int(lenbarvar)
				task.getbarxj(mosek.soltype.itr,0, barx)

				pobj, pviolcon, pviolvar, pviolbarvar, pviolcones, pviolitg,\
				dobj, dviolcon, dviolvar, dviolbarvar, dviolcones = task.getsolutioninfo(mosek.soltype.itr)
				# print(pobj)

				M,eigs,sigma = list2matrix(barx,barD)

				print('Negative element in M: ', M[M<-1e-9] )

				# print(M[1:401,0])
				ids1 = [0] + list(range(201,401,1))
				ids2 = [0,167,367,368]

				# plot_matrix(M,0)
				# plot_matrix(M[range(1,201,1),:][:,range(201,401,1)],1)
				# plot_matrix(M[ids1,:][:,ids1],1)
				# plot_matrix(M[ids2,:][:,ids2],2)

				# print("The residual of SDP matrix is:%s\n"%(sigma,))
				# print("The rank of SDP matrix is:%s\n"%(rank,))
				vio = ComplementaryVio(M,n)
				print("The Complementary Violation is:%s \n"%(vio,))
				apprObj =realObj(M,n,Qnpa)
				viobox,vio1,vio2,vio3,xactset = OrginalSol(M,n,chi,bdsnpa,munpa,rho,Qnpa)
				# print("xQx' is:%s\n"%(apprObj,))

			elif (solsta == mosek.solsta.dual_infeas_cer or solsta == mosek.solsta.prim_infeas_cer):
				print("Primal or dual infeasibility certificate found.\n")
			elif solsta == mosek.solsta.unknown:
				print("Unknown solution status")
			else:

				print("Other solution status")

	# try:
	# 	main()
	# except mosek.MosekException as e:
	# 	print("ERROR: %s" % str(e.errno))
	# 	if e.msg is not None:
	# 		print("\t%s" % e.msg)
	# 		sys.exit(1)
	# except:
	# 	import traceback
	# 	traceback.print_exc()
	# 	sys.exit(1)

	sys.stdout.close()
	return pobj,xactset,vio,time_elapsed,tm,eigs


for chi in [5,10]:

	results = {}
	instanceslist = []
	pobjlist = []
	activesetlist = []
	violist = []
	chivalue= []
	# chivalue = [2,5,10,20,40]
	timelist = []
	mosektimelist = []
	Eigslist = []
	# print(graphlist[:1])
	# syse.exit()



	for graphname in graphlist:
		# print('Chi is:%s'%(chi)
		pobj,actset,vio,cput,tm,Eigs = rSDP(graphname,chi,sizefile)
		instanceslist.append(graphname)
		chivalue.append(chi)
		pobjlist.append(pobj)
		violist.append(vio)
		activesetlist.append(actset)
		timelist.append(cput)
		mosektimelist.append(tm)
		Eigslist.append(Eigs)

	results['graphname'] = instanceslist
	results['Chi'] = chivalue
	results['pObj'] = pobjlist
	results['Xact'] = activesetlist
	results['violation'] = violist
	results['Computation Time (CPU)'] = timelist
	results['Computation Time (clock)']= mosektimelist
	results['Largest 5 eignvalues'] = Eigslist

	# sys.exit()

	csv_file = datetime.now().strftime("%Y%m%d-%H%M%S")+"_rSDP_Cardinality{}_{}_{}_ineq.csv".format(size,instanceslist[0],chi)


	df = pd.DataFrame.from_dict(results)


	df.to_csv(csv_file)
