#imports usual modules
import numpy
import scipy
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pandas as pd
import tqdm

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})


'''This module will calculate the reciprocal lattice vecotrs and plot the energies for silicon (fcc) and maybe bcc and simple'''


############################################################


'''constants are input here'''

#defines the fraction of germanium
prop=1

print('Compound = Si',(1-prop),'Ge',(prop))

#lattice parameter
asil = 5.43*10**-10 
ager = 5.65*10**-10
a = ((1-prop)*asil) + (prop*ager)
#lattice constant of material
c =2*numpy.pi/a 
#lattice constant divided by half bc python likes integers
melec= 9.11*10**-31 
#mass of an electron
e = 1.61*10**-19 
#elementary charge
hbar= 1.0534*10**-34 
#hbar

c_energy = c**2*(hbar**2)/(2*melec*e)
#constant in front of the delta funtcion in the s.e


############################################################


'''reciprocal lattice vectors are genertaed here'''


#a function to find the magnitude of a vector
def mags(value):
    return numpy.sqrt(abs(value.dot(value)))


#a function to generate reciprocal lattice vectors
def RLVS(N,struc):

    rlvs = []
    nxs=[]
    for i in range(-N,N+1):
        nxs.append(i)
    nys,nzs=nxs,nxs


    if struc == 'fcc':
    #for face centred cubic
        for nx in nxs:
            for ny in nys:
                for nz in nzs:
                    g = numpy.array([-nx+ny+nz, nx-ny+nz, nx+ny-nz])
                    rlvs.append((g))
  
    elif struc == 'bcc':
    #for body centred cubic
        for nx in nxs:
            for ny in nys:
                for nz in nzs:
                    g = numpy.array([nx+ny, nx+nz, ny+nz])
                    rlvs.append(g)

    elif struc == 'simple cubic':
    #for simple cubic
        for nx in nxs:
            for ny in nys:
                for nz in nzs:
                    g = numpy.array([nx, ny, nz])
                    rlvs.append(g)


    else:
    #those are the only ones I know
        print('Invalid input')

    
    
    # sorts the rlvs in order of magnitude
    rlvs = sorted(rlvs, key=lambda x: max(abs(x)))
    rlvs.sort(key=mags) 
    return(rlvs[:181])
    #for the purpose of the first excercise, this only returns the first 15 rlvs


result = RLVS(4,'fcc')


for i in result:
    print(i)
############################################################


'''potential are defined here'''

convert = 13.61
# #form factors germanium
V3G = -3.12931 
V4G = 0
V8G = 0.13606 
V11G = 0.81634 
ffg = numpy.array([V3G, V4G, V8G, V11G])

# V3G =  -0.2768 
# V4G = -00
# V8G = 0.0582
# V11G = 0.0152 
# ffg = numpy.array([V3G, V4G, V8G, V11G]) * convert

#form factors silicon
V3S = -3.04768
V4S = 0
V8S =  0.74831 
V11S = 0.97961
ffs = numpy.array([V3S, V4S, V8S, V11S])

#compound form factors
ffc = (1-prop)*ffs + prop*ffg


#a function to caluculate structure factor
def strucfact(g):
    nxprime, nyprime, nzprime = g
    #this g will be the vector g - gprime; corresponds to the position in the matrix
    sf = numpy.cos((numpy.pi/4)*(nxprime + nyprime + nzprime))

    return sf


############################################################


'''schrodinger equation is calculated here'''


def matrix(k,V):
#utilise a matrix formulation to find the energies of the bands
    gs = result
    gprimes = gs 

    matrix = []


    for i in range(0,len(gs)):
        matrix.append([])
        #creates a blank row that is filled
        for j in range(0,len(gprimes)):
        #indexes the coordinates within the matrix

            deltag = gs[i] - gprimes[j] 
            
            #g'' in the notes

            energy = 0 
            #initialises energy value

            if numpy.array_equiv(gs[i],gprimes[j])==True:
            # this calculates the diagonal energy terms with the condition if g = gprime
                modkgs = (k+(gs[i])).dot(k+(gprimes[j]))
                energy += c_energy*(modkgs)


            #adds the potential terms with the condiditon that 'g' has a certain value
            if deltag.dot(deltag) == 3:
                energy += (V[0] * strucfact(deltag)) 
                
            if deltag.dot(deltag) == 8:
                energy += (V[2] * strucfact(deltag)) 

            if deltag.dot(deltag) == 11:
                energy += (V[3] * strucfact(deltag))   


            
            #all other terms are appended with 0
            matrix[i].append(energy)

    return (matrix)

############################################################


'''this finds the eignenvalues of the s.e. matrix'''


#a function that calculates the eigenvalues of a matrix 
def eigenvalues(k, V):
    se = matrix(k, V)
    #extra line of code to make it look a bit cleaner
    eigenvalues = numpy.real(numpy.sort(scipy.linalg.eigvals(se)))


    return(eigenvalues)


############################################################


'''this generates the energy levels as functions of k'''


alpha  = numpy.arange(0.0,1, 0.02)
#defines the delineation of the k values

kxs = []
kls = []
kks = []
#empty lists to be filled with vectors of k for calculating eigenvectors
plotkx = []
plotkl = []
plotkk = []
#empty lists to be filled with magnitudes of k for plotting

for i in range(len(alpha)):
#loop that generates k vectors in the x direction
    kx = numpy.array([1,0,0])*alpha[i]
    magkx = mags(kx)
    kxs.append(kx)
    plotkx.append(magkx)

    kl = numpy.array([.5,.5,.5])*alpha[i]
    magkl = (-1)*mags(kl)
    kls.append(kl)
    plotkl.append(magkl)

    #forbidden direction
    kk = numpy.array([.75,.75,0])*alpha[i]
    magkk = mags(kk)
    kks.append(kk)
    plotkk.append(magkk)


######    ######    ######    ######    ######    ###### 


N = 3
#declares the range of N values (and hence the number of reciprocal lattice vectors)
index=iter(numpy.arange(1,344))


for rlv in result:

    xenergies = []
    lenergies = []
    #an empty list to be filled with energies for each reciprocal lattice vector
    
    for x in range(len(kxs)):
    
        #loop calculates energies for each k vector
        
        xenergy = eigenvalues(kxs[x], ffc) 
        xenergies.append(xenergy)
        #appends the energy values to the energy list

        #repeats for lambda axis
        lenergy = eigenvalues(kls[x], ffc) 
        lenergies.append(lenergy)

    #reversed to follow plotting convention
    lenergies.reverse()
    print(next(index))
    


energies = lenergies + xenergies 
#places all energy values in one list for the plot
energies = numpy.array(energies)
#converts the list of energies to an array so it can be enumaerated
plote = [energies[:,i] for i,e in enumerate(energies[0])]
#enumarates the list to separate the different energies out for different values of k to be plotted

plotks = plotkl + plotkx
plotks.sort()


#creates a list of odered ks to plot against
numpy.savetxt('data.csv', (plotks, plote[4]), delimiter=',', fmt='%d')


############################################################


'bandgap is determined here'


#graphically

l4 = numpy.max(plote[3]) #valence band
max_k = plotks[plote[3].argmax()] 

l5 = numpy.min(plote[4]) #conduction band
min_k = plotks[plote[4].argmin()]

gap = l5-l4
print(gap, max_k, min_k)


############################################################


'effective mass'

delta = 5

fitk = numpy.array(plotks[:])*c
cbe = plote[4][:]*e

# cbe = plote[4] #condutcion band energy

# y = numpy.zeros((10,10))
y = []

for i in range(len(fitk)):
    y.append([])
    for j in range(len(fitk)):
        cb = cbe[i]+ cbe[j]
            
        y[i].append(cb)


k0x = numpy.where(y == numpy.min(y))[0]
k0y = numpy.where(y == numpy.min(y))[1]
print(k0x,k0y)

dk0x = k0x[0] + delta
dk0xm = k0x[0] - delta
dk0y = k0y[0] + delta
dk0ym = k0y[0] - delta
print(dk0x,dk0y)





######    ######    ######    ######    ######    ###### 

# fitk = numpy.array(plotks[150:])*c
# fite = plote[4][150:]*e
# # print(fitk,fite)
# p=numpy.polyfit(fitk,fite,2)


# fitk = numpy.array(fitk)
# d2e =  numpy.polyfit(fitk,fite,2)[0]
# ee= p[0]*fitk**2 + p[1]*fitk + p[2]

# meff = (hbar**2) / (d2e*melec)
# print(meff)

#https://scicomp.stackexchange.com/questions/2246/recommendation-for-finite-difference-method-in-scientific-python

#https://hplgit.github.io/fdm-book/doc/pub/book/pdf/fdm-book-4print-2up.pdf


############################################################


'graph time'


plt.figure(figsize=(6,6))
plote = plote[:14]
#don't want to be overkill
colour=iter(cm.magma(numpy.linspace(0,1,15)))
level = iter(numpy.linspace(1,14,14))

#to label
for e,label in zip(plote, level):
    c= next(colour)
    # label=next(level)
    plt.plot(plotks,e)
    # plots the energies over each k value
# plt.plot(fitk, ee,'r', linewidth=3)
# plt.axhline(l4,linewidth=1,color='grey',linestyle='--')
# plt.axhline(l5,linewidth=1,color='grey',linestyle='--')


#usual graph stuffs
plt.xlabel('k')
plt.ylabel('Energy')
plt.title(1-prop)
plt.legend()
# plt.savefig('interpolv.png')
plt.show()

