#imports usual modules
import numpy
import scipy
from scipy import linalg
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re

'''This module will calculate the reciprocal lattice vecotrs and plot the energies for silicon (fcc) and maybe bcc and simple'''


############################################################


'''constants are input here'''

prop=0
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


    if struc == numpy.str('fcc'):
    #for face centred cubic
        for nx in nxs:
            for ny in nys:
                for nz in nzs:
                    g = numpy.array([-nx+ny+nz, nx-ny+nz, nx+ny-nz])
                    rlvs.append(g)
  
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

    
    rlvs.sort(key=mags) 
    # sorts the rlvs in order of magnitude



    return(rlvs[:51]) 
    #for the purpose of the first excercise, this only returns the first 15 rlvs

############################################################


'''potential are defined here'''

convert = 13.6056980659
#form factors germanium
V3G = -0.238 * convert
V8G = 0.0038 * convert
V11G = 0.068 * convert
ffg = numpy.array([V3G, V8G, V11G])

#form factors silicon
# V3S = -3.04768 
# V8S =  0.74831
# V11S = 0.97961
V3S = -0.224 * convert
V8S =  0.055 * convert
V11S = 0.072 * convert
ffs = numpy.array([V3S, V8S, V11S])




ffg = ffg*(prop)
ffs = ffs*(1-prop)

ffc = numpy.add(ffg,ffs)
#compound form factor


#a function to caluculate structure factor
def strucfact(g):
    nxprime, nyprime, nzprime = g
    #this g will be the vector g - gprime; corresponds to the position in the matrix
    sf = numpy.cos((numpy.pi/4)*(nxprime + nyprime + nzprime))

    return sf


############################################################


'''schrodinger equation is calculated here'''


def matrix(N,struc,k,V):
#utilise a matrix formulation to find the energies of the bands
    gs = RLVS(N,struc)  
    gprimes = gs 

    pop = []
    #a list of numbers that we be reshaped to become the appropriate matrix as numpy doesn't like using matrices.


    

    for i in range(0,len(gs)):
        for j in range(0,len(gprimes)):
    #indexes the coordinates within the matrix

            deltag = gs[i] - gprimes[j] 
            #g'' in the notes

            energy = 0 
            #initialises energy value

            if deltag.dot(deltag) == 0:
            # this calculates the diagonal energy terms with the condition if g = grpime
                modkgs = (k+(gs[i])).dot(k+(gs[i]))
                energy += c_energy*abs(modkgs)
            


            #adds the potential terms with the condiditon that g'' has a certain value
            if deltag.dot(deltag) == 3:
                energy += (V[0] * strucfact(deltag))           
            
            if deltag.dot(deltag) == 8:
                energy += (V[1] * strucfact(deltag))     

            if deltag.dot(deltag) == 11:
                energy += (V[2] * strucfact(deltag))            


            
            #all other terms are appended with 0
            pop.append(energy)

    matrix = numpy.matrix(numpy.array(pop).reshape(len(gs),len(gprimes)))
    #creates a matrix of size g x gprime with the appropriate values in each spot

    return (matrix)

# print(matrix(3,'fcc', numpy.array([1,0,0]), ffs))
############################################################


'''this finds the eignenvalues of the s.e. matrix'''


#a function that calculates the eigenvalues of a matrix 
def eigenvalues(N, struc, k, V):
    se = matrix(N, struc, k, V)
    #extra line of code to make it look a bit cleaner
    eigenvalues = numpy.real(numpy.sort(scipy.linalg.eigvals(se)))


    return(eigenvalues)

print(eigenvalues(3,'fcc', numpy.array([1,0,0]), ffs))

############################################################


'''graph'''


alpha  = numpy.arange(0.0,1, 0.01)
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
    kx = (numpy.array([1,0,0])*alpha[i])
    magkx = mags(kx)
    kxs.append(kx)
    plotkx.append(magkx)

    kl = numpy.array([.5,.5,.5])*alpha[i]
    magkl = (-1)*mags(kl)
    kls.append(kl)
    plotkl.append(magkl)


    kk = numpy.array([.75,.75,.75])*alpha[i]
    magkk = mags(kk)
    kks.append(kk)
    plotkk.append(magkk)


######    ######    ######    ######    ######    ###### 


N = 3
#declares the range of N values (and hence the number of reciprocal lattice vectors)

for rlv in range(len(RLVS(N, 'fcc'))):

    xenergies = []
    #an empty list to be filled with energies for each reciprocal lattice vector
    
    for x in range(len(kxs)):
        #loop calculates energies for each k vector
        
        xenergy = eigenvalues(N, 'fcc', kxs[x], ffc) 
        xenergies.append(xenergy)
        #appends the energy values to the energy list

    #repeats for lambda axis
    lenergies = []

    for l in range(len(kls)):

        lenergy = eigenvalues(N, 'fcc', kls[l], ffc) 
        lenergies.append(lenergy)

    lenergies.reverse()


energies = lenergies + xenergies 
#places all energy values in one list for the plot
energies = numpy.array(energies)
#converts the list of energies to an array so it can be enumaerated

plote = [energies[:,i] for i,e in enumerate(energies[0])]
#enumarates the list to separate the different energies out for different values of k to be plotted



plotks = (plotkl + plotkx)
plotks.sort()
#creates a list of odered ks to plot against

#########          #########          #########          #########          

plt.figure(figsize=(6,6))
#graph time

# for e in plote:
    # plt.plot(plotks,e,label='j')
    #plots the energies over each k value

#usual graph stuffs
plt.xlabel('k')
plt.ylabel('Energy')
plt.legend()
# plt.savefig('interpolv.png')


############################################################


'bandgap is determined here'


#analytically

l4 = numpy.max(plote[3])

plt.plot(plotks, plote[3])

max_k = plotks[plote[3].argmax()] 

l5 = numpy.min(plote[4])
plt.plot(plotks,plote[4])
min_k = plotks[plote[4].argmin()]
l5d = plote[4]
gap = l5-l4
print(gap, max_k, min_k)

plt.show()

