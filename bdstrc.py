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

#lattice parameter
asil = 5.43*10**-10 
ager = 5.65*10**-10

#lattice constant divided by half bc python likes integers
melec= 9.11*10**-31 
#mass of an electron
e = 1.61*10**-19 
#elementary charge
hbar= 1.0534*10**-34 
#hbar




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
    
    return(rlvs[:137])
    #for the purpose of the first excercise, this only returns the first 15 rlvs


rlvs = RLVS(4,'fcc')


############################################################
#defines the fraction of germanium


'''potential are defined here'''

convert = 13.61
#form factors germanium
# V3G = -3.12931 
# V8G = 0.13606 
# V11G = 0.81634 
# ffg = numpy.array([V3G, V8G, V11G])

V3G =  -0.2769
V8G = 0.0583
V11G = 0.0153
ffg = numpy.array([V3G, V8G, V11G]) * convert



#form factors silicon
V3S = -3.04768
V8S =  0.74831 
V11S = 0.97961
ffs = numpy.array([V3S, V8S, V11S])




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
    gs = rlvs

    matrix = []


    for i in range(len(gs)):
        matrix.append([])
        #creates a blank row that is filled
        for j in range(len(gs)):
        #indexes the coordinates within the matrix

            deltag = gs[i] - gs[j] 
            
            #g'' in the notes

            energy = 0 
            #initialises energy value

            if deltag.dot(deltag) == 0:
            # this calculates the diagonal energy terms with the condition if g = gprime
                modkgs = (k+(gs[i])).dot(k+(gs[j]))
                energy += c_energy*(modkgs)


            #adds the potential terms with the condiditon that 'g' has a certain value

            if deltag.dot(deltag) == 3:
                energy += (V[0] * strucfact(deltag))           
                
            if deltag.dot(deltag) == 8:
                energy += (V[1] * strucfact(deltag)) 

            if deltag.dot(deltag) == 11:
                energy += (V[2] * strucfact(deltag))   


            
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




alpha  = numpy.arange(0.0,1, 0.01)
#defines the delineation of the k values

kxs = []
kls = []
kks = []

ktransx = []
ktransl = []
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
    kk = numpy.array([0.75,0.75,0])*alpha[i]
    magkk = mags(kk)
    kks.append(kk)
    plotkk.append(magkk)

    kt = numpy.array([0,1,0])*alpha[i]
    ktransx.append(kt)

    klo = numpy.array([0,-0,1])*alpha[i]
    ktransl.append(klo)

######    ######    ######    ######    ######    ###### 


props = numpy.array([0,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1])

bandgaps = []
max_ks = []
mls = []
mts = []
ms = []
for prop in props:
    print('Compound = Si',(1-prop),'Ge',(prop))

    x=prop
    a = ( 5.431 + 0.20*x + 0.045*x**2) *10**-10

    #lattice constant of material
    c =2*numpy.pi/a 
    c_energy = c**2*(hbar**2)/(2*melec*e)
    #constant in front of the delta funtcion in the s.e

    #compound form factors
    ffc = (1-prop)*ffs + prop*ffg 


    #declares the range of N values (and hence the number of reciprocal lattice vectors)
    index=iter(numpy.arange(1,344))

    #an empty list to be filled with energies for each reciprocal lattice vector

    xenergies = []
    lenergies = []    


    for x in range(len(kxs)):
        #loop calculates energies for each k vector
        
        xenergy = eigenvalues(kxs[x], ffc) 
        xenergies.append(xenergy)
        #appends the energy values to the energy list

        #repeats for lambda axis
        lenergy = eigenvalues(kls[x], ffc) 
        lenergies.append(lenergy)

        print(next(index))

    #reversed to follow plotting convention
    lenergies.reverse()
    plotkl.reverse()

    energies = lenergies + xenergies 
    #places all energy values in one list for the plot
    energies = numpy.array(energies)

    #converts the list of energies to an array so it can be enumaerated
    plote = [energies[:,i] for i,e in enumerate(energies[0])]
    # print(plote)
    #enumarates the list to separate the different energies out for different values of k to be plotted

    plotks = plotkl + plotkx

    #creates a list of odered ks to plot against
    d=list(zip(plotks,plote[3],plote[4]))
    df = pd.DataFrame(d)

    df.to_csv('{}xdatagood.csv'.format(prop))


    ############################################################


    'bandgap is determined here'


    #graphically

    l4 = numpy.max(plote[3]) #valence band
    max_k = plotks[plote[3].argmax()] 

    l5 = numpy.min(plote[4]) #conduction band
    min_k = plotks[plote[4].argmin()]

    gap = l5-l4
    bandgaps.append(gap)
    print(gap, max_k, min_k)


    ############################################################


    'effective mass'
    # min_k=max_k
    delta = 2

    fitk = numpy.array(plotks[:])*c**2

    cbe = plote[4][:]*e
    kxs = numpy.array(kxs)

    if min_k > 0:
        ktran = (list(numpy.array([min_k,0,0])-ktransx))[::-1] + list(ktransx + numpy.array([min_k,0,0])) 

    

    else:
        min_k = numpy.abs(min_k)
        ktran = (list(numpy.array([min_k/(2**.5),min_k/(2**.5),0])-ktransl))[::-1] + list(ktransl + numpy.array([min_k/(2**.5),min_k/(2**.5),0])) 


    # for i in ktran:
    #     print(i) 
    etrans = []
    for x in range(len(ktran)):
        etran = eigenvalues(ktran[x], ffc) 
        etrans.append(etran)
    etrans = numpy.array(etrans)
    etrans2 = [etrans[:,i] for i,e in enumerate(etrans[0])]

    cbey = etrans2[4][:]*e

    d2=list(zip(ktran,cbey/e))
    df2 = pd.DataFrame(d2)

    df2.to_csv('{}ydatagood.csv'.format(prop))


    # y = numpy.zeros((10,10))
    y = []

    for i in range(len(ktran)):
        y.append([])
        for j in range(len(ktran)):
            cb = cbe[i] + cbey[j]
                
            y[i].append(cb)


    k0x = numpy.where(y == numpy.min(y))[0]
    k0y = numpy.where(y == numpy.min(y))[1]
    # print(k0x,k0y)



    d2e = (y[k0x[0]+delta][delta] + y[k0x[0]+delta][k0x[0]-delta] + y[k0x[0]-delta][k0x[0]+delta] + y[k0x[0]-delta][k0x[0]-delta])/(4*delta**2)
    m =  (hbar**2)*c**2 / (melec*d2e) *10**-1
    ms.append(m)
    print('x&y=',m)
    d2ex = (y[k0x[0]+delta][k0x[0]] - 2*y[k0x[0]][k0x[0]] + y[k0x[0]-delta][k0x[0]])/delta**2
    mt =  (hbar**2)*c**2 / (d2ex * melec) *10**-4
    mts.append(mt)
    print('x only =',mt)
    d2ey = (y[k0x[0]][k0x[0]+delta] - 2*y[k0x[0]][k0x[0]] + y[k0x[0]][k0x[0]-delta])/delta**2
    ml = (hbar**2)*c**2 / (d2ey * melec) *10**-4
    mls.append(ml)
    print('y only =',ml)


############################################################

for i in bandgaps:
    print (i)
print()
for i in ms:
    print (i)
print()
for i in mts:
    print (i)
print()
for i in mls:
    print (i)


'graph time'


# plt.figure(figsize=(6,6))
# plote = plote[:15]


# #don't want to be overkill
# colour=iter(cm.magma(numpy.linspace(0,1,15)))
# level = iter(numpy.linspace(1,15,15))

# #to label
# for e in (plote):
#     c= next(colour)
#     label=next(level)
#     plt.plot(plotks,e-l4, label=label)
#     # plots the energies over each k value
# # plt.plot(fitk, ee,'r', linewidth=3)
# # plt.axhline(l4,linewidth=1,color='grey',linestyle='--')
# # plt.axhline(l5,linewidth=1,color='grey',linestyle='--')


# #usual graph stuffs
# plt.xlabel('k')
# plt.ylabel('Energy')
# plt.title(1-prop)
# plt.legend()
# plt.savefig('interpolv.png')
# plt.show()

