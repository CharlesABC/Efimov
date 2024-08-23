using ITensors
using ITensorGaussianMPS
using LinearAlgebra
using ITensorVisualizationBase
using LayeredLayouts
using Graphs
using Plots
using LightGraphs

N = 8
bond3= Index[Index(2) for a in 1:N]
bonds= Index[Index(2) for a in 1:N-1]
bond = Index[Index(2) for a in 1:N-1]
bond0 = Index[Index(2) for a in 1:N-1]
s=2
bond2 = []
threshold= 10^-15
abcdef = [(a, b, c, d, e, f) for a in 1:2, b in 1:2, c in 1:2, d in 1:2, e in 1:2, f in 1:2]
ghijkl = [(g,h,i,j,k,l) for g in 1:2, h in 1:2, i in 1:2, j in 1:2, k in 1:2, l in 1:2]
opqrst = [(o,p,q,r,s,t) for o in 1:2, p in 1:2, q in 1:2, r in 1:2, s in 1:2, t in 1:2]
abcdefghij = [(a, b, c, d, e, f, g, h, i ,j) for a in 1:2, b in 1:2, c in 1:2, d in 1:2, e in 1:2, f in 1:2, g in 1:2, h in 1:2, i in 1:2, j in 1:2]
sites = siteinds("S=1/2", N)
FMPO = MPO(sites)
IMPO = MPO(sites)
BMPO = MPO(sites)
EMPO = MPO(sites)
EEMPO = MPO(sites)

F=ITensor(sites[1],prime(sites[1]),bond[1])
F[sites[1]=>2,prime(sites[1])=>1,bond[1]=>1]=1.0
F[sites[1]=>1,prime(sites[1])=>2,bond[1]=>2]=1.0
FMPO[1]=F

for a in 2:N-1
 F=ITensor(sites[a],prime(sites[a]),bond[a-1],bond[a])   
 F[sites[a]=>2,prime(sites[a])=>1,bond[a-1]=>2,bond[a]=>1]=1.0
 F[sites[a]=>1,prime(sites[a])=>2,bond[a-1]=>2,bond[a]=>2]=1.0
 F[sites[a]=>1,prime(sites[a])=>1,bond[a-1]=>1,bond[a]=>1]=1.0
 F[sites[a]=>2,prime(sites[a])=>2,bond[a-1]=>1,bond[a]=>1]=1.0
 FMPO[a]=F
end

F=ITensor(sites[N],prime(sites[N]),bond[N-1])
F[sites[N]=>2,prime(sites[N])=>1,bond[N-1]=>2]=1.0
F[sites[N]=>1,prime(sites[N])=>1,bond[N-1]=>1]=1.0
F[sites[N]=>2,prime(sites[N])=>2,bond[N-1]=>1]=1.0
FMPO[N]=F

B=ITensor(sites[1],prime(sites[1]),bond[1])
B[sites[1]=>2,prime(sites[1])=>1,bond[1]=>2]=1.0
B[sites[1]=>1,prime(sites[1])=>2,bond[1]=>1]=1.0
BMPO[1]=B

for a in 2:N-1
 B=ITensor(sites[a],prime(sites[a]),bond[a-1],bond[a])   
 B[sites[a]=>2,prime(sites[a])=>1,bond[a-1]=>2,bond[a]=>2]=1.0
 B[sites[a]=>1,prime(sites[a])=>2,bond[a-1]=>2,bond[a]=>1]=1.0
 B[sites[a]=>1,prime(sites[a])=>1,bond[a-1]=>1,bond[a]=>1]=1.0
 B[sites[a]=>2,prime(sites[a])=>2,bond[a-1]=>1,bond[a]=>1]=1.0
 BMPO[a]=B
end

B=ITensor(sites[N],prime(sites[N]),bond[N-1])
B[sites[N]=>1,prime(sites[N])=>2,bond[N-1]=>2]=1.0
B[sites[N]=>1,prime(sites[N])=>1,bond[N-1]=>1]=1.0
B[sites[N]=>2,prime(sites[N])=>2,bond[N-1]=>1]=1.0
BMPO[N]=B

I=ITensor(sites[1],prime(sites[1]))
I[sites[1]=>1,prime(sites[1])=>1]=1.0
I[sites[1]=>2,prime(sites[1])=>2]=1.0
IMPO[1]=I

for a in 2:N-1
 I=ITensor(sites[a],prime(sites[a]))   
 I[sites[a]=>1,prime(sites[a])=>1]=1.0
 I[sites[a]=>2,prime(sites[a])=>2]=1.0
 IMPO[a]=I
end

I=ITensor(sites[N],prime(sites[N]))
I[sites[N]=>2,prime(sites[N])=>2]=1.0
I[sites[N]=>1,prime(sites[N])=>1]=1.0
IMPO[N]=I


E=ITensor(sites[1],prime(sites[1]),bond[1])
E[sites[1]=>1,prime(sites[1])=>1,bond[1]=>1]=1.0
E[sites[1]=>2,prime(sites[1])=>2,bond[1]=>2]=1.0
EMPO[1]=E

for a in 2:N-1
 E=ITensor(sites[a],prime(sites[a]),bond[a-1],bond[a])   
 E[sites[a]=>1,prime(sites[a])=>1,bond[a-1]=>1,bond[a]=>1]=1.0
 E[sites[a]=>2,prime(sites[a])=>2,bond[a-1]=>2,bond[a]=>2]=1.0
 EMPO[a]=E
end

E=ITensor(sites[N],prime(sites[N]),bond[N-1])
E[sites[N]=>1,prime(sites[N])=>1,bond[N-1]=>1]=1.0
E[sites[N]=>2,prime(sites[N])=>2,bond[N-1]=>2]=1.0
EMPO[N]=E

##period choice
E=ITensor(sites[1],prime(sites[1]),bond[1])
E[sites[1]=>1,prime(sites[1])=>2,bond[1]=>1]=1.0
E[sites[1]=>2,prime(sites[1])=>1,bond[1]=>2]=1.0
EEMPO[1]=E

for a in 2:N-1
 E=ITensor(sites[a],prime(sites[a]),bond[a-1],bond[a])   
 E[sites[a]=>1,prime(sites[a])=>2,bond[a-1]=>1,bond[a]=>1]=1.0
 E[sites[a]=>2,prime(sites[a])=>1,bond[a-1]=>2,bond[a]=>2]=1.0
 EEMPO[a]=E
end

E=ITensor(sites[N],prime(sites[N]),bond[N-1])
E[sites[N]=>1,prime(sites[N])=>2,bond[N-1]=>1]=1.0
E[sites[N]=>2,prime(sites[N])=>1,bond[N-1]=>2]=1.0
EEMPO[N]=E

#=
CSMPO=prime(SMPO)
CIMPO=prime(IMPO)
S2MPO=SMPO*CSMPO
SSMPO=CIMPO*S2MPO
@show SMPO
@show IMPO
@show SSMPO
=#
DMPO=(BMPO+FMPO-2.0IMPO+EEMPO)
#TDMPO=DMPO[1]*DMPO[2]*DMPO[3]*DMPO[4]*DMPO[5]*DMPO[6]
@show DMPO

M = ITensor(sites)
for (a, b, c, d, e, f, g, h) in abcdefghij
    M[sites[1]=>a,sites[2]=>b,sites[3]=>c,sites[4]=>d,sites[5]=>e,sites[6]=>f,sites[7]=>g,sites[8]=>h]=cos(2.0*pi*((a-1)/s^N+(b-1)/s^(N-1)+(c-1)/s^(N-2)+(d-1)/s^(N-3)+(e-1)/s^(N-4)+(f-1)/s^(N-5)+(g-1)/s^(N-6)+(h-1)/s^(N-7))/(255.0/256.0))
end
#=
U, S, V = svd(M,(sites[1],sites[2],sites[3],sites[4],sites[5]))
vector = diag(S)
@show vector
n=count(abs.(vector) .> threshold)
push!(bond2, Index(n))
MABCDE = ITensor(sites[1],sites[2],sites[3],sites[4],sites[5],bond2[1])
MF = ITensor(sites[6],bond2[1])
for (a, b, c, d, e, f) in abcdef  
  for w in 1:n
     MABCDE[sites[1]=>a,sites[2]=>b,sites[3]=>c,sites[4]=>d,sites[5]=>e,bond2[1]=>w] = U[a,b,c,d,e,w]
     MF[sites[6]=>a,bond2[1]=>w] =vector[w]*V[a,w]
  end  
end 

U, S, V = svd(MABCDE,(sites[1],sites[2],sites[3],sites[4]))
vector = diag(S)
@show vector
m=n
n=count(abs.(vector) .> threshold)
push!(bond2, Index(n))
MABCD = ITensor(sites[1],sites[2],sites[3],sites[4],bond2[2])
ME = ITensor(sites[5],bond2[1],bond2[2])
for (a, b, c, d, e, f) in abcdef
 for w in 1:n
   for x in 1:m  
    MABCD[sites[1]=>a,sites[2]=>b,sites[3]=>c,sites[4]=>d,bond2[2]=>w] = U[a,b,c,d,w]
    ME[sites[5]=>e,bond2[1]=>x,bond2[2]=>w] = vector[w]*V[e,x,w] 
   end     
 end   
end


U, S, V = svd(MABCD,(sites[1],sites[2],sites[3]))
vector = diag(S)
@show vector
m=n
n=count(abs.(vector) .> threshold)
push!(bond2, Index(n))
MABC = ITensor(sites[1],sites[2],sites[3],bond2[3])
MD = ITensor(sites[4],bond2[2],bond2[3])

for (a, b, c, d, e, f) in abcdef  
 for w in 1:n
  for x in 1:m   
    MD[sites[4]=>d,bond2[2]=>x,bond2[3]=>w] = vector[w]*V[d,x,w]
    MABC[sites[1]=>a,sites[2]=>b,sites[3]=>c,bond2[3]=>w] = U[a,b,c,w] 
  end  
 end   
end

U, S, V = svd(MABC,(sites[1],sites[2]))
vector = diag(S)
@show vector
m=n
n=count(abs.(vector) .> threshold)
push!(bond2, Index(n))
MAB = ITensor(sites[1],sites[2],bond2[4])
MC = ITensor(sites[3],bond2[3],bond2[4]) 
for (a, b, c, d, e, f) in abcdef  
 for w in 1:n
  for x in 1:m    
    MAB[sites[1]=>a,sites[2]=>b,bond2[4]=>w] = U[a,b,w]
    MC[sites[3]=>c,bond2[3]=>x,bond2[4]=>w] =vector[w]*V[c,x,w]   
  end  
 end  
end

U, S, V = svd(MAB,(sites[1]))
vector = diag(S)
@show vector
m=n
n=count(abs.(vector) .> threshold)
push!(bond2, Index(n))
MA = ITensor(sites[1],bond2[5])
MB = ITensor(sites[2],bond2[4],bond2[5])
for (a, b, c, d, e, f) in abcdef  
 for w in 1:n
  for x in 1:m     
    MB[sites[2]=>b,bond2[4]=>x,bond2[5]=>w] = vector[w]*V[b,x,w]
    MA[sites[1]=>a,bond2[5]=>w] = U[a,w]
  end  
 end
end

FMPS=MPS(sites)
FMPS[1] = MA 
FMPS[2] = MB 
FMPS[3] = MC 
FMPS[4] = MD 
FMPS[5] = ME 
FMPS[6] = MF 
=#

function FF(tensor::ITensor)
  F = MPS(sites)
  for a in N:-1:2
    U, S, V = svd(tensor,[sites[i] for i in 1:a-1]; cutoff=threshold)
    tensor=U
    F[a]=S*V
    F[1]=U
  end  
  return F
 end

FMPS = MPS(sites) 
FMPS=FF(M)
@show FMPS


real_stdout = stdout
rmps=randomMPS(sites,2) 
#(rd, wr) = redirect_stdout() # Noisy code here
energy2, mps2 = dmrg(-DMPO, rmps, cutoff=1e-15,maxdim=2,sweeps=100,nsweeps=200,outputlevel = 2)
energy3, mps3 = dmrg(-DMPO,[mps2], rmps, cutoff=1e-15,maxdim=2,sweeps=100,nsweeps=200)
energy, mps = dmrg(-DMPO,[mps2,mps3], rmps, cutoff=1e-15,maxdim=2,sweeps=100,nsweeps=200)
#redirect_stdout(real_stdout)
@show (energy*2.0^(2N))^0.5/pi/(1.0-1/2.0^N),(energy2*2.0^(2N))^0.5/pi/(1.0-1/2.0^N),(energy3*2.0^(2N))^0.5/pi/(1.0-1/2.0^N)

b0=ITensor(sites[1])
b1=ITensor(sites[2])
b2=ITensor(sites[3])
b3=ITensor(sites[4])
b4=ITensor(sites[5])
b5=ITensor(sites[6])
b0[sites[1]=>1]=1.0
b0[sites[1]=>2]=1.0
b1[sites[2]=>1]=1.0
b1[sites[2]=>2]=1.0
b2[sites[3]=>1]=1.0
b2[sites[3]=>2]=1.0
b3[sites[4]=>1]=1.0
b3[sites[4]=>2]=1.0
b4[sites[5]=>1]=1.0
b4[sites[5]=>2]=1.0
b5[sites[6]=>1]=1.0
b5[sites[6]=>2]=1.0
nor=(((((mps[1]*b0)*(mps[2]*b1))*(mps[3]*b2))*(mps[4]*b3))*(mps[5]*b4))*(mps[6]*b5)/64.0
@show nor
TT=noprime(DMPO*FMPS)
@show TT
function mps3f(mps::MPS)
  x=[]
  y=[]
  b0=ITensor(sites[1])
  b1=ITensor(sites[2])
  b2=ITensor(sites[3])
  b3=ITensor(sites[4])
  b4=ITensor(sites[5])
  b5=ITensor(sites[6])
  b6=ITensor(sites[7])
  b7=ITensor(sites[8])

  for n in 1:1
  nor=0.0  
  for a in 1:2
   b0[sites[1]=>a]=1.0  
   for b in 1:2
    b1[sites[2]=>b]=1.0  
    for c in 1:2
     b2[sites[3]=>c]=1.0 
     for d in 1:2
      b3[sites[4]=>d]=1.0
      for e in 1:2
       b4[sites[5]=>e]=1.0   
       for f in 1:2 
          b5[sites[6]=>f]=1.0
          for g in 1:2
            b6[sites[7]=>g]=1.0 
            for h in 1:2
                 b7[sites[8]=>h]=1.0
             #for i in 1:2
              #b8[sites[9]=>i]=1.0   
              #for j in 1:2 
                 #b9[sites[10]=>j]=1.0       
                 push!(x, s^2*((a-1)/s^10+(b-1)/s^9+(c-1)/s^8+(d-1)/s^7+(e-1)/s^6+(f-1)/s^5+(g-1)/s^4+(h-1)/s^3))
                 tmps=(mps[1]*b0)*(mps[2]*b1)*(mps[3]*b2)*(mps[4]*b3)*(mps[5]*b4)*(mps[6]*b5)*(mps[7]*b6)*(mps[8]*b7)
                 nor+=tmps[1]^2
                 push!(y, tmps[1])
                 #b9[sites[10]=>j]=0.0 
              #end
              #b8[sites[9]=>i]=0.0
              #end
                 b7[sites[8]=>h]=0.0
            end
           b6[sites[7]=>g]=0.0
          end          
         b5[sites[6]=>f]=0.0 
       end
       b4[sites[5]=>e]=0.0
      end
      b3[sites[4]=>d]=0.0
     end
     b2[sites[3]=>c]=0.0
    end
    b1[sites[2]=>b]=0.0
   end
   b0[sites[1]=>a]=0.0   
  end
  return x, y, nor
  nor=0.0
  end
 end




x,y,nor= mps3f(FMPS)
x,z,nor3= mps3f(mps3)
x,w,nor0= mps3f(mps)

if z[1]<0.0
  mps=-mps
end

#display(scatter( x, y/nor^0.5, legend=true, xlabel="X", ylabel="Y", title="Scatter Plot", color=:green, label="Ground stste(sweeps=7)" ))  # Display the updated plot
display(scatter( x, z, legend=true, xlabel="X", ylabel="Y", title="Scatter Plot", color=:red, label="First exicted stste(sweeps=10)" ))  # Display the updated plot
display(scatter!( x, w, legend=true, xlabel="X", ylabel="Y", title="Free Particle", color=:blue, label="Secod exicted stste(sweeps=12)" ))  # Display the updated plot
#savefig("C:/Users/User/Downloads/julia/free particle.png")

@show nor




for n in 1:1
 error=[]
 terr=0.0
 for i in 1:2^N
  if abs(y[i])>threshold*100.0
    push!(error,abs((z[i]-(y[i]/nor^0.5))/(y[i]/nor^0.5)))
    terr+=error[i]
  else
    push!(error,0.0)
    terr+=error[i]
  end
 end 
 @show terr
end

for i in 1:6
  max=argmax(z)
  @show z[max],max
  z[max]=0
  max=argmax(y)
  @show y[max]/(nor^0.5),max
  y[max]=0
end
@show z[1],w[1]
#display(scatter( x, error, label="x(d/dx)x", legend=true, xlabel="X", ylabel="Y", title="Error", color=:red))  # Display the updated plot