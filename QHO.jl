using ITensors
using ITensorGaussianMPS
using LinearAlgebra
using LayeredLayouts
using Graphs
using GLM
using DataFrames
using CSV
using StatsModels
using HDF5
using Printf
using Plots
using LightGraphs
using IterTools
using SharedArrays

##The quantum harmonic oscillator

#set up 
#######
N = 10 #Number of qubits 
sites = siteinds("S=1/2", N) #Build the sites of two-level system
s = 2.0 #A constant for binary representation remains from the early program, only uses in *mps2f() now 

n = 4 #Number of states you want DMRG to search
m = 1 #Number of sweep interval you want to calculate the error (unfinish and only 1 is available now)
maxbdim = 64 #maxdim in dmrg()
threshold = 1e-16 #cutoff in dmrg()
sweep = 20 #nsweeps in dmrg()
kdim = 16 #eigsolve_krylovdim in dmrg()
#######

include("C:/Users/User/Downloads/julia/functions.jl") #Include some functions (noted by "*")
IMPS, IMPO = I(N) #*Build identity MPS and MPO
DDMPO = DD(N, :NBC) #*Build second-order differential operator into MPO
XMPO = X(N, -0.5+2.0^(-N-1)) #*Build f(x)=x-0.5+2.0^(-N-1) into MPO
XXMPO = MPO(sites)
for i in 1:N
	XXMPO[i] = (XMPO[i]*prime(XMPO[i]))*prime(IMPO[i]) #Build f(x)=(x-0.5+2.0^(-N-1))^2 from XMPO (julia will contract automatically if you write mpo1*mpo2)
end
XXMPO=cfc(XXMPO) #*Conbine the bond indices of MPO


#The data need to store from DMRG
#######
e = Matrix{Float64}(undef, n, sweep + 1) #energy for each state
err = Matrix{Float64}(undef, n, Int(sweep / m) + 1) #Frobenius norm error for each state
bd = Array{Int}(undef, n, sweep + 1, N - 1) #each bond dimension for each state
ele = Vector{Int}(undef, n) #total elements in MPS for each state
y = Matrix{Float64}(undef, n, 2^N) #function for each state
#######

rmps = randomMPS(sites, 2)

#Solve eigen problem by eigen()
#######
HM = zeros(2^N, 2^N)
for i in 1:2^N
	HM[i, i] = -2.0 *2.0^(2N)  - 2.0^(2N)*((i-0.5)/2.0^N-0.5)^2
	if i < 2^N
		HM[i, i+1] = 1.0 *2.0^(2N) 
		HM[i+1, i] = 1.0 *2.0^(2N) 
	end
end

val, vec = eigen(-HM)
vec = permutedims(vec, (2, 1))
#######


H = ADD(-DDMPO, XXMPO) #Add two MPOs without contraction (julia will contract automatically if you write mpo1+mpo2)

#Solve eigen problem by *muti_rdmrg()
@time begin
	e, tmps, err, bd, ele, y = muti_rdmrg(H, rmps, sweep, kdim, maxbdim, threshold, vec, n, m)
end
e=e*2.0^(2N) #Put the overall factor from second-order differential operator back

#Store the results by HDF5
#######
for i in 1:n
	for j in 1:sweep+1
	h5write("C:/Users/User/Downloads/julia/mps/QHO/MPS/QHO_MPS_($i,$(j-1))(N,sweep,kdim)=($N,$sweep,$(kdim)).h5", "MPS", tmps[i,j])
end
end

h5write("C:/Users/User/Downloads/julia/mps/QHO/QHO_energy(N,sweep,kdim)=($N,$sweep,$(kdim)).h5", "energy", e)
h5write("C:/Users/User/Downloads/julia/mps/QHO/QHO_error(N,sweep,kdim)=($N,$sweep,$(kdim)).h5", "error", err)
h5write("C:/Users/User/Downloads/julia/mps/QHO/QHO_bond_dim(N,sweep,kdim)=($N,$sweep,$(kdim)).h5", "bond_dim", bd)
h5write("C:/Users/User/Downloads/julia/mps/QHO/QHO_element(N,sweep,kdim)=($N,$sweep,$(kdim)).h5", "element", ele)
h5write("C:/Users/User/Downloads/julia/mps/QHO/QHO_function(N,sweep,kdim)=($N,$sweep,$(kdim)).h5", "function", y)
#######

#Plot labels
#######
slab = Matrix{String}(undef, 1, n)
elab = Matrix{String}(undef, 1, n)
flab = Matrix{String}(undef, 1, n)
blab = Matrix{String}(undef, 1, n)

lab = ["Ground state", "First excited state", "Second excited state", "Third excited state", "Fourth excited state"]
slab[1, 1] = "$(lab[1])"
elab[1, 1] = "$(lab[1])(energy=$(@sprintf("%.4e", e[1,sweep+1])))"
flab[1, 1] = "$(lab[1])(error=$(@sprintf("%.4e", err[1,sweep+1])))"
blab[1, 1] = "$(lab[1])(total elements=$(ele[1]))"
for i ∈ 2:n
	slab[1, i] = "$(lab[i])"
	elab[1, i] = "$(lab[i])(energy=$(@sprintf("%.4e", e[i,sweep+1])))"
	flab[1, i] = "$(lab[i])(error=$(@sprintf("%.4e", err[i,sweep+1])))"
	blab[1, i] = "$(lab[i])(total elements=$(ele[i]))"
end
#######

#Plots
#######
plot(
	0:Int(sweep / m),
	[e[i, 1:end] .- 1.0 * minimum(e[i, :]) .+ 1e-18 for i ∈ 1:n],
	legend = true,
	xlabel = "Sweep",
	ylabel = "Energy+ε",
	label = elab,
	#ylim = (minimum(e), maximum(e),
	yscale = :log10,
)

#savefig("C:/Users/User/Downloads/julia/mps/plot/QHO/QHO_energy(N,sweep,kdim)=($N,$sweep,$(kdim)).png")

err = h5read("C:/Users/User/Downloads/julia/mps/QHO/QHO_error(N,sweep,kdim)=($N,$sweep,$(kdim)).h5", "error")
plot(
	0:Int(sweep / m),
	[err[i, :] for i ∈ 1:n],
	legend = true,
	xlabel = "Sweep",
	ylabel = "Error",
	label = flab,
	yscale = :log10,
	xlims = (0, sweep),
	yticks = ([10.0^(-5i) for i in 0:4]),
)

#savefig("C:/Users/User/Downloads/julia/mps/plot/QHO/QHO_error(N,sweep,kdim)=($N,$sweep,$(kdim)).png")

x=collect(Float64,0.0:2.0^(-N):1.0-2.0^(-N))
plot(
	x,
	[y[i, :] for i ∈ 1:n],
	legend = true,
	xlabel = "x",
	ylabel = "ψ(x)",
	#title = "Particle in the box",
	label = slab,
	xlims = (minimum(x), 1.0),
)
#savefig("C:/Users/User/Downloads/julia/mps/plot/QHO/QHO_function(N,sweep,kdim)=($N,$sweep,$(kdim)).png")

#=
for i in 1:n
	display(
		heatmap(collect(1:sweep+1), collect(1:N-1), permutedims(bd, (1, 3, 2))[i, :, :], xlabel = "Sweep", ylabel = "Bond number", yticks = (1:N-1, 1:N-1),
			clims = (2, 34),
			label = blab),
	)
	savefig("C:/Users/User/Downloads/julia/mps/plot/QHO/QHO_bond_dim$i(N,sweep,kdim)=($N,$sweep,$(kdim)).png")
end
=#

plot(
	0:sweep,
	[maximum(bd[i, j, :]) for j ∈ 1:sweep+1, i ∈ 1:n],
	legend = true,
	xlabel = "Sweep",
	ylabel = "Max bond dimension",
	#title = "Particle in the box",
	label = slab,
)
#savefig("C:/Users/User/Downloads/julia/mps/plot/QHO/QHO_bond_dim(N,sweep,kdim)=($N,$sweep,$(kdim)).png")

eref=[2.0^(11)*(0.5+i-1) for i in 1:4]
ni= collect(1:n)
nv=val[1:n]
df = DataFrame(X = (ni), Y = (nv))
lin = lm(@formula(Y ~ X), df)
c0, cx = coef(lin)
y0= ( cx .*(ni) .+ c0 ) 
en = Vector{Float64}(undef, n)
for i in 1:n
	en[i] = e[i,sweep+1]
end
df2 = DataFrame(X=(ni), Y=(en))
lin = lm(@formula(Y ~ X), df2)
c02,cx2=coef(lin)
y2= ( cx2 .*(ni) .+ c02 ) 
scatter(ni, nv,label="", xlabel = "Bound state number", ylabel = "Energy", color = :blue, markershape = :circle)
plot!(ni, y0, label="ED: y=" * @sprintf("%.4f", cx) * "n +" * @sprintf("%.4f", c0), xlabel = "Bound state number ", ylabel = "Energy", xticks=(ni,[@sprintf("%.1i", xval) for xval in ni]), color = :blue)
scatter!(ni, en,label="", xlabel = "Bound state number", ylabel = "Energy", color = :red, markershape = :xcross)
plot!(ni, y2, label="MPS: y=" * @sprintf("%.4f", cx2) * "n +" * @sprintf("%.4f", c02), xlabel = "Bound state number ", ylabel = "Energy", xticks=(ni,[@sprintf("%.1i", xval) for xval in ni]), color = :red,
yticks=(eref, ["E\u2080","2E\u2080","3E\u2080","4E\u2080"]), ytickfontsize = 10)
#savefig("C:/Users/User/Downloads/julia/mps/plot/QHO/QHO_spectrum(N,sweep,kdim)=($N,$sweep,$(kdim)).png")


tmps = Matrix{Any}(undef, n, sweep + 1) #MPS for each state
for i in 1:n
	for j in 1:sweep+1
		file = "C:/Users/User/Downloads/julia/mps/QHO/MPS/QHO_MPS_($i,$(j-1))(N,sweep,kdim)=($N,$sweep,$(kdim)).h5"
		tmps[i, j] = reading_mps2(file)
	end
end

H1 = Matrix{Float64}(undef, n, sweep + 1) #energy for each state
H2 = Matrix{Float64}(undef, n, sweep + 1) #energy for each state
SD = Matrix{Float64}(undef, n, sweep + 1) #energy for each state
for i in 1:n
	for j in 1:sweep+1
		hmps = MPS(sites)
		for k in 1:N
			hmps[k] = H[k] * tmps[i, j][k]
		end
		H1[i, j] = inner(tmps[i, j], hmps) * 2.0^(2N)
		H2[i, j] = inner(hmps, hmps) * 2.0^(4N)
		SD[i, j] = abs((H2[i, j] - H1[i, j]^2) / 2.0^N)^0.5
	end
end

sdref=[10.0^(6-2i)/(2.0^(11)*(0.5+i-1)) for i in 1:5]
ref=[10.0^(6-2i) for i in 1:5]
plot(
	0:Int(sweep / m),
	[SD[i, :] for i ∈ 1:n],
	legend = true,
	xlabel = "Sweep",
	ylabel = "SD(E)",
	label = elab,
	yscale = :log10,
	xlims = (0, sweep),
	yticks=(ref, [@sprintf("%.2E", rref) * "E\u2080" for rref in sdref]),
	ytickfontsize = 10
)
#savefig("C:/Users/User/Downloads/julia/mps/plot/QHO/QHO_SD(E)(N,sweep,kdim)=($N,$sweep,$(kdim)).png")
