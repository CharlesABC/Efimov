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
using KrylovKit
using JuliaFormatter
using IterTools
using NDTensors
using Distributed

N = 20
s = 2.0
shift = 10
s0 = 2.0

n = 4
maxbdim = 64
threshold = 1e-16
sweep = 20
kdim = 256
m = 1


x0 = 2.0^(-shift)
p = (2.0^(N) - 1.0) / log(2.0^(shift) + 1.0 - 2.0^(-N+shift))#Compare to the thesis k, k=2.0^N/p
d = -p * log(x0)#Compare to the thesis d', d'=d/2.0^N
dx2 = ((exp(-2.0 / p) * (exp(1.0 / p) + 1.0) * (exp(1.0 / p) - 1.0)^2) / 2.0)
sites = siteinds("S=1/2", N)

include("C:/Users/User/Downloads/julia/functions.jl")
IMPS, IMPO = I(N)
CMPS = COS(N, 200.0)
NEXMPO = EX(N, -0.5 * (2.0^N / p))
N2EXMPO = EX(N, -2.0 * (2.0^N / p))


function EDD(N::Int, c::Float64)
	bond3 = Index[Index(3, "link, l=$a") for a ∈ 1:N-1]
	DDMPO = MPO(sites)

	DD = ITensor(sites[1], prime(sites[1]), bond3[1])
	DD[sites[1]=>2, prime(sites[1])=>1, bond3[1]=>1] = 1.0 * exp(-2.0c * 2.0^(0) + 0.5c)
	DD[sites[1]=>1, prime(sites[1])=>2, bond3[1]=>3] = 1.0 * exp(0.5c)
	DD[sites[1]=>2, prime(sites[1])=>1, bond3[1]=>2] = 1.0 * exp(0.5c)
	DD[sites[1]=>1, prime(sites[1])=>2, bond3[1]=>1] = 1.0 * exp(-2.0c * 2.0^(0) + 0.5c)
	DD[sites[1]=>1, prime(sites[1])=>1, bond3[1]=>1] = -1.0 * (exp(-c) + 1.0)
	DD[sites[1]=>2, prime(sites[1])=>2, bond3[1]=>1] =
		-1.0 * (exp(-c) + 1.0) * exp(-2.0c * 2.0^(0))
	DDMPO[1] = DD

	for a ∈ 2:N-1
		DD = ITensor(sites[a], prime(sites[a]), bond3[a-1], bond3[a])
		DD[sites[a]=>2, prime(sites[a])=>1, bond3[a-1]=>3, bond3[a]=>1] =
			1.0 * exp(-2.0c * 2.0^(a - 1))
		DD[sites[a]=>1, prime(sites[a])=>2, bond3[a-1]=>3, bond3[a]=>3] = 1.0
		DD[sites[a]=>2, prime(sites[a])=>1, bond3[a-1]=>2, bond3[a]=>2] = 1.0
		DD[sites[a]=>1, prime(sites[a])=>2, bond3[a-1]=>2, bond3[a]=>1] =
			1.0 * exp(-2.0c * 2.0^(a - 1))
		DD[sites[a]=>1, prime(sites[a])=>1, bond3[a-1]=>1, bond3[a]=>1] = 1.0
		DD[sites[a]=>2, prime(sites[a])=>2, bond3[a-1]=>1, bond3[a]=>1] =
			1.0 * exp(-2.0c * 2.0^(a - 1))
		DDMPO[a] = DD
	end

	DD = ITensor(sites[N], prime(sites[N]), bond3[N-1])
	DD[sites[N]=>2, prime(sites[N])=>1, bond3[N-1]=>3] = 1.0 * exp(-2.0c * 2.0^(N - 1))
	DD[sites[N]=>1, prime(sites[N])=>1, bond3[N-1]=>1] = 1.0
	DD[sites[N]=>2, prime(sites[N])=>2, bond3[N-1]=>1] = 1.0 * exp(-2.0c * 2.0^(N - 1))
	DD[sites[N]=>1, prime(sites[N])=>2, bond3[N-1]=>2] = 1.0 * exp(-2.0c * 2.0^(N - 1))
	DDMPO[N] = DD
	return DDMPO
end

EDDMPO = EDD(N, 1.0 / p)


#=
HM = zeros(2^N, 2^N)
f(i) = exp(-2.0 * (i - 1.0) / p)
V(i) = exp(-2.0 * (i - 1.0) / p)
for i in 1:2^N
	HM[i, i] = -(1.0 + exp(-1.0 / p)) * f(i)/ dx2 + (s0^2 + 0.25) * V(i)
	if i < 2^N
		HM[i, i+1] = exp(-1.5 / p) * f(i)/ dx2
		HM[i+1, i] = exp(-1.5 / p) * f(i)/ dx2
	end
end

val, vec = eigen(-HM)
vec = permutedims(vec, (2, 1))

val = val / exp(-2.0d/p)
nv = filter(x -> x < 0, val)
ni = findall(x -> x < 0, val)
df = DataFrame(X = ni, Y = -nv)
lin = lm(@formula(Y ~ X), df)
c0, cx = coef(lin)
@show c0, cx
scatter(df.X, df.Y, label="exact diagonalization", xlabel = "number of state", ylabel = "-energy", color = :blue, yscale=:log10)
=#
for i in 1:N
	EDDMPO[i] = EDDMPO[i] #/ dx2^(0.5/N) 
	N2EXMPO[i] = N2EXMPO[i] #* dx2^(0.5/N) * (s0^2 + 0.25)^(1/N)
end
H = ADD(-EDDMPO / dx2^(0.5), -(s0^2 + 0.25) * dx2^(0.5) * N2EXMPO)

@time begin
	e, tmps, bd, ele, y = muti_rdmrg(H, CMPS, sweep, kdim, maxbdim, threshold, nothing, n, m)
end
e = e / dx2^0.5 / exp(-2.0d/p)

for i in 1:n
	for j in 1:sweep+1
		h5write("C:/Users/User/Downloads/julia/mps/efimov_e^x/MPS/efimov_e^x_MPS_($i,$(j-1))(N,sweep,shift,kdim)=($N,$sweep,$(shift),$(kdim)).h5", "MPS", tmps[i,j])
	end
end

h5write("C:/Users/User/Downloads/julia/mps/efimov_e^x/efimov_e^x_energy(N,sweep,shift,kdim)=($N,$sweep,$(shift),$(kdim)).h5", "energy", e)
#h5write("C:/Users/User/Downloads/julia/mps/efimov_e^x/efimov_e^x_error(N,sweep,shift,kdim)=($N,$sweep,$(shift),$(kdim)).h5", "error", err)
h5write("C:/Users/User/Downloads/julia/mps/efimov_e^x/efimov_e^x_bond_dim(N,sweep,shift,kdim)=($N,$sweep,$(shift),$(kdim)).h5", "bond_dim", bd)
h5write("C:/Users/User/Downloads/julia/mps/efimov_e^x/efimov_e^x_element(N,sweep,shift,kdim)=($N,$sweep,$(shift),$(kdim)).h5", "element", ele)
h5write("C:/Users/User/Downloads/julia/mps/efimov_e^x/efimov_e^x_function(N,sweep,shift,kdim)=($N,$sweep,$(shift),$(kdim)).h5", "function", y)



slab = Matrix{String}(undef, 1, n)
flab = Matrix{String}(undef, 1, n)
elab = Matrix{String}(undef, 1, n)
blab = Matrix{String}(undef, 1, n)
markers = Matrix{Symbol}(undef, 1, n)

lab = ["Ground state", "First excited state", "Second excited state", "Third excited state", "Fourth excited state"]
slab[1, 1] = "$(lab[1])"
elab[1, 1] = "$(lab[1])(energy=$(@sprintf("%.4e", e[1,sweep+1])))"
#flab[1, 1] = "$(lab[1])(error=$(@sprintf("%.4e", err[1,sweep+1])))"
blab[1, 1] = "$(lab[1])(total elements=$(ele[1]))"
for i ∈ 2:n
	slab[1, i] = "$(lab[i])"
	elab[1, i] = "$(lab[i])(energy=$(@sprintf("%.4e", e[i,sweep+1])))"
	#flab[1, i] = "$(lab[i])(error=$(@sprintf("%.4e", err[i,sweep+1])))"
	blab[1, i] = "$(lab[i])(total elements=$(ele[i]))"
end

#=
plot(
	0:Int(sweep / m),
	[e[i, 1:end] .- 1.0 * minimum(e[i, :]) .+ 1e-18 for i ∈ 1:n],
	legend = true,
	xlabel = "Sweep",
	ylabel = "Energy+ε",
	label = elab,
	#ylim = (minimum(e), maximum(e),
	yscale = :log10,
	markersize = 2.0,
)

savefig("C:/Users/User/Downloads/julia/mps/plot/efimov_e^x/efimov_e^x_energy(N,sweep,shift,kdim)=($N,$sweep,$(shift),$(kdim)).png")

#=
err=h5read("C:/Users/User/Downloads/julia/mps/efimov_e^x/efimov_e^x_error(N,sweep,shift,kdim)=($N,$sweep,$(shift),$(kdim)).h5", "error")
plot(
	0:Int(sweep / m),
	[err[i, :] for i ∈ 1:n],
	legend = true,
	xlabel = "Sweep",
	ylabel = "Error",
	label = flab,
	yscale = :log10,
	xlims = (0, sweep),
)

#savefig("C:/Users/User/Downloads/julia/mps/plot/efimov_e^x/efimov_e^x_error(N,sweep,shift,kdim)=($N,$sweep,$(shift),$(kdim)).png")
=#

y=h5read("C:/Users/User/Downloads/julia/mps/efimov_e^x/efimov_e^x_function(N,sweep,shift,kdim)=($N,$sweep,$(shift),$(kdim)).h5", "function")
dx = Vector{Float64}(undef, 2^N)
x = Vector{Float64}(undef, 2^N)
z = Matrix{Float64}(undef, n , 2^N)

for i ∈ 1:2^N
	#x[i] = exp(((i - 1.0) / s^N * 2.0^N - d) / p)
	x[i] = (i - 1.0) / s^N + 2.0^(-shift)
end
for i ∈ 1:2^N-1
	dx[i] = x[i+1] - x[i]
end
dx[2^N]=dx[1]
#dx[2^N] = exp(((2^N) / s ^ N * 2.0 ^ N - d ) / p) - exp(((2^N - 1.0) / s ^ N * 2.0^N - d ) / p)
for i in 1:n
	for j in 1:2^N
		y[i, j]=y[i, j]*exp(-(j - 1.0)/ p )
	end
	enor = (y[i, :] .* dx)'*(y[i, :] .* dx)*2.0^(2N)
	y[i,:] = y[i,:] / enor^0.5
end

plot(
	x,
	[y[i, :] for i ∈ 1:n],
	legend = true,
	xlabel = "x",
	ylabel = "ψ(x)",
	label = slab,
	#ylims = (-0.001, 0.01),
	xlims = (minimum(x),maximum(x)),
)
xmax=[argmax(y[i,:]) for i in 1:n]
vline!(x[xmax], label="", color=:red, xticks=(x[xmax], [@sprintf("%.4f", xval) for xval in x[xmax]]))
#savefig("C:/Users/User/Downloads/julia/mps/plot/efimov_e^x/efimov_e^x_function0(N,sweep,shift,kdim)=($N,$sweep,$(shift),$(kdim)).png")

for i in 1:n
	display(
		heatmap(collect(1:sweep+1), collect(1:N-1), permutedims(bd, (1, 3, 2))[i, :, :], xlabel = "Sweep", ylabel = "Bond number", yticks = (1:N-1, 1:N-1),
			clims = (2, 32),
			label = blab),
	)
	savefig("C:/Users/User/Downloads/julia/mps/plot/efimov_e^x/efimov_e^x_bond_dim$i(N,sweep,shift,kdim)=($N,$sweep,$(shift),$(kdim)).png")
end
=#
#=
plot(
	0:sweep,
	[maximum(bd[i, j, :]) for j ∈ 1:sweep+1, i ∈ 1:n],
	legend = true,
	xlabel = "Sweep",
	ylabel = "Max bond dimension",
	#title = "Particle in the box",
	label = slab,
	yticks=(2:4:32)
)
#savefig("C:/Users/User/Downloads/julia/mps/plot/efimov_e^x/efimov_e^x_bond_dim(N,sweep,shift,kdim)=($N,$sweep,$shift,$(kdim)).png")
=#
#=
tmps = Matrix{Any}(undef, n, sweep + 1) #MPS for each state
for i in 1:n
	for j in 1:sweep+1
		file = "C:/Users/User/Downloads/julia/mps/efimov_e^x/MPS/efimov_e^x_MPS_($i,$(j-1))(N,sweep,shift,kdim)=($N,$sweep,$(shift),$(kdim)).h5"
		tmps[i, j] = reading_mps2(file)
	end
end

H1 = Matrix{Float64}(undef, n, sweep + 1) #energy for each state
H2 = Matrix{Float64}(undef, n, sweep + 1) #energy for each state
SD = Matrix{Float64}(undef, n, sweep + 1) #energy for each state
for i in 1:n
	for j in 1:sweep+1
		hmps = MPS(sites)
		gmps = MPS(sites)
		for k in 1:N
			hmps[k] = H[k] * tmps[i, j][k]
			gmps[k] = H[k] * prime(tmps[i, j])[k]
		end
		hmps=prime(csc(noprime(hmps)))
		gmps=csc(gmps)
		H1[i, j] = inner(tmps[i, j], hmps) 
		H2[i, j] = inner(gmps, hmps) 
		SD[i, j] = sqrt(abs((H2[i, j] - H1[i, j]^2) / 2.0^N)) / dx2^0.5 / exp(-2.0d/p)
	end
end

plot(
	0:Int(sweep / m),
	[SD[i, :] for i ∈ 1:n],
	legend = true,
	xlabel = "Sweep",
	ylabel = "SD(E)",
	label = elab,
	yscale = :log10,
	xlims = (0, sweep),
)
#savefig("C:/Users/User/Downloads/julia/mps/plot/efimov_e^x/efimov_e^x_SD(E)(N,sweep,shift,kdim)=($N,$sweep,$shift,$(kdim)).png")
#=
HH = MPO(sites)
for k in 1:N
	HH[k]=(H[k]*prime(H[k]))*prime(IMPO[k])
end
HH=cfc(HH)
inner(tmps[4,21],HH*tmps[4,21])

a=[]
k=N
push!(a,prime(ED[k]*prime(tmps[1,21])[k])*((ED[k]*tmps[1,21][k])))
for k in 2:N-1
	push!(a,prime(ED[N-k+1]*prime(tmps[1,21])[N-k+1])*((ED[N-k+1]*tmps[1,21][N-k+1])))
	a[k]=a[k]*a[k-1]
end
k=1
inner(a[N-1],prime(ED[k]*prime(tmps[1,21])[k])*((ED[k]*tmps[1,21][k])))