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

N = 10
s0 = 2.0
shift = 10

n = 5
maxbdim = 64
threshold = 1e-16
sweep = 20
kdim = 32
m = 1
s = 2.0

shi = Vector{Float64}()

bond = Index[Index(2, "link, l=$a") for a ∈ 1:N-1]
bond1 = Index[Index(1, "link, l=$a") for a ∈ 1:N-1]
sites = siteinds("S=1/2", N)
XXMPO = MPO(sites)

include("C:/Users/User/Downloads/julia/functions.jl")
file = "C:/Users/User/Downloads/julia/mps/(x+a)^-1/(x+a)^-1_MPS(N,sweep,shift,kdim)=($N,25,$shift,32).h5"
XMPS = reading_mps(file)
XMPO = mps2mpo(XMPS)

IMPS, IMPO = I(N)
DDMPO = DD(N, :DBC)
CMPS = COS(N, 200.0)

function calculate_sum(N, a)
	total = 0.0
	for i ∈ 0:2^N-1
		total = total + 1.0 / (i / 2.0^N + 2.0^(-a))^2
	end
	return total
end
nor0 = calculate_sum(N, shift)


for i ∈ 1:N
	XMPO[i] = XMPO[i]
	XXMPO[i] = (XMPO[i] * prime(XMPO[i])) * prime(IMPO[i])
end
XXMPO = cfc(XXMPO)
nor=inner(XMPO,XMPO)
ratio = Float64(2.0^(2N) * nor / nor0 / (s0^2 + 0.25))
for i in 1:N
	DDMPO[i] = DDMPO[i] * ratio^(0.5 / N)
	XXMPO[i] = XXMPO[i] / ratio^(0.5 / N)
end
EFI = ADD(DDMPO, XXMPO)
@show EFI

sq=Vector{Float64}(undef,2^N)
f(i)=1.0/sqrt((i - 1) / 2.0^N+2^(-10))
for i in 1:2^N
	sq[i]=f(i)
end
#=
HM = zeros(2^N, 2^N)
V(i) = 1.0 / ((i - 1) / 2.0^N + 2.0^(-shift))^2
for i in 1:2^N
	HM[i, i] = -2.0 * 2.0^(2N) + (s0^2 + 0.25) * V(i)
	if i < 2^N
		HM[i, i+1] = 1.0 * 2.0^(2N)
		HM[i+1, i] = 1.0 * 2.0^(2N)
	end
end


val, vec = eigen(-HM)
vec = permutedims(vec, (2, 1))
nv = filter(x -> x < 0, val)
ni = findall(x -> x < 0, val)
df = DataFrame(X = ni, Y = log10.(-nv))
lin = lm(@formula(Y ~ X), df)
c0, cx = coef(lin)
@show c0, cx
Y1 = cx * df.X .+ c0
for i in 1:n
	df.Y[i] = 10.0^df.Y[i]
	Y1[i] = 10.0^Y1[i]
end
=#
#scatter!(df.X, df.Y, label="exact diagonalization", xlabel = "number of state", ylabel = "-energy", color = :blue, yscale=:log10)
#plot!(df.X, Y1, label="y=" * @sprintf("%.5f", cx) * "n +" * @sprintf("%.5f", c0), color = :blue,yscale=:log10)


#err = Matrix{Float64}(undef, n, sweep + 1)
bd = Array{Int}(undef, n, sweep + 1, N - 1)
e = Matrix{Float64}(undef, n, sweep + 1)
ele = Vector{Int}(undef, n)
y = Matrix{Float64}(undef, n, 2^N)
tmps = []

e, tmps, bd, ele, y = muti_rdmrg(-EFI, CMPS, sweep, kdim, maxbdim, threshold, nothing, n, m)
e = e / ratio^(0.5) *2^(2N)

for i in 1:n
	for j in 1:sweep+1
	h5write("C:/Users/User/Downloads/julia/mps/efimov/MPS/efimov_MPS_($i,$(j-1))(N,sweep,shift,kdim)=($N,$sweep,$(shift),$(kdim)).h5", "MPS", tmps[i,j])
end
end

h5write("C:/Users/User/Downloads/julia/mps/efimov/efimov_energy(N,sweep,shift,kdim)=($N,$sweep,$(shift),$(kdim)).h5", "energy", e)
#h5write("C:/Users/User/Downloads/julia/mps/efimov/efimov_error(N,sweep,shift,kdim)=($N,$sweep,$(shift),$(kdim)).h5", "error", err)
h5write("C:/Users/User/Downloads/julia/mps/efimov/efimov_bond_dim(N,sweep,shift,kdim)=($N,$sweep,$(shift),$(kdim)).h5", "bond_dim", bd)
h5write("C:/Users/User/Downloads/julia/mps/efimov/efimov_element(N,sweep,shift,kdim)=($N,$sweep,$(shift),$(kdim)).h5", "element", ele)
h5write("C:/Users/User/Downloads/julia/mps/efimov/efimov_function(N,sweep,shift,kdim)=($N,$sweep,$(shift),$(kdim)).h5", "function", y)

bd=h5read("C:/Users/User/Downloads/julia/mps/efimov/efimov_bond_dim(N,sweep,shift,kdim)=($N,$sweep,$(shift),$(kdim)).h5", "bond_dim")


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
		collect(0:Int(sweep / m)),
		[e[i, 1:end] .- 1.0 * minimum(e[i, :]) .+ 1e-18 for i ∈ 1:n],
		legend = true,
		xlabel = "Sweep",
		ylabel = "Energy-ε",
		label = elab,
		#ylim = (minimum(e), maximum(e),
		yscale = :log10,
	)

#savefig("C:/Users/User/Downloads/julia/mps/efimov/efimov_energy(N,sweep,shift,kdim)=($N,$sweep,$(shift),$(kdim)).png")


display(
	scatter(
		collect(0:Int(sweep / m)),
		[err[i, :] for i ∈ 1:n],
		legend = true,
		xlabel = "Sweep",
		ylabel = "Error",
		label = flab,
		yscale = :log10,
		xlims = (0, sweep),
		markersize = 2.5,
	),
)  # Display the updated plot
#savefig("C:/Users/User/Downloads/julia/mps/efimov/efimov_error(N,sweep,shift,kdim)=($N,$sweep,$(shift),$(kdim)).png")



x = Vector{Float64}(undef, 2^N)
for i ∈ 1:2^N
	x[i] = (i - 1.0) / s^N
end
y1 = []
x1 = x[1:2^1:2^N]
for i in 1:n
	push!(y1, y[i, 1:2^1:2^N])
end

	plot(
		x1,
		[y1[i, :] for i ∈ 1:n],
		legend = true,
		xlabel = "X",
		ylabel = "Y",
		#title = "Particle in the box",
		label = slab,
		xlims = (minimum(x), maximum(x)),
		markersize = 2.0
	)
#savefig("C:/Users/User/Downloads/julia/mps/efimov/efimov_function(N,sweep,shift,kdim)=($N,$sweep,$(shift),$(kdim)).png")

for i in 1:n
		heatmap(collect(1:sweep+1), collect(1:N-1), permutedims(bd,(1,3,2))[i, :, :], xlabel = "Sweep", ylabel = "Bond number", yticks = (1:N-1, 1:N-1),
			label = blab,)
	#savefig("C:/Users/User/Downloads/julia/mps/efimov/efimov_bond_dim$i(N,sweep,shift,kdim)=($N,$sweep,$(shift),$(kdim)).png")
end
#=
z = Matrix{Float64}(undef, n, 2^N)
for j in 1:n
	for i in 1:2^N
		z[j, i] = y[j, i] * ((i - 1) / 2.0^N + 2.0^(-shift))^(0.5)
	end
	nor = sum(z[j, i]^2 for i in 1:2^N)
	z[j, :] = z[j, :] / nor^0.5
end

display(
	scatter(
		x,
		[z[i, :] for i ∈ 1:n],
		legend = true,
		xlabel = "X",
		ylabel = "Y",
		#title = "Particle in the box",
		#label = slab,
		xlims = (minimum(x), maximum(x)),
		markersize = 1.5,
		markerstrokewidth = 0,
	),
)  # Display the updated plot
=#
=#

plot(
	0:sweep,
	[maximum(bd[i, j, :]) for j ∈ 1:sweep+1, i ∈ 1:n],
	legend = true,
	xlabel = "Sweep",
	ylabel = "Max bond dimension",
	label = slab,
	yticks=(2:4:18)
)
#savefig("C:/Users/User/Downloads/julia/mps/plot/efimov/efimov_bond_dim(N,sweep,shift,kdim)=($N,$sweep,$shift,$(kdim)).png")

tmps = Matrix{Any}(undef, n, sweep + 1) #MPS for each state
for i in 1:n
	for j in 1:sweep+1
		file = "C:/Users/User/Downloads/julia/mps/efimov/MPS/efimov_MPS_($i,$(j-1))(N,sweep,shift,kdim)=($N,$sweep,$(shift),$(kdim)).h5"
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
			hmps[k] = EFI[k] * tmps[i, j][k]
			gmps[k] = EFI[k] * prime(tmps[i, j])[k]
		end
		hmps=prime(csc(noprime(hmps)))
		gmps=csc(gmps)
		H1[i, j] = inner(tmps[i, j], -hmps) 
		H2[i, j] = inner(-gmps, -hmps) 
		SD[i, j] = sqrt(abs((H2[i, j] - H1[i, j]^2) / 2.0^N))/ ratio^(0.5) *2^(2N)
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
#savefig("C:/Users/User/Downloads/julia/mps/plot/efimov/efimov_SD(E)(N,sweep,shift,kdim)=($N,$sweep,$shift,$(kdim)).png")