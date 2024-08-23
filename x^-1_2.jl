using Distributed
cpu = 4
if nworkers() < cpu
	addprocs(cpu - nworkers())
elseif nworkers() > cpu
	rmprocs(workers()[cpu+1:end])  # Remove excess workers
end

@everywhere begin
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

	N = 10
	n = 4

	maxbdim = 64
	threshold = 1e-10
	sweep = 200
	kdim = 16
	m = 1
	s = 2.0

	shi = Vector{Float64}()
	shift = Vector{Int}()
	for i in 1:n
		push!(shift, (i - 1) * 5)
		push!(shi, Float64(2.0^(-shift[i])))
	end
	bond = Index[Index(2, "link, l=$a") for a ∈ 1:N-1]
	bond1 = Index[Index(1, "link, l=$a") for a ∈ 1:N-1]
	sites = siteinds("S=1/2", N)
	
	include("C:/Users/User/Downloads/julia/functions.jl")

	
	function rdmrg(mpo::MPO, mps::MPS, sweep::Int, kdim::Int, maxbdim::Int, threshold::Float64, z::Vector{Float64}, m::Int)
	err = Vector{Float64}(undef, sweep + 1)
	bd = Matrix{Int}(undef, sweep + 1, N - 1)
	e = Vector{Float64}(undef, sweep + 1)
	mps0 = MPS(sites)
	r = Int(sweep / m)

	##
	for j ∈ 1:N
		mps0[j] = noprime(mps[j] * NEXMPO[j])
	end
	mps0 = csc(mps0)
	x, y = mps2f(mps0)
	##

	err[1] = ERROR(N, y, z)
	bdim, ele = BDIM(N, mps)
	bd[1, :] = bdim
	energy = inner(prime(mps), mpo * mps)
	e[1] = energy

	for i in 1:r
		energy, mps = dmrg(
			mpo,
			mps,
			cutoff = threshold,
			maxdim = maxbdim,
			mindim = 2,
			noise = 10.0^(-15),
			nsweeps = m,
			outputlevel = 2,
			eigsolve_krylovdim = kdim,
		)

		##
		for j ∈ 1:N
			mps0[j] = noprime(mps[j] * NEXMPO[j])
		end
		mps0 = csc(mps0)
		x, y, nor = mps2f(mps0)
		y = y / nor^(0.5)
		##
		err[i+1] = ERROR(N, y, z)
		bdim, ele = BDIM(N, mps)
		bd[i+1, :] = bdim
		e[i+1] = energy
	end

	return e, mps0, err, bd, ele, y
	end

	#mps0 = MPS(sites)
	IMPS, IMPO = I(N)
	DDMPO = DD(N, :NBC)
	NEXMPO = EX(N, -6.0)



	e = SharedArray{Float64}(n, Int(sweep / m) + 1)
	err = SharedArray{Float64}(n, Int(sweep / m) + 1)
	bd = SharedArray{Int}(n, Int(sweep / m) + 1, N - 1)
	ele = SharedArray{Int}(n)
	y = SharedArray{Float64}(n, 2^N)

	slab = Matrix{String}(undef, 1, n)
	elab = Matrix{String}(undef, 1, n)
	flab = Matrix{String}(undef, 1, n)
	blab = Matrix{String}(undef, 1, n)
end

@sync begin
	@distributed for i ∈ 1:n
		F = MPO(sites)
		H = MPO(sites)
		z = Vector{Float64}(undef, 2^N)


		RMPS = randomMPS(sites, 1)
		for i in 1:N
			RMPS[i] = NEXMPO[i] * prime(IMPS[i])
		end
		RMPS = csc(RMPS)



		for j ∈ 1:2^N
			z[j] = 1.0 / ((j - 1.0) / s^N + shi[i])
		end
		XIMPO = X(N, shi[i])
		for j ∈ 1:N
			F[j] = (prime(NEXMPO[j]) * XIMPO[j]) * prime(IMPO[j])
		end
		F = cfc(F)
		for j ∈ 1:N
			H[j] =
				prime(IMPO[j]) *
				(prime(prime(IMPO[j])) * ((prime(prime(F[j]))) * (prime(DDMPO[j]) * F[j])))
		end
		H = -cfc(H)
		@time begin
			e[i, :], mps, err[i, :], bd[i, :, :], ele[i], y[i, :] = rdmrg(H, RMPS, sweep, kdim, maxbdim, threshold, z, m)
		end
		h5write("C:/Users/User/Downloads/julia/mps/(x+a)^-1/(x+a)^-1_MPS(N,sweep,shift,kdim)=($N,$sweep,$(shift[i]),$(kdim)).h5", "MPS", mps)

		@show ele[i]
		@show mps
		@show H
		@show RMPS
	end
end
@show 1
h5write("C:/Users/User/Downloads/julia/mps/(x+a)^-1/(x+a)^-1_energy(N,sweep,kdim)=($N,$sweep,$(kdim)).h5", "energy", e)
h5write("C:/Users/User/Downloads/julia/mps/(x+a)^-1/(x+a)^-1_error(N,sweep,kdim)=($N,$sweep,$(kdim)).h5", "error", err)
h5write("C:/Users/User/Downloads/julia/mps/(x+a)^-1/(x+a)^-1_bond_dim(N,sweep,kdim)=($N,$sweep,$(kdim)).h5", "bond_dim", bd)
h5write("C:/Users/User/Downloads/julia/mps/(x+a)^-1/(x+a)^-1_element(N,sweep,kdim)=($N,$sweep,$(kdim)).h5", "element", ele)
h5write("C:/Users/User/Downloads/julia/mps/(x+a)^-1/(x+a)^-1_function(N,sweep,kdim)=($N,$sweep,$(kdim)).h5", "function", y)


for i ∈ 1:n
	slab[1, i] = "(x+2^(-$(shift[i])))^-1"
	elab[1, i] = "(x+2^(-$(shift[i])))^-1 (energy=$(@sprintf("%.4e", e[i,sweep+1])))"
	flab[1, i] = "(x+2^(-$(shift[i])))^-1 (error=$(@sprintf("%.4e", err[i,sweep+1])))"
	blab[1, i] = "(x+2^(-$(shift[i])))^-1 (total elements=$(ele[i]))"
end


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
#savefig("C:/Users/User/Downloads/julia/mps/(x+a)^-1/(x+a)^-1_error(N,shift,kdim)=($N,$(shift),$(kdim)).png")



x = Vector{Float64}(undef, 2^N)
for i ∈ 1:2^N
	x[i] = (i - 1.0) / s^N
end
y1 = []
x1 = x[1:2^1:2^N]
for i in 1:n
	push!(y1, y[i, 1:2^1:2^N])
end
display(
	scatter(
		x1,
		[y1[i, :] for i ∈ 1:n],
		legend = true,
		xlabel = "X",
		ylabel = "Y",
		#title = "Particle in the box",
		label = slab,
		xlims = (minimum(x), maximum(x)),
		markersize = 1.5,
		markerstrokewidth=0
	),
)  # Display the updated plot
#savefig("C:/Users/User/Downloads/julia/mps/(x+a)^-1/(x+a)^-1_function(N,shift,kdim)=($N,$(shift),$(kdim)).png")

for i in 1:n
	display(
		heatmap(collect(1:sweep+1), collect(1:N-1), permutedims(bd,(1,3,2))[i, :, :], xlabel = "Sweep", ylabel = "Bond number", ylim = (1, N - 1),
		label = blab),
	)
	#savefig("C:/Users/User/Downloads/julia/mps/(x+a)^-1/(x+a)^-1_bond_dim$i(N,shift,kdim)=($N,$(shift[i]),$(kdim)).png")
end


display(
	scatter(
		collect(1:Int(sweep / m)),
		[e[i, 1:end] for i ∈ 1:n],
		legend = true,
		xlabel = "Sweep",
		ylabel = "Energy",
		label = elab,
		#ylim = (1e-18, 1e-4),
		yscale = :log10,
		markersize = 2.0,
	),
)  # Display the updated plot
#savefig("C:/Users/User/Downloads/julia/mps/(x+a)^-1/(x+a)^-1_energy(N,shift,kdim)=($N,$(shift),$(kdim)).png")

bd=h5read("C:/Users/User/Downloads/julia/mps/(x+a)^-1/(x+a)^-1_bond_dim(N,sweep,kdim)=($N,$sweep,$(kdim)).h5", "bond_dim")
plot(
	0:sweep,
	[maximum(bd[i, j, :]) for j ∈ 1:sweep+1, i ∈ 1:n],
	legend = true,
	xlabel = "Sweep",
	ylabel = "Max bond dimension",
	#title = "Particle in the box",
	label = slab,
	xticks=(0:20:200)
)
#savefig("C:/Users/User/Downloads/julia/mps/plot/(x+a)^-1/(x+a)^-1_bond_dim(N,sweep,kdim)=($N,$sweep,$(kdim)).png")
