using Distributed
cpu = 9
if nworkers() < cpu
	addprocs(cpu - nworkers())
elseif nworkers() > cpu
	rmprocs(workers()[(cpu+1):end])  # Remove excess workers
end

@everywhere begin
	using ITensors
	using ITensorMPS
	using LinearAlgebra
	using DataFrames
	using HDF5
	using Printf
	using Plots
	using IterTools
	using SharedArrays
	using LaTeXStrings

	#dmrg() setup 
	#######
	N = 12 # Number of qubits 
	n = 2 # Number of lowest states to search in dmrg()
	maxbdim = 128 # maxdim in dmrg()
	cutoff = 1.0e-16 # cutoff in dmrg()
	sweep = 20 # nsweeps in dmrg()
	kdim = 128 # eigsolve_krylovdim in dmrg()
	sites = ITensors.siteinds("S=1/2", N) # Build the sites of two-level system
	str = "efimov_e^x"
	include(joinpath(@__DIR__, "functions.jl"))
	include(joinpath(@__DIR__, str*"_setup.jl"))
	#######

	energy = SharedArray{Float64}(m, l, n, sweep)
	maxerr = SharedArray{Float64}(m, l, n, sweep)
	maxlinkdim = SharedArray{Int}(m, l, n, sweep)
	y = SharedArray{Float64}(m, l, n, 2^N)
	err = SharedArray{Float64}(m, l, n, sweep)
end


@time begin
	@distributed for q in 1:(m*l)
		i = (q - 1) ÷ l + 1
		j = (q - 1) % l + 1
		tmps = Vector{MPS}()
		for k in 1:n
			open(joinpath(@__DIR__, str, "info.txt", str*"_info($k)(s0, N,sweep,shift,kdim,cutoff)=($(s0[i]),$(N),$(sweep),$(shift[j]),$(kdim),$(cutoff)).txt"), "w") do file
				# Redirect standard output to the file
				redirect_stdout(file) do
					energy, mps = dmrg(H[i, j], tmps, INMPS[j]; nsweeps = sweep, mindim = 2, maxdim = maxbdim, eigsolve_krylovdim = kdim, cutoff = cutoff)
					push!(tmps, mps)
					x, y[i, j, k, :], nor = mps2f(mps)
					#h5write(joinpath(@__DIR__, str, "mps.h5", str*"_MPS($i)(N,sweep,shift,kdim,cutoff,weight)=($(s0[j]), $(N),$(sweep),$(shift),$(kdim),$(cutoff)).h5"), "MPS", mps)
				end
			end
			output_str = read(joinpath(@__DIR__, str, "info.txt", str*"_info($k)(s0, N,sweep,shift,kdim,cutoff)=($(s0[i]),$(N),$(sweep),$(shift[j]),$(kdim),$(cutoff)).txt"), String)
			lines = split(output_str, '\n')
			maxerr[i, j, k, :] = [parse(Float64, match(r"maxerr=([0-9.eE+-]+)", line).captures[1])
								  for line in lines if occursin("maxerr=", line)]
			maxlinkdim[i, j, k, :] = [parse(Int, match(r"maxlinkdim=([0-9]+)", line).captures[1])
									  for line in lines if occursin("maxerr=", line)]
			energy[i, j, k, :] = [parse(Float64, match(r"energy=([0-9.eE+-]+)", line).captures[1])
								  for line in lines if occursin("maxerr=", line)]/factor[j]
			@show energy[i, j, k, end]
		end
	end
end

# Normalize eigenstates in linear or exponential coordinate.
#######
x, z = let
	dx = Vector{Float64}(undef, 2^N) # dx[i] = x[i+1] - x[i] 
	x = Vector{Float64}(undef, 2^N) # Position 
	z = copy(y)
	for j in 1:l
		dx = [x0[j] * xscale[j]^(i) * (1.0 - 1.0 / xscale[j]) for i in 1:(2^N)] # Exponential x
		#dx = [2.0^(-N) for i in 1:(2^N-1)] # linear x	

		#Put the Jacobi back
		for k in 1:n, i in 1:m
			z[i, j, k, :] = z[i, j, k, :] * exp(-0.5 * (i - 1.0) * log(xscale[j])) # exp(-1.0 * (i - 1.0) * log(xscale)) for \psi ~ cos(s0*log(x)), and exp(-0.5 * (i - 1.0) * log(xscale)) for \psi ~ sqrt(2) * cos(s0*log(x)) 
			nor = (z[i, j, k, :] .* z[i, j, k, :])' * dx * 2.0^(N) # normalization factor nor
			@show dx
			#z[i, j, k, :] = z[i, j, k, :] / nor^0.5
		end
	end
	x, z
end
#######

# Plots
#######
j = 5
k = 1
elab = [latexstring("g = $(g[i]), E_$(k)=$(@sprintf("%.3e", energy[i,j,k,end]))") for a in 1:1, i in 1:m]

for i in 1:m
	if z[i, j, k, end-1] < 0
		z[i, j, k, :] = -z[i, j, k, :]
	end
end

plot(
	[x0[j] * xscale[j]^i for i in 0:(2^N-1)], [z[i, j, k, :] for i in 1:m],
	xlabel = L"x", ylabel = latexstring("\\psi_$(k)(e^{x})"), label = elab[:, 1:m],
	xguidefontsize = 24, yguidefontsize = 24,
	xtickfontsize = 18, ytickfontsize = 18,
	legendfont = 16,
	linewidth = 3,
	#legend = false,
)
#######
