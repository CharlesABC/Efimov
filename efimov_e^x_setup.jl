# Efimov Hamiltonian in exponential coordinate setup

"Build second-order differential operator into MPO in exponetial coordinate for N qubits with log scaling factor c."
function EDD(sites::Vector{Index{Int64}}, c::Float64)
	N = length(sites)
	bond3 = [Index(3, "Link, l=$a") for a ∈ 1:(N-1)]
	DDMPO = MPO(sites)

	DD = ITensor(sites[1], prime(sites[1]), bond3[1])
	DD[sites[1]=>2, prime(sites[1])=>1, bond3[1]=>1] = 1.0 * exp(-2.0c * 2.0^(0) + 0.5c)
	DD[sites[1]=>1, prime(sites[1])=>2, bond3[1]=>3] = 1.0 * exp(0.5c)
	DD[sites[1]=>2, prime(sites[1])=>1, bond3[1]=>2] = 1.0 * exp(0.5c)
	DD[sites[1]=>1, prime(sites[1])=>2, bond3[1]=>1] = 1.0 * exp(-2.0c * 2.0^(0) + 0.5c)
	DD[sites[1]=>1, prime(sites[1])=>1, bond3[1]=>1] = -1.0 * (exp(-c) + 1.0)
	DD[sites[1]=>2, prime(sites[1])=>2, bond3[1]=>1] = -1.0 * (exp(-c) + 1.0) * exp(-2.0c * 2.0^(0))
	DDMPO[1] = DD

	for a ∈ 2:(N-1)
		DD = ITensor(sites[a], prime(sites[a]), bond3[a-1], bond3[a])
		DD[sites[a]=>2, prime(sites[a])=>1, bond3[a-1]=>3, bond3[a]=>1] = 1.0 * exp(-2.0c * 2.0^(a - 1))
		DD[sites[a]=>1, prime(sites[a])=>2, bond3[a-1]=>3, bond3[a]=>3] = 1.0
		DD[sites[a]=>2, prime(sites[a])=>1, bond3[a-1]=>2, bond3[a]=>2] = 1.0
		DD[sites[a]=>1, prime(sites[a])=>2, bond3[a-1]=>2, bond3[a]=>1] = 1.0 * exp(-2.0c * 2.0^(a - 1))
		DD[sites[a]=>1, prime(sites[a])=>1, bond3[a-1]=>1, bond3[a]=>1] = 1.0
		DD[sites[a]=>2, prime(sites[a])=>2, bond3[a-1]=>1, bond3[a]=>1] = 1.0 * exp(-2.0c * 2.0^(a - 1))
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

# x \in [x0, x0 * xscale^{2^N-1}})
shift = [-6, -8, -10, -12, -14, -16]
l = length(shift) # Number of differnt x0 cutoff
x0 = [2.0^(-shift[j]) for j in 1:l] # cutoff of x
xscale = [2.0^(-log2(x0[j]) / 2.0^N) for j in 1:l] 

# dx2 = x0^2 * delta
delta = [xscale[j] * (1.0 - 1.0 / xscale[j])^2 * ((xscale[j] + 1.0) / 2.0) for j in 1:l]

# Scaling factor
s0 = [i*0.05*im for i in 0:6] # Scaling factor
m = length(s0) # Number of differnt coupling
scale = [exp(pi / real(s0[i]^2)) / xscale[j] for i in 1:m, j in 1:l] # The number of grids reelated to one discrete scaling by s0

# Construct Hamiltonian and initial MPS
IMPS, IMPO = I(sites)
NEXMPO = [EX(sites, -0.5 * (2.0^N * log(xscale[j]))) for j in 1:l]
N2EXMPO = [EX(sites, -2.0 * (2.0^N * log(xscale[j]))) for j in 1:l]
K = [EDD(sites, log(xscale[j])) for j in 1:l]
for j in 1:l, i in 1:N
	N2EXMPO[j][i] = N2EXMPO[j][i] * delta[j]^(0.0 / N) 
	K[j][i] = K[j][i] / delta[j]^(1.0 / N)
end

INMPS = [noprime(NEXMPO[j] * COS(sites, 1000.0)) for j in 1:l] # Initial MPS in dmrg()
g = [real(s0[i]^2) - 0.25 for i in 1:m] # coupling constant
H = [ADD(-K[j], g[i] * N2EXMPO[j]) for i in 1:m, j in 1:l] # H = K + (s0^2 - 0.25)/r^2
factor = [x0[j]^2 for j in 1:l] # real  = E / factor
