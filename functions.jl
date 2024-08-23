using ITensors
using LinearAlgebra
using DataFrames
using Plots

"Build identity MPS and MPO for N qubits"
function I(N::Int)
	IMPO = MPO(sites)
	IMPS = MPS(sites)
	for a ∈ 1:N
		I = ITensor(sites[a], prime(sites[a]))
		I[sites[a]=>1, prime(sites[a])=>1] = 1.0
		I[sites[a]=>2, prime(sites[a])=>2] = 1.0
		IMPO[a] = I
		I = ITensor(sites[a])
		I[sites[a]=>1] = 1.0
		I[sites[a]=>2] = 1.0
		IMPS[a] = I
	end
	return IMPS, IMPO
end

"build f(x)=e^{cx} MPS for N qubits"
function EX(N::Int, c::Float64)
	EXMPO = MPO(sites)
	bond1 = Index[Index(1, "link, l=$a") for a ∈ 1:N-1]

	A = ITensor(sites[1], prime(sites[1]), bond1[1])
	A[sites[1]=>1, prime(sites[1])=>1, bond1[1]=>1] = 1.0
	A[sites[1]=>2, prime(sites[1])=>2, bond1[1]=>1] = exp(c / 2.0^N)
	EXMPO[1] = A

	for a ∈ 2:N-1
		A = ITensor(sites[a], prime(sites[a]), bond1[a-1], bond1[a])
		A[sites[a]=>1, prime(sites[a])=>1, bond1[a-1]=>1, bond1[a]=>1] = 1.0
		A[sites[a]=>2, prime(sites[a])=>2, bond1[a-1]=>1, bond1[a]=>1] =
			exp(c / 2.0^(N - a + 1))
		EXMPO[a] = A
	end

	A = ITensor(sites[N], prime(sites[N]), bond1[N-1])
	A[sites[N]=>1, prime(sites[N])=>1, bond1[N-1]=>1] = 1.0
	A[sites[N]=>2, prime(sites[N])=>2, bond1[N-1]=>1] = exp(c / 2.0)
	EXMPO[N] = A
	return EXMPO
end

"build f(x)=cos(kx) MPS for N qubits"
function COS(N::Int, k::Float64)
	CMPS = MPS(sites)
	bond = Index[Index(2, "link, l=$a") for a ∈ 1:N-1]

	A = ITensor(sites[1], bond[1])
	A[sites[1]=>1, bond[1]=>1] = 1.0
	A[sites[1]=>1, bond[1]=>2] = 0.0
	A[sites[1]=>2, bond[1]=>1] = cos(k * 1.0 / 2.0^N)
	A[sites[1]=>2, bond[1]=>2] = -sin(k * 1.0 / 2.0^N)
	CMPS[1] = A

	for a in 2:N-1
		A = ITensor(sites[a], bond[a-1], bond[a])
		A[sites[a]=>1, bond[a-1]=>1, bond[a]=>1] = 1.0
		A[sites[a]=>1, bond[a-1]=>1, bond[a]=>2] = 0.0
		A[sites[a]=>1, bond[a-1]=>2, bond[a]=>1] = 0.0
		A[sites[a]=>1, bond[a-1]=>2, bond[a]=>2] = 1.0
		A[sites[a]=>2, bond[a-1]=>1, bond[a]=>1] = cos(k * 1.0 / 2.0^(N - a + 1))
		A[sites[a]=>2, bond[a-1]=>1, bond[a]=>2] = -sin(k * 1.0 / 2.0^(N - a + 1))
		A[sites[a]=>2, bond[a-1]=>2, bond[a]=>1] = sin(k * 1.0 / 2.0^(N - a + 1))
		A[sites[a]=>2, bond[a-1]=>2, bond[a]=>2] = cos(k * 1.0 / 2.0^(N - a + 1))
		CMPS[a] = A
	end

	A = ITensor(sites[N], bond[N-1])
	A[sites[N]=>1, bond[N-1]=>1] = 1.0
	A[sites[N]=>1, bond[N-1]=>2] = 0.0
	A[sites[N]=>2, bond[N-1]=>1] = cos(k * 1.0 / 2.0)
	A[sites[N]=>2, bond[N-1]=>2] = sin(k * 1.0 / 2.0)
	CMPS[N] = A

	return CMPS
end

"Build f(x)=x+shi MPS for N qubits"
function X(N::Int, shi::Float64)
	XMPO = MPO(sites)
	bond = Index[Index(2, "link, l=$a") for a ∈ 1:N-1]
	A = ITensor(sites[1], prime(sites[1]), bond[1])
	A[sites[1]=>2, prime(sites[1])=>2, bond[1]=>1] = 1.0 / 2.0^(N) + shi / N
	A[sites[1]=>2, prime(sites[1])=>2, bond[1]=>2] = 1.0
	A[sites[1]=>1, prime(sites[1])=>1, bond[1]=>1] = 0.0 + shi / N
	A[sites[1]=>1, prime(sites[1])=>1, bond[1]=>2] = 1.0
	XMPO[1] = A

	for a ∈ 2:N-1
		A = ITensor(sites[a], prime(sites[a]), bond[a-1], bond[a])
		A[sites[a]=>2, prime(sites[a])=>2, bond[a-1]=>1, bond[a]=>1] = 1.0
		A[sites[a]=>2, prime(sites[a])=>2, bond[a-1]=>1, bond[a]=>2] = 0.0
		A[sites[a]=>2, prime(sites[a])=>2, bond[a-1]=>2, bond[a]=>1] = 1.0 / 2.0^(N - a + 1) + shi / N
		A[sites[a]=>2, prime(sites[a])=>2, bond[a-1]=>2, bond[a]=>2] = 1.0
		A[sites[a]=>1, prime(sites[a])=>1, bond[a-1]=>1, bond[a]=>1] = 1.0
		A[sites[a]=>1, prime(sites[a])=>1, bond[a-1]=>1, bond[a]=>2] = 0.0
		A[sites[a]=>1, prime(sites[a])=>1, bond[a-1]=>2, bond[a]=>1] = 0.0 + shi / N
		A[sites[a]=>1, prime(sites[a])=>1, bond[a-1]=>2, bond[a]=>2] = 1.0
		XMPO[a] = A
	end

	A = ITensor(sites[N], prime(sites[N]), bond[N-1])
	A[sites[N]=>2, prime(sites[N])=>2, bond[N-1]=>1] = 1.0
	A[sites[N]=>2, prime(sites[N])=>2, bond[N-1]=>2] = 1.0 / 2.0 + shi / N
	A[sites[N]=>1, prime(sites[N])=>1, bond[N-1]=>1] = 1.0
	A[sites[N]=>1, prime(sites[N])=>1, bond[N-1]=>2] = 0.0 + shi / N
	XMPO[N] = A

	return XMPO
end

"Build step function u(x-x') where pos is a vector store each binary number of x'."
function STEP(N::Int, pos::Vector{Int})
	SMPO = MPO(sites)
	bond2 = Index[Index(2, "link, l=$a") for a ∈ 1:N-1]
	S = ITensor(sites[N], prime(sites[N]), bond2[N-1])
	if pos[N] == 1
		S[sites[N]=>2, prime(sites[N])=>2, bond2[N-1]=>2] = 1.0
	else
		S[sites[N]=>2, prime(sites[N])=>2, bond2[N-1]=>1] = 1.0
		S[sites[N]=>1, prime(sites[N])=>1, bond2[N-1]=>2] = 1.0
	end
	SMPO[N] = S

	for a ∈ 2:N-1
		S = ITensor(sites[a], prime(sites[a]), bond2[a-1], bond2[a])
		if pos[a] == 1
			S[sites[a]=>2, prime(sites[a])=>2, bond2[a-1]=>2, bond2[a]=>2] = 1.0
			S[sites[a]=>2, prime(sites[a])=>2, bond2[a-1]=>1, bond2[a]=>1] = 1.0
			S[sites[a]=>1, prime(sites[a])=>1, bond2[a-1]=>1, bond2[a]=>1] = 1.0
		else
			S[sites[a]=>2, prime(sites[a])=>2, bond2[a-1]=>1, bond2[a]=>2] = 1.0
			S[sites[a]=>1, prime(sites[a])=>1, bond2[a-1]=>2, bond2[a]=>2] = 1.0
			S[sites[a]=>2, prime(sites[a])=>2, bond2[a-1]=>1, bond2[a]=>1] = 1.0
			S[sites[a]=>1, prime(sites[a])=>1, bond2[a-1]=>1, bond2[a]=>1] = 1.0
		end
		SMPO[a] = S
	end

	S = ITensor(sites[1], prime(sites[1]), bond2[1])
	if pos[1] == 1
		S[sites[1]=>2, prime(sites[1])=>2, bond2[1]=>2] = 1.0
		S[sites[1]=>2, prime(sites[1])=>2, bond2[1]=>1] = 1.0
		S[sites[1]=>1, prime(sites[1])=>1, bond2[1]=>1] = 1.0
	else
		S[sites[1]=>2, prime(sites[1])=>2, bond2[1]=>2] = 1.0
		S[sites[1]=>1, prime(sites[1])=>1, bond2[1]=>2] = 1.0
		S[sites[1]=>2, prime(sites[1])=>2, bond2[1]=>1] = 1.0
		S[sites[1]=>1, prime(sites[1])=>1, bond2[1]=>1] = 1.0
	end
	SMPO[1] = S
	return SMPO
end
#update PBC
"Build second-order differential operator into MPO for N qubits without overall factor 2^(2N). 

NBC for Neumann B.C., DBC for Dirichlet B.C., and, PBC for periodic B.C."
function DD(N::Int, which::Symbol = :NBC)
	DDMPO = MPO(sites)
	if which == :NBC
		bond5 = Index[Index(5, "link, l=$a") for a ∈ 1:N-1]

		DD = ITensor(sites[1], prime(sites[1]), bond5[1])
		DD[sites[1]=>2, prime(sites[1])=>1, bond5[1]=>1] = 1.0
		DD[sites[1]=>1, prime(sites[1])=>2, bond5[1]=>3] = 1.0
		DD[sites[1]=>2, prime(sites[1])=>1, bond5[1]=>2] = 1.0
		DD[sites[1]=>1, prime(sites[1])=>2, bond5[1]=>1] = 1.0
		DD[sites[1]=>1, prime(sites[1])=>1, bond5[1]=>1] = -2.0
		DD[sites[1]=>2, prime(sites[1])=>2, bond5[1]=>1] = -2.0
		DD[sites[1]=>1, prime(sites[1])=>1, bond5[1]=>4] = 1.0
		DD[sites[1]=>2, prime(sites[1])=>2, bond5[1]=>5] = 1.0
		DDMPO[1] = DD

		for a ∈ 2:N-1
			DD = ITensor(sites[a], prime(sites[a]), bond5[a-1], bond5[a])
			DD[sites[a]=>2, prime(sites[a])=>1, bond5[a-1]=>3, bond5[a]=>1] = 1.0
			DD[sites[a]=>1, prime(sites[a])=>2, bond5[a-1]=>3, bond5[a]=>3] = 1.0
			DD[sites[a]=>2, prime(sites[a])=>1, bond5[a-1]=>2, bond5[a]=>2] = 1.0
			DD[sites[a]=>1, prime(sites[a])=>2, bond5[a-1]=>2, bond5[a]=>1] = 1.0
			DD[sites[a]=>1, prime(sites[a])=>1, bond5[a-1]=>1, bond5[a]=>1] = 1.0
			DD[sites[a]=>2, prime(sites[a])=>2, bond5[a-1]=>1, bond5[a]=>1] = 1.0
			DD[sites[a]=>1, prime(sites[a])=>1, bond5[a-1]=>4, bond5[a]=>4] = 1.0
			DD[sites[a]=>2, prime(sites[a])=>2, bond5[a-1]=>5, bond5[a]=>5] = 1.0
			DDMPO[a] = DD
		end

		DD = ITensor(sites[N], prime(sites[N]), bond5[N-1])
		DD[sites[N]=>2, prime(sites[N])=>1, bond5[N-1]=>3] = 1.0
		DD[sites[N]=>1, prime(sites[N])=>1, bond5[N-1]=>1] = 1.0
		DD[sites[N]=>2, prime(sites[N])=>2, bond5[N-1]=>1] = 1.0
		DD[sites[N]=>1, prime(sites[N])=>2, bond5[N-1]=>2] = 1.0
		DD[sites[N]=>1, prime(sites[N])=>1, bond5[N-1]=>4] = 1.0
		DD[sites[N]=>2, prime(sites[N])=>2, bond5[N-1]=>5] = 1.0
		DDMPO[N] = DD
	elseif which == :DBC
		bond3 = Index[Index(3, "link, l=$a") for a ∈ 1:N-1]

		DD = ITensor(sites[1], prime(sites[1]), bond3[1])
		DD[sites[1]=>2, prime(sites[1])=>1, bond3[1]=>1] = 1.0
		DD[sites[1]=>1, prime(sites[1])=>2, bond3[1]=>3] = 1.0
		DD[sites[1]=>2, prime(sites[1])=>1, bond3[1]=>2] = 1.0
		DD[sites[1]=>1, prime(sites[1])=>2, bond3[1]=>1] = 1.0
		DD[sites[1]=>1, prime(sites[1])=>1, bond3[1]=>1] = -2.0
		DD[sites[1]=>2, prime(sites[1])=>2, bond3[1]=>1] = -2.0
		DDMPO[1] = DD

		for a ∈ 2:N-1
			DD = ITensor(sites[a], prime(sites[a]), bond3[a-1], bond3[a])
			DD[sites[a]=>2, prime(sites[a])=>1, bond3[a-1]=>3, bond3[a]=>1] =
				1.0
			DD[sites[a]=>1, prime(sites[a])=>2, bond3[a-1]=>3, bond3[a]=>3] = 1.0
			DD[sites[a]=>2, prime(sites[a])=>1, bond3[a-1]=>2, bond3[a]=>2] = 1.0
			DD[sites[a]=>1, prime(sites[a])=>2, bond3[a-1]=>2, bond3[a]=>1] = 1.0
			DD[sites[a]=>1, prime(sites[a])=>1, bond3[a-1]=>1, bond3[a]=>1] = 1.0
			DD[sites[a]=>2, prime(sites[a])=>2, bond3[a-1]=>1, bond3[a]=>1] = 1.0
			DDMPO[a] = DD
		end

		DD = ITensor(sites[N], prime(sites[N]), bond3[N-1])
		DD[sites[N]=>2, prime(sites[N])=>1, bond3[N-1]=>3] = 1.0
		DD[sites[N]=>1, prime(sites[N])=>1, bond3[N-1]=>1] = 1.0
		DD[sites[N]=>2, prime(sites[N])=>2, bond3[N-1]=>1] = 1.0
		DD[sites[N]=>1, prime(sites[N])=>2, bond3[N-1]=>2] = 1.0
		DDMPO[N] = DD
	elseif which == :PBC
		bond3 = Index[Index(3, "link, l=$a") for a ∈ 1:N-1]

		DD = ITensor(sites[1], prime(sites[1]), bond3[1])
		DD[sites[1]=>2, prime(sites[1])=>1, bond3[1]=>1] = 1.0
		DD[sites[1]=>1, prime(sites[1])=>2, bond3[1]=>3] = 1.0
		DD[sites[1]=>2, prime(sites[1])=>1, bond3[1]=>2] = 1.0
		DD[sites[1]=>1, prime(sites[1])=>2, bond3[1]=>1] = 1.0
		DD[sites[1]=>1, prime(sites[1])=>1, bond3[1]=>1] = -2.0
		DD[sites[1]=>2, prime(sites[1])=>2, bond3[1]=>1] = -2.0
		DDMPO[1] = DD

		for a ∈ 2:N-1
			DD = ITensor(sites[a], prime(sites[a]), bond3[a-1], bond3[a])
			DD[sites[a]=>2, prime(sites[a])=>1, bond3[a-1]=>3, bond3[a]=>1] =
				1.0
			DD[sites[a]=>1, prime(sites[a])=>2, bond3[a-1]=>3, bond3[a]=>3] = 1.0
			DD[sites[a]=>2, prime(sites[a])=>1, bond3[a-1]=>2, bond3[a]=>2] = 1.0
			DD[sites[a]=>1, prime(sites[a])=>2, bond3[a-1]=>2, bond3[a]=>1] = 1.0
			DD[sites[a]=>1, prime(sites[a])=>1, bond3[a-1]=>1, bond3[a]=>1] = 1.0
			DD[sites[a]=>2, prime(sites[a])=>2, bond3[a-1]=>1, bond3[a]=>1] = 1.0
			DDMPO[a] = DD
		end

		DD = ITensor(sites[N], prime(sites[N]), bond3[N-1])
		DD[sites[N]=>2, prime(sites[N])=>1, bond3[N-1]=>3] = 1.0
		DD[sites[N]=>1, prime(sites[N])=>2, bond3[N-1]=>3] = 1.0
		DD[sites[N]=>1, prime(sites[N])=>1, bond3[N-1]=>1] = 1.0
		DD[sites[N]=>2, prime(sites[N])=>2, bond3[N-1]=>1] = 1.0
		DD[sites[N]=>1, prime(sites[N])=>2, bond3[N-1]=>2] = 1.0
		DD[sites[N]=>2, prime(sites[N])=>1, bond3[N-1]=>2] = 1.0
		DDMPO[N] = DD
	else
		error("Invalid which: $which. Must be :NBC or :DBC.")
	end
	return DDMPO
end

"Conbine the bond indices of MPO"
function cfc(mpo::MPO)
	C = []
	c = mpo[1] * IMPO[1]
	indices = ITensors.inds(c)
	push!(C, combiner(indices))
	mpo[1] = mpo[1] * C[1]
	for i ∈ 2:N-1
		c = mpo[i] * IMPO[i] * c
		indices = ITensors.inds(c)
		push!(C, combiner(indices))
		mpo[i] = C[i-1] * mpo[i] * C[i]
	end
	mpo[N] = C[N-1] * mpo[N]
	return mpo
end

"Conbine the bond indices of MPS"
function csc(mps::MPS)
	C = []
	c = mps[1] * IMPS[1]
	indices = ITensors.inds(c)
	push!(C, combiner(indices))
	mps[1] = mps[1] * C[1]
	for i ∈ 2:N-1
		c = mps[i] * IMPS[i] * c
		indices = ITensors.inds(c)
		push!(C, combiner(indices))
		mps[i] = C[i-1] * mps[i] * C[i]
	end
	mps[N] = C[N-1] * mps[N]
	return mps
end

"Add two MPOs without contraction (julia will contract automatically if you write mpo1+mpo2)"
function ADD(mpo1::MPO, mpo2::MPO)
	mpo = MPO(sites)
	b = []
	b1 = []
	b2 = []
	push!(b, Int(dim(mpo1[1]) / 4 + dim(mpo2[1]) / 4))
	push!(b1, Int(dim(mpo1[1]) / 4))
	push!(b2, Int(dim(mpo2[1]) / 4))
	for i ∈ 2:N-1
		push!(b, Int(dim(mpo1[i]) / (4 * b1[i-1]) + dim(mpo2[i]) / (4 * b2[i-1])))
		push!(b1, Int(dim(mpo1[i]) / (4 * b1[i-1])))
		push!(b2, Int(dim(mpo2[i]) / (4 * b2[i-1])))
	end
	nbond = Index[Index(b[i], "link, l=$i") for i ∈ 1:N-1]
	bond1 = []
	bond2 = []
	push!(bond1, linkind(mpo1, 1))
	push!(bond2, linkind(mpo2, 1))

	A = ITensor(sites[1], prime(sites[1]), nbond[1])
	for a ∈ 1:2, b ∈ 1:2
		for i ∈ 1:b1[1]
			A[sites[1]=>a, prime(sites[1])=>b, nbond[1]=>i] =
				mpo1[1][sites[1]=>a, prime(sites[1])=>b, bond1[1]=>i]
		end
		for j ∈ 1:b2[1]
			A[sites[1]=>a, prime(sites[1])=>b, nbond[1]=>(j+b1[1])] =
				mpo2[1][sites[1]=>a, prime(sites[1])=>b, bond2[1]=>j]
		end
	end
	mpo[1] = A

	for k ∈ 2:N-1
		push!(bond1, linkind(mpo1, k))
		push!(bond2, linkind(mpo2, k))
		A = ITensor(sites[k], prime(sites[k]), nbond[k-1], nbond[k])
		for a ∈ 1:2, b ∈ 1:2
			for i ∈ 1:b1[k], p ∈ 1:b1[k-1]
				A[sites[k]=>a, prime(sites[k])=>b, nbond[k-1]=>p, nbond[k]=>i] =
					mpo1[k][sites[k]=>a, bond1[k-1]=>p, prime(sites[k])=>b, bond1[k]=>i]
			end
			for j ∈ 1:b2[k], q ∈ 1:b2[k-1]
				A[sites[k]=>a, prime(sites[k])=>b, nbond[k-1]=>(q+b1[k-1]), nbond[k]=>(j+b1[k])] =
					mpo2[k][sites[k]=>a, bond2[k-1]=>q, prime(sites[k])=>b, bond2[k]=>j]
			end
		end
		mpo[k] = A
	end

	A = ITensor(sites[N], prime(sites[N]), nbond[N-1])
	for a ∈ 1:2, b ∈ 1:2
		for i ∈ 1:b1[N-1]
			A[sites[N]=>a, prime(sites[N])=>b, nbond[N-1]=>i] =
				mpo1[N][sites[N]=>a, prime(sites[N])=>b, bond1[N-1]=>i]
		end
		for j ∈ 1:b2[N-1]
			A[sites[N]=>a, prime(sites[N])=>b, nbond[N-1]=>(j+b1[N-1])] =
				mpo2[N][sites[N]=>a, prime(sites[N])=>b, bond2[N-1]=>j]
		end
	end
	mpo[N] = A
	return mpo
end

"Transform mps into function"
function mps2f(mps::MPS)
	x = Vector{Float64}()
	y = Vector{Float64}()
	B = [ITensor(sites[i]) for i ∈ 1:N]
	s = 2.0
	nor = 0.0
	p = []
	permute = collect(IterTools.product(fill(1:2, N)...))
	for p in permute
		for i ∈ 1:N
			B[i][sites[i]=>p[i]] = 1.0
		end
		push!(x, sum(((p[i] - 1.0) / s^(N - i + 1)) for i ∈ 1:N))
		tmps = prod((mps[i] * B[i]) for i ∈ 1:N)
		for i ∈ 1:N
			B[i][sites[i]=>1] = 0.0
			B[i][sites[i]=>2] = 0.0
		end
		nor += tmps[1] * tmps[1]
		push!(y, tmps[1])
	end
	return x, y, nor
end

"Calculate Frobenius norm error between two vectors"
function ERROR(N::Int, y::Vector{Float64}, z::Vector{Float64})
	nory = 0.0
	norz = 0.0
	err1 = 0.0
	err2 = 0.0
	nory = sum(y[i]^2 for i in 1:2^N)
	norz = sum(z[i]^2 for i in 1:2^N)
	for i ∈ 1:2^N
		err1 = err1 + abs((z[i] / norz^0.5 - y[i] / nory^0.5)^2)
		err2 = err2 + abs((z[i] / norz^0.5 + y[i] / nory^0.5)^2)
	end
	if err1 < err2
		err = err1
	else
		err = err2
	end
	return err
end

"Calculate the each bond dimension and totol elements of MPS"
function BDIM(N::Int, mps::MPS)
	bdim = Vector{Int64}()
	for i in 1:N-1
		push!(bdim, dim(linkind(mps, i)))
	end
	data = Vector{Int64}()
	push!(data, bdim[1] * 2)
	for i in 2:N-2
		push!(data, 2 * (bdim[i-1] * bdim[i]))
	end
	push!(data, 2 * bdim[N-1])
	ele = sum(data)
	return bdim, ele
end

"Read MPS in HDF5 file"
function reading_mps(file::String)
	bond = []
	rmps = h5read(file, "MPS")
	remps = MPS(sites)
	for n ∈ 1:1
		label = 1
		dim1 = rmps["MPS[$n]"]["inds"]["index_1"]["dim"]
		dim2 = rmps["MPS[$n]"]["inds"]["index_2"]["dim"]
		push!(bond, Index(dim2))
		A = ITensor(sites[n], bond[n])
		for b ∈ 1:dim2
			for a ∈ 1:dim1
				A[sites[n]=>a, bond[n]=>b] = rmps["MPS[$n]"]["storage"]["data"][label]
				label += 1
			end
		end
		remps[n] = A
	end

	for n ∈ 2:N-1
		label = 1
		dim1 = rmps["MPS[$n]"]["inds"]["index_1"]["dim"]
		dim2 = rmps["MPS[$n]"]["inds"]["index_2"]["dim"]
		dim3 = rmps["MPS[$n]"]["inds"]["index_3"]["dim"]
		push!(bond, Index(dim3))
		A = ITensor(sites[n], bond[n], bond[n-1])
		for c ∈ 1:dim3
			for b ∈ 1:dim2
				for a ∈ 1:dim1
					A[sites[n]=>b, bond[n]=>c, bond[n-1]=>a] =
						rmps["MPS[$n]"]["storage"]["data"][label]
					label += 1
				end
			end
		end
		remps[n] = A
	end

	for n ∈ N:N
		label = 1
		dim1 = rmps["MPS[$n]"]["inds"]["index_1"]["dim"]
		dim2 = rmps["MPS[$n]"]["inds"]["index_2"]["dim"]
		push!(bond, Index(dim1))
		A = ITensor(sites[n], bond[n-1])
		for b ∈ 1:dim2
			for a ∈ 1:dim1
				A[sites[n]=>b, bond[n-1]=>a] = rmps["MPS[$n]"]["storage"]["data"][label]
				label += 1
			end
		end
		remps[n] = A
	end
	IRMPS = noprime(remps)
	@show IRMPS
	return IRMPS
end

function reading_mps2(file::String)
	bond = []
	rmps = h5read(file, "MPS")
	remps = MPS(sites)
	for n ∈ 1:1
		label = 1
		dim1 = rmps["MPS[$n]"]["inds"]["index_1"]["dim"]
		dim2 = rmps["MPS[$n]"]["inds"]["index_2"]["dim"]
		push!(bond, Index(dim1))
		A = ITensor(sites[n], bond[n])
		for b ∈ 1:dim2
			for a ∈ 1:dim1
				A[sites[n]=>b, bond[n]=>a] = rmps["MPS[$n]"]["storage"]["data"][label]
				label += 1
			end
		end
		remps[n] = A
	end

	for n ∈ 2:2
		label = 1
		dim1 = rmps["MPS[$n]"]["inds"]["index_1"]["dim"]
		dim2 = rmps["MPS[$n]"]["inds"]["index_2"]["dim"]
		dim3 = rmps["MPS[$n]"]["inds"]["index_3"]["dim"]
		push!(bond, Index(dim1))
		A = ITensor(sites[n], bond[n], bond[n-1])
		for c ∈ 1:dim3
			for b ∈ 1:dim2
				for a ∈ 1:dim1
					A[sites[n]=>b, bond[n]=>a, bond[n-1]=>c] =
						rmps["MPS[$n]"]["storage"]["data"][label]
					label += 1
				end
			end
		end
		remps[n] = A
	end


	for n ∈ 3:N-1
		label = 1
		dim1 = rmps["MPS[$n]"]["inds"]["index_1"]["dim"]
		dim2 = rmps["MPS[$n]"]["inds"]["index_2"]["dim"]
		dim3 = rmps["MPS[$n]"]["inds"]["index_3"]["dim"]
		push!(bond, Index(dim2))
		A = ITensor(sites[n], bond[n], bond[n-1])
		for c ∈ 1:dim3
			for b ∈ 1:dim2
				for a ∈ 1:dim1
					A[sites[n]=>a, bond[n]=>b, bond[n-1]=>c] =
						rmps["MPS[$n]"]["storage"]["data"][label]
					label += 1
				end
			end
		end
		remps[n] = A
	end

	for n ∈ N:N
		label = 1
		dim1 = rmps["MPS[$n]"]["inds"]["index_1"]["dim"]
		dim2 = rmps["MPS[$n]"]["inds"]["index_2"]["dim"]
		A = ITensor(sites[n], bond[n-1])
		for b ∈ 1:dim2
			for a ∈ 1:dim1
				A[sites[n]=>a, bond[n-1]=>b] = rmps["MPS[$n]"]["storage"]["data"][label]
				label += 1
			end
		end
		remps[n] = A
	end
	IRMPS = noprime(remps)
	@show IRMPS
	return IRMPS
end


"Transform MPS into MPO, which means the operator of diagonal matrix"
function mps2mpo(IRMPS::MPS)
	IRMPO = MPO(sites)
	bonds = []
	push!(bonds, linkind(IRMPS, 1))
	A = ITensor(sites[1], prime(sites[1]), bonds[1])
	for i in 1:2
		for j in 1:dim(linkind(IRMPS, 1))
			A[sites[1]=>i, prime(sites[1])=>i, bonds[1]=>j] = IRMPS[1][sites[1]=>i, linkind(IRMPS, 1)=>j]
		end
	end
	IRMPO[1] = A

	for n in 2:N-1
		push!(bonds, linkind(IRMPS, n))
		A = ITensor(sites[n], prime(sites[n]), bonds[n-1], bonds[n])
		for i in 1:2
			for j in 1:dim(linkind(IRMPS, n - 1))
				for k in 1:dim(linkind(IRMPS, n))
					A[sites[n]=>i, prime(sites[n])=>i, bonds[n-1]=>j, bonds[n]=>k] = IRMPS[n][sites[n]=>i, linkind(IRMPS, n - 1)=>j, linkind(IRMPS, n)=>k]
				end
			end
		end
		IRMPO[n] = A
	end

	A = ITensor(sites[N], prime(sites[N]), bonds[N-1])
	for i in 1:2
		for j in 1:dim(linkind(IRMPS, N - 1))
			A[sites[N]=>i, prime(sites[N])=>i, bonds[N-1]=>j] = IRMPS[N][sites[N]=>i, linkind(IRMPS, N - 1)=>j]
		end
	end
	IRMPO[N] = A
	return IRMPO
end


"use dmrg() to solve the first n states

z is the vector you want calculate the Frobenius norm error

m is the number of sweep interval you want to calculate the error(unfinish and remain 1 now)"
function muti_rdmrg(mpo::MPO, mps0::MPS, sweep::Int, kdim::Int, maxbdim::Int, threshold::Float64, z::Matrix{Float64}, n::Int, m::Int)
	err = Matrix{Float64}(undef, n, sweep + 1) #Frobenius norm error for each state
	bd = Array{Int}(undef, n, sweep + 1, N - 1) #each bond dimension for each state
	e = Matrix{Float64}(undef, n, sweep + 1) #energy for each state
	ele = Vector{Int}(undef, n) #total elements in MPS for each state
	f = Matrix{Float64}(undef, n, 2^N) #function for each state
	tmps = Matrix{Any}(undef, n, sweep + 1) #MPS for each state
	
	x, y = mps2f(mps0)
	energy = inner(prime(mps0), mpo * mps0)
	for i in 1:n
		err[i, 1] = ERROR(N, y, z[1, :])
		bd[i, 1, :], el = BDIM(N, mps0)
		e[i, 1] = energy
		tmps[i,1] = mps0
	end

	mps=mps0
	for j in 1:sweep
		energy, mps = dmrg(
			mpo,
			mps,
			cutoff = threshold,
			maxdim = maxbdim,
			mindim = 2,
			noise = 10.0^(-10),
			nsweeps = m,
			outputlevel = 2,
			eigsolve_krylovdim = kdim,
		)
		x, f[1, :] = mps2f(mps)
		energy = inner(prime(mps), mpo * mps)
		err[1, j+1] = ERROR(N, f[1, :], z[1, :])
		bd[1, j+1, :], ele[1] = BDIM(N, mps)
		e[1, j+1] = energy
		tmps[1,j+1] = mps
	end
	#push!(tmps, mps)

	
	for i ∈ 1:n-1
		mps=mps0
		for j ∈ 1:sweep
			energy, mps = dmrg(
				mpo,
				[tmps[k,sweep+1] for k in 1:i],
				mps,
				cutoff = threshold,
				maxdim = maxbdim,
				mindim = 2,
				noise = 10.0^(-10),
				nsweeps = m,
				outputlevel = 2,
				eigsolve_krylovdim = kdim,
			)
			x, f[i+1, :] = mps2f(mps)
			energy = inner(prime(mps), mpo * mps)
			err[i+1, j+1] = ERROR(N, f[i+1, :], z[i+1, :])
			bd[i+1, j+1, :], ele[i+1] = BDIM(N, mps)
			e[i+1, j+1] = energy
			tmps[i+1,j+1] = mps
		end
		#push!(tmps, mps)
	end
	return e, tmps, err, bd, ele, f
end

#same to muti_rdmrg() but without calculating the error
function muti_rdmrg(mpo::MPO, mps0::MPS, sweep::Int, kdim::Int, maxbdim::Int, threshold::Float64, z::Nothing, n::Int, m::Int)
	bd = Array{Int}(undef, n, sweep + 1, N - 1)
	e = Matrix{Float64}(undef, n, sweep + 1)
	ele = Vector{Int}(undef, n)
	f = Matrix{Float64}(undef, n, 2^N)
	tmps = Matrix{Any}(undef, n, sweep + 1) #MPS for each state

	
	energy = inner(prime(mps0), mpo * mps0)
	for i in 1:n
		bd[i, 1, :], el = BDIM(N, mps0)
		e[i, 1] = energy
		tmps[i,1] = mps0
	end
	

	mps=mps0
	for j in 1:sweep
		energy, mps = dmrg(
			mpo,
			mps,
			cutoff = threshold,
			maxdim = maxbdim,
			mindim = 2,
			noise = 10.0^(-10),
			nsweeps = m,
			outputlevel = 2,
			eigsolve_krylovdim = kdim,
		)
		bd[1, j+1, :], ele[1] = BDIM(N, mps)
		e[1, j+1] = energy
		tmps[1,j+1] = mps
	end
	x, f[1,:] = mps2f(mps)
	#push!(tmps, mps)
	
	for i ∈ 1:n-1
		mps=mps0
		for j ∈ 1:sweep
			energy, mps = dmrg(
				mpo,
				[tmps[k,sweep+1] for k in 1:i],
				mps,
				cutoff = threshold,
				maxdim = maxbdim,
				mindim = 2,
				noise = 10.0^(-10),
				nsweeps = m,
				outputlevel = 2,
				eigsolve_krylovdim = kdim,
			)
			
			bd[i+1, j+1, :], ele[i+1] = BDIM(N, mps)
			e[i+1, j+1] = energy
			tmps[i+1,j+1] = mps
		end
		x, f[i+1, :] = mps2f(mps)
		#push!(tmps, mps)
	end
	return e, tmps, bd, ele, f
end