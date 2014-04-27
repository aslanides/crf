#############################
# Gradient checks
#############################
function finiteGrad(func,x;epsilon=1e-5)
	tmp = x

	grad = similar(x)
	for i=1:length(x)
		tmp[i] += epsilon
		f1 = func(tmp)[1]
		tmp[i] -= 2*epsilon
		f2 = func(tmp)[1]
		tmp[i] += epsilon
		grad[i] = (f1 - f2)/(2 * epsilon)
	end
	return grad
end

function gradCheck(func,x)
	finite = finiteGrad(func,x)
	return norm(func(x)[2] - finite,Inf)
end
#################################
# Functions to initialize storage
#################################
function init_stats{T}(::Type{T},N,K)
	s = MyTypes.Stuff(Array(Array{T,2},N-1),Array(Array{T,1},N))
	for i=1:N-1
		s.pairs[i] = zeros(T,K,K)
	end
	for i=1:N
		s.single[i] = zeros(T,K)
	end
	return s
end

function init_feats{T}(::Type{T},N)
	feat_s = Array(Array{Array{T,1},1},N)
	feat_p = Array(Array{Array{T,1},2},N-1)
	F = MyTypes.Stuff(feat_p,feat_s)

	for node=1:N-1
		F.pairs[node] = Array(Array{T,1},2,2)
	end
	for node=1:N
		F.single[node] = Array(Array{T,1},2)
	end
	return F
end

function init_messages{T}(::Type{T},N)
	m = MyTypes.Messages(Array(Array{T,1},N),Array(Array{T,1},N))
	return m
end

function init_storage{T}(::Type{T},N)
	s = MyTypes.Storage(init_messages(T,N),init_stats(T,N,2),init_stats(T,N,2),init_stats(T,N,2))
	for i=1:N
		s.messages.forward[i] = ones(T,2)
		s.messages.backward[i] = ones(T,2)
	end
	return s
end

#############################
# Rasterization/undo functions
#############################
function array_to_rows{T}(::Type{T},array::Array{T,2})
	height,width = size(array)
	rows = Array(Array{T,1},height)
	for i=1:height
		rows[i] = Array(T,width)
		for j=1:width
			rows[i][j] = array[i,j]
		end
	end
	return rows::Array{Array{T,1},1}
end

function rows_to_array{T}(rows::Array{Array{T,1},1})
	height = length(rows)
	width = length(rows[1])
	array = Array(T,height,width)
	for i=1:height
		array[i,:] = rows[i]
	end
	return array
end

#######################################################
# Functions for operations on stats data structures
#######################################################

function big_exp!(B::MyTypes.Stats,S::MyTypes.Stats)
	N = length(B.single)
	K = 2
	for i=1:N-1
		for y_m=1:K
			for y_n=1:K
				S.pairs[i][y_m,y_n] = exp(B.pairs[i][y_m,y_n])
			end
		end
	end

	for i=1:N
		for y=1:K
			S.single[i][y] = exp(B.single[i][y])
		end
	end
	return S
end

function big_dot!{T}(A::MyTypes.Stats{T},B::MyTypes.Stats{T})
	N = length(A.single)
	K = 2
	t = zero(T)
	for i=1:N-1
		for y_m=1:K
			for y_n=1:K
				t += A.pairs[i][y_m,y_n] * B.pairs[i][y_m,y_n]
			end
		end
	end

	for i=1:N
		for y=1:K
			t += A.single[i][y] * B.single[i][y]
		end
	end
	return t
end

function big_dot!{T}(weights::Vector{T},A::MyTypes.Feats{T},B::MyTypes.Stats{T})
	N = length(A.single)
	M1 = length(A.single[1][1])
	M2 = length(A.pairs[1][1,1])
	M = length(weights)
	K = 2
	for i=1:N-1
		for y_m=1:K
			for y_n=1:K
				B.pairs[i][y_m,y_n] = zero(T)
				for j=1:M2
					B.pairs[i][y_m,y_n] += weights[j] * A.pairs[i][y_m,y_n][j]
				end
			end
		end
	end

	for i=1:N
		for y=1:K
			B.single[i][y] = zero(T)
			for j=1:M1
				B.single[i][y] += weights[M2+j] * A.single[i][y][j]
			end
		end
	end
end

function big_dot!{T}(A::MyTypes.Stats{T},B::MyTypes.Feats{T},S::Vector{T}) 
	N = length(A.single)
	M1 = length(B.single[1][1])
	M2 = length(B.pairs[1][1,1])
	M = length(S)
	K = 2
	for i=1:N-1
		for y_m=1:K
			for y_n=1:K
				for k=1:M2
					S[k] += A.pairs[i][y_m,y_n] * B.pairs[i][y_m,y_n][k]
				end
			end
		end
	end
	for i=1:N
		for y=1:K
			for k=1:M1
				S[M2+k] += A.single[i][y] * B.single[i][y][k]
			end
		end
	end
	return S
end
#############################
# Pack/unpack stats<-->array
#############################
function pack_stats{T}(stats::MyTypes.Stats{T})
	N = length(stats.single)
	arr_single = Array(T,N*2)
	arr_pairs =  Array(T,(N-1)*4)
	where = 1
	for i=1:N-1
		arr_pairs[where:where+3] = stats.pairs[i][:]
		where += 4
	end
	where = 1
	for i=1:N
		arr_single[where:where+1] = stats.single[i][:]
		where += 2
	end
	return vcat(arr_pairs,arr_single)
end

function unpack_stats{T}(vec::Vector{T})
	N = int((length(vec) + 4)/6)
	
	stats = init_stats(T,N,2)
	where = 1
	for i=1:N-1
		stats.pairs[i][:] = vec[where:where+3]
		where+=4
	end

	for i=1:N
		stats.single[i][:] = vec[where:where+1]
		where += 2
	end
	return stats
end
#############################
# Brute force marginals
#############################
function bruteMarginals{T}(theta::MyTypes.Stats{T}) # brute force marginalisation

	function log_sum_exp(x)
		c = maximum(x)
		return log(sum(exp(x-c)))+c
	end

	function make_table(L)
		x = [1:L[1]]'
		for i=2:length(L)
			y = repmat([1:L[i]]',size(x,2),1)
			y = y[:]'
			x = repmat(x,1,L[i])
			x = [x; y];
		end
		return x
	end

	function compute_inds(x)
		N = length(x)
		f = init_stats(T,length(x),2)
		for i=1:N-1
			f.pairs[i][x[i],x[i+1]]=1
		end
		for i=1:N
			f.single[i][x[i]] = 1
		end
		return f

	end

	N = length(theta.single)
	x = make_table([2 for i=1:N])
	mu = init_stats(T,N,2)
	E = zeros(T,1,size(x,2))
	
	for i=1:size(x,2)
		c = x[:,i]
		for j=1:N-1
			E[i] += theta.pairs[j][c[j],c[j+1]] 
		end
		for j=1:N
			E[i] += theta.single[j][c[j]] 
		end
	end

	A = log_sum_exp(E)

	P = exp(E-A)
	for i=1:size(x,2)
		c = x[:,i]
		f = compute_inds(c)
		for j=1:N-1
			mu.pairs[j] += f.pairs[j] * P[i]
		end
		for j=1:N
			mu.single[j] += f.single[j] *P[i]
		end
	end
	return mu.pairs, mu.single, A
end

#############################
# Functions for writing images to file
#############################
function dir_today()
	d = string(now(AEST))
	dirname = string(d[1:4],d[6:7],d[9:10],d[12:13],d[15:16],d[18:19])
	mkdir(dirname)
	cd(dirname)
end

function home()
	cd(string(homedir(),"/Git/crf/"))
	include("horses.jl")
end

function img_to_csv(predictions,truths)
	cd("output")
	dir_today()
	make_csv(predictions,"pred")
	make_csv(truths,"truth")
	println("Successfully written to ",pwd())
	cd("../..")
end

function make_csv(images,name::String)
	D = length(images)
	for i=1:D
		fname = string(name,i,".csv")
		file = open(fname,"w")
		tmp = images[i]

		writedlm(file,tmp)
		close(file)
	end
end
#############################
# Timing/profiling
#############################
macro mytime(ex)
    quote
        local b0 = Base.gc_bytes()
        local t0 = time_ns()
        local val = $(esc(ex))
        local t1 = time_ns()
        local b1 = Base.gc_bytes()
        (t1-t0)/1e9, b1-b0
    end
end