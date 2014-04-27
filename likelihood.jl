#####################
# Training function
#####################
function train{T}(::Type{T}=Float64,n=1,max_iter=100,λ=0.)

	function CG_gradient(g,weights::Vector)
		l, tmp = p_total_gradient(weights,features,labels)
		
		if !(g === nothing)
			gptr = pointer(g)
			g[:] = tmp
			if pointer(g) != gptr
				error("bad gradient pointer")
			end
		end
		return l
	end

	features,labels = p_prepare_data(T,"data/horses_train.mat",n)
	w = zeros(T,96)
	ops = @options display=Optim.ITER fcountmax=max_iter tol=1e-4
	@time w, lval, cnt, conv = cgdescent(CG_gradient,w,ops)

	predictions = [predict(w,features[i]) for i=1:n]
	truth = [rows_to_array(labels[i]) for i=1:n]
	img_to_csv(predictions,truth)

	return w
end
###################
# Main computation
###################
function p_total_gradient{T}(weights::Vector{T},feats::DArray,labs::DArray)
	#D = sum([length(feats[img]) for img=1:length(feats)])
	M = length(weights)
	gradient = zeros(T,M)
	likelihood = zero(T)
	rofl = map(fetch,{@spawnat p total_gradient(weights,localpart(feats),localpart(labs)) for p in feats.pmap})
	
	for i=1:length(rofl)
		likelihood += rofl[i][1]
		gradient += rofl[i][2]
	end

	# likelihood /= D
	# likelihood -= 0.5λ * norm(weights)^2
	# gradient /= D
	# gradient -= λ * norm(weights)
	return (-1*likelihood, -1.*gradient)
end

function total_gradient{T}(weights::Vector{T},feats::Vector{MyTypes.Features{T}},labs::Array{Array{Array{Int64,1},1},1},λ=0.)
	
	M = length(weights)
	gradient = zeros(T,M)
	likelihood = zero(T)

	for img=1:length(feats)
		a,b = single_gradient(weights,feats[img],labs[img])
		likelihood += a
		@devec gradient += b
	end

	if is(likelihood,NaN) || is(likelihood,-Inf) || is(likelihood,Inf)
		error("Bad likelihood!")
	end

	return likelihood, gradient
end

function single_gradient{T}(weights::Array{T,1},feat::MyTypes.Features{T},lab::Array{Array{Int64,1},1})
	
	height = length(feat)
	width = length(lab[1])

	M = length(weights)
	gradient = zeros(T,M)
	likelihood = zero(T)
	μ_model = zeros(T,M)
	μ_emp = zeros(T,M)
	

	sto = init_storage(T,width)

	for row=1:height
		fill!(μ_model,zero(T))
		fill!(μ_emp,zero(T))

		big_dot!(weights,feat[row],sto.θ)
		big_exp!(sto.θ,sto.ψ)
		getMessages!(sto)
		getMarginals!(sto)
		big_dot!(sto.μ,feat[row],μ_model)
		empiricals!(feat[row],lab[row],μ_emp)
		likelihood += dot(weights,μ_emp) - logPartition(sto.θ,sto.μ)
		@devec gradient += μ_emp - μ_model
	end
	return likelihood, gradient
end

####################
# Compute empiricals
####################
function empiricals!(features,labels,μ_emp::Vector)
	N = length(features.single)
	M = length(μ_emp)
	M1 = length(features.single[1][1])
	M2 = length(features.pairs[1][1,1])
	for node=1:N-1
		for i=1:M2
			μ_emp[i] += features.pairs[node][labels[node],labels[node+1]][i]
		end
	end

	for node=1:N
		for i=1:M1
			μ_emp[M2+i] += features.single[node][labels[node]][i]
		end
	end
	return μ_emp
end
###########
# Entropy
###########
function entropy{T}(μ::MyTypes.Stats{T})
	π = μ.pairs
	σ = μ.single
	N = length(σ)
	H = zero(T)
	I = zero(T)
	
	for i=1:N
		for j=1:2
			H += -one(T) * σ[i][j] * log(σ[i][j])
		end
	end
	
	for i=1:N-1
		for j=1:2
			for k=1:2
				I += π[i][j,k] * log(π[i][j,k] / (σ[i][j] * σ[i+1][k]))
			end
		end
	end

	return H - I
end
###############
# Log-partition
###############
logPartition(θ,μ) = big_dot!(θ,μ) + entropy(μ)
##############
# Predictions
##############
function predict{T}(weights::Array{T,1},feat)
	height = length(feat)
	width = length(feat[1].single)

	sto = init_storage(T,width)

	p = Array(Array{T,1},height)

	for row=1:height
		big_dot!(weights,feat[row],sto.θ)
		big_exp!(sto.θ,sto.ψ)
		getMessages!(sto)
		μ = getMarginals!(sto)
		p[row] = Array(T,width)
		for col=1:width
			p[row][col] = μ.single[col][2]
		end
	end
	return rows_to_array(p)
end