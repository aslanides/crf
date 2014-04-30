#####################
# Training function
#####################
function train{T}(n=200,::Type{T}=Float64)

	features,labels = prepare_data(n,T)
	D = sum(map(fetch,{@spawnat p sum([length(localpart(features)[i]) for i=1:length(localpart(features))]) for p in features.pmap}))
	println(D)
	func = MyTypes.hashed_function(x->p_total_gradient(x,features,labels))
	f(x) = func(x)[1]/D
	function g!(x::Vector,storage::Vector)
		grad = func(x)[2]
		storage[:] = grad
		storage ./= D
	end

	w_init = zeros(T,96)
	result = optimize(f,g!,w_init,method=:l_bfgs)
	return result.minimum,result.f_minimum
end

###################
# Main computation
###################
function p_total_gradient{T}(weights::Vector{T},feats::DArray,labs::DArray,λ=0.)
	M = length(weights)
	println(weights[1:20])
	gradient = zeros(T,M)
	likelihood = zero(T)
	rofl = map(fetch,{@spawnat p total_gradient(weights,localpart(feats),localpart(labs)) for p in feats.pmap})

	for i=1:length(rofl)
		likelihood += rofl[i][1]
		gradient += rofl[i][2]
	end
	likelihood -= 0.5λ * norm(weights)^2
	gradient -= λ * norm(weights)
	return (-1*likelihood, -1.*gradient)
end

function total_gradient{T}(weights::Vector{T},feats::Vector{MyTypes.Features{T}},labs::Array{Array{Array{Int32,1},1},1})

	M = length(weights)
	gradient = zeros(T,M)
	likelihood = zero(T)

	for img=1:length(feats)
		a,b = single_gradient(weights,feats[img],labs[img])
		likelihood += a
		@devec gradient += b

	end

	if isnan(likelihood) || isinf(likelihood)
		warn("likelihood is NaN/Inf!")
	end

	return likelihood, gradient
end

function single_gradient{T}(weights::Array{T,1},feat::MyTypes.Features{T},lab::Array{Array{Int32,1},1})
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
