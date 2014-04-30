#####################
# Training function
#####################
function train{T}(n=200;max_iter=200,λ=0.,ε=1e-3,typeske::Type{T}=Float64)

	function CG_gradient(g,weights::Vector)
		l, tmp = p_total_gradient(weights,features,labels,λ)
		
		if !(g === nothing)
			gptr = pointer(g)
			g[:] = tmp
			if pointer(g) != gptr
				error("bad gradient pointer")
			end
		end
		return l
	end

	initial_w = zeros(T,96)
	ops = @options display=Optim.ITER fcountmax=max_iter tol=ε

	println("Training on univariate terms...")
	features,labels = prepare_data(n,T,univ=true)
	@time new_w, lval, cnt, conv = cgdescent(CG_gradient,initial_w,ops) # first train on univariate terms only

	@everywhere (features = 0.; labels = 0.)
	println("Collecting trash...")
	@everywhere gc()

	println("Training full model...")
	features,labels = prepare_data(n,T,univ=false)
	@time w, lval, cnt, conv = cgdescent(CG_gradient,new_w,ops)	

	D = sum(map(fetch,{@spawnat p sum([length(localpart(features)[i]) for i=1:length(localpart(features))]) for p in feats.pmap}))
	println("Final likelihood (averaged over training set): ",lval/D)
	return w
end
###################
# Main computation
###################
function p_total_gradient{T}(weights::Vector{T},feats::DArray,labs::DArray,λ=0.)
	M = length(weights)
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
	μ_model = zeros(T,M)
	μ_emp = zeros(T,M)

	for img=1:length(feats)
		height = length(feats[img])
		width = length(labs[img][1])
		sto = init_storage(T,width)

		for row=1:height
			fill!(μ_model,zero(T))
			fill!(μ_emp,zero(T))

			big_dot!(weights,feats[img][row],sto.θ)
			big_exp!(sto.θ,sto.ψ)
			getMessages!(sto)
			getMarginals!(sto)
			big_dot!(sto.μ,feats[img][row],μ_model)
			empiricals!(feats[img][row],lab[row],μ_emp)
			likelihood += dot(weights,μ_emp) - logPartition(sto.θ,sto.μ)
			@devec gradient += μ_emp - μ_model
		end
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
