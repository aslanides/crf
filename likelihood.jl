#####################
# Training function
#####################
function train{T}(::Type{T}=Float64,n=1,max_iter=100,λ=0.)

	function CG_gradient(g,weights::Vector)
		l, tmp,jnk1,jnk2 = get_gradient(weights,features,labels,λ)
		
		if !(g === nothing)
			gptr = pointer(g)
			g[:] = tmp
			if pointer(g) != gptr
				error("bad gradient pointer")
			end
		end	
		return l
	end

	features,labels = prepare_data(T,"data/horses_train.mat",n)
	w = rand(T,96)
	ops = @options display=Optim.ITER fcountmax=max_iter tol=1e-2
	@time w, lval, cnt, conv = cgdescent(CG_gradient,w,ops)

	predictions = [predict(w,features[i]) for i=1:n]
	truth = [rows_to_array(labels[i]) for i=1:n]
	img_to_csv(predictions,truth)

	return w
end
###################
# Main computation
###################
function get_gradient{T}(weights::Array{T,1},feats::Array{MyTypes.Features{T},1},labs,λ=0.)
	M = length(weights) 
	gradient = zeros(T,M)
	μ_model = zeros(T,M)
	μ_emp = zeros(T,M)
	likelihood = zero(T)
	
	t = zeros(T,8)
	mem = zeros(Int64,8)
	
	for img=1:length(feats)
		height = length(feats[img])
		width = length(labs[img][1])

		a,b = @mytime (sto = init_storage(T,width))
		t[1] += a; mem[1] += b

		for row=1:height
			fill!(μ_model,zero(T))
			fill!(μ_emp,zero(T))

			a,b = @mytime big_dot!(weights,feats[img][row],sto.θ)
			t[2] += a; mem[2] += b
			
			a,b = @mytime big_exp!(sto.θ,sto.ψ)
			t[3] += a; mem[3] += b
			
			a,b = @mytime getMessages!(sto)
			t[4] += a; mem[4] += b
			
			a,b = @mytime getMarginals!(sto)
			t[5] += a; mem[5] += b
			
			a,b = @mytime big_dot!(sto.μ,feats[img][row],μ_model)
			t[6] += a; mem[6] += b
			
			a,b = @mytime empiricals!(feats[img][row],labs[img][row],μ_emp)
			t[7] += a; mem[7] += b
			
			a,b = @mytime (A = logPartition(sto.θ,sto.μ))
			t[8] += a; mem[8] += b
			
			likelihood += dot(weights,μ_emp) - A
			@devec gradient += μ_emp - μ_model
		end
	end

	D = sum([length(feats[img]) for img=1:length(feats)])
	likelihood /= D
	likelihood -= 0.5λ * norm(weights)^2
	gradient -= λ * norm(weights)

	return (-1.*likelihood,-1.*gradient,t,mem)
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