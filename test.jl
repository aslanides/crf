##############
# Test suite
##############
function test{T}(::Type{T}=Float64,ε=1e-8)

	function make_test_data(n=1)
		width = 15
		height = 20
		F = [rand(T,height,width,42) for i=1:n]
		G = [rand(T,height,width-1,3) for i=1:n]

		features = [represent_features(T,F[i],G[i]) for i=1:n]
		labels = [array_to_rows(int(rand(T,height,width)).+1) for i=1:n]
		return features, labels
	end

	function shizen!(feat,sto)
		fill!(μ_model,0.)
		fill!(μ_emp,0.)
		big_exp!(sto.θ,sto.ψ)
		getMessages!(sto)
		getMarginals!(sto)
		big_dot!(sto.μ,feat,μ_model)
	end

	function ∂A∂θ{T}(θ::Vector{T},feat)
		sto = init_storage(T,width)
		sto.θ = unpack_stats(θ)
		shizen!(feat,sto)
		return logPartition(sto.θ,sto.μ), pack_stats(sto.μ)
	end

	function ∂A∂w{T}(weights::Vector{T},feat)
		sto = init_storage(T,width)
		big_dot!(weights,feat,sto.θ)
		shizen!(feat,sto)
		return logPartition(sto.θ,sto.μ), μ_model
	end

	function ∂Λ∂w{T}(weights::Vector{T},feat,lab)
		sto = init_storage(T,width)
		big_dot!(weights,features[1][1],sto.θ)
		shizen!(feat,sto)
		empiricals!(feat,lab,μ_emp)
		return dot(weights,μ_emp) - logPartition(sto.θ,sto.μ), μ_emp - μ_model
	end

	features,labels = make_test_data()
	
	M = 96
	gradient = zeros(T,M)
	μ_model = zeros(T,M)
	μ_emp = zeros(T,M)
	likelihood = 0.
	w = rand(T,M)
	height = length(features[1])
	width = length(labels[1][1])

	sto = init_storage(T,width)
	big_dot!(w,features[1][1],sto.θ)
	big_exp!(sto.θ,sto.ψ)
	getMessages!(sto)
	getMarginals!(sto)
	
	μ_brute = init_stats(T,width,2); (μ_brute.pairs, μ_brute.single, bruteA) = bruteMarginals(sto.θ) # brute force marginals

	println("Marginalisation (pairs): ",maximum(map(x->norm(x,Inf),sto.μ.pairs - μ_brute.pairs)) < ε)
	println("Marginalisation (single): ",maximum(map(x->norm(x,Inf),sto.μ.single - μ_brute.single)) < ε)
	println("Log-partition: ",logPartition(sto.θ,sto.μ) - bruteA < ε)
	println("∂A/∂θ: ",gradCheck(x->∂A∂θ(x,features[1][1]),pack_stats(sto.θ)) < ε)
	println("∂A/∂w: ",gradCheck(x->∂A∂w(x,features[1][1]),w) < ε)
	println("∂Λ/∂w: ",gradCheck(x->∂Λ∂w(x,features[1][1],labels[1][1]),w) < ε)
end
##############
# Benchmark
##############
function bench_gradient{T}(::Type{T}=Float64;lst::Array{Int,1}=[10])
	srand(1)
	features,labels = prepare_data(T,"/Users/john/Dropbox/Machine\ Learning/Tutorials/Horses/data/horses_train.mat",maximum(lst))
	w = rand(T,96)
	ks = ["storage","mrf_params","potentials","messages","marginals","μ_model","μ_emp","log_partition"]
	times = zeros(T,length(ks),length(lst))
	memory = zeros(T,length(ks),length(lst))
	tic()
	for i=1:length(lst)
		println("Benchmarking on ",lst[i]," images (max ",maximum(lst),")...")
		(junk1,junk2,times[:,i],memory[:,i]) = get_gradient(w,features[1:lst[i]],labels[1:lst[i]])
	end
	toc()
	cd("benchmarks")
	dir_today()
	open("timing.csv","a") do lel
		writecsv(lel,lst[:])
		writecsv(lel,times)
	end
	open("memory.csv","a") do lel
		writecsv(lel,lst)
		writecsv(lel,memory)
	end
	writedlm("names.csv",ks,',')
	println("Written output to",pwd())
	cd("../..")

end
