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
		getMarginals!(sto)
		p[row] = Array(T,width)
		for col=1:width
			p[row][col] = sto.μ.single[col][2]
		end
	end
	return rows_to_array(p)
end



println("Writing predictions...")
predictions = map(fetch,{@spawnat p [predict(w,localpart(features)[i]) for i=1:length(localpart(features))] for p in features.pmap})
truth = map(fetch,{@spawnat p [rows_to_array(localpart(labels)[i]) for i=1:length(localpart(labels))] for p in features.pmap})