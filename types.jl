module MyTypes
	
	type Stuff{T}
		pairs::Array{Array{T,2},1}
		single::Array{Array{T,1},1}
	end

	typealias Stats{T} Stuff{T}
	typealias Feats{T} Stuff{Array{T,1}}
	typealias Features{T} Array{Feats{T},1}
		
	type Messages{T}
		forward::Array{Array{T,1},1}
		backward::Array{Array{T,1},1}
	end

	type Storage{T}
		messages::Messages{T}
		μ::Stats{T}
		ψ::Stats{T}
		θ::Stats{T}
		temp::Array{T,1}
	end

end