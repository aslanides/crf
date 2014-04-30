module MyTypes
	
export Stuff,Stats,Feats,Features,Messages,Storage,hashed_function

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
end
###########################
# Justin's caching function
###########################
type hashed_function
	fun::Function
	x  # inputs
	f  # outputs
	age
	hashed_function(fun::Function,N::Int=2) = (a = new(); a.fun=fun; a.x = Array(Any,N); a.f = Array(Any,N); a.age = Inf+zeros(Int,N); g(x) = cache_ref(a,x); g)
end

function cache_ref(h::MyTypes.hashed_function,x)
    N = length(h.f)
    # if we can find the value in the hash, return it
    for n=1:N
        if h.age[n]!=Inf && isequal(x,h.x[n])
            h.age += 1
            h.age[n] = 0
            return h.f[n]
        end
    end
    # otherwise kill off the oldest value and replace it
    damax = maximum(h.age)
    for n=1:N
        if h.age[n]==damax
            h.f[n] = h.fun(x)
            h.x[n] = x
            h.age += 1
            h.age[n] = 0
            return h.f[n]
        end
    end
end

end