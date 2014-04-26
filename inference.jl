##################
# Message passing
##################
function getMessages!(sto)
	
	psi = sto.ψ.pairs 
	phi = sto.ψ.single
	α = sto.messages.forward
	β = sto.messages.backward
	N = length(α)
	
	for i=2:N
		@devec begin
			α[i] = (phi[i-1] .*  α[i-1])
			β[N-i+1] = phi[N-i+2] .* β[N-i+2]
		end

		α[i] = psi[i-1]' * α[i] # can't devec matrix computations
		β[N-i+1] = psi[N-i+1] * β[N-i+1]
		
		@devec begin
			tmp = sum(α[i])
			α[i] ./= tmp
			tmp = sum(β[N-i+1])
			β[N-i+1] ./= tmp
		end
	end
end
##################
# Marginalization
##################
function getMarginals!(sto)
	
	psi = sto.ψ.pairs 
	phi = sto.ψ.single 
	α = sto.messages.forward
	β = sto.messages.backward
	π = sto.μ.pairs
	σ = sto.μ.single
	N = length(α)
	u = sto.temp
	
	for n=1:N-1
		@devec begin
			u = phi[n+1] .* β[n+1]
			α[n] .*= phi[n]
		end
			v = psi[n]
			tmp1 = α[n][1]
			tmp2 = α[n][2]
		@devec begin 
			v[1,:] .*= u .* tmp1
			v[2,:] .*= u .* tmp2
			π[n] = v

			tmp = sum(π[n])
			π[n] ./= tmp
		end
	end
	@devec α[N] .*= phi[N]
	for n=1:N
		@devec begin
			σ[n] = α[n] .* β[n]
			tmp = sum(σ[n])
			σ[n] ./= tmp
		end
	end
	return sto.μ
end
