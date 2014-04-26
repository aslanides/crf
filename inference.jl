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
		
		for j=1:2 α[i][j] = phi[i-1][j] *  α[i-1][j]; end
		for j=1:2 β[N-i+1][j] = phi[N-i+2][j] * β[N-i+2][j]; end
		
		#@time α[i] = psi[i-1]' * α[i]
		tmp = α[i][1]
		α[i][1] = α[i][1] * psi[i-1][1,1] + α[i][2] * psi[i-1][2,1]
		α[i][2] = tmp * psi[i-1][1,2] + α[i][2] * psi[i-1][2,2]

		tmp = β[N-i+1][1]
		β[N-i+1][1] = β[N-i+1][1] * psi[N-i+1][1,1] + β[N-i+1][2] * psi[N-i+1][1,2]
		β[N-i+1][2] = tmp * psi[N-i+1][2,1] + β[N-i+1][2] * psi[N-i+1][2,2]
					
		tmp = sum(α[i])
		for j=1:2 α[i][j] /= tmp; end
		tmp = sum(β[N-i+1])
		for j=1:2 β[N-i+1][j] /= tmp; end	
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
		
	for n=1:N-1
		
		#α[n] .*= phi[n] # ~2e-6 seconds, 752 bytes
		#@devec α[n] .*= phi[n] # 4e-7 seconds, 112 bytes
		for i=1:2 α[n][i] .*= phi[n][i]; end # ~2.5e-7 seconds, 0 bytes
		
		v = psi[n]
		tmp1 = α[n][1]
		tmp2 = α[n][2]
	
		@devec v[1,:] .*= phi[n+1] .* β[n+1] .* tmp1
		@devec v[2,:] .*= phi[n+1] .* β[n+1] .* tmp2
		π[n] = v

		tmp = sum(π[n])
		for i=1:4 π[n][i] /= tmp; end		
	end
	for i=1:2 α[N][i] .*= phi[N][i]; end
	
	for n=1:N
		
			for i=1:2 σ[n][i] = α[n][i] .* β[n][i]; end
			tmp = sum(σ[n])
			for i=1:2 σ[n][i] /= tmp; end
			#@time for i=1:2 σ[n][i] /= tmp; end
		
	end
	return sto.μ
end
