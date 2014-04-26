####################################
# Import, Build feature/label repr.
####################################
function prepare_data{T}(::Type{T},dataset::String,n_images=30) 
	data = get_data(dataset)
	imgs = randperm(length(data[1]))[1:n_images]
	println("Making features...")
	features = [represent_features(T,data[2][i],data[3][i]) for i in imgs]
	println("Making labels...")
	labels = Array(Array{Array{Int64,1},1},length(imgs))
	for i=1:length(imgs)
		labels[i] = array_to_rows(data[1][imgs[i]])
	end
	data = 0.
	return features::Array{MyTypes.Features{T},1},labels
end
#####################
# Build feature repr.
#####################
function represent_features{T}(::Type{T},F,G)
	height,width,f = size(F)
	g = size(G)[3]
	feature = [init_feats(T,width) for i=1:height]
	K=2
	for i=1:height
		for j=1:width-1
			for y_m=1:K
				for y_n=1:K
					feature[i].pairs[j][y_m,y_n] = zeros(T,4g)
					feature[i].pairs[j][y_m,y_n][(2*y_m + y_n -3)*g+1:(2*y_m + y_n -2)*g] = G[i,j,:][:]
				end
			end
		end
	end
	for i=1:height
		for j=1:width
			for y=1:K
				feature[i].single[j][y] = zeros(T,2f)
				feature[i].single[j][y][(y-1)*f+1:y*f] = F[i,j,:][:]
			end
		end
	end
	return feature
end
#####################
# Import matlab data
#####################
function get_data(file)
	println("Loading $file...")
	file = matopen(file)
	return (map(int,read(file,"Y")),read(file,"F"),read(file,"G_hor"))
end