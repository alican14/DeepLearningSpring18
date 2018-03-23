using PlyIO
using ArgParse
using Knet

path = "/home/alican/Desktop/baseline_folder/scapecomp/"
##trial
data0 = load_ply("/home/alican/Desktop/baseline_folder/scapecomp/mesh000.ply")
atype = gpu() >= 0 ? KnetArray{Float32} : Array{Float32}
datax = data0["vertex"]["x"]
datay = data0["vertex"]["y"]
dataz = data0["vertex"]["z"]
data1 = hcat(datax,datay)
data1 = hcat(data1,dataz)


files = readdir(path)
#println(files[1])
dataset=Any[]
for i in files
    data0 = load_ply(path * i)
    datax = data0["vertex"]["x"]
    datay = data0["vertex"]["y"]
    dataz = data0["vertex"]["z"]
    data1 = hcat(datax,datay)
    data1 = hcat(data1,dataz)
    dataset = push!(dataset,data1)
end

#println(size(dataset))
dataset = minibatch(dataset, 1; xtype=atype)
#println(length(dataset))
#println(dataset)
##end of trial
for p in ("Knet","ArgParse","Images")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using Knet
using ArgParse
using Images

const F = Float32

function encode(ϕ, x)
    x = mat(x)
    x = relu.(ϕ[1]*x .+ ϕ[2])
    μ = ϕ[3]*x .+ ϕ[4]
    logσ² = ϕ[5]*x .+ ϕ[6]
    return μ, logσ²
end

function decode(θ, z)
    z = relu.(θ[1]*z .+ θ[2])
    return sigm.(θ[3]*z .+ θ[4])
end

function binary_cross_entropy(x, x̂)
    s = @. x * log(x̂ + F(1e-10)) + (1-x) * log(1 - x̂ + F(1e-10))
    return -mean(s)
end

function output(w, x, nθ)
    θ, ϕ = w[1:nθ], w[nθ+1:end]
    μ, logσ² = encode(ϕ, x)
    nz, M = size(μ)
    σ² = exp.(logσ²)
    σ = sqrt.(σ²)

    KL =  -sum(@. 1 + logσ² - μ*μ - σ²) / 2
    # Normalise by same number of elements as in reconstruction
    KL /= M*28*28 ###1

    z = μ .+ randn!(similar(μ)) .* σ
    x̂ = decode(θ, z)
    BCE = binary_cross_entropy(mat(x), x̂) ####2

    return x̂
end

function loss(w, x, nθ)
    θ, ϕ = w[1:nθ], w[nθ+1:end]
    μ, logσ² = encode(ϕ, x)
    nz, M = size(μ)
    σ² = exp.(logσ²)
    σ = sqrt.(σ²)

    KL =  -sum(@. 1 + logσ² - μ*μ - σ²) / 2
    # Normalise by same number of elements as in reconstruction
    KL /= M*28*28 ###1

    z = μ .+ randn!(similar(μ)) .* σ
    x̂ = decode(θ, z)
    BCE = binary_cross_entropy(mat(x), x̂) ####2

    return BCE + KL
end

function aveloss(θ, ϕ, data) #####3
    ls = F(0)
    nθ = length(θ)
    #for (x, y) in data
    for x in data
        ls += loss([θ; ϕ], x, nθ)
    end
    return ls / length(data)
end

function train!(θ, ϕ, data, opt; epochs=1)
    w = [θ; ϕ]
    for epoch=1:epochs
        for (x, y) in data
            dw = grad(loss)(w, x, length(θ))
            update!(w, dw, opt)
        end
    end
    return θ, ϕ
end

function weights(nz, nh; atype=Array{F})
    θ = [  # z->x
        xavier(nh, nz),
        zeros(nh),
        xavier(12500*3, nh), #x
        zeros(12500*3)
        ]
    θ = map(a->convert(atype,a), θ)

    ϕ = [ # x->z
        xavier(nh, 12500*3),
        zeros(nh),
        xavier(nz, nh), #μ
        zeros(nz),
        xavier(nz, nh), #σ
        zeros(nz)
        ]
    ϕ = map(a->convert(atype,a), ϕ)

    return θ, ϕ
end

function main(args="")
    s = ArgParseSettings()
    s.description="Variational Auto Encoder on MNIST dataset."
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--seed"; arg_type=Int; default=-1; help="random number seed: use a nonnegative int for repeatable results")
        ("--batchsize"; arg_type=Int; default=100; help="minibatch size")
        ("--epochs"; arg_type=Int; default=100; help="number of epochs for training")
        ("--nh"; arg_type=Int; default=400; help="hidden layer dimension")
        ("--nz"; arg_type=Int; default=40; help="encoding dimention")
        ("--lr"; arg_type=Float64; default=1e-3; help="learning rate")
        ("--atype"; default=(gpu()>=0 ? "KnetArray{F}" : "Array{F}"); help="array type: Array for cpu, KnetArray for gpu")
        ("--infotime"; arg_type=Int; default=2; help="report every infotime epochs")
    end
    isa(args, String) && (args=split(args))
    if in("--help", args) || in("-h", args)
        ArgParse.show_help(s; exit_when_done=false)
        return
    end
    o = parse_args(args, s; as_symbols=true)

    atype = eval(parse(o[:atype]))
    info("using ", atype)
    o[:seed] > 0 && setseed(o[:seed])

    θ, ϕ = weights(o[:nz], o[:nh], atype=atype)
    w = [θ; ϕ]
    opt = optimizers(w, Adam, lr=o[:lr])

    path = "/home/alican/Desktop/baseline_folder/scapecomp/"
    files = readdir(path)
    dataset=Any[]
    for i in files
        data0 = load_ply(path * i)
        datax = data0["vertex"]["x"]
        datay = data0["vertex"]["y"]
        dataz = data0["vertex"]["z"]
        data1 = hcat(datax,datay)
        data1 = hcat(data1,dataz)
        data1 = reshape(data1,37500,1)
        dataset = push!(dataset,data1)
    end
    #println(typeof(dataset))
    #println(size(dataset))
    #dataset = reshape(dataset,12500,3,1,72)
    #println(size(dataset))
    #dataset = minibatch(dataset,1;xtype=atype())

    #for (d1,d2) in dataset
    #    println(d1)
    #    break
    #end
    #println(size(dataset[1]))
    randomout = output(w, atype(reshape(dataset[1],37500,1)), length(θ))
    return reshape(randomout,12500,3)

end

#return randomout

#end # module

randomout = main("--infotime 10 --seed 1 --epochs 0");
println(size(randomout))
println(randomout)
