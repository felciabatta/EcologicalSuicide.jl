module EcologicalSuicide

using DifferentialEquations:
    EM, SDEProblem, SKenCarp, SOSRA, SciMLBase.AbstractSDEAlgorithm, solve,
    vecvec_to_mat
using Distances: euclidean, pairwise
using Interpolations
using NaNStatistics: nansum
using Plots

include("utils.jl")
include("particle_terms.jl")
include("diffusion_terms.jl")
include("equations.jl")
include("visuals.jl")
include("examples.jl")

export example5

end
