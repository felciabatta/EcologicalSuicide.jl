module EcologicalSuicide

using DifferentialEquations:
    EM, ImplicitEM, SDEProblem, SKenCarp, SOSRA,
    SciMLBase.AbstractSDEAlgorithm, solve
using Interpolations
using Plots
using Random
using RecursiveArrayTools

include("modelspecs.jl")
include("utils.jl")
include("particle_terms.jl")
include("diffusion_terms.jl")
include("equations.jl")
include("visuals.jl")
include("examples.jl")

export EM, SKenCarp, SOSRA, ImplicitEM
export idx_pm, range_mod!, GenericZero, fill!, strides
export GaussianSpec, ActiveBrownianParticleSpec, DiffusiveChemicalSpec,
    PeriodicLattice, ActiveBrownianSystemSpec
export WCACache, DiffusionCache
export ∂wca!, ∇wca!, distances!, displacements!, normalize!, nansumforces!, ∇c
export diffuse!, Σδtrain, ddiff2, relposition!
export activebrownian!, randombrownian!, ṗ, ṗ!, ċ, ċ!
export example6, systemsetup
export particleplot, potentialplot, particle_animate, save

end
