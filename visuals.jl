@userplot ParticlePlot
@recipe function f(pp::ParticlePlot; boxwidth=20)
    p, = pp.args
    # idx = (idx isa Int) ? (idx:idx) : idx
    seriestype --> :scatter
    xlims --> (-1, 1) .* boxwidth ./ 2
    ylims --> (-1, 1) .* boxwidth ./ 2
    show --> true
    marker --> 280 / boxwidth
    markeralpha --> 0.5
    markercolor --> :black
    markerstrokewidth --> 0
    # margin --> (1, :mm)
    aspect_ratio --> :equal
    legend --> false
    size --> (540, 540)
    dpi --> 100.1
    return p[2, :], p[1, :]
end

@userplot PotentialPlot
@recipe function f(pp::PotentialPlot)
    x, z = pp.args
    seriestype --> :contourf
    color --> :RdPu
    levels --> 20
    # cbar --> false
    lw --> 0.2
    alpha --> 0.9
    z_order --> :back
    return x, x, z
end

function particle_animate(
    usol,
    grid::PeriodicLattice;
    name::Union{String, Bool}=false,
    fps=30,
    Δf=40,
)
    anim = @inbounds @animate for i ∈ eachindex(usol)
        potentialplot(grid.x, usol[i].x[2])
        particleplot!(usol[i].x[1]; boxwidth=grid.xlength)
    end every Δf
    display(anim)
    save(anim, name; fps)
    return anim
end

function save(anim::Plots.Animation, name::String="particle"; fps)
    mov(anim, "./Figures/$name.mov"; fps)
    return nothing
end

function save(anim::Plots.Animation, yes::Bool; fps)
    if yes
        xyz = randstring(3)
        mov(anim, "./Figures/particle-$xyz.mov"; fps)
    end
end

######################################################################
# DEPRECATED
######################################################################

@userplot _ParticlePlot
@recipe function f(pp::_ParticlePlot; boxwidth=20)
    sol, idx = pp.args
    # idx = (idx isa Int) ? (idx:idx) : idx
    seriestype --> :scatter
    xlims --> (-1, 1) .* boxwidth ./ 2
    ylims --> (-1, 1) .* boxwidth ./ 2
    show --> true
    marker --> 280 / boxwidth
    markeralpha --> 0.5
    markercolor --> :black
    markerstrokewidth --> 0
    # margin --> (1, :mm)
    aspect_ratio --> :equal
    legend --> false
    size --> (540, 540)
    dpi --> 100.1
    return sol[[1], :, idx], sol[[2], :, idx]
end

@userplot _PotentialPlot
@recipe function f(pp::_PotentialPlot)
    x, y, z, idx = pp.args
    seriestype --> :contourf
    color --> :RdPu
    levels --> 20
    # cbar --> false
    lw --> 0.2
    alpha --> 0.9
    z_order --> :back
    return x, y, z[:, :, idx]
end

function _particle_animate(
    usol,
    x=0,
    y=0,
    z=0;
    Δf=40,
    fps=30,
    save=false,
    width=20,
)
    nsol = size(usol)[3]
    anim = @animate for i ∈ 1:nsol
        if !any((x, y, z) .== false)
            _potentialplot(x, y, z, i)
        end
        _particleplot!(usol, i; boxwidth=width)
    end every Δf
    display(anim)
    if save
        mov(anim, "./Figures/particle.mov"; fps=fps)
    end
    return usol, anim
end
