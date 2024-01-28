@userplot ParticlePlot
@recipe function f(pp::ParticlePlot; boxwidth=20)
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
    sol[[1], :, idx], sol[[2], :, idx]
end

@userplot PotentialPlot
@recipe function f(pp::PotentialPlot)
    x, y, z, idx = pp.args
    seriestype --> :contourf
    color --> :RdPu
    levels --> 20
    # cbar --> false
    lw --> 0.2
    alpha --> 0.9
    z_order --> :back
    x, y, z[:, :, idx]
end

function particle_animate(
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
            potentialplot(x, y, z, i)
        end
        particleplot!(usol, i; boxwidth=width)
    end every Δf
    display(anim)
    if save
        mov(anim, "./Figures/particle.mov", fps=fps)
    end
    return usol, anim
end
