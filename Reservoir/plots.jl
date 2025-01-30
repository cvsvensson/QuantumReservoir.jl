function summary_gif2(result)
    @unpack signal, res, W, seed, spectrum, measurements, ts, input, temperature, evals, params, ztrain, ztest, targets, mses, memory_capacities, n_train_first, n_test_first, n_train, n_test, ediffs, average_gapratio, time_multiplexing, tspan = result
    leads = res.leads
    xlims = maximum(abs ∘ real, first(spectrum)) .* (-1.01, 0.01)
    ylims = maximum(abs ∘ imag, first(spectrum)) .* (-1.01, 1.01)
    leadlabels = permutedims(collect(keys(input(tspan[1]))))
    multiplexedlabels = permutedims(reduce(vcat, map(n -> map(l -> string("$l,$n"), leadlabels), 1:time_multiplexing)))
    pcurrent = plot(ts, stack(measurements)', xlabel="t", ylabel="current", legend=false, marker=true)#, label=multiplexedlabels,legendtitle="Lead", legendposition=:topright#)

    ptargets = map(eachcol(ztrain), eachcol(ztest), collect(pairs(targets))) do ztrain, ztest, (name, target)
        ptarget = plot(ts, target, label=name, xlabel="t", legendposition=:left)
        plot!(ptarget, ts, signal, label="input", c=:black, linestyle=:dash)
        plot!(ptarget, ts[n_train], ztrain, label="train")
        plot!(ptarget, ts[n_test], ztest, label="test")
        ptarget
    end
    pevals = plot(evals, markers=true, label="eigenvalues")
    # pecho = plot()#plot(ts, overlaps, xlabel="t", label="overlap of two solutions", yrange=(0, 1.01), legendposition=:bottomright)
    # vline!(pecho, [ts[n_train_first]], color=:red, label="start training")
    smallest_decay_rate = mean(spec -> abs(partialsort(spec, 2, by=abs)), spectrum)
    # vline!(pecho, [1 / smallest_decay_rate], label="1/(decay rate)")
    targetnames = collect(keys(targets))
    infos = (; seed, temperature=round(temperature, digits=2), average_gapratio=round(average_gapratio, digits=3), mse=round.(mses, digits=3), memory_capacity=round.(memory_capacities, digits=3))
    pinfo = plot([1:-1 for _ in infos]; framestyle=:none, la=0, label=permutedims(["$k = $v" for (k, v) in pairs(infos)]), legend_font_pointsize=10, legendposition=:top)
    pW = heatmap(log.(abs.(W')), color=:greys, yticks=(1:length(targets), targetnames), xticks=(1:length(multiplexedlabels)+1, [multiplexedlabels..., "bias"]), title="logabs(W)")
    # indices = 1:N#round.(Int, range(1, length(spectrum), Nfigs))
    N = length(signal)
    high_frequency_ts = range(tspan..., N * 10 + 1)[1:end-1]
    voltages = stack([[x.μ for x in values(input(t))] for t in high_frequency_ts])'
    # display(voltages)
    Nfigs = 50
    dn = max(1, div(N, Nfigs))
    anim = @animate for n in 1:dn:N-1#(s, t) in zip(spectrum, ts)
        s = spectrum[n]
        t = ts[n]
        pspec = scatter(real(s), imag(s); xlims, ylims, size=(800, 800), ylabel="im", xlabel="re", label="eigenvalues", legendposition=:topleft)
        boltz = stack([QuantumDots.fermidirac.(ediffs, leads[l].T, input(t)[l].μ) |> sort for l in keys(leads)])
        psignal = plot(high_frequency_ts, voltages, labels=leadlabels, xlabel="t", ylabel="voltage", legendtitle="Lead", legendposition=:right)
        vline!(psignal, [t], color=:red, label="t")
        vline!(psignal, [1 / smallest_decay_rate], label="t*")
        pboltz = plot(boltz, marker=true, ylims=(0, 1), labels=leadlabels, markersize=1, markerstrokewidth=0, legendposition=:left, ylabel="boltzmann factors")
        plot(psignal, pspec, pcurrent, pboltz, pevals, pinfo, pW, ptargets..., layout=(4 + div(length(targets), 2), 2))

    end
    gif(anim, "anim.gif", fps=div(length(1:dn:N-1), 5))
end