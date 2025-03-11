function summary_gif2(result)
    @unpack signal, res, W, seed, spectrum, measurements, ts, input, temperature, evals, params, ztrain, ztest, targets, mses, memory_capacities, n_train_first, n_test_first, n_train, n_test, ediffs, average_gapratio, time_multiplexing, tspan = result
    leads = res.leads
    xlims = maximum(abs ∘ real, first(spectrum)) .* (-1.01, 0.01)
    ylims = maximum(abs ∘ imag, first(spectrum)) .* (-1.01, 1.01)
    leadlabels = permutedims(collect(keys(input(tspan[1]))))
    multiplexedlabels = permutedims(reduce(vcat, map(n -> map(l -> string("$l,$n"), leadlabels), 1:time_multiplexing)))
    pcurrent = plot(ts, stack(measurements)', xlabel="t", ylabel="current", legend=false, marker=false)#, label=multiplexedlabels,legendtitle="Lead", legendposition=:topright#)
    # vline!(pcurrent, ts, color=:red)

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
        pboltz = plot(boltz, marker=true, ylims=(0, 1), labels=leadlabels, markersize=1, markerstrokewidth=0, legendposition=:left, ylabel="fermidirac")
        plot(psignal, pspec, pcurrent, pboltz, pevals, pinfo, pW, ptargets..., layout=(4 + div(length(targets), 2), 2))

    end
    gif(anim, "anim.gif", fps=div(length(1:dn:N-1), 5))
end

function summary_gif(result, Nfigs=100)
    @unpack leads, W, seed, spectrum, currents, sol, ts, input, H, sol2, temperature, evals, params, ztrain, ztest, targets, targetnames, mses, memory_capacities, overlaps, n_train_first, n_test_first, n_train, n_test, ediffs, average_gapratio = result
    xlims = maximum(abs ∘ real, first(spectrum)) .* (-1.01, 0.01)
    ylims = maximum(abs ∘ imag, first(spectrum)) .* (-1.01, 1.01)
    leadlabels = transpose(collect(keys(input(0))))
    signal = stack([collect(values(input.input.signal(t))) for t in ts])'
    pcurrent = plot(ts, stack(currents)', label=leadlabels, legendtitle="Lead", xlabel="t", ylabel="current", legendposition=:topright)

    inputsignal = [input.input.signal.signal(t) for t in ts]
    ptargets = map(eachcol(ztrain), eachcol(ztest), targets, targetnames) do ztrain, ztest, target, name
        ptarget = plot(ts, target.(ts), label="$name", xlabel="t")
        plot!(ptarget, ts, inputsignal, label="input", c=:black, linestyle=:dash)
        plot!(ptarget, ts[n_train], ztrain, label="train")
        plot!(ptarget, ts[n_test], ztest, label="test")
        ptarget
    end
    pecho = plot(ts, overlaps, xlabel="t", label="overlap of two solutions", yrange=(0, 1.01), legendposition=:bottomright)
    vline!(pecho, [ts[n_train_first]], color=:red, label="start training")
    smallest_decay_rate = mean(spec -> abs(partialsort(spec, 2, by=abs)), spectrum)
    vline!(pecho, [1 / smallest_decay_rate], label="1/(decay rate)")
    infos = (; seed, temperature=round(temperature, digits=2), average_gapratio=round(average_gapratio, digits=3), mse=round.(mses, digits=3), memory_capacity=round.(memory_capacities, digits=3))
    pinfo = plot([1:-1 for _ in infos]; framestyle=:none, la=0, label=permutedims(["$k = $v" for (k, v) in pairs(infos)]), legend_font_pointsize=10, legendposition=:top)
    pW = heatmap(log.(abs.(W')), color=:greys, yticks=(1:length(targets), targetnames), xticks=(1:length(leadlabels)+1, [leadlabels..., "bias"]), title="logabs(W)")
    indices = round.(Int, range(1, length(spectrum), Nfigs))
    anim = @animate for n in indices #(s, t) in zip(spectrum, ts)
        s = spectrum[n]
        t = ts[n]
        pspec = scatter(real(s), imag(s); xlims, ylims, size=(800, 800), ylabel="im", xlabel="re", label="eigenvalues", legendposition=:topleft)
        boltz = stack([QuantumDots.fermidirac.(ediffs, leads[l].T, input(t)[l].μ) |> sort for l in keys(leads)])
        psignal = plot(ts, signal, labels=leadlabels, xlabel="t", ylabel="voltage", legendtitle="Lead")
        vline!(psignal, [t], color=:red, label="t")
        pboltz = plot(boltz, marker=true, ylims=(0, 1), labels=leadlabels, markersize=1, markerstrokewidth=0, legendposition=:left, ylabel="fermidirac")
        plot(psignal, pspec, pcurrent, pboltz, pecho, pinfo, pW, ptargets..., layout=(4 + div(length(targets), 2), 2))
    end
    gif(anim, "anim.gif", fps=div(Nfigs, 5))
end


function summary_gif(reservoir, lead, input, opensystem, measurement, target, training; start=0.5, kwargs...)
    result = fullanalysis(reservoir, lead, input, opensystem, measurement, target, training; kwargs...)
    # @unpack signal, res, W, seed, spectrum, measurements, ts, input, temperature, evals, params, ztrain, ztest, targets, mses, memory_capacities, n_train_first, n_test_first, n_train, n_test, ediffs, average_gapratio, time_multiplexing, tspan = result
    # leads = res.leads
    reservoir_seed = reservoir.seed
    lead_seed = lead.seed
    temperature = lead.temperature
    @unpack ediffs, measurements, spectrum, evals, average_gapratio = result.simulation
    xlims = maximum(abs ∘ real, first(spectrum)) .* (-1.01, 0.01)
    ylims = maximum(abs ∘ imag, first(spectrum)) .* (-1.01, 1.01)
    leadlabels = permutedims(lead.labels)
    time_multiplexing = measurement.time_multiplexing
    @unpack ts, signal, tspan, voltage_input = input
    N = length(signal)
    nstart = Int(div(N, inv(start)))
    plot_tspan = (ts[nstart], ts[end])
    multiplexedlabels = permutedims(reduce(vcat, map(n -> map(l -> string("$l,$n"), leadlabels), 1:time_multiplexing)))
    pcurrent = plot(ts[1:end-1], stack(measurements)', xlabel="t", ylabel="current", legend=false, marker=false, xlims=plot_tspan)#, label=multiplexedlabels,legendtitle="Lead", legendposition=:topright#)
    # vline!(pcurrent, ts, color=:red)
    @unpack ztrain, ztest, targets, mses, memory_capacities, n_train, n_test, W = result.fit
    # ptargetkwargs = (; marker = true)
    ptargets = map(eachcol(ztrain), eachcol(ztest), collect(targets)) do ztrain, ztest, (name, target)
        ptarget = scatter(ts, target, label=name, xlabel="t", legendposition=:left, xlims=plot_tspan)
        scatter!(ptarget, ts, signal, label="input", c=:black, linestyle=:dash)
        scatter!(ptarget, ts[n_train], ztrain, label="train")
        scatter!(ptarget, ts[n_test], ztest, label="test")
        ptarget
    end
    pevals = plot(evals, markers=true, label="eigenvalues")
    # pecho = plot()#plot(ts, overlaps, xlabel="t", label="overlap of two solutions", yrange=(0, 1.01), legendposition=:bottomright)
    # vline!(pecho, [ts[n_train_first]], color=:red, label="start training")
    smallest_decay_rate = mean(spec -> abs(partialsort(spec, 2, by=abs)), spectrum)
    # vline!(pecho, [1 / smallest_decay_rate], label="1/(decay rate)")
    targetnames = collect(keys(targets))
    infos = (; reservoir_seed, lead_seed, temperature=round(temperature, digits=2), average_gapratio=round(average_gapratio, digits=3), mse=round.(mses, digits=3), memory_capacity=round.(memory_capacities, digits=3), smallest_decay_rate=round(smallest_decay_rate, digits=3))
    pinfo = plot([1:-1 for _ in infos]; framestyle=:none, la=0, label=permutedims(["$k = $v" for (k, v) in pairs(infos)]), legend_font_pointsize=10, legendposition=:top)
    pW = heatmap(log.(abs.(W')), color=:greys, yticks=(1:length(targets), targetnames), xticks=(1:length(multiplexedlabels)+1, [multiplexedlabels..., "bias"]), title="logabs(W)")
    # indices = 1:N#round.(Int, range(1, length(spectrum), Nfigs))
    high_frequency_ts = range(tspan..., N * 10 + 1)[1:end-1]
    voltages = stack([[x.μ for x in values(voltage_input(t))] for t in high_frequency_ts])'
    # display(voltages)
    Nfigs = 50
    dn = max(1, Int(div(N / 2, Nfigs)))
    first_t = ts[Int(div(N, 2))]
    anim = @animate for n in Int(div(N, 2)):dn:N-1#(s, t) in zip(spectrum, ts)
        s = spectrum[n]
        t = ts[n]
        pspec = scatter(real(s), imag(s); xlims, ylims, size=(800, 800), ylabel="im", xlabel="re", label="eigenvalues", legendposition=:topleft)
        boltz = stack([QuantumDots.fermidirac.(ediffs, temperature, voltage_input(t)[l].μ) |> sort for l in lead.labels])
        psignal = plot(high_frequency_ts, voltages, labels=leadlabels, xlabel="t", ylabel="voltage", legendtitle="Lead", legendposition=:right, xlims=plot_tspan)
        vline!(psignal, [t], color=:red, label="t")
        vline!(psignal, [first_t + 1 / smallest_decay_rate], color=:black, lw=2, label="t*")
        pboltz = plot(boltz, marker=true, ylims=(0, 1), labels=leadlabels, markersize=1, markerstrokewidth=0, legendposition=:left, ylabel="fermidirac")
        plot(psignal, pspec, pcurrent, pboltz, pevals, pinfo, pW, ptargets..., layout=(4 + div(length(targets), 2), 2))

    end
    gif(anim, "anim.gif", fps=div(length(1:dn:N-1), 5))
end