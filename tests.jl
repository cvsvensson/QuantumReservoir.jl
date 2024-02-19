## test measurement_matrix
@time mm1 = measurement_matrix(reservoir, qd_level_measurements, tmax, Exponentiation(); abstol, reltol);
@time mm2 = measurement_matrix(reservoir, qd_level_measurements, tmax, Exponentiation(EXP_sciml()); abstol, reltol);
@time mm3 = measurement_matrix(reservoir, qd_level_measurements, tmax, ODE(); int_alg, abstol, reltol);
@time mm4 = measurement_matrix(reservoir, qd_level_measurements, tmax, IntegratedODE(); abstol, reltol);

mm1.mat ≈ mm2.mat ≈ mm3.mat ≈ mm4.mat
[norm(mm1.mat - mm.mat) for mm in [mm2, mm3, mm4]] # ≈ [0.0, 0.0, 0.0
## rho evolution
@time sols1 = run_reservoir_ensemble(reservoir, qd_level_measurements, training_parameters[1:M_train], tmax, ODE(DP8()); abstol, reltol, int_alg=GaussLegendre(), lindbladian);
@time sols2 = run_reservoir_ensemble(reservoir, qd_level_measurements, training_parameters[1:M_train], tmax, IntegratedODE(DP8()); abstol, reltol, lindbladian);
@time sols3 = run_reservoir_ensemble(reservoir, qd_level_measurements, training_parameters[1:M_train], tmax, Exponentiation(); abstol, reltol, lindbladian);

[norm(sols1.integrated - sol.integrated) for sol in [sols2, sols3]] # ≈ [0.0, 0.0]

## compare measurement evolution and density matrix evolution
inte = mapreduce(rho0 -> map(Base.Fix1(dot, rho0), eachrow(mm1.mat)), hcat, sols1.vecensembleI.rho0s) |> real
sols1.integrated
inte - sols1.integrated .|> abs |> maximum
inte - sols2.integrated .|> abs |> maximum

##
