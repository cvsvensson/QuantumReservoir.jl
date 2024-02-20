lindbladian = DenseLindblad()
tmax = 100
abstol = 1e-12
reltol = 1e-12
## test measurement_matrix
@time mm1 = measurement_matrix(reservoir, qd_level_measurements, tmax, Exponentiation(); abstol, reltol, lindbladian);
@time mm2 = measurement_matrix(reservoir, qd_level_measurements, tmax, Exponentiation(EXP_sciml()); abstol, reltol, lindbladian);
@time mm3 = measurement_matrix(reservoir, qd_level_measurements, tmax, ODE(); int_alg, abstol, reltol, lindbladian);
@time mm4 = measurement_matrix(reservoir, qd_level_measurements, tmax, IntegratedODE(); abstol, reltol, lindbladian);
@time mm5 = measurement_matrix(reservoir, qd_level_measurements, tmax, Exponentiation(EXP_sciml_full()); abstol, reltol, lindbladian);

mm1.mat ≈ mm2.mat ≈ mm3.mat ≈ mm4.mat
[norm(mm1.mat - mm.mat) for mm in [mm2, mm3, mm4]] # ≈ [0.0, 0.0, 0.0
mm1.mat ≈ mm2.mat ≈ mm3.mat ≈ mm4.mat ≈ mm5.mat
[norm(mm1.mat - mm.mat) for mm in [mm2, mm3, mm4, mm5]] # ≈ [0.0, 0.0, 0.0
## rho evolution
abstol = 1e-12
@time sols1 = run_reservoir_ensemble(reservoir, qd_level_measurements, training_parameters[1:M_train], tmax, ODE(DP8()); abstol, reltol, int_alg=GaussLegendre(), lindbladian);
@time sols2 = run_reservoir_ensemble(reservoir, qd_level_measurements, training_parameters[1:M_train], tmax, IntegratedODE(DP8()); abstol, reltol, lindbladian);
@time sols3 = run_reservoir_ensemble(reservoir, qd_level_measurements, training_parameters[1:M_train], tmax, Exponentiation(); abstol, reltol, lindbladian);

[norm(sols1.integrated - sol.integrated) for sol in [sols2, sols3]] # ≈ [0.0, 0.0]

## compare measurement evolution and density matrix evolution
inte = mapreduce(rho0 -> map(Base.Fix2(dot, rho0), eachcol(mm1.mat)), hcat, sols1.vecensembleI.rho0s) |> real
sols1.integrated
inte - sols1.integrated .|> abs |> maximum
inte - sols2.integrated .|> abs |> maximum

inte - sols1.integrated # 
##TODO: check commutativity of wedge product and initial state preparation
