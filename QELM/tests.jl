lindbladian = LazyLindblad()
tmax = 100
abstol = 1e-12
reltol = 1e-12
reservoir = prepare_reservoir(rc, reservoir_parameters[1], (T, μs); abstol, lindbladian);
rho0s = training_input_states[1:50]
int_alg = GaussLegendre()
## test measurement_matrix
@time mm1 = measurement_matrix(reservoir, tmax, Exponentiation(); abstol, reltol);
@time mm2 = measurement_matrix(reservoir, tmax, Exponentiation(EXP_krylovkit()); abstol, reltol);
@time mm3 = measurement_matrix(reservoir, tmax, ODE(); int_alg, abstol, reltol);
@time mm4 = measurement_matrix(reservoir, tmax, IntegratedODE(); abstol, reltol);
@time mm5 = measurement_matrix(reservoir, tmax, Exponentiation(EXP_sciml_full()); abstol, reltol);

mm1 ≈ mm2 ≈ mm3 ≈ mm4
[norm(mm1 - mm) for mm in [mm2, mm3, mm4]] # ≈ [0.0, 0.0, 0.0
mm1 ≈ mm2 ≈ mm3 ≈ mm4 ≈ mm5
[norm(mm1 - mm) for mm in [mm2, mm3, mm4, mm5]] # ≈ [0.0, 0.0, 0.0
## rho evolution
@time sols1 = run_reservoir_trajectories(reservoir, rho0s, tmax, ODE(DP8()); abstol, reltol, int_alg=GaussLegendre(), ensemblealg=EnsembleThreads());
@time sols2 = run_reservoir_trajectories(reservoir, rho0s, tmax, IntegratedODE(DP8()); abstol, reltol);
@time sols3 = run_reservoir_trajectories(reservoir, rho0s, tmax, Exponentiation(); abstol, reltol, lindbladian);
@time sols4 = run_reservoir_trajectories(reservoir, rho0s, tmax, Exponentiation(EXP_krylovkit()); abstol, reltol, lindbladian);

[norm(sols1.integrated - sol.integrated) for sol in [sols2, sols3, sols4]] # ≈ [0.0, 0.0]

## compare measurement evolution and density matrix evolution
inte = mm1' * stack([vecrep(rho0, reservoir) for rho0 in rho0s])
sols1.integrated
inte - sols1.integrated .|> abs |> maximum
inte - sols2.integrated .|> abs |> maximum
inte - sols1.integrated # 
##TODO: check commutativity of wedge product and initial state preparation
