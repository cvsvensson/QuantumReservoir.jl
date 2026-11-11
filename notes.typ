#set page(paper: "a4", margin: 2.2cm, numbering: "1")
#set text(size: 10.5pt)
#set par(justify: true)
#set heading(numbering: "1.")
#set math.equation(numbering: "(1)")

#align(center)[
  #text(size: 15pt, weight: "bold")[Quantum-dot reservoir computing with voltage input and current readout] \
  #v(2pt)
  #text(size: 10pt)[Status notes]
]

= Setup

*Reservoir.* A few interacting spinless quantum dots with random, all-to-all hopping, interactions and levels,
$
  H = sum_(i<j) (t_(i j) c_i^dagger c_j + "h.c.") + sum_(i<j) V_(i j) n_i n_j + sum_i epsilon_i n_i .
$
Each dot $i$ is tunnel coupled to its own lead with rate $Gamma_i$, temperature $T$ and chemical potential $mu_i (t)$.

*Input.* A random binary sequence $s_k = plus.minus 1$ is held for a time $Delta t$ per step and applied as lead voltages through a fixed mask $m_i$ and amplitude $A$: $mu_i (t) = A m_i s_k$ for $k Delta t <= t < (k+1) Delta t$. The time scale $Delta t$ is measured in units of the slowest relaxation time $tau$ of the open system.

*Dynamics.* Two descriptions with identical tunnelling rates are compared.
- _Lindblad_: jump operators are the dot operators $c_i^dagger$, $c_i$ written in the eigenbasis of $H$ with each transition weighted by the Fermi function of its lead. Coherences between eigenstates in the same charge sector are kept, so interference between transport paths enters.
- _Pauli_: a rate equation for the eigenstate populations only, i.e. the same model with coherences dropped.

Since the input is piecewise constant, the reservoir is a switched linear map
$
  rho_k = P_(s_k) rho_(k-1), quad P_plus.minus = e^(cal(L)_plus.minus Delta t),
$ <map>
where $cal(L)_plus.minus$ is the generator at the two voltage configurations.

*Readout.* The features are the particle currents into all leads at the end of each step; with time multiplexing they are sampled at several times within each step. A linear ridge regression with bias is trained on the middle part of the sequence and tested on the remainder, after discarding a warm-up. Performance is the test mean-squared error; since the targets have unit variance, an MSE of one means no information.

*Tasks.* Delayed recall $y_k = s_(k-d)$, a nonlinear product of delayed inputs $s_(k-2) s_(k-3)$, and NARMA.

= Results

*Memory requires non-commuting propagators.* By @map, the state after many steps is an ordered product of $P_+$ and $P_-$ acting on the initial state. If the two commuted, the state would depend only on how many $+$ inputs occurred, not on their order, and no delayed input could be read out linearly. The commutator of $P_+$ and $P_-$ therefore measures the reservoir's ability to remember order. It vanishes as $Delta t -> 0$ and grows to saturation once $Delta t$ exceeds $tau$ (dashed lines in @fig-dt). The current itself responds instantaneously to the voltage, so recalling the present input ($d = 0$) is trivial at every $Delta t$.

*Optimal input time scale.* Delayed recall works only in a window of $Delta t$ (@fig-dt). For short steps the propagators nearly commute and the state barely moves. For long steps each $P_plus.minus$ relaxes the system to the stationary state of the current voltage, erasing the past. The optimum lies at $Delta t$ somewhat below $tau$, and the window shrinks as the delay grows.

*Coherences help.* The Lindblad reservoir outperforms the Pauli one on the longest delay, and the gap is far larger than the spread over random reservoirs. The two models have nearly the same commutator measure, so non-commutativity alone does not explain the advantage.

*Time multiplexing.* Sampling the current several times per step improves both models substantially and widens the useful $Delta t$ window, since samples taken early in a step still carry the transient left by earlier inputs. The Lindblad advantage on the longest delay persists (@fig-dt, right).

#figure(
  grid(
    columns: 2,
    gutter: 6pt,
    image("mse_vs_dt.svg"), image("mse_vs_dt_timemultiplex2.svg"),
  ),
  caption: [Test MSE on delayed recall versus step duration $Delta t \/ tau$, median and interquartile range over random reservoirs, for Lindblad (blue) and Pauli (orange). Left: one current sample per step; right: two samples per step. Dashed: normalized commutator of $P_+$ and $P_-$.],
) <fig-dt>

*What makes a good reservoir.* At fixed $Delta t$, ranking random reservoir–lead pairs by performance, the better ones have a faster slowest relaxation rate, a larger propagator commutator and stronger total lead coupling (@fig-rank). These quantities rise and fall together, so this does not yet separate their roles. Level statistics of $H$ (the mean gap ratio) show no correlation with performance.

#figure(
  image("smooth15.svg", width: 55%),
  caption: [Random reservoir–lead pairs sorted by test MSE on delayed recall (Lindblad, fixed $Delta t$), together with moving averages of the slowest relaxation rate, the propagator commutator and the summed couplings, each normalized to its mean.],
) <fig-rank>

= Open questions

+ *Origin of the coherence advantage.* Is it interference as such, or only the larger hidden state space of the Lindblad description at equal readout? A Pauli model with a matched number of hidden variables, or a Lindblad model with dephasing of selected coherences, would distinguish these.
+ *Validity of the master equation.* The couplings are comparable to level spacings, where neither the secular (Pauli) limit nor the Lindblad form used here is controlled. Does the advantage survive in a more accurate treatment of the leads?
+ *Predicting performance.* Is there a single spectral quantity of $cal(L)_plus.minus$ or $P_plus.minus$, combining memory (relaxation) and order sensitivity (commutator), that predicts the MSE of a given reservoir and the optimal $Delta t$?
+ *Voltage amplitude.* At small $A$ the two propagators differ only weakly and their commutator vanishes; at large $A$ the response saturates. The joint scan over $A$ and $Delta t$ is in progress.
+ *Nonlinear tasks.* Performance on products of delayed inputs and on NARMA has not been analysed. How do these depend on $Delta t$ and $A$, and do coherences help there too?
+ *Scaling and resources.* How does memory grow with the number of dots and leads, and how does that compare with time multiplexing as a resource?
+ *Measurement realism.* The readout uses exact average currents. Shot noise and finite averaging time set a minimum $Delta t$ per sample, which may conflict with the optimal window.
