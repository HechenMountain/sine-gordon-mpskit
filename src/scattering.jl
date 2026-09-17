using MPSKit, TensorKit, Printf
using TensorKit: truncrank
using LinearAlgebra: norm, kron, dot, I

# Energy density operator (on-site part)

"""
    energy_density_onsite(; d::Int, a::Real, m::Real, β::Real) → TensorMap

On-site part of the energy density:
    h_onsite = (1/2a) Π² + (1/a) φ² + (am²/β²)(I − cos βφ)

This excludes the nearest-neighbour coupling −(1/a) φⱼφⱼ₊₁.

Arguments
- `d` : Fock-space truncation
- `a` : lattice spacing
- `m` : mass parameter
- `β` : coupling constant

Returns
On-site energy operator as a `TensorMap`.
"""
function energy_density_onsite(; d::Int=30, a::Real=1.0, m::Real=1.0, β::Real=1.0)
    return (1 / (2a)) * fock_pi2(d) + (1 / a) * fock_phi2(d) +
           (a * m^2 / β^2) * fock_cosp(d; β)
end

"""
    energy_density_nn(; d::Int, a::Real) → TensorMap

Nearest-neighbour coupling as a two-site operator:
    h_nn = −(1/a) φⱼ φⱼ₊₁

Arguments
- `d` : Fock-space truncation
- `a` : lattice spacing

Returns
Two-site coupling operator as a `TensorMap`.
"""
function energy_density_nn(; d::Int=30, a::Real=1.0)
    V = boson_space(d)
    phi = _phi_matrix(d)
    return TensorMap(ComplexF64.((-1.0 / a) .* kron(phi, phi)), V ⊗ V, V ⊗ V)
end


# Measurement helpers

"""
    measure_energy_density(ψ::FiniteMPS; d::Int, a::Real, m::Real, β::Real,
                           ϕ_L::Real=0.0, ϕ_R::Real=0.0) → Vector{Float64}

Measure the energy density ε(j) at each site of the finite MPS `ψ`.

Arguments
- `ψ`         : finite MPS to measure
- `d`         : Fock-space truncation
- `a`         : lattice spacing
- `m`         : mass parameter
- `β`         : coupling constant
- `ϕ_L`, `ϕ_R` : ghost-site boundary field values

Returns
Per-site energy density as a `Vector{Float64}`.

"""
function measure_energy_density(ψ::FiniteMPS;
                                 d::Int=30, a::Real=1.0, m::Real=1.0, β::Real=1.0,
                                 ϕ_L::Real=0.0, ϕ_R::Real=0.0)
    N = length(ψ)
    h_on = energy_density_onsite(; d, a, m, β)
    h_nn = energy_density_nn(; d, a)

    ε = zeros(Float64, N)

    # On-site contributions
    for j in 1:N
        ε[j] += real(expectation_value(ψ, (j,) => h_on))
    end

    # NN contributions split equally between the two sites
    for j in 1:N-1
        e_nn = real(expectation_value(ψ, (j, j + 1) => h_nn))
        ε[j]   += e_nn / 2
        ε[j+1] += e_nn / 2
    end

    # Ghost-site boundary corrections (no-ops for ϕ_L = ϕ_R = 0).
    Phi = fock_phi(d)
    ε[1] += (-ϕ_L / a) * real(expectation_value(ψ, (1,) => Phi)) + ϕ_L^2 / (2a)
    ε[N] += (-ϕ_R / a) * real(expectation_value(ψ, (N,) => Phi)) + ϕ_R^2 / (2a)

    return ε
end

"""
    measure_field_profile(ψ::FiniteMPS; d::Int) → Vector{Float64}

Measure ⟨φ(j)⟩ at each site of the finite MPS.

Arguments
- `ψ` : finite MPS to measure
- `d` : Fock-space truncation

Returns
Per-site field expectation values as a `Vector{Float64}`.

"""
function measure_field_profile(ψ::FiniteMPS; d::Int=30)
    Phi = fock_phi(d)
    return real.([expectation_value(ψ, (j,) => Phi) for j in 1:length(ψ)])
end


"""
    momentum_density_nn(; d::Int, a::Real) → TensorMap

Two-site field momentum density operator:
    p(j, j+1) = (1/2a)(φⱼ πⱼ₊₁ − πⱼ φⱼ₊₁)

The total lattice momentum is P = Σⱼ p(j, j+1).

Arguments
- `d` : Fock-space truncation
- `a` : lattice spacing

Returns
Two-site momentum density operator as a `TensorMap`.
"""
function momentum_density_nn(; d::Int=30, a::Real=1.0)
    V = boson_space(d)
    phi = _phi_matrix(d)
    pi_m = Matrix{ComplexF64}(_pi_matrix(d))
    # Convention: for TensorMap(kron(A,B), V⊗V, V⊗V) with expectation_value(ψ,(j,j+1)=>op),
    # A acts on site j+1 and B acts on site j.
    # So kron(pi_m, phi) − kron(phi, pi_m) = π_{j+1}⊗φ_j − φ_{j+1}⊗π_j
    #                                      = φ_j π_{j+1} − π_j φ_{j+1}  (diff-site commutation)
    h = (1 / (2a)) .* (kron(pi_m, phi) - kron(phi, pi_m))
    return TensorMap(ComplexF64.(h), V ⊗ V, V ⊗ V)
end


"""
    measure_total_momentum(ψ::FiniteMPS; d::Int, a::Real) → Float64

Measure the total field momentum ⟨P⟩ = Σⱼ ⟨φⱼπⱼ₊₁ − πⱼφⱼ₊₁⟩/(2a).

Arguments
- `ψ` : finite MPS to measure
- `d` : Fock-space truncation
- `a` : lattice spacing

Returns
Real total field momentum.
"""
function measure_total_momentum(ψ::FiniteMPS; d::Int=30, a::Real=1.0)
    N = length(ψ)
    h_p = momentum_density_nn(; d, a)
    P = 0.0
    for j in 1:N-1
        P += real(expectation_value(ψ, (j, j + 1) => h_p))
    end
    return P
end


"""
    charge_density_op(; d::Int, a::Real, β::Real) → TensorMap

Topological charge density as a two-site operator on bond (j, j+1):
    ρ(j, j+1) = β/(2πa) (φⱼ₊₁ − φⱼ)

Arguments
- `d` : Fock-space truncation
- `a` : lattice spacing
- `β` : coupling constant

Returns
Two-site topological charge density operator as a `TensorMap`.
"""
function charge_density_op(; d::Int=30, a::Real=1.0, β::Real=2.0)
    V   = boson_space(d)
    phi = _phi_matrix(d)
    Id  = Matrix{Float64}(I, d, d)
    # ρ = β/(2πa) (φ_{j+1} − φ_j).
    # Note: MPSKit's expectation_value((j, j+1) => op) maps the FIRST kron factor
    # to site j+1 and the SECOND to site j, so kron(phi, Id) − kron(Id, phi)
    # correctly computes φ_{j+1} − φ_j.
    h = (β / (2π * a)) .* (kron(phi, Id) - kron(Id, phi))
    return TensorMap(ComplexF64.(h), V ⊗ V, V ⊗ V)
end

"""
    measure_charge_density(ψ::FiniteMPS; d::Int, a::Real, β::Real) → Vector{Float64}

Measure the topological charge density ρ(j, j+1) = β/(2πa)⟨φⱼ₊₁ − φⱼ⟩ at
each bond of the finite MPS `ψ`.

Returns a `Vector{Float64}` of length `N−1` (one value per bond), where
`N = length(ψ)`.  Positive values correspond to soliton-like regions, negative
values to anti-soliton-like regions. 

Used for trajectory extraction.

Arguments
- `ψ` : finite MPS to measure
- `d` : Fock-space truncation
- `a` : lattice spacing
- `β` : coupling constant

Returns
Bond charge densities as a `Vector{Float64}` of length `length(ψ) - 1`.
"""
function measure_charge_density(ψ::FiniteMPS; d::Int=30, a::Real=1.0, β::Real=2.0)
    N   = length(ψ)
    ρ_op = charge_density_op(; d, a, β)
    return real.([expectation_value(ψ, (j, j + 1) => ρ_op) for j in 1:N-1])
end


# Vacuum-referenced energy density

"""
    vacuum_energy_density(ψ_vac::InfiniteMPS, N::Int; d, a, m, β,
                          ϕ_L=0.0, ϕ_R=0.0) → Vector{Float64}

Static per-site energy-density baseline ε_vac(j) of the soliton-free vacuum on an
`N`-site open chain, built from the infinite-vacuum tensors (`AL` in the bulk,
`AR` closing the right edge). Used to subtract static offset.

Arguments
- `ψ_vac`     : infinite MPS vacuum state
- `N`         : number of finite-chain sites
- `d`         : Fock-space truncation
- `a`         : lattice spacing
- `m`         : mass parameter
- `β`         : coupling constant
- `ϕ_L`, `ϕ_R` : ghost-site boundary field values

Returns
Per-site vacuum energy baseline as a `Vector{Float64}`.
"""
function vacuum_energy_density(ψ_vac::InfiniteMPS, N::Int;
                                d::Int=30, a::Real=1.0, m::Real=1.0, β::Real=1.0,
                                ϕ_L::Real=0.0, ϕ_R::Real=0.0)
    # Build a finite vacuum chain in the SAME mixed gauge as the wavepacket
    # tails: AL on the left, AR on the right, with the orthogonality centre
    # tensor AC at the junction (as in the B-tensor construction, where the B-tensor
    # plays the role of AC).  Both far ends then close with the identity (the
    # AL/AR transfer-matrix fixed points), so the left and right edge spikes
    # match the wavepacket's exactly.  A bare AL→AR junction (no AC) instead
    # injects a spurious right-edge defect.
    AL = ψ_vac.AL[1]
    AR = ψ_vac.AR[1]
    AC = ψ_vac.AC[1]
    c  = N ÷ 2
    tensors = Vector{typeof(AL)}(undef, N)
    for j in 1:c-1
        tensors[j] = AL
    end
    tensors[c] = AC
    for j in c+1:N
        tensors[j] = AR
    end
    ψ = FiniteMPS(tensors)
    return measure_energy_density(ψ; d, a, m, β, ϕ_L, ϕ_R)
end


# Time evolution loop

# Append one snapshot row to a checkpoint TSV file.
# On the first call (file absent) the header line is written first.
# Each row: time value followed by the measurement vector.
function _ckpt_append(path::String, header::Vector{String}, t::Float64, vals::Vector{Float64})
    if !isfile(path)
        open(path, "w") do io
            println(io, "# ", join(header, "\t"))
        end
    end
    open(path, "a") do io
        print(io, @sprintf("%.10e", t))
        for v in vals
            print(io, "\t", @sprintf("%.10e", v))
        end
        println(io)
    end
end

# Bundle the current in-memory history into matrices and write a resume
# checkpoint (atomic, via save_state_checkpoint in helpers.jl).
function _save_tdvp_state(path, ψ, step, mpo_params, extras,
                          times, total_energy, total_momentum,
                          energy_density, field_profiles, charge_profiles)
    history = (; times          = copy(times),
                 total_energy   = copy(total_energy),
                 total_momentum = copy(total_momentum),
                 energy_density = hcat(energy_density...),
                 field_profile  = hcat(field_profiles...),
                 charge_density = hcat(charge_profiles...))
    return save_state_checkpoint(path; ψ, step, mpo_params, history, extras)
end

"""
    run_scattering(ψ::FiniteMPS, H_finite::FiniteMPOHamiltonian;
                   d, a, m, β, χ, dt, T_final, save_every,
                   two_site, verbosity,
                   checkpoint_dir, checkpoint_tag,
                   ϕ_L, ϕ_R,
                   state_every, extras, resume) → NamedTuple

Run TDVP time evolution of a prepared soliton–anti-soliton state on a finite chain.

The initial state `ψ` and finite Hamiltonian `H_finite` should be constructed
by `soliton_antisoliton_wavepacket` (or any other method).

When `two_site=true` (default), uses two-site TDVP (`TDVP2`) which adapts bond
dimension up to `χ` via SVD truncation. Otherwise uses cheaper single-site TDVP.

When `checkpoint_dir` is provided, each measurement snapshot is written to
disk immediately after it is taken.
Four files are written in TSV format (one row per snapshot, sites as columns):
  - `ckpt_scalars_<tag>.tsv`       : time, E_total, P_total
  - `ckpt_energy_<tag>.tsv`        : time, ε(1), …, ε(N)
  - `ckpt_field_<tag>.tsv`         : time, ⟨φ(1)⟩, …, ⟨φ(N)⟩
  - `ckpt_charge_<tag>.tsv`        : time, ρ(1,2), …, ρ(N−1,N)

Resume checkpointing (`state_every > 0`, requires `checkpoint_dir`):
every `state_every` TDVP steps (and at `t = 0` and the final step) the full MPS
state, step index, the scalars needed to rebuild `H_finite`
(`N, d, a, m, β, ϕ_L, ϕ_R`), the accumulated measurement history, and the
caller-supplied `extras` are written atomically to
`ckpt_state_<tag>.jld2` (overwriting the previous one — only the latest is kept).

`resume = true` ignores any passed `ψ`/`H_finite`, loads `ckpt_state_<tag>.jld2`
from `checkpoint_dir`, rebuilds `ψ`, `H_finite` (from the stored scalars) and the
MPO environments, restores the in-memory history, truncates the four `ckpt_*.tsv`
logs back to the last checkpointed time (dropping rows from steps that will be
recomputed), and continues from the next step. `T_final` may be increased
relative to the original run to extend it.  `extras` is echoed back unchanged in
the result so the caller's post-run save block can source ε_vac / M_K without
re-running setup.

Arguments
- `ψ`              : initial finite MPS, or `nothing` when resuming
- `H_finite`       : finite Hamiltonian, or `nothing` when resuming
- `d`, `a`, `m`, `β` : model parameters
- `χ`              : maximum bond dimension for two-site TDVP
- `dt`             : time step
- `T_final`        : final evolution time
- `save_every`     : measurement interval in TDVP steps
- `two_site`       : select two-site TDVP when `true`
- `verbosity`      : output level
- `checkpoint_dir` : directory for checkpoint files, or `nothing`
- `checkpoint_tag` : suffix for checkpoint filenames
- `ϕ_L`, `ϕ_R`    : ghost-site boundary field values
- `state_every`    : state checkpoint interval in TDVP steps
- `extras`         : static run artifacts carried in the result
- `resume`         : resume from a state checkpoint when `true`

Returns
Named tuple with fields:
- `times`           : saved time points  (Vector{Float64})
- `energy_density`  : ε(j, t_i)  matrix  (N × n_saves)
- `field_profile`   : ⟨φ(j, t_i)⟩ matrix  (N × n_saves)
- `charge_density`  : ρ(j, t_i)  matrix  ((N−1) × n_saves)  — bond-local
- `total_energy`    : ⟨H_finite⟩ at each saved time  (Vector{Float64})
- `total_momentum`  : total field momentum at each saved time  (Vector{Float64})
- `extras`          : the `extras` NamedTuple (passed in, or loaded on resume)

Boundary terms: `ϕ_L`/`ϕ_R` must match the ghost values used to build
`H_finite` (see `sine_gordon_finite_mpo`.
The total energy itself is measured directly as ⟨H_finite⟩.

Note:
- Works for n-particle wavepackets
"""
function run_scattering(
    ψ = nothing, H_finite = nothing;
    d::Int      = 30,
    a::Real     = 1.0,
    m::Real     = 1.0,
    β::Real     = 2.0,
    χ::Int      = 64,
    dt::Real    = 0.5,
    T_final::Real = 10.0,
    save_every::Int = 1,
    two_site::Bool  = true,
    verbosity::Int  = 1,
    checkpoint_dir::Union{String,Nothing} = nothing,
    checkpoint_tag::String = "",
    ϕ_L::Real   = 0.0,
    ϕ_R::Real   = 0.0,
    state_every::Int = 0,
    extras::NamedTuple = NamedTuple(),
    resume::Bool = false,
)
    # TDVP algorithm selection.
    alg = two_site ? TDVP2(; trscheme=truncrank(χ)) : TDVP()

    # Checkpoint file paths (nothing when checkpointing is disabled).
    if !isnothing(checkpoint_dir)
        mkpath(checkpoint_dir)
    end
    _ckpt_path(name) = joinpath(checkpoint_dir, "ckpt_$(name)_$(checkpoint_tag).tsv")
    _state_path = isnothing(checkpoint_dir) ? nothing :
                  joinpath(checkpoint_dir, "ckpt_state_$(checkpoint_tag).jld2")

    # Storage
    times           = Float64[]
    energy_density  = Vector{Vector{Float64}}()
    field_profiles  = Vector{Vector{Float64}}()
    charge_profiles = Vector{Vector{Float64}}()
    total_energy    = Float64[]
    total_momentum  = Float64[]

    # ── Establish the initial state: resume from disk, or start fresh ─────────
    if resume
        @assert _state_path !== nothing && isfile(_state_path) "resume=true but no state checkpoint at $(_state_path)"
        ckpt = load_state_checkpoint(_state_path)
        ψ    = ckpt.psi
        mp   = ckpt.mpo_params

        d, a, m, β = mp.d, mp.a, mp.m, mp.β
        ϕ_L, ϕ_R   = mp.ϕ_L, mp.ϕ_R
        extras     = ckpt.extras
        N          = length(ψ)
        H_finite   = sine_gordon_finite_mpo(N; d, a, m, β, ϕ_L, ϕ_R)

        # Restore the in-memory history (matrices → per-snapshot column vectors).
        h = ckpt.history
        append!(times,          h.times)
        append!(total_energy,   h.total_energy)
        append!(total_momentum, h.total_momentum)
        for k in axes(h.energy_density, 2)
            push!(energy_density,  h.energy_density[:, k])
            push!(field_profiles,  h.field_profile[:, k])
            push!(charge_profiles, h.charge_density[:, k])
        end
        start_step = ckpt.step + 1
        t_state    = times[end]

        # Drop ckpt_*.tsv rows written after the last state checkpoint.
        # Those steps are recomputed later on
        for nm in ("scalars", "energy", "field", "charge")
            truncate_ckpt_tsv(_ckpt_path(nm), t_state)
        end

        envs = environments(ψ, H_finite)
        verbosity > 0 && vprint(@sprintf("  Resuming from step %d (t = %.4f)  χ_max = %d",
                                          ckpt.step, t_state,
                                          maximum(dim, right_virtualspace.(Ref(ψ), 1:N))))
    else
        @assert ψ !== nothing && H_finite !== nothing "fresh run requires both ψ and H_finite (or pass resume=true)"
        N    = length(ψ)
        # environments(ψ, H) builds/caches the left/right boundary contractions
        # for expectation values and time stepping; timestep() threads them
        # through to avoid recomputation.
        envs = environments(ψ, H_finite)

        # Measure initial state.  The total energy is ⟨H_finite⟩ (exact, includes
        # any ghost boundary terms); ε(j) is a per-site split of the same
        # quantity (Σⱼ ε(j) ≈ ⟨H⟩ when ϕ_L/ϕ_R match H_finite).
        push!(times, 0.0)
        push!(energy_density, measure_energy_density(ψ; d, a, m, β, ϕ_L, ϕ_R))
        push!(field_profiles, measure_field_profile(ψ; d))
        push!(charge_profiles, measure_charge_density(ψ; d, a, β))
        push!(total_energy, real(expectation_value(ψ, H_finite, envs)))
        push!(total_momentum, measure_total_momentum(ψ; d, a))
        start_step = 1
    end

    n_steps     = round(Int, T_final / dt)
    _mpo_params = (; N, d, a, m, β, ϕ_L, ϕ_R)

    # Checkpoint headers + writer (need N, hence defined after the branch above).
    _hdr_scalars = ["time", "E_total", "P_total"]
    _hdr_energy  = vcat(["time"], ["e$j"   for j in 1:N])
    _hdr_field   = vcat(["time"], ["phi$j" for j in 1:N])
    _hdr_charge  = vcat(["time"], ["rho$j" for j in 1:N-1])
    function _write_checkpoint(t, ε_j, φ_j, ρ_j, E, P)
        isnothing(checkpoint_dir) && return
        _ckpt_append(_ckpt_path("scalars"), _hdr_scalars, t, [E, P])
        _ckpt_append(_ckpt_path("energy"),  _hdr_energy,  t, ε_j)
        _ckpt_append(_ckpt_path("field"),   _hdr_field,   t, φ_j)
        _ckpt_append(_ckpt_path("charge"),  _hdr_charge,  t, ρ_j)
    end
    # State-save is gated to measurement steps (it runs inside the snapshot
    # block), so state_every effectively rounds up to a save_every multiple.
    _state_due(step) = state_every > 0 && _state_path !== nothing &&
                       (step % state_every == 0 || step == n_steps)

    # On a fresh run, emit the t = 0 snapshot now (TSV + initial state).  Saving
    # the initial state means even a crash before the first state-save step can
    # resume without rebuilding the wavepacket.
    if !resume
        verbosity > 0 && vprint(@sprintf("t = %6.2f   E_tot = %.8f   P_tot = %+.6f   χ_max = %d",
                                          0.0, total_energy[end], total_momentum[end],
                                          maximum(dim, right_virtualspace.(Ref(ψ), 1:N))))
        _write_checkpoint(0.0, energy_density[end], field_profiles[end],
                          charge_profiles[end], total_energy[end], total_momentum[end])
        if state_every > 0 && _state_path !== nothing
            _save_tdvp_state(_state_path, ψ, 0, _mpo_params, extras,
                             times, total_energy, total_momentum,
                             energy_density, field_profiles, charge_profiles)
        end
    end

    for step in start_step:n_steps
        t = step * dt
        # timestep(ψ, H, t_current, dt, alg, envs) → (ψ_new, envs_new)
        # The 5 positional arguments are: MPS, Hamiltonian, current time,
        # time step size, algorithm.  The 6th is the cached environments.
        ψ, envs = timestep(ψ, H_finite, (step - 1) * dt, dt, alg, envs)

        if step % save_every == 0 || step == n_steps
            push!(times, t)
            ε_j = measure_energy_density(ψ; d, a, m, β, ϕ_L, ϕ_R)
            push!(energy_density, ε_j)
            push!(field_profiles, measure_field_profile(ψ; d))
            push!(charge_profiles, measure_charge_density(ψ; d, a, β))
            push!(total_energy, real(expectation_value(ψ, H_finite, envs)))
            push!(total_momentum, measure_total_momentum(ψ; d, a))

            # Write snapshot to checkpoint files immediately so data is not
            # lost if the job is killed at wall time.
            _write_checkpoint(t, energy_density[end], field_profiles[end],
                              charge_profiles[end], total_energy[end], total_momentum[end])

            # Persist the full state for resume.  Done after the TSV write above,
            # so a complete state file always implies its measurement rows exist.
            if _state_due(step)
                _save_tdvp_state(_state_path, ψ, step, _mpo_params, extras,
                                 times, total_energy, total_momentum,
                                 energy_density, field_profiles, charge_profiles)
            end

            verbosity > 0 && vprint(@sprintf("t = %6.2f   E_tot = %.8f   P_tot = %+.6f   χ_max = %d",
                                              t, total_energy[end], total_momentum[end],
                                              maximum(dim, right_virtualspace.(Ref(ψ), 1:N))))
        end
    end

    # Collect into matrices
    ε_mat = hcat(energy_density...)   # N × n_saves
    φ_mat = hcat(field_profiles...)   # N × n_saves
    ρ_mat = hcat(charge_profiles...)  # (N-1) × n_saves

    return (; times, energy_density=ε_mat, field_profile=φ_mat,
              charge_density=ρ_mat, total_energy, total_momentum, extras)
end
