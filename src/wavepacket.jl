using MPSKit, TensorKit, Printf
using TensorKit: truncrank
using LinearAlgebra: norm, kron, dot, I

# Reflection-symmetric gauge-fixing for B-tensors (Milsted et al. Eq. A17)

"""
    _symmetrize_B(B_arr, AL_arr, AR_arr, p) → B_new_arr

Apply the reflection-symmetric gauge condition (Milsted et al. arXiv:2012.07243,
Eq. A17, adapted from Vandamme & Vanderstraeten 2019) to a quasiparticle
B-tensor in dense array form.

The SGF condition chooses X to minimise the cost function

    C(X) = ‖Σ_s B_new^s ⊗ conj(A_L^s)‖²_F
         + ‖Σ_s B_new^s ⊗ conj(A_R^s)‖²_F

which simultaneously penalises components of B that overlap with both A_L
and A_R.  C(X) is quadratic in X, so the optimum satisfies the normal
equations  H vec(X) = −g which are solved via `\\`.

Arguments
- `B_arr`  : fused B tensor, shape `(D_L, d_phys, D_R)`.
             (D_L = left-vacuum bond dim, D_R = right-vacuum bond dim.)
- `AL_arr` : the tensor standing immediately left of B in the assembled
             block-matrix state, `(D_L, d_phys, D_L)` — the left-gauge tensor
             of the left ground state
- `AR_arr` : the tensor standing immediately right of B in the assembled
             block-matrix state, `(D_R, d_phys, D_R)` — the right-gauge tensor
             of the right ground state
- `p`      : Bloch momentum of the excitation

Returns
Dense `(D_L, d_phys, D_R)` array — the gauge-transformed B tensor.
"""
function _symmetrize_B(B_arr::Array{ComplexF64,3},
                        AL_arr::Array{ComplexF64,3},
                        AR_arr::Array{ComplexF64,3},
                        p::Real)
    DL, dp, DR = size(B_arr)
    @assert size(AL_arr) == (DL, dp, DL) "AL size mismatch: expected ($DL,$dp,$DL), got $(size(AL_arr))"
    @assert size(AR_arr) == (DR, dp, DR) "AR size mismatch: expected ($DR,$dp,$DR), got $(size(AR_arr))"

    nx = DL * DR
    φ  = exp(-im * p)

    # ── 1. Gram matrices ─────────────────────────────────────────────────────
    # G^L_{st} = tr(A_L^{s†} A_L^t),   G^R_{st} = tr(A_R^{s†} A_R^t)
    G = zeros(ComplexF64, dp, dp)
    for s in 1:dp, t in s:dp
        gl = dot(AL_arr[:, s, :], AL_arr[:, t, :])
        gr = dot(AR_arr[:, s, :], AR_arr[:, t, :])
        g_st = gl + gr
        G[s, t] = g_st
        G[t, s] = conj(g_st)
    end

    # ── 2. Hessian: diagonal blocks ──────────────────────────────────────────
    #
    # The Hessian of C(X) w.r.t. vec(X) is  H = Σ_{s,t} G_{ts} F^{s†} F^t,
    # where F^s = I⊗A_L^s − φ A_R^{sT}⊗I maps vec(X) → vec(A_L^s X − φ X A_R^s).
    #
    # Expanding F^{s†}F^t produces 4 terms.  The two diagonal block terms are:
    #   I_{DR} ⊗ P_LL   where  P_LL = Σ_{s,t} G_{ts} A_L^{s†} A_L^t
    #   P_RR ⊗ I_{DL}   where  P_RR = Σ_{s,t} G_{ts} A_R^{s*} A_R^{tT}
    #
    # Swapping the summation labels s↔t:
    #   P_LL = Σ_{s,t} G_{st} A_L^{t†} A_L^s
    #   P_RR = Σ_{s,t} G_{st} A_R^{t*} A_R^{sT}
    P_LL = zeros(ComplexF64, DL, DL)
    P_RR = zeros(ComplexF64, DR, DR)
    for s in 1:dp, t in 1:dp
        P_LL .+= G[s, t] .* (AL_arr[:, t, :]' * AL_arr[:, s, :])
        P_RR .+= G[s, t] .* (conj.(AR_arr[:, t, :]) * transpose(AR_arr[:, s, :]))
    end

    # Assemble H (nx × nx), column-major: vec(X)[(b-1)*DL + a] = X[a, b]
    H = zeros(ComplexF64, nx, nx)

    # δ_{b₁,b₂} P_LL[a₁,a₂]
    for b in 1:DR
        rng = (b - 1) * DL + 1 : b * DL
        H[rng, rng] .+= P_LL
    end

    # δ_{a₁,a₂} P_RR[b₁,b₂]
    for a in 1:DL, b1 in 1:DR, b2 in 1:DR
        H[(b1 - 1) * DL + a, (b2 - 1) * DL + a] += P_RR[b1, b2]
    end

    # ── 3. Hessian: cross terms ──────────────────────────────────────────────
    # The two off-diagonal terms are −φ M₁ − φ* M₁†, where:
    #   M₁ = Σ_{s,t} G_{ts} (A_R^{tT} ⊗ A_L^{s†})
    #      = Σ_{s,t} G_{st} (A_R^{sT} ⊗ A_L^{t†})   [after s↔t swap]
    M1 = zeros(ComplexF64, nx, nx)
    for s in 1:dp, t in 1:dp
        M1 .+= G[s, t] .* kron(transpose(AR_arr[:, s, :]), AL_arr[:, t, :]')
    end
    H .-= φ .* M1 .+ conj(φ) .* M1'

    # ── 4. Gradient at X=0 ───────────────────────────────────────────────────
    # g_mat = Σ_{s,t} G_{ts} A_L^{s†} B^t − φ* Σ_{s,t} G_{ts} B^t A_R^{s†}
    #       = Σ_{s,t} G_{st} A_L^{t†} B^s − φ* Σ_{s,t} G_{st} B^s A_R^{t†}
    g_mat = zeros(ComplexF64, DL, DR)
    for s in 1:dp, t in 1:dp
        g_mat .+= G[s, t] .* (AL_arr[:, t, :]' * B_arr[:, s, :])
        g_mat .-= conj(φ) .* G[s, t] .* (B_arr[:, s, :] * AR_arr[:, t, :]')
    end
    g = vec(g_mat)

    # ── 5. Solve normal equations ────────────────────────────────────────────
    X_vec = H \ (-g)
    X = reshape(X_vec, DL, DR)

    # ── 6. Apply gauge transformation (*) ────────────────────────────────────
    B_new = copy(B_arr)
    for s in 1:dp
        B_new[:, s, :] .+= AL_arr[:, s, :] * X .- φ .* X * AR_arr[:, s, :]
    end

    return B_new
end


# Gaussian wavepacket coefficients

"""
    _gaussian_coeffs(N::Int, x0::Real, p0::Real, σ::Real) → Vector{ComplexF64}

Gaussian wavepacket coefficients:
    f_j = exp(−(j − x₀)²/(2σ²)) exp(ip₀ j)

Normalised so that Σ|f_j|² = 1.

Arguments
- `N`  : number of lattice sites
- `x0` : wavepacket centre position
- `p0` : central Bloch momentum
- `σ`  : Gaussian width in lattice units

Returns
Normalized complex coefficient vector of length `N`.
"""
function _gaussian_coeffs(N::Int, x0::Real, p0::Real, σ::Real)
    f = [exp(-(j - x0)^2 / (2σ^2)) * exp(im * p0 * j) for j in 1:N]
    f ./= norm(f)
    return f
end


# Gaussian wavepacket via block-matrix MPS

"""
    soliton_wavepacket(; H, ψ_vac0, ψ_vac1, N, d, a, m, β, x_K, p0, σ, χ, verbosity)
        → (; psi, H_finite, M_K, ϕ_L, ϕ_R)

Build a Gaussian soliton wavepacket on `N` sites using the block-matrix MPS
construction of Milsted et al. (arXiv:2012.07243)

    |ψ⟩ ∝ Σⱼ f_j |soliton_j⟩,    f_j = exp(−(j−x_K)²/2σ²) exp(ip₀j),

where |soliton_j⟩ has the QP B-tensor (at fixed central momentum p₀) placed at
site j.  The sum is encoded in a 2×2 block-matrix MPS with bond dimension
D₀+D₁, where D₀ and D₁ are the bond dimensions of the left and right vacua.
The tail left of the B-tensor uses the left-gauge tensor of the left vacuum,
the tail right of it the right-gauge tensor of the right vacuum (matching
the quasiparticle-ansatz convention |soliton_j⟩ = A_L^{j−1} B A_R^{N−j}).
The B-tensor is reflection-symmetrically gauge-fixed.

The returned `H_finite` carries ghost-site boundary terms matched to the two
vacua (ϕ_L ≈ 0, ϕ_R ≈ 2π/β), so the chain ends exert no mean force on the
topological state — see `sine_gordon_finite_mpo`.

Arguments
- `H`       : `InfiniteMPOHamiltonian` (from `sine_gordon_mpo`)
- `ψ_vac0`  : left vacuum (⟨φ⟩ ≈ 0)
- `ψ_vac1`  : right vacuum (⟨φ⟩ ≈ 2π/β)
- `N`       : chain length
- `x_K`     : wavepacket centre (lattice units)
- `p0`      : central momentum
- `σ`       : spatial width of the Gaussian envelope (lattice units)
- `χ`       : max bond dimension after SVD compression
- `d`, `a`, `m`, `β` : model parameters (needed for `H_finite`)
- `verbosity`: output level

Returns
Named tuple `(; psi, H_finite, M_K, ϕ_L, ϕ_R)`:
- `psi`      : compressed `FiniteMPS`
- `H_finite` : `FiniteMPOHamiltonian` on `N` sites (ghost values ϕ_L, ϕ_R)
- `M_K`      : soliton rest mass E(p=0)
- `ϕ_L`, `ϕ_R` : ghost boundary values used in `H_finite` — pass these to
  `run_scattering` so the energy-density measurement matches the Hamiltonian
"""
function soliton_wavepacket(;
    H::InfiniteMPOHamiltonian,
    ψ_vac0::InfiniteMPS,
    ψ_vac1::InfiniteMPS,
    N::Int         = 120,
    d::Int         = 30,
    a::Real        = 1.0,
    m::Real        = 1.0,
    β::Real        = 2.0,
    x_K::Real      = N / 3,
    p0::Real       = 0.1,
    σ::Real        = 8.0,
    χ::Int         = 64,
    verbosity::Int = 1,
)
    verbosity > 0 && vprint("  Computing soliton B-tensor at p = [0.0, +$p0] ...")
    Es_K, qps_K = find_soliton_excitation(H, ψ_vac0, ψ_vac1;
                                        momenta=[0.0, p0], num=1, verbosity=0)
    Es_K_vec    = real.(Es_K[:,1])
    qps_K_vec   = qps_K[:,1]
    verbosity > 0 && vprint(@sprintf("  M_soliton(p=%.3f) = %.6f", 0.0, Es_K_vec[1]))
    verbosity > 0 && vprint(@sprintf("  E_soliton(p=%.3f) = %.6f", p0,  Es_K_vec[2]))

    # ── 2. Extract MPS tensors ────────────────────────────────────────────────
    qp_soliton = qps_K_vec[2]

    AL  = ψ_vac0.AL[1]           # A_L : left-gauge true vacuum
    AR  = ψ_vac1.AR[1]           # A_R : right-gauge false vacuum

    B_K = qp_soliton[1]          # soliton B-tensor  (V_vac0⊗P ← aux⊗V_vac1)

    fuser_K = isomorphism(storagetype(B_K), fuse(domain(B_K)), domain(B_K))
    B_K_f   = B_K * fuser_K'

    AL_arr  = convert(Array, AL)        # (D₀, d_phys, D₀)
    AR_arr  = convert(Array, AR)        # (D₁, d_phys, D₁)
    BK_arr  = convert(Array, B_K_f)     # (D₀, d_phys, D₁)

    D0     = size(AL_arr, 1)
    D1     = size(AR_arr, 1)
    DB_K   = size(BK_arr, 3)
    d_phys = size(AL_arr, 2)

    @assert DB_K == D1 "Expected fused soliton domain = D₁=$D1, got $DB_K " *
                        "(aux dim > 1 is not supported — use num=1)"

    D_total = D0 + D1

    verbosity > 0 && vprint(@sprintf("  Block-matrix bond dim: %d  (D₀=%d, D₁=%d)",
                              D_total, D0, D1))

    # ── 3. Gauge-fix B-tensor (reflection-symmetric gauge, SGF) ──────────────
    verbosity > 0 && vprint("  Gauge-fixing soliton B-tensor (SGF) ...")
    BK_arr = _symmetrize_B(BK_arr, AL_arr, AR_arr, +p0)
    verbosity > 0 && vprint("    done")

    # ── 4. Gaussian envelope coefficients ─────────────────────────────────────
    f_soliton = _gaussian_coeffs(N, x_K, +p0, σ)   # soliton (moving right)

    # ── 5. Build block-matrix MPS tensors ─────────────────────────────────────
    #
    #  Block layout (rows = left virtual, cols = right virtual):
    #
    #    ┌──────────┬──────────────┐
    #    │  A_L     │ f_j B_K      │
    #    │ (D₀×D₀)  │  (D₀×D₁)     │
    #    ├──────────┼──────────────┤
    #    │    0     │  A_R (D₁×D₁) │
    #    └──────────┴──────────────┘
    #
    #  Left boundary:  project onto first  D₀ rows  (true vacuum input)
    #  Right boundary: project onto last   D₁ cols  (false vacuum output)
    #
    #  A_R is the right-gauge tensor of vacuum 1, so closing the right
    #  boundary leg with the identity reproduces the vacuum tail.
    V_phys  = codomain(AL)[2]
    V_block = ℂ^D_total
    V_L     = ℂ^D0   # left boundary space
    V_R     = ℂ^D1   # right boundary space

    tensors = Vector{typeof(AL)}(undef, N)
    for j in 1:N
        block = zeros(ComplexF64, D_total, d_phys, D_total)

        for s in 1:d_phys
            # Block (1,1): A_L  — propagates true vacuum from the left
            block[1:D0, s, 1:D0]               .= AL_arr[:, s, :]
            # Block (1,2): f_j * B_K  — injects soliton (vac0 → vac1)
            block[1:D0, s, D0+1:D_total]        .= f_soliton[j] .* BK_arr[:, s, :]
            # Block (2,2): A_R  — propagates false vacuum to the right
            block[D0+1:D_total, s, D0+1:D_total] .= AR_arr[:, s, :]
        end

        if j == 1
            # Left boundary: select first D₀ rows
            tensors[j] = TensorMap(block[1:D0, :, :], V_L ⊗ V_phys ← V_block)
        elseif j == N
            # Right boundary: select last D₁ columns (identity closure of the
            # right-gauge vacuum-1 tail — exact, see block-layout note above)
            tensors[j] = TensorMap(block[:, :, D0+1:D_total],
                                    V_block ⊗ V_phys ← V_R)
        else
            tensors[j] = TensorMap(block, V_block ⊗ V_phys ← V_block)
        end
    end

    ψ = FiniteMPS(tensors)

    # ── 6. SVD compress to target bond dimension χ ────────────────────────────
    verbosity > 0 && vprint("  SVD compressing to χ = $χ ...")
    changebonds!(ψ, SvdCut(; trscheme=truncrank(χ)))

    # ── 7. Build finite Hamiltonian for time evolution ────────────────────────
    ϕ_L = vacuum_phi(ψ_vac0, d)
    ϕ_R = vacuum_phi(ψ_vac1, d)
    H_finite = sine_gordon_finite_mpo(N; d, a, m, β, ϕ_L, ϕ_R) # Include ghost field boundary  values

    verbosity > 0 && vprint(@sprintf("  ⟨φ⟩_soliton ≈ %.4f",
                              real(expectation_value(ψ, (round(Int, x_K),) => fock_phi(d)))))
    verbosity > 0 && vprint(@sprintf("  ⟨H⟩ = %.8f",
                              real(sum(expectation_value(ψ, H_finite)))))
    return (; psi=ψ, H_finite, M_K=Es_K_vec[1], ϕ_L, ϕ_R)
end

"""
    antisoliton_wavepacket(; H, ψ_vac0, ψ_vac1, N, d, a, m, β, x_AK, p0, σ, χ, verbosity)
        → (; psi, H_finite, M_K, ϕ_L, ϕ_R)

Gaussian anti-soliton wavepacket. Delegates to `soliton_wavepacket` with the two
vacua swapped: the anti-soliton interpolates from vacuum 1 (left, ⟨φ⟩ ≈ 2π/β)
to vacuum 0 (right, ⟨φ⟩ ≈ 0).  All parameters have the same meaning as in
`soliton_wavepacket`; `x_AK` sets the wavepacket centre.

Arguments
- `H`         : infinite Hamiltonian from `sine_gordon_mpo`
- `ψ_vac0`    : vacuum 0, used as the right vacuum
- `ψ_vac1`    : vacuum 1, used as the left vacuum
- `N`         : chain length
- `d`, `a`, `m`, `β` : model parameters
- `x_AK`      : anti-soliton centre position
- `p0`        : central momentum
- `σ`         : Gaussian width in lattice units
- `χ`         : maximum bond dimension after compression
- `verbosity` : output level

Returns
Named tuple `(; psi, H_finite, M_K, ϕ_L, ϕ_R)` as in `soliton_wavepacket`.
"""
function antisoliton_wavepacket(;
    H::InfiniteMPOHamiltonian,
    ψ_vac0::InfiniteMPS,
    ψ_vac1::InfiniteMPS,
    N::Int         = 120,
    d::Int         = 30,
    a::Real        = 1.0,
    m::Real        = 1.0,
    β::Real        = 2.0,
    x_AK::Real     = N / 3,
    p0::Real       = 0.1,
    σ::Real        = 8.0,
    χ::Int         = 64,
    verbosity::Int = 1,
)
    return soliton_wavepacket(;
        H, ψ_vac0=ψ_vac1, ψ_vac1=ψ_vac0,
        N, d, a, m, β,
        x_K=x_AK, p0, σ, χ, verbosity,
    )
end

# Two-particle soliton–anti-soliton wavepacket (paper Eq. A34 + block matrix)

"""
    soliton_antisoliton_wavepacket(; <keyword arguments>) → NamedTuple

Construct a quantum soliton–anti-soliton two-particle wavepacket following the
approach of Milsted et al. (arXiv:2012.07243, §A.4).

The state is built using the block-matrix MPS construction, which embeds the
two-particle sum over all soliton/anti-soliton positions into a single MPS with
bond dimension `2D₀ + D₁`, where `D₀` and `D₁` are the bond dimensions of
the true and false vacuum infinite-MPS respectively.

The 3×3 block matrix at each site `j` is (Eq. A28 generalized):

    A_j^s = [ A_L^s    f_j B_K^s    0            ]
            [   0       A_C^s       g_j B_AK^s   ]
            [   0         0          A_R^s       ]

where:
  - `A_L`  = left-gauge true vacuum       (vac0, ⟨φ⟩ ≈ 0)
  - `A_C`  = left-gauge false vacuum      (vac1, ⟨φ⟩ ≈ 2π/β)  ["centre" vacuum]
  - `A_R`  = right-gauge true vacuum
  - `B_K`  = soliton excitation tensor        (vac0 → vac1)
  - `B_AK` = anti-soliton excitation tensor   (vac1 → vac0)
  - `f_j`  = Gaussian soliton envelope        exp(ip₀j) exp(−(j−x_K)²/2σ²)
  - `g_j`  = Gaussian anti-soliton envelope   exp(−ip₀j) exp(−(j−x_AK)²/2σ²)

The left boundary projects onto the first `D₀` rows (true vacuum) and the
right boundary onto the last `D₀` columns.  

Arguments
- `H::InfiniteMPOHamiltonian` : Hamiltonian MPO (from `sine_gordon_mpo`)
- `ψ_vac0`   : InfiniteMPS for vacuum 0  (⟨φ⟩ ≈ 0)
- `ψ_vac1`   : InfiniteMPS for vacuum 1  (⟨φ⟩ ≈ 2π/β)
- `N`        : chain length
- `x_K`      : soliton centre position
- `x_AK`     : anti-soliton centre position (must satisfy x_AK > x_K)
- `p0`       : magnitude of wavepacket momentum (soliton gets +p0, antisoliton −p0)
- `σ`        : spatial width of each Gaussian envelope (lattice units)
- `χ`        : max bond dimension after SVD compression
- `d`, `a`, `m`, `β` : model parameters (needed for H_finite)
- `verbosity`: output level

Returns
Named tuple with:
- `psi`      : compressed `FiniteMPS`
- `H_finite` : `FiniteMPOHamiltonian` on `N` sites
"""
function soliton_antisoliton_wavepacket(;
    H::InfiniteMPOHamiltonian,
    ψ_vac0::InfiniteMPS,
    ψ_vac1::InfiniteMPS,
    N::Int      = 120,
    d::Int      = 30,
    a::Real     = 1.0,
    m::Real     = 1.0,
    β::Real     = 2.0,
    x_K::Real   = N / 3,
    x_AK::Real  = 2N / 3,
    p0::Real    = 0.1,
    σ::Real     = 8.0,
    χ::Int      = 64,
    verbosity::Int = 1,
)
    @assert x_AK > x_K "anti-soliton must be to the right of soliton (x_AK > x_K)"

    # ── 1. Compute soliton and anti-soliton excitation tensors ──────────────────────
    verbosity > 0 && vprint("  Computing soliton B-tensor at p = +$p0 ...")
    Es_K, qps_K = find_soliton_excitation(H, ψ_vac0, ψ_vac1;
                                        momenta=[p0], num=1, verbosity=0)
    verbosity > 0 && vprint(@sprintf("  E_soliton(p=%.3f) = %.6f", p0, real(Es_K[1,1])))

    verbosity > 0 && vprint("  Computing antisoliton B-tensor at p = -$p0 ...")
    Es_AK, qps_AK = find_antisoliton_excitation(H, ψ_vac1, ψ_vac0;
                                              momenta=[-p0], num=1, verbosity=0)
    verbosity > 0 && vprint(@sprintf("  E_antisoliton(p=%.3f) = %.6f", -p0, real(Es_AK[1,1])))

    # ── 2. Extract MPS tensors ────────────────────────────────────────────────
    qp_soliton  = qps_K[1, 1]
    qp_asoliton = qps_AK[1, 1]

    AL  = ψ_vac0.AL[1]           # A_L : left-gauge true vacuum
    AR  = ψ_vac0.AR[1]           # A_R : right-gauge true vacuum
    AC  = ψ_vac1.AL[1]           # A_C : left-gauge false ("centre") vacuum

    B_K  = qp_soliton[1]            # soliton B-tensor     (V_vac0⊗P ← aux⊗V_vac1)
    B_AK = qp_asoliton[1]           # antisoliton B-tensor  (V_vac1⊗P ← aux⊗V_vac0)

    # Fuse auxiliary leg into right-virtual leg.
    fuser_K  = isomorphism(storagetype(B_K),  fuse(domain(B_K)),  domain(B_K))
    fuser_AK = isomorphism(storagetype(B_AK), fuse(domain(B_AK)), domain(B_AK))
    B_K_f  = B_K  * fuser_K'
    B_AK_f = B_AK * fuser_AK'

    # Convert TensorKit TensorMaps to plain dense arrays for block assembly.
    AL_arr   = convert(Array, AL)       # (D₀, d_phys, D₀)
    AR_arr   = convert(Array, AR)       # (D₀, d_phys, D₀)
    AC_arr   = convert(Array, AC)       # (D₁, d_phys, D₁)
    BK_arr   = convert(Array, B_K_f)    # (D₀, d_phys, DB_K)
    BAK_arr  = convert(Array, B_AK_f)   # (D₁, d_phys, DB_AK)

    D0      = size(AL_arr, 1)      # true-vacuum bond dim
    D1      = size(AC_arr, 1)      # false-vacuum bond dim
    DB_K    = size(BK_arr,  3)     # fused soliton domain dim   (= D₁ when aux=ℂ^1)
    DB_AK   = size(BAK_arr, 3)     # fused antisoliton domain   (= D₀ when aux=ℂ^1)
    d_phys  = size(AL_arr, 2)

    @assert DB_K == D1 "Expected fused soliton domain = D₁=$D1, got $DB_K " *
                        "(aux dim > 1 is not supported — use num=1)"
    @assert DB_AK == D0 "Expected fused antisoliton domain = D₀=$D0, got $DB_AK"

    D_total = D0 + D1 + D0         # 2D₀ + D₁ (three row/column blocks)

    verbosity > 0 && vprint(@sprintf("  Block-matrix bond dim: %d  (D₀=%d, D₁=%d)",
                              D_total, D0, D1))

    # ── 3b. Reflection-symmetric gauge-fixing (Milsted et al. Eq. A17) ───────
    verbosity > 0 && vprint("  Gauge-fixing soliton B-tensor (SGF) ...")
    BK_arr  = _symmetrize_B(BK_arr,  AL_arr, AC_arr, +p0)
    verbosity > 0 && vprint("    done")

    verbosity > 0 && vprint("  Gauge-fixing antisoliton B-tensor (SGF) ...")
    BAK_arr = _symmetrize_B(BAK_arr, AC_arr, AR_arr, -p0)
    verbosity > 0 && vprint("    done")

    # ── 4. Gaussian envelope coefficients ─────────────────────────────────────
    f_soliton  = _gaussian_coeffs(N, x_K,  +p0, σ)   # soliton (moving right)
    g_asoliton = _gaussian_coeffs(N, x_AK, -p0, σ)   # anti-soliton (moving left)

    # ── 5. Build block-matrix MPS tensors ─────────────────────────────────────
    #
    #  Block layout (rows = left virtual, cols = right virtual):
    #
    #    ┌──────────┬──────────────┬────────────┐
    #    │  A_L     │ f_j B_K      │     0      │
    #    │ (D₀×D₀)  │  (D₀×D₁)     │            │
    #    ├──────────┼──────────────┼────────────┤
    #    │    0     │  A_C (D₁×D₁) │ g_j B_AK   │
    #    │          │              │  (D₁×D₀)   │
    #    ├──────────┼──────────────┼────────────┤
    #    │    0     │     0        │ A_R (D₀×D₀)│
    #    └──────────┴──────────────┴────────────┘
    #
    #  Left boundary:  project onto first  D₀ rows  (true vacuum input)
    #  Right boundary: project onto last   D₀ cols  (true vacuum output)

    # Wrap assembled block arrays back into TensorMaps for FiniteMPS.
    V_phys = codomain(AL)[2]
    V_block = ℂ^D_total
    V_L = ℂ^D0   # left boundary space
    V_R = ℂ^D0   # right boundary space

    tensors = Vector{typeof(AL)}(undef, N)
    for j in 1:N
        block = zeros(ComplexF64, D_total, d_phys, D_total)

        for s in 1:d_phys
            # Block (1,1): A_L  — propagates true vacuum from the left
            block[1:D0, s, 1:D0]     .= AL_arr[:, s, :]
            # Block (1,2): f_j * B_K  — injects soliton (vac0 → vac1)
            block[1:D0, s, D0+1:D0+D1] .= f_soliton[j] .* BK_arr[:, s, :]
            # Block (2,2): A_C  — propagates false vacuum in the centre
            block[D0+1:D0+D1, s, D0+1:D0+D1] .= AC_arr[:, s, :]
            # Block (2,3): g_j * B_AK  — injects anti-soliton (vac1 → vac0)
            block[D0+1:D0+D1, s, D0+D1+1:D_total] .= g_asoliton[j] .* BAK_arr[:, s, :]
            # Block (3,3): A_R  — propagates true vacuum to the right
            block[D0+D1+1:D_total, s, D0+D1+1:D_total] .= AR_arr[:, s, :]
        end

        if j == 1
            # Left boundary: select first D₀ rows
            tensors[j] = TensorMap(block[1:D0, :, :], V_L ⊗ V_phys ← V_block)
        elseif j == N
            # Right boundary: select last D₀ columns
            tensors[j] = TensorMap(block[:, :, D0+D1+1:D_total],
                                    V_block ⊗ V_phys ← V_R)
        else
            tensors[j] = TensorMap(block, V_block ⊗ V_phys ← V_block)
        end
    end

    ψ = FiniteMPS(tensors)

    # ── 6. SVD compress to target bond dimension χ ────────────────────────────
    # The block-matrix MPS has bond dim 2D₀ + D₁ (e.g. 60 for D₀=D₁=20).
    # SVD truncation compresses each bond to at most χ, introducing a small
    # controlled truncation error.
    verbosity > 0 && vprint("  SVD compressing to χ = $χ ...")
    changebonds!(ψ, SvdCut(; trscheme=truncrank(χ)))

    # ── 7. Build finite Hamiltonian for time evolution ────────────────────────
    H_finite = sine_gordon_finite_mpo(N; d, a, m, β) # Use default vacuum ghost field values

    verbosity > 0 && vprint(@sprintf("  ⟨φ⟩_soliton ≈ %.4f,   ⟨φ⟩_asoliton ≈ %.4f",
                              real(expectation_value(ψ, (round(Int, x_K),) => fock_phi(d))),
                              real(expectation_value(ψ, (round(Int, x_AK),) => fock_phi(d)))))
    verbosity > 0 && vprint(@sprintf("  ⟨H⟩ = %.8f",
                              real(sum(expectation_value(ψ, H_finite)))))

    return (; psi=ψ, H_finite)
end
