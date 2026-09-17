using Dates
using DelimitedFiles
using Printf
using JLD2

const DATA_DIR = joinpath(@__DIR__, "..", "data")
mkpath(DATA_DIR)


# Environment-variable parameters
#
# The run scripts declare every parameter as `env_get("NAME", default)` with the
# previous hard-coded value as the default, so a job is fully specified at submit
# time and needs no source edit:
#
#   sbatch --export=ALL,A_VAL=0.75,D_VAL=30,CHI=80 scripts/run_soliton_free.slurm

"""
    _env_raw(name)

Read and strip an environment variable.

Arguments
- `name` : environment variable name

Returns
Nonblank string value, or `nothing` when unset or blank.
"""
_env_raw(name::AbstractString) =
    (s = strip(get(ENV, name, "")); isempty(s) ? nothing : String(s))

"""
    env_get(name, default) → value

Value of the environment variable `name`, parsed to the type of `default`;
`default` when the variable is unset or empty.  Booleans accept `1`/`0`,
`true`/`false`, `yes`/`no`, `on`/`off` (case-insensitive).

A value that does not parse raises rather than falling back to `default`.

Arguments
- `name`    : environment variable name
- `default` : value used when the variable is unset or blank; its type selects the parser

Returns
Parsed value, or `default` when the variable is unset or blank.
"""
function env_get end

env_get(name::AbstractString, default::AbstractString) = something(_env_raw(name), default)

function env_get(name::AbstractString, default::Bool)
    s = _env_raw(name)
    s === nothing && return default
    ls = lowercase(s)
    ls in ("1", "true", "yes", "on")  && return true
    ls in ("0", "false", "no", "off") && return false
    error("env_get: $name = \"$s\" is not a boolean (use 1/0, true/false, yes/no, on/off)")
end

function env_get(name::AbstractString, default::Integer)
    s = _env_raw(name)
    s === nothing && return Int(default)
    v = tryparse(Int, s)
    v === nothing && error("env_get: $name = \"$s\" is not an integer")
    return v
end

function env_get(name::AbstractString, default::Real)
    s = _env_raw(name)
    s === nothing && return Float64(default)
    v = tryparse(Float64, s)
    v === nothing && error("env_get: $name = \"$s\" is not a number")
    return v
end


# Logging

const _LOG_IO = Ref{IOStream}()

function init_log(prefix::AbstractString)
  log_dir = joinpath(@__DIR__, "..", "logs")
  mkpath(log_dir)
  path = joinpath(log_dir, "$(prefix)_$(Dates.format(now(), "yyyymmdd_HHMMSS")).log")
  _LOG_IO[] = open(path, "w")
  println(_LOG_IO[], "Log file: $path")
  println(_LOG_IO[], "Started:  $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
  flush(_LOG_IO[])
  return path
end

function close_log()
  if isassigned(_LOG_IO) && isopen(_LOG_IO[])
    println(_LOG_IO[], "Finished: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    close(_LOG_IO[])
  end
end

function tprint(msg::AbstractString="")
  isassigned(_LOG_IO) || return
  println(_LOG_IO[], msg)
  flush(_LOG_IO[])
end
tprint(args...) = tprint(string(args...))

function vprint(msg::AbstractString="")
  tprint(msg)
  println(msg)
  flush(stdout)
end
vprint(args...) = vprint(string(args...))

"""
    vcapture(f)

Capture and log the output of `f()` to both the console and the log file.

Arguments
- `f` : zero-argument function to execute

Returns
`nothing` after logging the captured output.
"""
function vcapture(f)
    pipe = Pipe()
    Base.link_pipe!(pipe; reader_supports_async=true, writer_supports_async=true)
    reader = @async read(pipe.out, String)
    redirect_stdout(pipe.in) do
        f()
        flush(stdout)
    end
    close(pipe.in)
    s = fetch(reader)
    close(pipe.out)
    vprint(s)          # already writes to both stdout and _LOG_IO[]
end

function timestamp(label::AbstractString)
  vprint("[$label]  $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
end


# Data I/O

function save_data(filename, header, data; dir::AbstractString=DATA_DIR)
  path = joinpath(dir, filename)
  open(path, "w") do io
    println(io, "# ", join(header, "\t"))
    writedlm(io, data, '\t')
  end
  println("  ✓ $path")
  return path
end

load_data(filename) = readdlm(joinpath(DATA_DIR, filename), '\t', Float64; comments=true)

data_exists(filename) = isfile(joinpath(DATA_DIR, filename))


# Run directory

"""
    resolve_run_dir(parent) → String

Run directory for this job.  `RESUME_DIR`, when set, is taken verbatim (continue
or extend an earlier run)

Arguments
- `parent` : parent directory for a new dated run

Returns
Existing `RESUME_DIR` or a dated run directory under `parent`.
"""
function resolve_run_dir(parent::AbstractString)
  dir = env_get("RESUME_DIR", "")
  isempty(dir) && return make_dated_run_dir(parent)
  isdir(dir) || error("resolve_run_dir: RESUME_DIR=$dir is not an existing directory")
  return dir
end

"""
    make_dated_run_dir(parent) → String

Create (if needed) and return a date-stamped subdirectory of `parent`,
named `YYYYMMDD`.  If a different SLURM job already owns that folder,
a suffix `_2`, `_3`, … is appended until a free slot is found.  Tasks
from the same SLURM array always resolve to the same folder.

Arguments
- `parent` : parent directory for the dated run

Returns
Path of the created or reused run directory.
"""
function make_dated_run_dir(parent::AbstractString)
  mkpath(parent)
  datestr = Dates.format(now(), "yyyymmdd")
  job_id  = get(ENV, "SLURM_ARRAY_JOB_ID", get(ENV, "SLURM_JOB_ID", ""))

  # Claim by job id first, ignoring the date (see docstring).
  if !isempty(job_id)
    for name in sort(readdir(parent))
      dir      = joinpath(parent, name)
      sentinel = joinpath(dir, ".slurm_job_id")
      isdir(dir) && isfile(sentinel) &&
        strip(read(sentinel, String)) == job_id && return dir
    end
  end

  for n in 1:99
    name     = n == 1 ? datestr : "$(datestr)_$(n)"
    dir      = joinpath(parent, name)
    sentinel = joinpath(dir, ".slurm_job_id")

    if !isdir(dir)
      mkpath(dir)
      isempty(job_id) || write(sentinel, job_id)
      return dir
    end

    # Folder exists — local run: reuse the first folder unconditionally
    isempty(job_id) && return dir

    # SLURM run: check whether this folder belongs to our job
    if isfile(sentinel)
      strip(read(sentinel, String)) == job_id && return dir
      # Different job owns this slot — try the next one
      continue
    end

    # No sentinel yet (e.g. created by a local run); claim it
    write(sentinel, job_id)
    return dir
  end

  error("make_dated_run_dir: too many same-day run directories under $parent")
end


# MPS serialization + TDVP resume checkpoints

"""
    mps_to_dicts(ψ) → Vector{Dict}

Serialize a finite MPS to one `Dict` per site via TensorKit's `convert(Dict, ·)`.
The state is captured in mixed canonical gauge with the orthogonality centre on
site 1 (`AC[1]` carries the norm); the remaining sites are the right-canonical
`AR` tensors. 

Arguments
- `ψ` : finite MPS to serialize

Returns
Vector of per-site tensor dictionaries.
"""
function mps_to_dicts(ψ)
    As = [ψ.AC[1]; [ψ.AR[i] for i in 2:length(ψ)]]
    return [convert(Dict, A) for A in As]
end

"""
    dicts_to_mps(dicts) → FiniteMPS

Inverse of [`mps_to_dicts`](@ref): rebuild a `FiniteMPS` from per-site `Dict`s.
`FiniteMPS(::Vector)` re-gauges the (generally non-canonical) tensor list.

Arguments
- `dicts` : per-site tensor dictionaries from `mps_to_dicts`

Returns
Reconstructed `FiniteMPS`.
"""
dicts_to_mps(dicts) = FiniteMPS([convert(TensorMap, d) for d in dicts]; overwrite = true)

"""
    save_state_checkpoint(path; ψ, step, mpo_params, history, extras)

Atomically write a TDVP resume checkpoint to `path` (a `.jld2` file).

Stored keys:
  - `site_tensors` : `mps_to_dicts(ψ)` — the evolved state
  - `step`         : TDVP step index of this snapshot (`Int`)
  - `mpo_params`   : `NamedTuple` (N, d, a, m, β, ϕ_L, ϕ_R) — enough to rebuild
                     `H_finite` exactly via `sine_gordon_finite_mpo`
  - `history`      : `NamedTuple` of accumulated measurement arrays
  - `extras`       : `NamedTuple` of static run artifacts (e.g. ε_vac, M_K)

Arguments
- `path`       : destination `.jld2` path
- `ψ`          : finite MPS state
- `step`       : TDVP step index
- `mpo_params` : parameters for rebuilding the finite Hamiltonian
- `history`    : accumulated measurements
- `extras`     : static run artifacts

Returns
Checkpoint path.
"""
function save_state_checkpoint(path; ψ, step, mpo_params, history, extras)
    # Create tmp file
    tmp = path * ".tmp"
    jldsave(tmp; site_tensors = mps_to_dicts(ψ), step, mpo_params, history, extras)
    # Atomic move
    mv(tmp, path; force = true)
    return path
end

"""
    load_state_checkpoint(path) → NamedTuple

Read a checkpoint written by [`save_state_checkpoint`](@ref).  Returns
`(; psi, step, mpo_params, history, extras)` with `psi` already rebuilt as a
`FiniteMPS`.

Arguments
- `path` : checkpoint `.jld2` path

Returns
Named tuple `(; psi, step, mpo_params, history, extras)`.
"""
function load_state_checkpoint(path)
    data = load(path)
    return (; psi        = dicts_to_mps(data["site_tensors"]),
              step       = data["step"],
              mpo_params = data["mpo_params"],
              history    = data["history"],
              extras     = data["extras"])
end

"""
    truncate_ckpt_tsv(path, t_max; atol=1e-9)

Rewrite an append-mode `ckpt_*.tsv` log in place, keeping the header (`#…`) and
only the data rows whose leading time column is ≤ `t_max + atol`.  Used on resume
to drop measurement rows written after the last state checkpoint, since those
steps will be recomputed as evolution continues.  No-op if the file is absent.

Arguments
- `path`  : checkpoint TSV path
- `t_max` : last time to retain
- `atol`  : time comparison tolerance

Returns
Input path, whether or not the file exists.
"""
function truncate_ckpt_tsv(path, t_max; atol = 1e-9)
    isfile(path) || return path
    kept = String[]
    for ln in readlines(path)
        if isempty(strip(ln)) || startswith(ln, "#")
            push!(kept, ln)
        elseif parse(Float64, first(split(ln, '\t'))) <= t_max + atol
            push!(kept, ln)
        end
    end
    open(path, "w") do io
        for ln in kept
            println(io, ln)
        end
    end
    return path
end
