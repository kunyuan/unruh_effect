using ITensors
let
    N = 10
    sites = siteinds("S=1/2", N;conserve_qns=true)

    ampo = OpSum()
    for j = 1:N - 1
        ampo += "Sz", j, "Sz", j + 1
        ampo += 1 / 2, "S+", j, "S-", j + 1
        ampo += 1 / 2, "S-", j, "S+", j + 1
    end
    H = MPO(ampo, sites)

    state = [isodd(n) ? "Up" : "Dn" for n = 1:N]
  psi0 = productMPS(sites, state)
  @show flux(psi0)

  sweeps = Sweeps(5)
  setmaxdim!(sweeps, 10, 20, 100, 100, 200)
  setcutoff!(sweeps, 1E-10)

  energy, psi = dmrg(H, psi0, sweeps)

  return
end