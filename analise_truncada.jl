#=
analise_truncada.jl

Script de apoio para o artigo "O formato da curva: distribuições truncadas,
de Poincaré ao caso Madoff" — Pílulas de Quant.

Exercícios numéricos:
 1. Parábola do padeiro: simular distribuição original N(950, 50²) vs. truncada.
 2. Caso Madoff: estatísticas de Fairfield Sentry + probabilidade binomial.
 3. Testes de formato aplicados à série Madoff.
 4. MLE corrigido de Cohen (1950) para normal truncada.
 5. Viés por circuit breaker: volatilidade estimada em série censurada.
=#

using Random
using Statistics
using Distributions
using HypothesisTests
using StatsBase
using Optim
using Printf

Random.seed!(42)

println("="^68)
println("Artigo PQ: distribuições truncadas — Poincaré e Madoff")
println("="^68)

# ============================================================
# 1. Parábola do padeiro
# ============================================================
println("\n[1] Parábola do padeiro")
println("-"^40)

μ_real, σ_real = 950.0, 50.0
N = 365
c = 1000.0  # limiar de seleção: padeiro entrega só pães ≥ 1 kg

# Ano 1: padeiro entrega pães sem seleção (média 950)
pão_ano1 = rand(Normal(μ_real, σ_real), N)

# Ano 2: padeiro seleciona; Poincaré observa amostra truncada em c
# Simula: gera muitos pães e Poincaré só recebe os ≥ c
pão_produzido = rand(Normal(μ_real, σ_real), 10*N)
pão_ano2 = first(filter(x -> x ≥ c, pão_produzido), N)

@printf("Ano 1 — média observada: %.2f g,  σ: %.2f g\n",
        mean(pão_ano1), std(pão_ano1))
@printf("Ano 2 (truncado) — média: %.2f g,  σ: %.2f g\n",
        mean(pão_ano2), std(pão_ano2))
@printf("Ano 2 — skewness: %.3f,  excess kurtosis: %.3f\n",
        skewness(pão_ano2), kurtosis(pão_ano2))

# Teste Shapiro-Wilk: ano 1 aceita normalidade, ano 2 rejeita
sw1 = ShapiroWilkTest(pão_ano1)
sw2 = ShapiroWilkTest(pão_ano2)
@printf("Shapiro-Wilk ano 1: W=%.4f, p=%.4g\n", sw1.W, pvalue(sw1))
@printf("Shapiro-Wilk ano 2: W=%.4f, p=%.4g\n", sw2.W, pvalue(sw2))

# Momentos teóricos da normal truncada à esquerda em c
# (fórmulas de Johnson-Kotz-Balakrishnan 1994)
α_ = (c - μ_real) / σ_real
λα = pdf(Normal(), α_) / ccdf(Normal(), α_)   # inverse Mills ratio
E_trunc = μ_real + σ_real * λα
V_trunc = σ_real^2 * (1 + α_ * λα - λα^2)
@printf("Momentos teóricos (trunc. esq. em %.0f): μ_T=%.2f, σ_T=%.2f\n",
        c, E_trunc, sqrt(V_trunc))

# ============================================================
# 2. Caso Madoff — Fairfield Sentry (dez/1990 – out/2008)
# ============================================================
println("\n[2] Caso Madoff — Fairfield Sentry")
println("-"^40)

# Estatísticas reportadas (Bernard-Boyle 2009; Bollen-Pool 2012):
N_meses = 215
μ_madoff_m = 0.0084      # 0,84% média mensal
σ_madoff_m = 0.0071      # 0,71% vol mensal
pior_mes  = -0.0064      # -0,64%
neg_obs   = 10           # meses negativos observados

# Sharpe anualizado; a convenção dos papers (Bernard-Boyle 2009;
# Clauss-Roncalli-Weisang 2009) usa Rf ~ 4% a.a. (média T-bill EUA 1990-2008).
Rf_ann = 0.04
μ_excess_m = μ_madoff_m - Rf_ann/12
sharpe_ann = (μ_excess_m / σ_madoff_m) * sqrt(12)
@printf("N = %d meses;  Sharpe anualizado (Rf=4%%): %.2f\n",
        N_meses, sharpe_ann)

# Probabilidade de R<0 sob normal calibrada nos números de Madoff
p_neg_madoff = cdf(Normal(μ_madoff_m, σ_madoff_m), 0.0)
esperado_neg_madoff = p_neg_madoff * N_meses
@printf("Sob N(0.84%%, 0.71%%): P(R<0) = %.3f → esperado %.1f meses negativos\n",
        p_neg_madoff, esperado_neg_madoff)

# Probabilidade binomial de observar ≤10 em 215
p_bin_madoff = cdf(Binomial(N_meses, p_neg_madoff), neg_obs)
@printf("P(≤%d meses neg.) sob a própria distribuição reportada: %.3g\n",
        neg_obs, p_bin_madoff)

# Sob volatilidade "honesta" de split-strike (σ anual ≥ 8% ⇒ σ mensal ≈ 2,3%)
σ_honesto_m = 0.08 / sqrt(12)
p_neg_honesto = cdf(Normal(μ_madoff_m, σ_honesto_m), 0.0)
esperado_neg_honesto = p_neg_honesto * N_meses
p_bin_honesto = cdf(Binomial(N_meses, p_neg_honesto), neg_obs)
@printf("Sob σ honesto (8%% a.a.): P(R<0) = %.3f → esperado %.1f; P(≤10)=%.3g\n",
        p_neg_honesto, esperado_neg_honesto, p_bin_honesto)

# Bound teórico de Bernard-Boyle (2009) para split-strike legítimo: Sharpe ~0,66
sharpe_bb = 0.66
@printf("Bound Bernard-Boyle para split-strike legítimo: Sharpe ≈ %.2f\n", sharpe_bb)
@printf("Observado: %.2f  (≈ %.1fx o teto teórico)\n",
        sharpe_ann, sharpe_ann / sharpe_bb)

# ============================================================
# 3. Testes de formato numa série Madoff sintética
# ============================================================
println("\n[3] Testes de formato (série sintética calibrada)")
println("-"^40)

# Cenário A — série "crua" com volatilidade HONESTA de split-strike (σ anual 8%)
# Reproduz o que Madoff "deveria" ter reportado.
Random.seed!(7)
σ_honesto_m_serie = 0.08 / sqrt(12)
serie_honesta = rand(Normal(μ_madoff_m, σ_honesto_m_serie), 215)

# Cenário B — série "maquiada": média e σ iguais às reportadas por Madoff
# (0,84% / 0,71%) com apenas 10 meses negativos pequenos.
# Construção: truncagem à esquerda em 0 (quase) + massa residual em (-0.64%, 0)
# para bater exatamente com média e σ reportados.
function gera_serie_madoff(n, μ_alvo, σ_alvo, n_neg, pior; seed=9)
    Random.seed!(seed)
    # retornos positivos da cauda direita (N truncada em 0)
    n_pos = n - n_neg
    pos = rand(truncated(Normal(μ_alvo, σ_alvo); lower=0.0), n_pos)
    # pequenos negativos uniformemente entre pior e 0
    neg = rand(Uniform(pior, 0.0), n_neg)
    s = vcat(pos, neg)
    # rescala linearmente para casar média/σ alvo
    s = (s .- mean(s)) ./ std(s) .* σ_alvo .+ μ_alvo
    shuffle!(s)
    return s
end

serie_madoff = gera_serie_madoff(215, μ_madoff_m, σ_madoff_m, 10, pior_mes)

@printf("Série honesta (σ ano 8%%): μ=%.4f, σ=%.4f, skew=%.2f, %% neg=%.1f%%\n",
        mean(serie_honesta), std(serie_honesta),
        skewness(serie_honesta),
        100 * mean(serie_honesta .< 0))
@printf("Série Madoff sintética:  μ=%.4f, σ=%.4f, skew=%.2f, %% neg=%.1f%%\n",
        mean(serie_madoff), std(serie_madoff),
        skewness(serie_madoff),
        100 * mean(serie_madoff .< 0))

# Testes de normalidade
for (nome, s) in [("honesta", serie_honesta), ("Madoff ", serie_madoff)]
    jb = JarqueBeraTest(s)
    sw = ShapiroWilkTest(s)
    ad = OneSampleADTest(s, Normal(mean(s), std(s)))
    @printf("[%s]  JB=%.2f (p=%.3g);  SW W=%.4f (p=%.3g);  AD A²=%.2f (p=%.3g)\n",
            nome, jb.JB, pvalue(jb), sw.W, pvalue(sw), ad.A², pvalue(ad))
end

# ============================================================
# 4. MLE corrigido de Cohen (1950) para normal truncada
# ============================================================
println("\n[4] MLE corrigido de Cohen (1950)")
println("-"^40)

# Amostra truncada à esquerda em c=970 de N(950, 50²)
Random.seed!(11)
c_trunc = 970.0
pool = rand(Normal(μ_real, σ_real), 50_000)
amostra_trunc = first(filter(x -> x ≥ c_trunc, pool), 1000)

μ_ingenuo = mean(amostra_trunc)
σ_ingenuo = std(amostra_trunc)
@printf("Estimativa ingênua (ignora truncagem): μ̂=%.2f, σ̂=%.2f\n",
        μ_ingenuo, σ_ingenuo)

# Log-verossimilhança corrigida: ℓ = -n log σ - Σ(x-μ)²/(2σ²) - n log[1-Φ(α)]
function neg_loglik_trunc(θ, x, c)
    μ, σ = θ[1], exp(θ[2])   # σ > 0 via exp
    n = length(x)
    α = (c - μ) / σ
    logZ = logccdf(Normal(), α)        # log(1 - Φ(α))
    ℓ = -n*log(σ) - sum((x .- μ).^2) / (2σ^2) - n*logZ
    return -ℓ
end

θ0 = [μ_ingenuo, log(σ_ingenuo)]
opt = optimize(θ -> neg_loglik_trunc(θ, amostra_trunc, c_trunc),
               θ0, NelderMead())
μ_cohen = Optim.minimizer(opt)[1]
σ_cohen = exp(Optim.minimizer(opt)[2])
@printf("Estimativa de Cohen (corrigida):       μ̂=%.2f, σ̂=%.2f\n",
        μ_cohen, σ_cohen)
@printf("Verdadeiros parâmetros:                μ =%.2f, σ =%.2f\n",
        μ_real, σ_real)

# ============================================================
# 5. Viés de volatilidade por circuit breaker
# ============================================================
println("\n[5] Viés de σ por circuit breaker (simulação)")
println("-"^40)

# Retornos diários em regime de stress (σ verdadeiro 4,5% a.d.) — calibra no
# perfil observado no Ibovespa em março/2020, quando o halt foi acionado seis
# vezes em oito pregões.
Random.seed!(21)
σ_true = 0.045  # 4,5% a.d.
N_dias = 5_000
r_true = rand(Normal(0.0, σ_true), N_dias)

# Regra B3: halt em -10%; no dia do halt, retorno registrado é exatamente -10%
lim = -0.10
r_obs = clamp.(r_true, lim, Inf)        # censura à esquerda

# Viés tratando dados observados como se fossem verdadeiros
σ_obs   = std(r_obs)
# Ou analista deleta dias de halt
r_dropped = r_obs[r_obs .> lim]
σ_drop   = std(r_dropped)

frac_halt = mean(r_true .< lim)
@printf("Fração de dias de halt: %.2f%%  (σ verdadeiro %.2f%%)\n",
        100*frac_halt, 100*σ_true)
@printf("σ estimado usando dados censurados: %.4f  (viés %.2f%%)\n",
        σ_obs, 100*(σ_obs - σ_true) / σ_true)
@printf("σ estimado deletando dias de halt:  %.4f  (viés %.2f%%)\n",
        σ_drop, 100*(σ_drop - σ_true) / σ_true)

# Sob cauda gorda (t-Student escalada para σ_true), o viés aumenta
t_raw = rand(TDist(4), N_dias)
r_t = t_raw .* (σ_true / std(t_raw))   # força σ amostral = σ_true
r_t_obs = clamp.(r_t, lim, Inf)
r_t_drop = r_t_obs[r_t_obs .> lim]
σ_t_true = std(r_t)
σ_t_obs = std(r_t_obs)
σ_t_drop = std(r_t_drop)
frac_halt_t = mean(r_t .< lim)
@printf("\nCauda gorda (t-Student ν=4): frac halt %.2f%%,  σ verdadeiro %.4f\n",
        100*frac_halt_t, σ_t_true)
@printf("  σ com dados censurados: %.4f  (viés %.2f%%)\n",
        σ_t_obs, 100*(σ_t_obs - σ_t_true)/σ_t_true)
@printf("  σ deletando dias de halt:  %.4f  (viés %.2f%%)\n",
        σ_t_drop, 100*(σ_t_drop - σ_t_true)/σ_t_true)

println("\n" * "="^68)
println("Fim da análise.")
println("="^68)
