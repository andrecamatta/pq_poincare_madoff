# pq_poincare_madoff

Código complementar ao artigo **"O formato da curva: distribuições truncadas, de Poincaré ao caso Madoff"** — [Pílulas de Quant](https://pilulasdequant.substack.com) (André Camatta).

## Conteúdo

O script [`analise_truncada.jl`](analise_truncada.jl) reproduz os cinco exercícios numéricos discutidos no artigo:

1. **Parábola do padeiro.** Simula a amostra original *N*(950, 50²) e a amostra truncada à esquerda que um Poincaré-fictício receberia de um padeiro desonesto que só entrega os pães mais pesados da fornada. Aplica o teste de Shapiro-Wilk aos dois casos e confronta os momentos empíricos com as fórmulas fechadas da normal truncada (Johnson-Kotz-Balakrishnan, 1994).

2. **Caso Madoff — Fairfield Sentry.** Calcula o Sharpe anualizado e a probabilidade binomial de observar apenas 10 meses negativos em 215 sob a própria distribuição reportada pelo fundo, e sob uma volatilidade honesta para *split-strike conversion*. Confronta com o bound teórico de Bernard e Boyle (2009).

3. **Testes de formato.** Compara uma série simulada "honesta" (calibrada em volatilidade realista) com uma série "maquiada" (calibrada nas estatísticas reportadas por Madoff e com apenas 10 meses negativos pequenos). Aplica Jarque-Bera, Shapiro-Wilk e Anderson-Darling.

4. **MLE corrigido de Cohen (1950).** Resolve o problema de máxima verossimilhança para amostra truncada à esquerda, usando a log-verossimilhança corrigida com o termo `-n·log(1-Φ(α))`. Compara com o estimador ingênuo (média e desvio padrão amostrais).

5. **Viés de σ por circuit breaker.** Monte Carlo de 5.000 retornos diários com volatilidade verdadeira 4,5 % a.d. e regra de halt em −10 %. Mede o viés nas duas estratégias ingênuas (usar os dados censurados como se fossem crus; deletar os dias de halt). Repete sob cauda gorda (t-Student com 4 graus de liberdade), reproduzindo a ordem de grandeza dos resultados de Hsieh e Yang (2009).

## Reprodução

Requer Julia 1.10+.

```julia
cd("codigo")
using Pkg
Pkg.activate(".")
Pkg.instantiate()
include("analise_truncada.jl")
```

A saída completa da última execução está em [`resultados.txt`](resultados.txt).

## Pacotes utilizados

- `Distributions.jl` — distribuições, inclusive `truncated(Normal(μ,σ); lower=a, upper=b)`
- `HypothesisTests.jl` — `ShapiroWilkTest`, `JarqueBeraTest`, `OneSampleADTest`
- `StatsBase.jl` — `skewness`, `kurtosis`
- `Optim.jl` — otimização do MLE corrigido
- `Random`, `Statistics`, `Printf` — base

## Licença

MIT.
