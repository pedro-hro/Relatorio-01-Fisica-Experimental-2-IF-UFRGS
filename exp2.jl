import Pkg
Pkg.activate(".")  # Ativa o ambiente no diretório atual

using CSV
using DataFrames
using Plots
using GLM

# Constantes do experimento
L = 2.45     # Comprimento do pêndulo em metros
g = 9.81     # Aceleração da gravidade em m/s²
m = 0.0024   # Massa da bolinha em kg
d = 0.03762  # Diâmetro da bolinha em metros

# Leitura dos dados
data = CSV.read("dados.csv", DataFrame)

# Função para converter tempo no formato "00:00.00" para segundos
function converter_tempo_para_segundos(tempo_str)
    partes_minutos_segundos = split(tempo_str, ":")
    minutos = parse(Int, partes_minutos_segundos[1])
    segundos_centis = split(partes_minutos_segundos[2], ".")
    segundos = parse(Int, segundos_centis[1])
    centesimos = parse(Int, segundos_centis[2])
    total_segundos = minutos * 60 + segundos + centesimos / 100
    return total_segundos
end

# Função para calcular velocidades usando tempos acumulados de travessia do laser
function calculate_velocities_accumulated(traversal_times)
    velocities = Float64[]
    
    if length(traversal_times) < 2
        return velocities  # Retorna um array vazio se não houver elementos suficientes
    end
    
    for i in 2:eachindex(traversal_times)[end]
        # Calcular o tempo de travessia individual
        t = traversal_times[i] - traversal_times[i-1]
        v = d / t  # Velocidade calculada
        push!(velocities, v)
    end
    
    return velocities
end

# Cálculo das velocidades para cada vídeo
results = Dict()

for video in 1:8
    col_video = "Video $video"   # Coluna com tempos acumulados de travessia
    col_tempos = "Tempos $video" # Coluna com tempos individuais
    
    # Filtrar valores não faltantes e converter tempos individuais
    traversal_times = filter(!ismissing, data[!, col_video])
    tempos_individuais = [converter_tempo_para_segundos(t) for t in filter(!ismissing, data[!, col_tempos])]
    tempos_acumulados = cumsum(tempos_individuais)
    
    # Calcular as velocidades individuais
    velocities = calculate_velocities_accumulated(traversal_times)
    results[video] = (tempos_acumulados[2:end], velocities)  # Ajustar o tempo acumulado
end

# Função para obter o título do gráfico
function titles(video)
    if video == 1 || video == 5
        return "653 ± 0,5mm"
    elseif video == 2 || video == 6
        return "489 ± 0,5mm"
    elseif video == 3 || video == 7
        return "326 ± 0,5mm"
    elseif video == 4 || video == 8
        return "162 ± 0,5mm"
    else
        return ""
    end
end

# Função para calcular o Beta e o ln(v0)
function calculate_beta(times, velocities)
    ln_velocities = log.(abs.(velocities))
    model = lm(@formula(Y ~ X), DataFrame(X=times, Y=ln_velocities))
    beta = -coef(model)[2]
    ln_v0 = coef(model)[1]
    return beta, ln_v0
end

# Listas para armazenar os gráficos
velocity_plots = []
ln_velocity_plots = []

# Loop para plotar e salvar gráficos individuais e coletar para gráficos combinados
for video in 1:8
    tempos_acumulados, velocities = results[video]
    
    ## Gráfico de velocidade vs tempo
    pv = plot(tempos_acumulados, velocities,
              title="Lançamento $video - Distância Inicial $(titles(video))",
              xlabel="Tempo Acumulado (s)",
              ylabel="Velocidade (m/s)",
              legend=false,
              marker=:circle,
              markersize=3,
              palette=:Dark2_3)
    
    # Salvar gráfico individual
    savefig(pv, "velocidade_vs_tempo_video_$video.png")
    
    # Adicionar à lista de plots
    push!(velocity_plots, pv)
    
    ## Cálculo do Beta e ln(v0)
    beta, ln_v0 = calculate_beta(tempos_acumulados, velocities)
    
    ## Gráfico de ln|v(t)| vs tempo
    ln_velocities = log.(abs.(velocities))
    p_ln = plot(tempos_acumulados, ln_velocities,
                title="Lançamento $video - Distância Inicial $(titles(video))",
                xlabel="Tempo Acumulado (s)",
                ylabel="ln|Velocidade|",
                legend=true,
                marker=:circle,
                markersize=3,
                label="Dados",
                palette=:Dark2_3)
    
    # Adicionar linha de regressão
    t_range = range(minimum(tempos_acumulados), maximum(tempos_acumulados), length=100)
    ln_v_fit = ln_v0 .- beta .* t_range
    plot!(p_ln, t_range, ln_v_fit,
          line=:solid,
          color=:red,
          label="Regressão: β = $(round(beta, digits=4))")
    
    # Salvar gráfico individual
    savefig(p_ln, "ln_velocidade_vs_tempo_video_$video.png")
    
    # Adicionar à lista de plots
    push!(ln_velocity_plots, p_ln)
end

# Plotar todos os gráficos de velocidade vs tempo juntos
plot(velocity_plots..., layout=(4,2), size=(1366, 1820), dpi=300, palette=:Dark2_3)
savefig("todos_velocidade_vs_tempo.png")

# Plotar todos os gráficos de ln(velocidade) vs tempo juntos
plot(ln_velocity_plots..., layout=(4,2), size=(1366, 1820), dpi =300, palette=:Dark2_3)
savefig("todos_ln_velocidade_vs_tempo.png")

# Imprimir os valores de Beta para cada vídeo
println("Valores de Beta para cada vídeo:")
for video in 1:8
    tempos_acumulados, velocities = results[video]
    beta, _ = calculate_beta(tempos_acumulados, velocities)
    println("Vídeo $video: β = $(round(beta, digits=4))")
end

# Função para calcular o R²
function calculate_r_squared(times, velocities)
    ln_velocities = log.(abs.(velocities))
    model = lm(@formula(Y ~ X), DataFrame(X=times, Y=ln_velocities))
    r_squared = r2(model)
    return r_squared
end

# Imprimir os valores de R² para cada vídeo
println("Valores de R² para cada vídeo:")
for video in 1:8
    tempos_acumulados, velocities = results[video]
    r_squared = calculate_r_squared(tempos_acumulados, velocities)
    println("Vídeo $video: R² = $(round(r_squared, digits=4))")
end