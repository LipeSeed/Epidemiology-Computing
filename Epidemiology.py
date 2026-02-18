import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap, BoundaryNorm

"""
Cellular Automaton SIR(S) model on a 2D grid.

States:
0 = Susceptible (S)
1 = Infected (I)
2 = Recovered/Immune (R)
3 = Dead (D)

Transition ideas (per generation, synchronous update):
- S -> I with probability p_inf = 1 - (1 - beta) ** (number of infected neighbors)
- I -> R with probability gamma
- I -> D with probability mu (mortality), evaluated before recovery by default
- R -> S with probability omega (waning immunity) [optional, set to 0 to disable]
"""

def criar_grade(linhas: int, colunas: int, frac_infectados: float = 0.01) -> np.ndarray:
    grade = np.zeros((linhas, colunas), dtype=np.int8)
    n = linhas * colunas
    k = max(1, int(n * frac_infectados))
    idx = np.random.choice(n, size=k, replace=False)
    grade.flat[idx] = 1
    return grade

def contar_vizinhos_infectados(grade: np.ndarray, linha: int, coluna: int) -> int:
    linhas, colunas = grade.shape
    infectados = 0
    for i in range(linha - 1, linha + 2):
        for j in range(coluna - 1, coluna + 2):
            if (i == linha and j == coluna) or i < 0 or i >= linhas or j < 0 or j >= colunas:
                continue
            infectados += (grade[i, j] == 1)
    return int(infectados)

def atualizar_grade(
    grade: np.ndarray,
    beta: float = 0.25,    
    gamma: float = 0.06,   
    mu: float = 0.003,     
    omega: float = 0.0      
) -> np.ndarray:
    linhas, colunas = grade.shape
    nova = np.copy(grade)

    for i in range(linhas):
        for j in range(colunas):
            estado = grade[i, j]

            if estado == 0:
                n_inf = contar_vizinhos_infectados(grade, i, j)
                if n_inf > 0:
                    p_inf = 1.0 - (1.0 - beta) ** n_inf
                    if np.random.random() < p_inf:
                        nova[i, j] = 1

            elif estado == 1:
                r = np.random.random()
                if r < mu:
                    nova[i, j] = 3
                elif r < mu + gamma:
                    nova[i, j] = 2
                else:
                    nova[i, j] = 1

            elif estado == 2:
                if omega > 0 and np.random.random() < omega:
                    nova[i, j] = 0

            else:
                nova[i, j] = 3

    return nova

def animar(frame, grade, img, params):
    grade[:] = atualizar_grade(grade, **params)
    img.set_array(grade)
    return (img,)

def executar_simulacao(
    linhas: int = 80,
    colunas: int = 80,
    geracoes: int = 250,
    frac_infectados: float = 0.01,
    beta: float = 0.25,
    gamma: float = 0.06,
    mu: float = 0.003,
    omega: float = 0.0,
    intervalo_ms: int = 80
):
    grade = criar_grade(linhas, colunas, frac_infectados=frac_infectados)

    cmap = ListedColormap(["#f2f2f2", "#d62728", "#2ca02c", "#111111"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Cellular Automaton SIR(S): dinâmica espacial de surto")
    ax.set_xticks([])
    ax.set_yticks([])

    img = ax.imshow(grade, interpolation="nearest", cmap=cmap, norm=norm)

    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color=cmap(0), label="Suscetível (S)"),
        mpatches.Patch(color=cmap(1), label="Infectado (I)"),
        mpatches.Patch(color=cmap(2), label="Recuperado (R)"),
        mpatches.Patch(color=cmap(3), label="Óbito (D)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=8)

    params = dict(beta=beta, gamma=gamma, mu=mu, omega=omega)
    animacao = animation.FuncAnimation(
        fig,
        animar,
        fargs=(grade, img, params),
        frames=geracoes,
        interval=intervalo_ms,
        blit=True,
    )

    plt.show()

if __name__ == "__main__":
    executar_simulacao(
        linhas=80,
        colunas=80,
        geracoes=250,
        frac_infectados=0.01,
        beta=0.23,
        gamma=0.06,
        mu=0.003,
        omega=0.0,
        intervalo_ms=80,
    )
