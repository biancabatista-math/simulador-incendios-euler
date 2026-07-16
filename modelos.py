"""Modelos matemáticos de propagação de incêndios florestais."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from configuracoes import ParametrosSimulacao


def soma_vizinhos_von_neumann(matriz: np.ndarray) -> np.ndarray:
    """Calcula a soma dos vizinhos de Von Neumann de cada célula.

    A vizinhança de Von Neumann contém as células imediatamente acima,
    abaixo, à esquerda e à direita. Fora dos limites da matriz, considera-se
    intensidade zero. A implementação usa operações vetorizadas do NumPy e
    evita laços Python sobre as células.

    Args:
        matriz: Matriz bidimensional com as intensidades atuais.

    Returns:
        Matriz com a soma dos quatro vizinhos de cada posição.
    """
    if matriz.ndim != 2:
        raise ValueError("A matriz de intensidades deve ser bidimensional.")

    soma = np.zeros_like(matriz, dtype=np.float64)
    soma[1:, :] += matriz[:-1, :]   # Vizinho acima.
    soma[:-1, :] += matriz[1:, :]   # Vizinho abaixo.
    soma[:, 1:] += matriz[:, :-1]   # Vizinho à esquerda.
    soma[:, :-1] += matriz[:, 1:]   # Vizinho à direita.
    return soma


class ModeloIncendio(ABC):
    """Classe-base para uma etapa explícita do Método de Euler."""

    nome = "Modelo abstrato"

    def __init__(self, parametros: ParametrosSimulacao) -> None:
        self.parametros = parametros

    def _finalizar_passo(self, proxima: np.ndarray) -> np.ndarray:
        """Aplica opcionalmente a projeção física no intervalo [0, 1]."""
        if self.parametros.limitar_intervalo:
            return np.clip(proxima, 0.0, 1.0)
        return proxima

    @abstractmethod
    def passo(self, atual: np.ndarray) -> np.ndarray:
        """Calcula a matriz da iteração seguinte."""


class ModeloSemControle(ModeloIncendio):
    """Modelo de propagação sem atuação dos bombeiros."""

    nome = "Sem controle"

    def passo(self, atual: np.ndarray) -> np.ndarray:
        r"""Executa um passo de Euler do modelo sem controle.

        Equação implementada:

        y_i^(n+1) = y_i^n + h*k*sum_{j em N(i)}(y_j^n)*(1-y_i^n)
        """
        vizinhos = soma_vizinhos_von_neumann(atual)
        incremento = (
            self.parametros.passo_euler
            * self.parametros.constante_k
            * vizinhos
            * (1.0 - atual)
        )
        return self._finalizar_passo(atual + incremento)


class ModeloComControle(ModeloIncendio):
    """Modelo de propagação com termo de controle dos bombeiros."""

    nome = "Com controle"

    def passo(self, atual: np.ndarray) -> np.ndarray:
        r"""Executa um passo de Euler do modelo com controle.

        Equação implementada:

        y_i^(n+1) = y_i^n + h*[
            k*sum_{j em N(i)}(y_j^n)*(1-y_i^n) - gamma*(y_i^n)^2
        ]
        """
        vizinhos = soma_vizinhos_von_neumann(atual)
        propagacao = (
            self.parametros.constante_k
            * vizinhos
            * (1.0 - atual)
        )
        controle = self.parametros.constante_gamma * np.square(atual)
        proxima = atual + self.parametros.passo_euler * (
            propagacao - controle
        )
        return self._finalizar_passo(proxima)
