"""Motores de simulação e coleta de indicadores numéricos."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from configuracoes import ParametrosSimulacao
from modelos import ModeloComControle, ModeloIncendio, ModeloSemControle


@dataclass
class HistoricoSimulacao:
    """Séries temporais geradas durante uma simulação."""

    iteracoes: list[int] = field(default_factory=list)
    tempos: list[float] = field(default_factory=list)
    intensidades_medias: list[float] = field(default_factory=list)
    intensidades_maximas: list[float] = field(default_factory=list)
    celulas_queimando: list[int] = field(default_factory=list)

    def registrar(
        self,
        iteracao: int,
        tempo: float,
        matriz: np.ndarray,
        limiar_queima: float,
    ) -> None:
        """Registra os principais indicadores da matriz atual."""
        self.iteracoes.append(iteracao)
        self.tempos.append(tempo)
        self.intensidades_medias.append(float(np.mean(matriz)))
        self.intensidades_maximas.append(float(np.max(matriz)))
        self.celulas_queimando.append(
            int(np.count_nonzero(matriz >= limiar_queima))
        )


class MotorSimulacao:
    """Controla estado, iterações e histórico de um único modelo."""

    def __init__(
        self,
        matriz_inicial: np.ndarray,
        modelo: ModeloIncendio,
        parametros: ParametrosSimulacao,
    ) -> None:
        self.parametros = parametros
        self.modelo = modelo
        self.matriz_inicial = np.array(
            matriz_inicial, dtype=np.float64, copy=True
        )
        self.matriz_atual = self.matriz_inicial.copy()
        self.iteracao_atual = 0
        self.historico = HistoricoSimulacao()
        self._registrar_estado()

    @property
    def concluida(self) -> bool:
        """Indica se o número máximo de iterações foi alcançado."""
        return self.iteracao_atual >= self.parametros.iteracoes

    def _registrar_estado(self) -> None:
        tempo = self.iteracao_atual * self.parametros.passo_euler
        self.historico.registrar(
            self.iteracao_atual,
            tempo,
            self.matriz_atual,
            self.parametros.limiar_queima,
        )

    def executar_passo(self) -> np.ndarray:
        """Executa uma iteração síncrona do modelo."""
        if self.concluida:
            return self.matriz_atual

        self.matriz_atual = self.modelo.passo(self.matriz_atual)
        self.iteracao_atual += 1
        self._registrar_estado()
        return self.matriz_atual

    def reiniciar(self) -> None:
        """Restaura matriz, contador e histórico."""
        self.matriz_atual = self.matriz_inicial.copy()
        self.iteracao_atual = 0
        self.historico = HistoricoSimulacao()
        self._registrar_estado()


class MotorComparacao:
    """Executa os dois modelos sobre a mesma condição inicial."""

    def __init__(
        self,
        matriz_inicial: np.ndarray,
        parametros: ParametrosSimulacao,
    ) -> None:
        self.parametros = parametros
        self.sem_controle = MotorSimulacao(
            matriz_inicial,
            ModeloSemControle(parametros),
            parametros,
        )
        self.com_controle = MotorSimulacao(
            matriz_inicial,
            ModeloComControle(parametros),
            parametros,
        )

    @property
    def iteracao_atual(self) -> int:
        """Retorna a iteração comum aos dois motores."""
        return self.sem_controle.iteracao_atual

    @property
    def concluida(self) -> bool:
        """Indica se os dois modelos terminaram."""
        return self.sem_controle.concluida

    def executar_passo(self) -> tuple[np.ndarray, np.ndarray]:
        """Avança simultaneamente os dois modelos."""
        sem = self.sem_controle.executar_passo()
        com = self.com_controle.executar_passo()
        return sem, com

    def reiniciar(self) -> None:
        """Reinicia os dois modelos."""
        self.sem_controle.reiniciar()
        self.com_controle.reiniciar()
