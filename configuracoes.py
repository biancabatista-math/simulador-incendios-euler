"""Configurações, enumerações e validações centrais da aplicação."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TipoModelo(str, Enum):
    """Modelos matemáticos disponíveis na interface."""

    SEM_CONTROLE = "Sem controle"
    COM_CONTROLE = "Com controle"
    COMPARACAO = "Comparação lado a lado"


@dataclass(frozen=True)
class ConfiguracoesPadrao:
    """Valores iniciais exibidos na interface gráfica."""

    linhas: int = 50
    colunas: int = 50
    passo_euler: float = 0.05
    constante_k: float = 0.35
    constante_gamma: float = 0.15
    iteracoes: int = 300
    velocidade_ms: int = 50
    intensidade_inicial: float = 0.80
    quantidade_focos: int = 1
    limiar_queima: float = 0.01
    limite_frames_gif: int = 120


PADRAO = ConfiguracoesPadrao()


@dataclass(frozen=True)
class ParametrosSimulacao:
    """Parâmetros numéricos usados pelos modelos e pelo simulador."""

    linhas: int
    colunas: int
    passo_euler: float
    constante_k: float
    constante_gamma: float
    iteracoes: int
    velocidade_ms: int
    limiar_queima: float = PADRAO.limiar_queima
    limitar_intervalo: bool = False

    def validar(self) -> None:
        """Valida os parâmetros e gera mensagens adequadas à interface."""
        if self.linhas < 1 or self.colunas < 1:
            raise ValueError("O número de linhas e colunas deve ser positivo.")
        if self.linhas > 1000 or self.colunas > 1000:
            raise ValueError(
                "Por segurança, cada dimensão deve ser menor ou igual a 1000."
            )
        if self.passo_euler <= 0:
            raise ValueError("O passo de Euler h deve ser positivo.")
        if self.constante_k < 0:
            raise ValueError("A constante k não pode ser negativa.")
        if self.constante_gamma < 0:
            raise ValueError("A constante γ não pode ser negativa.")
        if self.iteracoes < 1:
            raise ValueError("O número de iterações deve ser positivo.")
        if self.velocidade_ms < 1:
            raise ValueError("A velocidade deve ser de pelo menos 1 ms.")
        if not 0 <= self.limiar_queima <= 1:
            raise ValueError("O limiar de queima deve estar entre 0 e 1.")
