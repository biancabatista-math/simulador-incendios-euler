"""Funções utilitárias de entrada, saída e preparação de matrizes."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np

from configuracoes import ParametrosSimulacao
from simulacao import HistoricoSimulacao


def interpretar_posicao(texto: str) -> Optional[tuple[int, int]]:
    """Converte 'linha,coluna' em uma tupla ou retorna None se vazio."""
    texto = texto.strip()
    if not texto:
        return None

    partes = texto.replace(";", ",").split(",")
    if len(partes) != 2:
        raise ValueError("Use o formato linha,coluna para o foco inicial.")

    try:
        linha, coluna = (int(parte.strip()) for parte in partes)
    except ValueError as erro:
        raise ValueError("Linha e coluna devem ser números inteiros.") from erro

    return linha, coluna


def criar_matriz_inicial(
    linhas: int,
    colunas: int,
    intensidade: float,
    quantidade_focos: int,
    foco_preferencial: Optional[tuple[int, int]] = None,
    semente: Optional[int] = None,
) -> np.ndarray:
    """Cria floresta saudável e posiciona focos iniciais.

    Todas as células começam com intensidade zero. Um foco informado pelo
    usuário é inserido primeiro; os focos restantes são escolhidos sem
    repetição por um gerador aleatório.
    """
    if linhas < 1 or colunas < 1:
        raise ValueError("As dimensões da matriz devem ser positivas.")
    if not 0 <= intensidade <= 1:
        raise ValueError("A intensidade inicial deve estar entre 0 e 1.")
    if quantidade_focos < 1:
        raise ValueError("A quantidade de focos deve ser positiva.")
    if quantidade_focos > linhas * colunas:
        raise ValueError("Há mais focos do que células disponíveis.")

    matriz = np.zeros((linhas, colunas), dtype=np.float64)
    ocupadas: set[tuple[int, int]] = set()

    if foco_preferencial is not None:
        linha, coluna = foco_preferencial
        if not (0 <= linha < linhas and 0 <= coluna < colunas):
            raise ValueError(
                "A posição inicial está fora dos limites da matriz. "
                "Os índices começam em zero."
            )
        ocupadas.add((linha, coluna))

    gerador = np.random.default_rng(semente)
    faltantes = quantidade_focos - len(ocupadas)
    if faltantes > 0:
        livres = np.array(
            [
                indice
                for indice in range(linhas * colunas)
                if divmod(indice, colunas) not in ocupadas
            ],
            dtype=int,
        )
        escolhidos = gerador.choice(livres, size=faltantes, replace=False)
        ocupadas.update(divmod(int(indice), colunas) for indice in escolhidos)

    for linha, coluna in ocupadas:
        matriz[linha, coluna] = intensidade

    return matriz


def validar_matriz(matriz: np.ndarray) -> np.ndarray:
    """Valida e normaliza uma matriz carregada pelo usuário."""
    matriz = np.asarray(matriz, dtype=np.float64)
    if matriz.ndim != 2:
        raise ValueError("O arquivo deve conter uma matriz bidimensional.")
    if matriz.size == 0:
        raise ValueError("A matriz carregada está vazia.")
    if not np.all(np.isfinite(matriz)):
        raise ValueError("A matriz contém valores não finitos.")
    if np.any((matriz < 0) | (matriz > 1)):
        raise ValueError("Todos os valores da matriz devem estar em [0, 1].")
    return matriz


def carregar_matriz(caminho: str | Path) -> np.ndarray:
    """Carrega matriz nos formatos CSV, TXT ou NPY."""
    caminho = Path(caminho)
    extensao = caminho.suffix.lower()

    if extensao == ".npy":
        matriz = np.load(caminho, allow_pickle=False)
    elif extensao == ".csv":
        matriz = np.loadtxt(caminho, delimiter=",")
    elif extensao == ".txt":
        try:
            matriz = np.loadtxt(caminho, delimiter=",")
        except ValueError:
            matriz = np.loadtxt(caminho)
    else:
        raise ValueError("Formato não suportado. Use CSV, TXT ou NPY.")

    return validar_matriz(matriz)


def salvar_matriz_csv(caminho: str | Path, matriz: np.ndarray) -> None:
    """Salva uma matriz com precisão adequada para análise posterior."""
    np.savetxt(caminho, matriz, delimiter=",", fmt="%.8f")


def _linhas_historico(historico: HistoricoSimulacao) -> list[list[object]]:
    return [
        [
            iteracao,
            tempo,
            media,
            maxima,
            queimando,
        ]
        for iteracao, tempo, media, maxima, queimando in zip(
            historico.iteracoes,
            historico.tempos,
            historico.intensidades_medias,
            historico.intensidades_maximas,
            historico.celulas_queimando,
        )
    ]


def exportar_resultado_unico(
    caminho_resumo: str | Path,
    historico: HistoricoSimulacao,
    matriz_final: np.ndarray,
    parametros: ParametrosSimulacao,
    nome_modelo: str,
) -> list[Path]:
    """Exporta resumo temporal, matriz final e metadados."""
    caminho_resumo = Path(caminho_resumo).with_suffix(".csv")
    base = caminho_resumo.with_suffix("")
    caminho_matriz = Path(f"{base}_matriz_final.csv")
    caminho_metadados = Path(f"{base}_metadados.json")

    with caminho_resumo.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(
            [
                "iteracao",
                "tempo",
                "intensidade_media",
                "intensidade_maxima",
                "celulas_queimando",
            ]
        )
        escritor.writerows(_linhas_historico(historico))

    salvar_matriz_csv(caminho_matriz, matriz_final)
    caminho_metadados.write_text(
        json.dumps(
            {
                "modelo": nome_modelo,
                "parametros": parametros.__dict__,
                "forma_matriz": list(matriz_final.shape),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return [caminho_resumo, caminho_matriz, caminho_metadados]


def exportar_resultado_comparacao(
    caminho_resumo: str | Path,
    historico_sem: HistoricoSimulacao,
    historico_com: HistoricoSimulacao,
    matriz_sem: np.ndarray,
    matriz_com: np.ndarray,
    parametros: ParametrosSimulacao,
) -> list[Path]:
    """Exporta séries e matrizes finais dos dois modelos."""
    caminho_resumo = Path(caminho_resumo).with_suffix(".csv")
    base = caminho_resumo.with_suffix("")
    caminho_sem = Path(f"{base}_sem_controle.csv")
    caminho_com = Path(f"{base}_com_controle.csv")
    caminho_metadados = Path(f"{base}_metadados.json")

    with caminho_resumo.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(
            [
                "iteracao",
                "tempo",
                "media_sem_controle",
                "media_com_controle",
                "maxima_sem_controle",
                "maxima_com_controle",
                "queimando_sem_controle",
                "queimando_com_controle",
            ]
        )
        for indice in range(len(historico_sem.iteracoes)):
            escritor.writerow(
                [
                    historico_sem.iteracoes[indice],
                    historico_sem.tempos[indice],
                    historico_sem.intensidades_medias[indice],
                    historico_com.intensidades_medias[indice],
                    historico_sem.intensidades_maximas[indice],
                    historico_com.intensidades_maximas[indice],
                    historico_sem.celulas_queimando[indice],
                    historico_com.celulas_queimando[indice],
                ]
            )

    salvar_matriz_csv(caminho_sem, matriz_sem)
    salvar_matriz_csv(caminho_com, matriz_com)
    caminho_metadados.write_text(
        json.dumps(
            {
                "modelo": "Comparação lado a lado",
                "parametros": parametros.__dict__,
                "forma_matriz": list(matriz_sem.shape),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return [caminho_resumo, caminho_sem, caminho_com, caminho_metadados]


def formatar_tempo(segundos: float) -> str:
    """Formata segundos como HH:MM:SS."""
    segundos_inteiros = max(0, int(segundos))
    horas, resto = divmod(segundos_inteiros, 3600)
    minutos, segundos_finais = divmod(resto, 60)
    return f"{horas:02d}:{minutos:02d}:{segundos_finais:02d}"
