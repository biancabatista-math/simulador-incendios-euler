"""Componentes de visualização, mapas de calor e gráficos finais."""

from __future__ import annotations

from typing import Callable, Optional

import customtkinter as ctk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.cm import ScalarMappable
from PIL import Image, ImageDraw

from simulacao import HistoricoSimulacao


CORES_FOGO = [
    (0.00, "#0b3d0b"),
    (0.18, "#2e8b57"),
    (0.38, "#9acd32"),
    (0.58, "#ffd700"),
    (0.78, "#ff8c00"),
    (1.00, "#d00000"),
]

MAPA_CORES_FOGO = LinearSegmentedColormap.from_list(
    "floresta_fogo",
    CORES_FOGO,
)


class PainelFloresta:
    """Mapa de calor Matplotlib incorporado à interface Tk."""

    def __init__(self, master: ctk.CTkFrame) -> None:
        self.figura = Figure(figsize=(8.5, 6.2), dpi=100)
        self.eixos = [self.figura.add_subplot(111)]
        self.comparacao = False
        self.imagens = []
        self.callback_edicao: Optional[Callable[[int, int], None]] = None

        self.canvas = FigureCanvasTkAgg(self.figura, master=master)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self._ao_clicar)

    def definir_callback_edicao(
        self,
        callback: Callable[[int, int], None],
    ) -> None:
        """Define função chamada quando o usuário clica em uma célula."""
        self.callback_edicao = callback

    def _recriar_eixos(self, comparacao: bool) -> None:
        if comparacao == self.comparacao and self.eixos:
            return

        self.figura.clear()
        self.comparacao = comparacao
        if comparacao:
            self.eixos = [
                self.figura.add_subplot(121),
                self.figura.add_subplot(122),
            ]
        else:
            self.eixos = [self.figura.add_subplot(111)]
        self.imagens = []

    def _configurar_eixo(self, eixo, titulo: str) -> None:
        eixo.set_title(titulo, fontsize=11, fontweight="bold")
        eixo.set_xlabel("Coluna")
        eixo.set_ylabel("Linha")

    def mostrar_matriz(
        self,
        matriz: np.ndarray,
        titulo: str = "Condição inicial",
    ) -> None:
        """Mostra uma única matriz no mapa de calor.

        A figura é reconstruída ao carregar ou gerar uma nova matriz. Isso
        remove barras de cores antigas e impede que elas sejam acumuladas
        após vários cliques em "Gerar floresta aleatória".
        """
        self.figura.clear()
        self.comparacao = False
        self.eixos = [self.figura.add_subplot(111)]
        self.imagens = []

        eixo = self.eixos[0]
        imagem = eixo.imshow(
            matriz,
            cmap=MAPA_CORES_FOGO,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            origin="upper",
        )
        self.imagens = [imagem]
        self._configurar_eixo(eixo, titulo)
        self._adicionar_barra_cores()
        self.figura.tight_layout()
        self.canvas.draw_idle()

    def mostrar_comparacao(
        self,
        matriz_sem: np.ndarray,
        matriz_com: np.ndarray,
        iteracao: int,
    ) -> None:
        """Mostra os dois modelos lado a lado."""
        self._recriar_eixos(True)
        titulos = [
            f"Sem controle — iteração {iteracao}",
            f"Com controle — iteração {iteracao}",
        ]

        if not self.imagens:
            for eixo, matriz, titulo in zip(
                self.eixos,
                [matriz_sem, matriz_com],
                titulos,
            ):
                imagem = eixo.imshow(
                    matriz,
                    cmap=MAPA_CORES_FOGO,
                    vmin=0.0,
                    vmax=1.0,
                    interpolation="nearest",
                    origin="upper",
                )
                self.imagens.append(imagem)
                self._configurar_eixo(eixo, titulo)
            self._adicionar_barra_cores()
        else:
            for eixo, imagem, matriz, titulo in zip(
                self.eixos,
                self.imagens,
                [matriz_sem, matriz_com],
                titulos,
            ):
                imagem.set_data(matriz)
                eixo.set_title(titulo, fontsize=11, fontweight="bold")

        self.figura.tight_layout()
        self.canvas.draw_idle()

    def atualizar_matriz(
        self,
        matriz: np.ndarray,
        titulo: str,
    ) -> None:
        """Atualiza o mapa sem recriar seus objetos gráficos."""
        if self.comparacao or not self.imagens:
            self.mostrar_matriz(matriz, titulo)
            return

        self.imagens[0].set_data(matriz)
        self.eixos[0].set_title(titulo, fontsize=11, fontweight="bold")
        self.canvas.draw_idle()

    def _adicionar_barra_cores(self) -> None:
        normalizador = Normalize(vmin=0.0, vmax=1.0)
        mapeavel = ScalarMappable(
            norm=normalizador,
            cmap=MAPA_CORES_FOGO,
        )
        barra = self.figura.colorbar(
            mapeavel,
            ax=self.eixos,
            fraction=0.035,
            pad=0.04,
        )
        barra.set_label("Intensidade do fogo")

    def _ao_clicar(self, evento) -> None:
        if (
            self.callback_edicao is None
            or evento.inaxes not in self.eixos
            or evento.xdata is None
            or evento.ydata is None
        ):
            return
        linha = int(round(evento.ydata))
        coluna = int(round(evento.xdata))
        self.callback_edicao(linha, coluna)

    def salvar_imagem(self, caminho: str) -> None:
        """Salva o estado visual atual em formato de imagem."""
        self.figura.savefig(caminho, dpi=220, bbox_inches="tight")

    @staticmethod
    def matriz_para_imagem(
        matriz: np.ndarray,
        escala: int = 4,
    ) -> Image.Image:
        """Converte uma matriz em imagem RGB usando o gradiente da aplicação."""
        normalizada = np.clip(matriz, 0.0, 1.0)
        rgba = MAPA_CORES_FOGO(normalizada, bytes=True)
        imagem = Image.fromarray(rgba[:, :, :3], mode="RGB")
        largura = max(1, matriz.shape[1] * escala)
        altura = max(1, matriz.shape[0] * escala)
        return imagem.resize((largura, altura), Image.Resampling.NEAREST)

    @classmethod
    def comparacao_para_imagem(
        cls,
        matriz_sem: np.ndarray,
        matriz_com: np.ndarray,
        escala: int = 4,
    ) -> Image.Image:
        """Cria quadro GIF com os dois modelos e seus rótulos."""
        esquerda = cls.matriz_para_imagem(matriz_sem, escala)
        direita = cls.matriz_para_imagem(matriz_com, escala)
        margem = 8
        cabecalho = 28
        quadro = Image.new(
            "RGB",
            (
                esquerda.width + direita.width + 3 * margem,
                max(esquerda.height, direita.height) + cabecalho + margem,
            ),
            "white",
        )
        desenho = ImageDraw.Draw(quadro)
        desenho.text((margem, 7), "Sem controle", fill="black")
        desenho.text(
            (esquerda.width + 2 * margem, 7),
            "Com controle",
            fill="black",
        )
        quadro.paste(esquerda, (margem, cabecalho))
        quadro.paste(
            direita,
            (esquerda.width + 2 * margem, cabecalho),
        )
        return quadro


class JanelaResultados(ctk.CTkToplevel):
    """Janela com gráficos finais dos indicadores da simulação."""

    def __init__(
        self,
        master,
        historico_sem: HistoricoSimulacao,
        historico_com: Optional[HistoricoSimulacao] = None,
        titulo_modelo: str = "Simulação",
    ) -> None:
        super().__init__(master)
        self.title("Resultados da simulação")
        self.geometry("1050x760")
        self.minsize(800, 600)

        figura = Figure(figsize=(10, 7), dpi=100)
        eixo_media = figura.add_subplot(221)
        eixo_maxima = figura.add_subplot(222)
        eixo_queimando = figura.add_subplot(223)
        eixo_comparacao = figura.add_subplot(224)

        self._plotar_historico(
            historico_sem,
            eixo_media,
            eixo_maxima,
            eixo_queimando,
            rotulo=titulo_modelo,
        )

        if historico_com is not None:
            self._plotar_historico(
                historico_com,
                eixo_media,
                eixo_maxima,
                eixo_queimando,
                rotulo="Com controle",
            )
            eixo_comparacao.plot(
                historico_sem.tempos,
                np.asarray(historico_sem.intensidades_medias)
                - np.asarray(historico_com.intensidades_medias),
                label="Redução média",
            )
            eixo_comparacao.set_title(
                "Diferença: sem controle − com controle"
            )
            eixo_comparacao.set_ylabel("Diferença de intensidade")
        else:
            eixo_comparacao.axis("off")
            eixo_comparacao.text(
                0.5,
                0.5,
                "Execute “Comparação lado a lado”\n"
                "para comparar os dois modelos.",
                ha="center",
                va="center",
                fontsize=12,
            )

        for eixo in [eixo_media, eixo_maxima, eixo_queimando]:
            eixo.legend()
            eixo.grid(True, alpha=0.25)
        eixo_comparacao.grid(True, alpha=0.25)
        eixo_comparacao.legend() if historico_com is not None else None

        figura.tight_layout()
        canvas = FigureCanvasTkAgg(figura, master=self)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=12, pady=12)
        canvas.draw()

    @staticmethod
    def _plotar_historico(
        historico: HistoricoSimulacao,
        eixo_media,
        eixo_maxima,
        eixo_queimando,
        rotulo: str,
    ) -> None:
        eixo_media.plot(
            historico.tempos,
            historico.intensidades_medias,
            label=rotulo,
        )
        eixo_media.set_title("Intensidade média")
        eixo_media.set_xlabel("Tempo")
        eixo_media.set_ylabel("Intensidade")

        eixo_maxima.plot(
            historico.tempos,
            historico.intensidades_maximas,
            label=rotulo,
        )
        eixo_maxima.set_title("Intensidade máxima")
        eixo_maxima.set_xlabel("Tempo")
        eixo_maxima.set_ylabel("Intensidade")

        eixo_queimando.plot(
            historico.tempos,
            historico.celulas_queimando,
            label=rotulo,
        )
        eixo_queimando.set_title("Quantidade de células queimando")
        eixo_queimando.set_xlabel("Tempo")
        eixo_queimando.set_ylabel("Células")
