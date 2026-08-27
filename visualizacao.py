"""Componentes de visualização, mapas de calor e gráficos finais."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import customtkinter as ctk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont

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
    """Mapa de calor Matplotlib incorporado à interface Tk.

    Além da exibição da floresta, o painel oferece edição por clique e uma
    informação flutuante ao passar o ponteiro sobre qualquer célula.
    """

    FORMATOS_IMAGEM = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}

    def __init__(self, master: ctk.CTkFrame) -> None:
        self.figura = Figure(figsize=(8.5, 6.2), dpi=100)
        self.eixos = []
        self.eixo_barra_cores = None
        self.comparacao = False
        self.imagens = []
        self.matrizes_exibidas: list[np.ndarray] = []
        self.anotacoes = {}
        self.callback_edicao: Optional[Callable[[int, int], None]] = None

        self.canvas = FigureCanvasTkAgg(self.figura, master=master)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self._ao_clicar)
        self.canvas.mpl_connect("motion_notify_event", self._ao_mover_mouse)
        self.canvas.mpl_connect("figure_leave_event", self._ocultar_anotacoes)

    def definir_callback_edicao(
        self,
        callback: Callable[[int, int], None],
    ) -> None:
        """Define a função chamada quando o usuário clica em uma célula."""
        self.callback_edicao = callback

    def _recriar_eixos(self, comparacao: bool) -> None:
        """Reconstrói a figura com espaço reservado para a barra de cores.

        Os mapas ocupam uma área um pouco menor que a janela. Isso evita que
        a barra de cores ou as informações flutuantes cubram as últimas
        colunas da matriz.
        """
        if comparacao == self.comparacao and self.eixos:
            return

        self.figura.clear()
        self.comparacao = comparacao

        if comparacao:
            self.eixos = [
                self.figura.add_axes([0.06, 0.17, 0.34, 0.69]),
                self.figura.add_axes([0.49, 0.17, 0.34, 0.69]),
            ]
            self.eixo_barra_cores = self.figura.add_axes(
                [0.87, 0.23, 0.020, 0.57]
            )
        else:
            self.eixos = [
                self.figura.add_axes([0.17, 0.14, 0.59, 0.73])
            ]
            self.eixo_barra_cores = self.figura.add_axes(
                [0.81, 0.22, 0.022, 0.58]
            )

        self.imagens = []
        self.matrizes_exibidas = []
        self.anotacoes = {}

    @staticmethod
    def _configurar_eixo(eixo, titulo: str) -> None:
        eixo.set_title(titulo, fontsize=11, fontweight="bold", pad=8)
        eixo.set_xlabel("Coluna")
        eixo.set_ylabel("Linha")
        eixo.set_aspect("equal", adjustable="box")

    def mostrar_matriz(
        self,
        matriz: np.ndarray,
        titulo: str = "Condição inicial",
    ) -> None:
        """Mostra uma única matriz e remove barras de cores anteriores."""
        self.comparacao = True
        self.eixos = []
        self._recriar_eixos(False)
        self.matrizes_exibidas = [matriz]

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
        self.canvas.draw_idle()

    def mostrar_comparacao(
        self,
        matriz_sem: np.ndarray,
        matriz_com: np.ndarray,
        iteracao: int,
    ) -> None:
        """Mostra os modelos sem e com controle lado a lado."""
        self._recriar_eixos(True)
        self.matrizes_exibidas = [matriz_sem, matriz_com]
        titulos = [
            f"Sem controle — iteração {iteracao}",
            f"Com controle — iteração {iteracao}",
        ]

        if not self.imagens:
            for eixo, matriz, titulo in zip(
                self.eixos,
                self.matrizes_exibidas,
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
                self.matrizes_exibidas,
                titulos,
            ):
                imagem.set_data(matriz)
                eixo.set_title(titulo, fontsize=11, fontweight="bold")

        self.canvas.draw_idle()

    def atualizar_matriz(
        self,
        matriz: np.ndarray,
        titulo: str,
    ) -> None:
        """Atualiza um mapa existente sem recriar os objetos gráficos."""
        if self.comparacao or not self.imagens:
            self.mostrar_matriz(matriz, titulo)
            return

        self.matrizes_exibidas = [matriz]
        self.imagens[0].set_data(matriz)
        self.eixos[0].set_title(titulo, fontsize=11, fontweight="bold")
        self.canvas.draw_idle()

    def _adicionar_barra_cores(self) -> None:
        """Desenha uma única barra de cores em um eixo exclusivo."""
        if self.eixo_barra_cores is None:
            return

        self.eixo_barra_cores.clear()
        normalizador = Normalize(vmin=0.0, vmax=1.0)
        mapeavel = ScalarMappable(norm=normalizador, cmap=MAPA_CORES_FOGO)
        barra = self.figura.colorbar(
            mapeavel,
            cax=self.eixo_barra_cores,
            ticks=np.linspace(0.0, 1.0, 6),
        )
        barra.ax.set_yticklabels(
            [f"{valor:.1f}".replace(".", ",") for valor in np.linspace(0, 1, 6)]
        )
        barra.set_label("Intensidade do fogo", labelpad=10)

    def _obter_indice_e_matriz(self, eixo) -> tuple[int, np.ndarray] | None:
        try:
            indice = self.eixos.index(eixo)
            matriz = self.matrizes_exibidas[indice]
        except (ValueError, IndexError):
            return None
        return indice, matriz

    def _ao_mover_mouse(self, evento) -> None:
        """Exibe linha, coluna e intensidade da célula sob o ponteiro."""
        dados = self._obter_indice_e_matriz(evento.inaxes)
        if (
            dados is None
            or evento.xdata is None
            or evento.ydata is None
        ):
            self._ocultar_anotacoes()
            return

        _, matriz = dados
        coluna = int(np.floor(evento.xdata + 0.5))
        linha = int(np.floor(evento.ydata + 0.5))

        if not (
            0 <= linha < matriz.shape[0]
            and 0 <= coluna < matriz.shape[1]
        ):
            self._ocultar_anotacoes()
            return

        eixo = evento.inaxes
        anotacao = self.anotacoes.get(eixo)
        if anotacao is None:
            anotacao = eixo.annotate(
                "",
                xy=(coluna, linha),
                xytext=(16, 16),
                textcoords="offset points",
                fontsize=9,
                annotation_clip=False,
                color="#111111",
                zorder=10,
                bbox={
                    "boxstyle": "round,pad=0.4",
                    "facecolor": "white",
                    "edgecolor": "#555555",
                    "alpha": 0.88,
                },
                arrowprops={
                    "arrowstyle": "->",
                    "color": "#555555",
                    "alpha": 0.8,
                },
            )
            self.anotacoes[eixo] = anotacao

        for outro_eixo, outra_anotacao in self.anotacoes.items():
            if outro_eixo is not eixo:
                outra_anotacao.set_visible(False)

        valor = float(matriz[linha, coluna])
        texto_valor = f"{valor:.3f}".replace(".", ",")

        # Nas últimas colunas, a caixa é mostrada à esquerda do ponteiro;
        # na primeira linha, ela é mostrada abaixo. Assim, o texto permanece
        # visível mesmo nas bordas da matriz.
        deslocamento_x = -18 if coluna >= matriz.shape[1] / 2 else 18
        deslocamento_y = -20 if linha == 0 else 18
        anotacao.xy = (coluna, linha)
        anotacao.set_position((deslocamento_x, deslocamento_y))
        anotacao.set_ha("right" if deslocamento_x < 0 else "left")
        anotacao.set_va("top" if deslocamento_y < 0 else "bottom")
        anotacao.set_text(
            f"Célula ({linha}, {coluna})\nIntensidade: {texto_valor}"
        )
        anotacao.set_visible(True)
        self.canvas.draw_idle()

    def _ocultar_anotacoes(self, _evento=None) -> None:
        alterou = False
        for anotacao in self.anotacoes.values():
            if anotacao.get_visible():
                anotacao.set_visible(False)
                alterou = True
        if alterou:
            self.canvas.draw_idle()

    def _ao_clicar(self, evento) -> None:
        if (
            self.callback_edicao is None
            or evento.inaxes not in self.eixos
            or evento.xdata is None
            or evento.ydata is None
        ):
            return

        coluna = int(np.floor(evento.xdata + 0.5))
        linha = int(np.floor(evento.ydata + 0.5))
        self.callback_edicao(linha, coluna)

    def salvar_imagem(self, caminho: str | Path) -> Path:
        """Salva o estado visual em PNG, JPEG, PDF ou SVG."""
        caminho = Path(caminho)
        if not caminho.suffix:
            caminho = caminho.with_suffix(".png")

        extensao = caminho.suffix.lower()
        if extensao not in self.FORMATOS_IMAGEM:
            raise ValueError(
                "Formato não suportado. Use PNG, JPG, JPEG, PDF ou SVG."
            )

        self.figura.savefig(
            caminho,
            dpi=260,
            bbox_inches="tight",
            facecolor="white",
            format=extensao.lstrip("."),
        )
        return caminho

    @staticmethod
    def _carregar_fonte(tamanho: int, negrito: bool = False):
        nomes = (
            ["DejaVuSans-Bold.ttf", "arialbd.ttf"]
            if negrito
            else ["DejaVuSans.ttf", "arial.ttf"]
        )
        for nome in nomes:
            try:
                return ImageFont.truetype(nome, tamanho)
            except OSError:
                continue
        return ImageFont.load_default()

    @classmethod
    def _desenhar_texto_centralizado(
        cls,
        desenho: ImageDraw.ImageDraw,
        texto: str,
        centro_x: int,
        y: int,
        tamanho: int,
        negrito: bool = False,
    ) -> None:
        fonte = cls._carregar_fonte(tamanho, negrito)
        caixa = desenho.textbbox((0, 0), texto, font=fonte)
        largura = caixa[2] - caixa[0]
        desenho.text(
            (centro_x - largura // 2, y),
            texto,
            fill="#171717",
            font=fonte,
        )

    @staticmethod
    def _matriz_colorida(matriz: np.ndarray) -> Image.Image:
        normalizada = np.clip(matriz, 0.0, 1.0)
        rgba = MAPA_CORES_FOGO(normalizada, bytes=True)
        return Image.fromarray(rgba[:, :, :3])

    @staticmethod
    def _ajustar_dimensoes(
        forma: tuple[int, int],
        largura_maxima: int,
        altura_maxima: int,
    ) -> tuple[int, int]:
        linhas, colunas = forma
        escala = min(largura_maxima / colunas, altura_maxima / linhas)
        largura = max(1, int(round(colunas * escala)))
        altura = max(1, int(round(linhas * escala)))
        return largura, altura

    @classmethod
    def _desenhar_barra_cores(
        cls,
        quadro: Image.Image,
        x: int,
        y: int,
        altura: int,
        largura: int = 28,
    ) -> None:
        gradiente = np.linspace(1.0, 0.0, altura, dtype=float)[:, None]
        gradiente = np.repeat(gradiente, largura, axis=1)
        rgba = MAPA_CORES_FOGO(gradiente, bytes=True)
        imagem_gradiente = Image.fromarray(rgba[:, :, :3])
        quadro.paste(imagem_gradiente, (x, y))

        desenho = ImageDraw.Draw(quadro)
        desenho.rectangle(
            (x, y, x + largura, y + altura),
            outline="#333333",
            width=1,
        )
        fonte = cls._carregar_fonte(14)
        for valor in (1.0, 0.75, 0.50, 0.25, 0.0):
            posicao_y = y + int((1.0 - valor) * altura)
            desenho.line(
                (x + largura, posicao_y, x + largura + 7, posicao_y),
                fill="#333333",
                width=1,
            )
            desenho.text(
                (x + largura + 10, posicao_y - 8),
                f"{valor:.2f}".replace(".", ","),
                fill="#222222",
                font=fonte,
            )

        fonte_rotulo = cls._carregar_fonte(15, negrito=True)
        desenho.text(
            (x - 7, y + altura + 12),
            "Intensidade",
            fill="#222222",
            font=fonte_rotulo,
        )

    @classmethod
    def matriz_para_imagem(
        cls,
        matriz: np.ndarray,
        titulo: str = "Simulação",
        iteracao: int = 0,
    ) -> Image.Image:
        """Cria um quadro GIF grande e legível para um único modelo."""
        largura_quadro, altura_quadro = 820, 680
        quadro = Image.new(
            "RGB",
            (largura_quadro, altura_quadro),
            "white",
        )
        desenho = ImageDraw.Draw(quadro)

        cls._desenhar_texto_centralizado(
            desenho,
            f"{titulo} — iteração {iteracao}",
            centro_x=largura_quadro // 2,
            y=18,
            tamanho=24,
            negrito=True,
        )

        largura_mapa, altura_mapa = cls._ajustar_dimensoes(
            matriz.shape,
            largura_maxima=650,
            altura_maxima=560,
        )
        mapa = cls._matriz_colorida(matriz).resize(
            (largura_mapa, altura_mapa),
            Image.Resampling.NEAREST,
        )
        area_centro_x = 355
        x_mapa = area_centro_x - largura_mapa // 2
        y_mapa = 75 + (560 - altura_mapa) // 2
        quadro.paste(mapa, (x_mapa, y_mapa))
        desenho.rectangle(
            (
                x_mapa,
                y_mapa,
                x_mapa + largura_mapa,
                y_mapa + altura_mapa,
            ),
            outline="#222222",
            width=2,
        )
        cls._desenhar_barra_cores(quadro, 720, 90, 500)
        return quadro

    @classmethod
    def comparacao_para_imagem(
        cls,
        matriz_sem: np.ndarray,
        matriz_com: np.ndarray,
        iteracao: int = 0,
    ) -> Image.Image:
        """Cria um quadro GIF amplo com os dois modelos lado a lado."""
        largura_quadro, altura_quadro = 1240, 680
        quadro = Image.new(
            "RGB",
            (largura_quadro, altura_quadro),
            "white",
        )
        desenho = ImageDraw.Draw(quadro)

        cls._desenhar_texto_centralizado(
            desenho,
            f"Comparação dos modelos — iteração {iteracao}",
            centro_x=largura_quadro // 2,
            y=15,
            tamanho=24,
            negrito=True,
        )

        matrizes = [matriz_sem, matriz_com]
        titulos = ["Sem controle", "Com controle"]
        centros_x = [300, 850]

        for matriz, titulo, centro_x in zip(matrizes, titulos, centros_x):
            cls._desenhar_texto_centralizado(
                desenho,
                titulo,
                centro_x=centro_x,
                y=56,
                tamanho=20,
                negrito=True,
            )
            largura_mapa, altura_mapa = cls._ajustar_dimensoes(
                matriz.shape,
                largura_maxima=500,
                altura_maxima=535,
            )
            mapa = cls._matriz_colorida(matriz).resize(
                (largura_mapa, altura_mapa),
                Image.Resampling.NEAREST,
            )
            x_mapa = centro_x - largura_mapa // 2
            y_mapa = 100 + (535 - altura_mapa) // 2
            quadro.paste(mapa, (x_mapa, y_mapa))
            desenho.rectangle(
                (
                    x_mapa,
                    y_mapa,
                    x_mapa + largura_mapa,
                    y_mapa + altura_mapa,
                ),
                outline="#222222",
                width=2,
            )

        cls._desenhar_barra_cores(quadro, 1125, 110, 470)
        return quadro


class JanelaResultados(ctk.CTkToplevel):
    """Janela com um gráfico comparativo e uma tabela de valores finais."""

    def __init__(
        self,
        master,
        historico_sem: HistoricoSimulacao,
        historico_com: Optional[HistoricoSimulacao] = None,
        titulo_modelo: str = "Simulação",
    ) -> None:
        super().__init__(master)
        self.title("Resultados da simulação")
        self.geometry("1080x720")
        self.minsize(850, 600)

        figura = Figure(figsize=(10.5, 6.6), dpi=100)
        grade = figura.add_gridspec(
            nrows=2,
            ncols=1,
            height_ratios=[3.2, 2.0],
        )
        eixo_grafico = figura.add_subplot(grade[0, 0])
        eixo_tabela = figura.add_subplot(grade[1, 0])

        self._plotar_media(
            historico_sem,
            eixo_grafico,
            rotulo=titulo_modelo,
        )
        if historico_com is not None:
            self._plotar_media(
                historico_com,
                eixo_grafico,
                rotulo="Com controle",
            )

        eixo_grafico.set_title(
            "Intensidade média do fogo por iteração",
            fontsize=13,
            fontweight="bold",
        )
        eixo_grafico.set_xlabel("Iteração")
        eixo_grafico.set_ylabel("Intensidade média")
        eixo_grafico.grid(True, alpha=0.25)
        eixo_grafico.legend()

        self._criar_tabela_final(
            eixo_tabela,
            historico_sem,
            historico_com,
            titulo_modelo,
        )

        figura.subplots_adjust(
            left=0.08,
            right=0.97,
            top=0.94,
            bottom=0.06,
            hspace=0.38,
        )
        canvas = FigureCanvasTkAgg(figura, master=self)
        canvas.get_tk_widget().pack(
            fill="both",
            expand=True,
            padx=12,
            pady=12,
        )
        canvas.draw()

    @staticmethod
    def _plotar_media(
        historico: HistoricoSimulacao,
        eixo,
        rotulo: str,
    ) -> None:
        eixo.plot(
            historico.iteracoes,
            historico.intensidades_medias,
            linewidth=2.0,
            label=rotulo,
        )

    @staticmethod
    def _formatar_decimal(valor: float) -> str:
        return f"{valor:.3f}".replace(".", ",")

    @classmethod
    def _valores_finais(cls, historico: HistoricoSimulacao) -> list[str]:
        return [
            str(historico.iteracoes[-1]),
            cls._formatar_decimal(historico.tempos[-1]),
            cls._formatar_decimal(historico.intensidades_medias[-1]),
            cls._formatar_decimal(historico.intensidades_maximas[-1]),
            str(historico.celulas_queimando[-1]),
        ]

    @classmethod
    def _criar_tabela_final(
        cls,
        eixo,
        historico_sem: HistoricoSimulacao,
        historico_com: Optional[HistoricoSimulacao],
        titulo_modelo: str,
    ) -> None:
        eixo.axis("off")
        eixo.set_title(
            "Valores finais da simulação",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )

        indicadores = [
            "Iteração final",
            "Instante final tₙ",
            "Intensidade média final",
            "Intensidade máxima final",
            "Células queimando ao final",
        ]
        valores_sem = cls._valores_finais(historico_sem)

        if historico_com is not None:
            valores_com = cls._valores_finais(historico_com)
            media_sem = historico_sem.intensidades_medias[-1]
            media_com = historico_com.intensidades_medias[-1]
            reducao = media_sem - media_com
            percentual = 100.0 * reducao / media_sem if media_sem else 0.0

            indicadores.append("Redução média pelo controle")
            valores_sem.append("—")
            valores_com.append(
                f"{cls._formatar_decimal(reducao)} "
                f"({cls._formatar_decimal(percentual)}%)"
            )
            dados = [
                [indicador, sem, com]
                for indicador, sem, com in zip(
                    indicadores,
                    valores_sem,
                    valores_com,
                )
            ]
            colunas = ["Indicador", titulo_modelo, "Com controle"]
            larguras = [0.50, 0.25, 0.25]
        else:
            dados = [
                [indicador, valor]
                for indicador, valor in zip(indicadores, valores_sem)
            ]
            colunas = ["Indicador", titulo_modelo]
            larguras = [0.65, 0.35]

        tabela = eixo.table(
            cellText=dados,
            colLabels=colunas,
            cellLoc="center",
            colLoc="center",
            colWidths=larguras,
            loc="center",
            bbox=[0.03, 0.02, 0.94, 0.88],
        )
        tabela.auto_set_font_size(False)
        tabela.set_fontsize(10)

        for coluna in range(len(colunas)):
            tabela[(0, coluna)].set_text_props(weight="bold")
        for linha in range(1, len(dados) + 1):
            tabela[(linha, 0)].set_text_props(ha="left")
