"""Interface gráfica principal do simulador."""

from __future__ import annotations

import math
import time
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import numpy as np
from PIL import Image

from configuracoes import PADRAO, ParametrosSimulacao, TipoModelo
from modelos import ModeloComControle, ModeloSemControle
from simulacao import MotorComparacao, MotorSimulacao
from utils import (
    carregar_matriz,
    criar_matriz_inicial,
    exportar_resultado_comparacao,
    exportar_resultado_unico,
    formatar_tempo,
    interpretar_posicao,
)
from visualizacao import JanelaResultados, PainelFloresta


class AplicacaoIncendio(ctk.CTk):
    """Janela principal e controladora da simulação."""

    def __init__(self) -> None:
        super().__init__()

        self.title("Simulador de Incêndios Florestais — Método de Euler")
        self.geometry("1420x860")
        self.minsize(1120, 720)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")

        self.matriz_inicial: np.ndarray | None = None
        self.motor: MotorSimulacao | None = None
        self.comparacao: MotorComparacao | None = None
        self.parametros: ParametrosSimulacao | None = None

        self.em_execucao = False
        self.pausada = False
        self.id_after: str | None = None
        self.inicio_cronometro = 0.0
        self.tempo_acumulado = 0.0
        self.inicio_pausa = 0.0
        self.frames_gif = []
        self.intervalo_frames = 1

        self._criar_variaveis()
        self._construir_interface()
        self._gerar_floresta_inicial()

    def _criar_variaveis(self) -> None:
        self.modelo_var = ctk.StringVar(value=TipoModelo.SEM_CONTROLE.value)
        self.linhas_var = ctk.StringVar(value=str(PADRAO.linhas))
        self.colunas_var = ctk.StringVar(value=str(PADRAO.colunas))
        self.h_var = ctk.StringVar(value=str(PADRAO.passo_euler))
        self.k_var = ctk.StringVar(value=str(PADRAO.constante_k))
        self.gamma_var = ctk.StringVar(value=str(PADRAO.constante_gamma))
        self.iteracoes_var = ctk.StringVar(value=str(PADRAO.iteracoes))
        self.velocidade_var = ctk.StringVar(value=str(PADRAO.velocidade_ms))
        self.intensidade_var = ctk.StringVar(
            value=str(PADRAO.intensidade_inicial)
        )
        self.focos_var = ctk.StringVar(value=str(PADRAO.quantidade_focos))
        self.posicao_var = ctk.StringVar(value="")
        self.semente_var = ctk.StringVar(value="")
        self.limitar_var = ctk.BooleanVar(value=False)
        self.tema_var = ctk.StringVar(value="Escuro")

        self.status_var = ctk.StringVar(value="Pronto")
        self.iteracao_var = ctk.StringVar(value="Iteração: 0")
        self.cronometro_var = ctk.StringVar(value="Tempo: 00:00:00")

    def _construir_interface(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.barra_lateral = ctk.CTkScrollableFrame(
            self,
            width=330,
            corner_radius=0,
        )
        self.barra_lateral.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        self.area_principal = ctk.CTkFrame(self, corner_radius=0)
        self.area_principal.grid(
            row=0,
            column=1,
            sticky="nsew",
        )
        self.area_principal.grid_columnconfigure(0, weight=1)
        self.area_principal.grid_rowconfigure(1, weight=1)

        self._construir_controles()
        self._construir_area_visual()
        self._construir_barra_status()

    def _construir_controles(self) -> None:
        titulo = ctk.CTkLabel(
            self.barra_lateral,
            text="Simulador de Incêndios",
            font=ctk.CTkFont(size=21, weight="bold"),
        )
        titulo.pack(padx=16, pady=(18, 4))

        subtitulo = ctk.CTkLabel(
            self.barra_lateral,
            text="Método de Euler • Vizinhança de Von Neumann",
            wraplength=290,
            text_color=("gray35", "gray70"),
        )
        subtitulo.pack(padx=16, pady=(0, 18))

        self._rotulo_secao("Modelo")
        self.combo_modelo = ctk.CTkComboBox(
            self.barra_lateral,
            variable=self.modelo_var,
            values=[tipo.value for tipo in TipoModelo],
            command=self._ao_mudar_modelo,
            state="readonly",
        )
        self.combo_modelo.pack(fill="x", padx=16, pady=(0, 10))

        self._rotulo_secao("Matriz")
        grade_dimensoes = ctk.CTkFrame(
            self.barra_lateral,
            fg_color="transparent",
        )
        grade_dimensoes.pack(fill="x", padx=16)
        grade_dimensoes.grid_columnconfigure((0, 1), weight=1)
        self._campo(
            grade_dimensoes,
            "Linhas",
            self.linhas_var,
            0,
            0,
        )
        self._campo(
            grade_dimensoes,
            "Colunas",
            self.colunas_var,
            0,
            1,
        )

        self._campo_empilhado(
            "Intensidade inicial [0, 1]",
            self.intensidade_var,
        )
        self._campo_empilhado("Quantidade de focos", self.focos_var)
        self._campo_empilhado(
            "Foco preferencial (linha,coluna)",
            self.posicao_var,
            placeholder="Ex.: 25,25; vazio = aleatório",
        )
        self._campo_empilhado(
            "Semente aleatória (opcional)",
            self.semente_var,
            placeholder="Ex.: 42",
        )

        self._rotulo_secao("Parâmetros numéricos")
        self._campo_empilhado("Passo de Euler h", self.h_var)
        self._campo_empilhado("Constante k", self.k_var)
        self._campo_empilhado("Constante γ", self.gamma_var)
        self._campo_empilhado(
            "Número máximo de iterações",
            self.iteracoes_var,
        )
        self._campo_empilhado(
            "Velocidade da animação (ms)",
            self.velocidade_var,
        )

        self.check_limite = ctk.CTkCheckBox(
            self.barra_lateral,
            text="Limitar intensidades ao intervalo [0, 1]",
            variable=self.limitar_var,
        )
        self.check_limite.pack(
            fill="x",
            padx=16,
            pady=(8, 12),
        )

        self._rotulo_secao("Preparação")
        self._botao(
            "Gerar floresta aleatória",
            self._gerar_floresta_inicial,
        )
        self._botao("Carregar matriz", self._carregar_matriz)
        self._botao(
            "Editar: clique em uma célula do mapa",
            self._mostrar_instrucao_edicao,
            estilo="secundario",
        )

        self._rotulo_secao("Simulação")
        self._botao("Iniciar simulação", self._iniciar_simulacao)
        linha_pausa = ctk.CTkFrame(
            self.barra_lateral,
            fg_color="transparent",
        )
        linha_pausa.pack(fill="x", padx=16, pady=4)
        linha_pausa.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(
            linha_pausa,
            text="Pausar",
            command=self._pausar,
        ).grid(row=0, column=0, padx=(0, 4), sticky="ew")
        ctk.CTkButton(
            linha_pausa,
            text="Continuar",
            command=self._continuar,
        ).grid(row=0, column=1, padx=(4, 0), sticky="ew")
        self._botao("Reiniciar", self._reiniciar, estilo="secundario")

        self._rotulo_secao("Exportação")
        self._botao("Salvar imagem", self._salvar_imagem)
        self._botao("Exportar resultados", self._exportar_resultados)
        self._botao("Exportar animação GIF", self._exportar_gif)

        self._rotulo_secao("Aplicação")
        linha_tema = ctk.CTkFrame(
            self.barra_lateral,
            fg_color="transparent",
        )
        linha_tema.pack(fill="x", padx=16, pady=4)
        linha_tema.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(linha_tema, text="Tema").grid(
            row=0,
            column=0,
            padx=(0, 8),
        )
        ctk.CTkOptionMenu(
            linha_tema,
            values=["Escuro", "Claro", "Sistema"],
            variable=self.tema_var,
            command=self._alterar_tema,
        ).grid(row=0, column=1, sticky="ew")
        self._botao("Sobre o modelo", self._abrir_sobre, estilo="secundario")

        ctk.CTkLabel(
            self.barra_lateral,
            text="Universidade Federal de Viçosa",
            text_color=("gray45", "gray65"),
        ).pack(pady=20)

    def _construir_area_visual(self) -> None:
        cabecalho = ctk.CTkFrame(
            self.area_principal,
            height=70,
            corner_radius=0,
        )
        cabecalho.grid(row=0, column=0, sticky="ew")
        cabecalho.grid_columnconfigure(3, weight=1)

        ctk.CTkLabel(
            cabecalho,
            textvariable=self.status_var,
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, padx=18, pady=18)

        ctk.CTkLabel(
            cabecalho,
            textvariable=self.iteracao_var,
        ).grid(row=0, column=1, padx=12)

        ctk.CTkLabel(
            cabecalho,
            textvariable=self.cronometro_var,
        ).grid(row=0, column=2, padx=12)

        self.barra_progresso = ctk.CTkProgressBar(cabecalho)
        self.barra_progresso.grid(
            row=0,
            column=3,
            padx=18,
            sticky="ew",
        )
        self.barra_progresso.set(0)

        quadro_visual = ctk.CTkFrame(self.area_principal)
        quadro_visual.grid(
            row=1,
            column=0,
            padx=12,
            pady=12,
            sticky="nsew",
        )
        self.painel = PainelFloresta(quadro_visual)
        self.painel.definir_callback_edicao(self._editar_celula)

    def _construir_barra_status(self) -> None:
        rodape = ctk.CTkLabel(
            self.area_principal,
            text=(
                "Verde: baixa intensidade • Amarelo: moderada • "
                "Laranja/vermelho: alta"
            ),
            anchor="w",
            text_color=("gray35", "gray70"),
        )
        rodape.grid(
            row=2,
            column=0,
            padx=18,
            pady=(0, 10),
            sticky="ew",
        )

    def _rotulo_secao(self, texto: str) -> None:
        ctk.CTkLabel(
            self.barra_lateral,
            text=texto.upper(),
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w",
            text_color=("gray30", "gray70"),
        ).pack(fill="x", padx=16, pady=(16, 6))

    @staticmethod
    def _campo(
        master,
        rotulo: str,
        variavel,
        linha: int,
        coluna: int,
    ) -> None:
        quadro = ctk.CTkFrame(master, fg_color="transparent")
        quadro.grid(
            row=linha,
            column=coluna,
            padx=(0, 6) if coluna == 0 else (6, 0),
            sticky="ew",
        )
        ctk.CTkLabel(quadro, text=rotulo, anchor="w").pack(fill="x")
        ctk.CTkEntry(quadro, textvariable=variavel).pack(fill="x")

    def _campo_empilhado(
        self,
        rotulo: str,
        variavel,
        placeholder: str = "",
    ) -> None:
        ctk.CTkLabel(
            self.barra_lateral,
            text=rotulo,
            anchor="w",
        ).pack(fill="x", padx=16, pady=(7, 2))
        entrada = ctk.CTkEntry(
            self.barra_lateral,
            textvariable=variavel,
            placeholder_text=placeholder,
        )
        entrada.pack(fill="x", padx=16)

        if rotulo == "Constante γ":
            self.entrada_gamma = entrada

    def _botao(
        self,
        texto: str,
        comando,
        estilo: str = "principal",
    ) -> None:
        opcoes = {}
        if estilo == "secundario":
            opcoes = {
                "fg_color": "transparent",
                "border_width": 1,
                "text_color": ("gray10", "gray90"),
            }
        ctk.CTkButton(
            self.barra_lateral,
            text=texto,
            command=comando,
            **opcoes,
        ).pack(fill="x", padx=16, pady=4)

    def _ao_mudar_modelo(self, valor: str) -> None:
        estado = (
            "disabled"
            if valor == TipoModelo.SEM_CONTROLE.value
            else "normal"
        )
        self.entrada_gamma.configure(state=estado)

    def _ler_inteiro(self, variavel, nome: str) -> int:
        try:
            return int(variavel.get().strip())
        except ValueError as erro:
            raise ValueError(f"{nome} deve ser um número inteiro.") from erro

    def _ler_float(self, variavel, nome: str) -> float:
        try:
            return float(variavel.get().strip().replace(",", "."))
        except ValueError as erro:
            raise ValueError(f"{nome} deve ser um número real.") from erro

    def _obter_parametros(self) -> ParametrosSimulacao:
        parametros = ParametrosSimulacao(
            linhas=self._ler_inteiro(self.linhas_var, "Linhas"),
            colunas=self._ler_inteiro(self.colunas_var, "Colunas"),
            passo_euler=self._ler_float(self.h_var, "h"),
            constante_k=self._ler_float(self.k_var, "k"),
            constante_gamma=self._ler_float(self.gamma_var, "γ"),
            iteracoes=self._ler_inteiro(
                self.iteracoes_var,
                "Iterações",
            ),
            velocidade_ms=self._ler_inteiro(
                self.velocidade_var,
                "Velocidade",
            ),
            limitar_intervalo=bool(self.limitar_var.get()),
        )
        parametros.validar()
        return parametros

    def _gerar_floresta_inicial(self) -> None:
        if self.em_execucao:
            messagebox.showwarning(
                "Simulação ativa",
                "Reinicie a simulação antes de gerar outra floresta.",
            )
            return

        try:
            linhas = self._ler_inteiro(self.linhas_var, "Linhas")
            colunas = self._ler_inteiro(self.colunas_var, "Colunas")
            intensidade = self._ler_float(
                self.intensidade_var,
                "Intensidade inicial",
            )
            focos = self._ler_inteiro(
                self.focos_var,
                "Quantidade de focos",
            )
            posicao = interpretar_posicao(self.posicao_var.get())
            texto_semente = self.semente_var.get().strip()
            semente = int(texto_semente) if texto_semente else None
            self.matriz_inicial = criar_matriz_inicial(
                linhas,
                colunas,
                intensidade,
                focos,
                posicao,
                semente,
            )
        except ValueError as erro:
            messagebox.showerror("Dados inválidos", str(erro))
            return

        self._limpar_motores()
        self.painel.mostrar_matriz(
            self.matriz_inicial,
            "Condição inicial — clique para editar",
        )
        self.status_var.set("Floresta gerada")
        self._atualizar_status(0)

    def _carregar_matriz(self) -> None:
        if self.em_execucao:
            messagebox.showwarning(
                "Simulação ativa",
                "Reinicie antes de carregar outra matriz.",
            )
            return

        caminho = filedialog.askopenfilename(
            title="Carregar matriz",
            filetypes=[
                ("Matrizes", "*.csv *.txt *.npy"),
                ("CSV", "*.csv"),
                ("Texto", "*.txt"),
                ("NumPy", "*.npy"),
            ],
        )
        if not caminho:
            return

        try:
            matriz = carregar_matriz(caminho)
        except (OSError, ValueError) as erro:
            messagebox.showerror("Falha ao carregar", str(erro))
            return

        self.matriz_inicial = matriz
        self.linhas_var.set(str(matriz.shape[0]))
        self.colunas_var.set(str(matriz.shape[1]))
        self._limpar_motores()
        self.painel.mostrar_matriz(
            matriz,
            f"Matriz carregada — {Path(caminho).name}",
        )
        self.status_var.set("Matriz carregada")

    def _mostrar_instrucao_edicao(self) -> None:
        messagebox.showinfo(
            "Edição manual",
            "Antes de iniciar a simulação, clique em qualquer célula do mapa "
            "e informe uma intensidade entre 0 e 1. Os índices começam em 0. "
            "Ao passar o mouse sobre uma célula, o valor atual é exibido.",
        )

    def _editar_celula(self, linha: int, coluna: int) -> None:
        if self.em_execucao or self.matriz_inicial is None:
            return
        if not (
            0 <= linha < self.matriz_inicial.shape[0]
            and 0 <= coluna < self.matriz_inicial.shape[1]
        ):
            return

        atual = self.matriz_inicial[linha, coluna]
        dialogo = ctk.CTkInputDialog(
            title="Editar intensidade",
            text=(
                f"Célula ({linha}, {coluna})\n"
                f"Valor atual: {atual:.4f}\n"
                "Novo valor entre 0 e 1:"
            ),
        )
        texto = dialogo.get_input()
        if texto is None:
            return

        try:
            valor = float(texto.replace(",", "."))
            if not 0 <= valor <= 1:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Valor inválido",
                "A intensidade deve ser um número entre 0 e 1.",
            )
            return

        self.matriz_inicial[linha, coluna] = valor
        self.painel.atualizar_matriz(
            self.matriz_inicial,
            "Condição inicial — clique para editar",
        )
        self.status_var.set(f"Célula ({linha}, {coluna}) atualizada")

    def _iniciar_simulacao(self) -> None:
        if self.em_execucao and not self.pausada:
            return

        try:
            parametros = self._obter_parametros()
        except ValueError as erro:
            messagebox.showerror("Parâmetros inválidos", str(erro))
            return

        if (
            self.matriz_inicial is None
            or self.matriz_inicial.shape
            != (parametros.linhas, parametros.colunas)
        ):
            messagebox.showerror(
                "Matriz incompatível",
                "Gere ou carregue uma matriz com as dimensões informadas.",
            )
            return

        self.parametros = parametros
        tipo = TipoModelo(self.modelo_var.get())

        if tipo == TipoModelo.COMPARACAO:
            self.comparacao = MotorComparacao(
                self.matriz_inicial,
                parametros,
            )
            self.motor = None
            self.painel.mostrar_comparacao(
                self.matriz_inicial,
                self.matriz_inicial,
                0,
            )
        else:
            modelo = (
                ModeloSemControle(parametros)
                if tipo == TipoModelo.SEM_CONTROLE
                else ModeloComControle(parametros)
            )
            self.motor = MotorSimulacao(
                self.matriz_inicial,
                modelo,
                parametros,
            )
            self.comparacao = None
            self.painel.mostrar_matriz(
                self.matriz_inicial,
                f"{modelo.nome} — iteração 0",
            )

        self.frames_gif = []
        self.intervalo_frames = max(
            1,
            math.ceil(
                (parametros.iteracoes + 1) / PADRAO.limite_frames_gif
            ),
        )
        self._registrar_frame()
        self.em_execucao = True
        self.pausada = False
        self.inicio_cronometro = time.perf_counter()
        self.tempo_acumulado = 0.0
        self.status_var.set("Simulação em execução")
        self._agendar_passo()

    def _agendar_passo(self) -> None:
        if not self.em_execucao or self.pausada or self.parametros is None:
            return
        self.id_after = self.after(
            self.parametros.velocidade_ms,
            self._executar_passo,
        )

    def _executar_passo(self) -> None:
        if not self.em_execucao or self.pausada:
            return

        if self.comparacao is not None:
            matriz_sem, matriz_com = self.comparacao.executar_passo()
            iteracao = self.comparacao.iteracao_atual
            self.painel.mostrar_comparacao(
                matriz_sem,
                matriz_com,
                iteracao,
            )
            concluida = self.comparacao.concluida
        elif self.motor is not None:
            matriz = self.motor.executar_passo()
            iteracao = self.motor.iteracao_atual
            self.painel.atualizar_matriz(
                matriz,
                f"{self.motor.modelo.nome} — iteração {iteracao}",
            )
            concluida = self.motor.concluida
        else:
            return

        self._atualizar_status(iteracao)
        if iteracao % self.intervalo_frames == 0 or concluida:
            self._registrar_frame()

        if concluida:
            self._finalizar_simulacao()
        else:
            self._agendar_passo()

    def _atualizar_status(self, iteracao: int) -> None:
        total = (
            self.parametros.iteracoes
            if self.parametros is not None
            else max(1, self._ler_inteiro(self.iteracoes_var, "Iterações"))
        )
        self.iteracao_var.set(f"Iteração: {iteracao}/{total}")
        self.barra_progresso.set(min(1.0, iteracao / max(1, total)))

        if self.em_execucao and not self.pausada:
            decorrido = self.tempo_acumulado + (
                time.perf_counter() - self.inicio_cronometro
            )
        else:
            decorrido = self.tempo_acumulado
        self.cronometro_var.set(f"Tempo: {formatar_tempo(decorrido)}")

    def _pausar(self) -> None:
        if not self.em_execucao or self.pausada:
            return
        self.pausada = True
        if self.id_after is not None:
            self.after_cancel(self.id_after)
            self.id_after = None
        self.tempo_acumulado += (
            time.perf_counter() - self.inicio_cronometro
        )
        self.inicio_pausa = time.perf_counter()
        self.status_var.set("Simulação pausada")
        iteracao = self._iteracao_atual()
        self._atualizar_status(iteracao)

    def _continuar(self) -> None:
        if not self.em_execucao or not self.pausada:
            return
        self.pausada = False
        self.inicio_cronometro = time.perf_counter()
        self.status_var.set("Simulação em execução")
        self._agendar_passo()

    def _reiniciar(self) -> None:
        if self.id_after is not None:
            self.after_cancel(self.id_after)
            self.id_after = None

        self.em_execucao = False
        self.pausada = False
        self.tempo_acumulado = 0.0
        self.frames_gif = []

        if self.comparacao is not None:
            self.comparacao.reiniciar()
        if self.motor is not None:
            self.motor.reiniciar()

        self._limpar_motores()
        if self.matriz_inicial is not None:
            self.painel.mostrar_matriz(
                self.matriz_inicial,
                "Condição inicial — clique para editar",
            )
        self.status_var.set("Simulação reiniciada")
        self.iteracao_var.set("Iteração: 0")
        self.cronometro_var.set("Tempo: 00:00:00")
        self.barra_progresso.set(0)

    def _limpar_motores(self) -> None:
        self.motor = None
        self.comparacao = None
        self.parametros = None
        self.frames_gif = []
        self.em_execucao = False
        self.pausada = False

    def _iteracao_atual(self) -> int:
        if self.comparacao is not None:
            return self.comparacao.iteracao_atual
        if self.motor is not None:
            return self.motor.iteracao_atual
        return 0

    def _finalizar_simulacao(self) -> None:
        self.em_execucao = False
        self.pausada = False
        self.tempo_acumulado += (
            time.perf_counter() - self.inicio_cronometro
        )
        self.status_var.set("Simulação concluída")
        self._atualizar_status(self._iteracao_atual())
        self._mostrar_resultados()

    def _mostrar_resultados(self) -> None:
        if self.comparacao is not None:
            JanelaResultados(
                self,
                self.comparacao.sem_controle.historico,
                self.comparacao.com_controle.historico,
                "Sem controle",
            )
        elif self.motor is not None:
            JanelaResultados(
                self,
                self.motor.historico,
                titulo_modelo=self.motor.modelo.nome,
            )

    def _salvar_imagem(self) -> None:
        caminho = filedialog.asksaveasfilename(
            title="Salvar imagem da simulação",
            defaultextension=".png",
            initialfile="simulacao_incendio.png",
            filetypes=[
                ("Imagem PNG", "*.png"),
                ("Imagem JPEG", "*.jpg *.jpeg"),
                ("Documento PDF", "*.pdf"),
                ("Imagem vetorial SVG", "*.svg"),
            ],
        )
        if not caminho:
            return

        caminho_path = Path(caminho)
        if not caminho_path.suffix:
            caminho_path = caminho_path.with_suffix(".png")

        try:
            arquivo_criado = self.painel.salvar_imagem(caminho_path)
        except (OSError, ValueError) as erro:
            messagebox.showerror("Falha ao salvar", str(erro))
            return

        messagebox.showinfo(
            "Imagem salva",
            f"Arquivo criado em:\n{arquivo_criado}",
        )

    def _exportar_resultados(self) -> None:
        if self.parametros is None:
            messagebox.showwarning(
                "Sem resultados",
                "Inicie uma simulação antes de exportar.",
            )
            return

        caminho = filedialog.asksaveasfilename(
            title="Exportar resultados",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="resultados_incendio.csv",
        )
        if not caminho:
            return

        try:
            if self.comparacao is not None:
                arquivos = exportar_resultado_comparacao(
                    caminho,
                    self.comparacao.sem_controle.historico,
                    self.comparacao.com_controle.historico,
                    self.comparacao.sem_controle.matriz_atual,
                    self.comparacao.com_controle.matriz_atual,
                    self.parametros,
                )
            elif self.motor is not None:
                arquivos = exportar_resultado_unico(
                    caminho,
                    self.motor.historico,
                    self.motor.matriz_atual,
                    self.parametros,
                    self.motor.modelo.nome,
                )
            else:
                return
        except OSError as erro:
            messagebox.showerror("Falha na exportação", str(erro))
            return

        lista = "\n".join(str(arquivo) for arquivo in arquivos)
        messagebox.showinfo(
            "Exportação concluída",
            f"Arquivos criados:\n{lista}",
        )

    def _registrar_frame(self) -> None:
        iteracao = self._iteracao_atual()

        if self.comparacao is not None:
            frame = self.painel.comparacao_para_imagem(
                self.comparacao.sem_controle.matriz_atual,
                self.comparacao.com_controle.matriz_atual,
                iteracao=iteracao,
            )
        elif self.motor is not None:
            frame = self.painel.matriz_para_imagem(
                self.motor.matriz_atual,
                titulo=self.motor.modelo.nome,
                iteracao=iteracao,
            )
        elif self.matriz_inicial is not None:
            frame = self.painel.matriz_para_imagem(
                self.matriz_inicial,
                titulo="Condição inicial",
                iteracao=0,
            )
        else:
            return

        # A paleta reduz significativamente o consumo de memória sem perder
        # a legibilidade necessária para a animação em GIF.
        frame_gif = frame.convert(
            "P",
            palette=Image.Palette.ADAPTIVE,
            colors=256,
        )
        self.frames_gif.append(frame_gif)

    def _exportar_gif(self) -> None:
        if not self.frames_gif:
            messagebox.showwarning(
                "Sem animação",
                "Execute ao menos uma etapa da simulação antes de exportar.",
            )
            return

        caminho = filedialog.asksaveasfilename(
            title="Exportar animação",
            defaultextension=".gif",
            filetypes=[("GIF animado", "*.gif")],
            initialfile="animacao_incendio.gif",
        )
        if not caminho:
            return

        duracao = (
            self.parametros.velocidade_ms * self.intervalo_frames
            if self.parametros is not None
            else PADRAO.velocidade_ms
        )
        try:
            self.frames_gif[0].save(
                caminho,
                save_all=True,
                append_images=self.frames_gif[1:],
                duration=max(20, duracao),
                loop=0,
                optimize=False,
                disposal=2,
            )
        except OSError as erro:
            messagebox.showerror("Falha ao exportar GIF", str(erro))
            return
        messagebox.showinfo("GIF exportado", f"Arquivo criado:\n{caminho}")

    @staticmethod
    def _alterar_tema(valor: str) -> None:
        mapa = {
            "Escuro": "dark",
            "Claro": "light",
            "Sistema": "system",
        }
        ctk.set_appearance_mode(mapa[valor])

    def _abrir_sobre(self) -> None:
        janela = ctk.CTkToplevel(self)
        janela.title("Sobre o modelo matemático")
        janela.geometry("760x610")
        janela.minsize(620, 480)

        texto = ctk.CTkTextbox(
            janela,
            wrap="word",
            font=ctk.CTkFont(family="Consolas", size=14),
        )
        texto.pack(fill="both", expand=True, padx=16, pady=16)
        texto.insert(
            "1.0",
            (
                "SIMULADOR DE INCÊNDIOS FLORESTAIS\n"
                "Método de Euler explícito\n\n"
                "Estado:\n"
                "  y_i^n representa a intensidade do fogo na célula i "
                "durante a iteração n.\n\n"
                "Vizinhança de Von Neumann:\n"
                "  N(i) contém os vizinhos acima, abaixo, à esquerda e "
                "à direita.\n\n"
                "Modelo sem controle:\n"
                "  y_i^(n+1) = y_i^n + h*k*Σ y_j^n*(1-y_i^n)\n\n"
                "Modelo com controle:\n"
                "  y_i^(n+1) = y_i^n + h*["
                "k*Σ y_j^n*(1-y_i^n) - γ*(y_i^n)^2]\n\n"
                "A atualização é síncrona: a matriz inteira da iteração "
                "seguinte é calculada a partir de uma cópia lógica do estado "
                "atual. A soma dos vizinhos é vetorizada com NumPy.\n\n"
                "Quando a opção de limite físico está ativada, o resultado "
                "de cada passo é projetado no intervalo [0, 1]. Desative a "
                "opção para observar a fórmula de Euler sem essa salvaguarda "
                "numérica.\n\n"
                "Desenvolvido para apresentação científica no SIA-UFV."
            ),
        )
        texto.configure(state="disabled")
