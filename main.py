"""Ponto de entrada do Simulador de Incêndios Florestais."""

from interface import AplicacaoIncendio


def main() -> None:
    """Inicializa a interface gráfica e o laço de eventos."""
    aplicacao = AplicacaoIncendio()
    aplicacao.mainloop()


if __name__ == "__main__":
    main()
