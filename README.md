# Simulador de Incêndios Florestais — Método de Euler

Software científico com interface gráfica para visualizar a propagação de
incêndios florestais em uma malha bidimensional. O projeto implementa o modelo
sem controle e o modelo com controle dos bombeiros, ambos discretizados pelo
Método de Euler explícito e usando vizinhança de Von Neumann.


## Funcionalidades

- Modelo sem controle.
- Modelo com termo de controle `γ(y_i^n)^2`.
- Comparação animada lado a lado.
- Matrizes configuráveis, inclusive 100×100 e 200×200.
- Focos aleatórios ou posição inicial informada.
- Edição manual de qualquer célula por clique.
- Importação de matrizes CSV, TXT e NPY.
- Pausa, continuação e reinício.
- Barra de progresso, contador de iterações e cronômetro.
- Tema claro, escuro ou definido pelo sistema.
- Mapa de calor com legenda de intensidade.
- Gráficos finais de média, máximo e células queimando.
- Exportação dos resultados em CSV, da matriz final e dos metadados.
- Captura da visualização em PNG, PDF ou SVG.
- Exportação da animação em GIF.
- Janela “Sobre” com as equações e decisões numéricas.

## Modelos matemáticos

Para uma célula `i`, seja `N(i)` sua vizinhança de Von Neumann, formada por
cima, baixo, esquerda e direita.

### Sem controle

```text
y_i^(n+1) = y_i^n
          + h*k*Σ_{j ∈ N(i)} y_j^n*(1 - y_i^n)
```

### Com controle

```text
y_i^(n+1) = y_i^n
          + h*[k*Σ_{j ∈ N(i)} y_j^n*(1 - y_i^n)
               - γ*(y_i^n)^2]
```

A atualização é síncrona. Isso significa que todas as células de `n+1` são
calculadas usando somente valores da iteração `n`. Não há alteração de uma
célula que possa contaminar o cálculo das células seguintes no mesmo passo.

Por padrão, a interface executa literalmente o passo de Euler descrito no
relatório, sem projeção adicional. A opção de limitar o resultado ao intervalo
físico `[0, 1]` está disponível, mas permanece desativada por padrão.

## Vizinhança de Von Neumann vetorizada

O arquivo `modelos.py` cria uma matriz de soma inicialmente nula e acrescenta
quatro versões deslocadas do estado atual:

```python
soma[1:, :] += matriz[:-1, :]
soma[:-1, :] += matriz[1:, :]
soma[:, 1:] += matriz[:, :-1]
soma[:, :-1] += matriz[:, 1:]
```

Fora das bordas, a intensidade é considerada zero. A complexidade por iteração
é `O(linhas × colunas)`, mas as operações são executadas internamente pelo
NumPy, evitando um laço Python para cada célula.

## Estrutura do projeto

```text
simulador_incendios_euler/
├── main.py
├── interface.py
├── simulacao.py
├── modelos.py
├── visualizacao.py
├── utils.py
├── configuracoes.py
├── requirements.txt
├── README.md
├── executar_windows.bat
├── executar_linux_macos.sh
├── exemplos/
│   └── matriz_exemplo.csv
└── tests/
    ├── test_modelos.py
    └── test_simulacao.py
```

### `main.py`

É o ponto de entrada. Cria uma instância de `AplicacaoIncendio` e inicia o laço
de eventos do Tkinter. Nenhuma lógica matemática fica neste arquivo.

### `configuracoes.py`

Centraliza valores padrão, a enumeração dos modelos e a classe imutável
`ParametrosSimulacao`. A validação em um único local evita regras divergentes
entre interface e motor numérico.

### `modelos.py`

Contém somente o núcleo matemático:

- `soma_vizinhos_von_neumann`;
- `ModeloIncendio`;
- `ModeloSemControle`;
- `ModeloComControle`.

Cada chamada de `passo` recebe a matriz atual e devolve uma nova matriz. Esse
contrato torna os modelos independentes da interface.

### `simulacao.py`

Mantém a evolução temporal e coleta os indicadores. `MotorSimulacao` controla
um modelo; `MotorComparacao` mantém dois motores iniciados com a mesma matriz.
`HistoricoSimulacao` guarda as séries usadas nos gráficos e arquivos CSV.

### `visualizacao.py`

Incorpora figuras Matplotlib ao Tkinter com `FigureCanvasTkAgg`. Define o
gradiente verde–amarelo–laranja–vermelho, atualiza os mapas de calor, cria os
gráficos finais e converte matrizes em imagens para o GIF.

### `utils.py`

Reúne funções sem estado para:

- interpretar a posição do foco;
- criar a condição inicial;
- carregar e validar matrizes;
- salvar matrizes;
- exportar históricos e metadados;
- formatar o cronômetro.

### `interface.py`

Coordena as ações do usuário. A classe `AplicacaoIncendio` lê e valida campos,
instancia os motores, agenda cada passo com `after`, atualiza a visualização e
chama as rotinas de exportação. O cálculo não usa `sleep`, portanto a janela
permanece responsiva.

## Comunicação entre os módulos

```text
main
  └── interface
        ├── configuracoes
        ├── modelos
        ├── simulacao
        │     └── modelos
        ├── visualizacao
        │     └── simulacao
        └── utils
              └── simulacao
```

A interface é a camada coordenadora. Os modelos não conhecem Tkinter,
Matplotlib nem arquivos, o que facilita testes e futuras alterações.

## Instalação

Requer Python 3.10 ou superior.

### Windows

```powershell
cd simulador_incendios_euler
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

Também é possível executar `executar_windows.bat`.

### Linux/macOS

```bash
cd simulador_incendios_euler
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

No Linux, o pacote do sistema para Tk pode ser necessário. Em distribuições
baseadas em Debian/Ubuntu, normalmente é fornecido por `python3-tk`.

## Uso

1. Escolha o modelo.
2. Defina linhas, colunas, `h`, `k`, `γ`, iterações e velocidade.
3. Informe a intensidade e a quantidade de focos.
4. Opcionalmente, informe `linha,coluna`. Os índices começam em zero.
5. Gere a floresta ou carregue uma matriz.
6. Clique em uma célula para editar sua intensidade.
7. Inicie a simulação.
8. Pause, continue ou reinicie quando necessário.
9. Ao final, analise os gráficos e exporte os resultados.

No modo comparação, os dois modelos usam exatamente a mesma condição inicial.

## Formatos de entrada

### CSV

```text
0,0,0,0
0,0.8,0,0
0,0,0,0
```

### TXT

Pode usar valores separados por vírgulas ou espaços.

### NPY

Matriz NumPy bidimensional, sem objetos serializados.

Todos os valores de entrada devem pertencer ao intervalo `[0, 1]`.

## Arquivos exportados

No modelo único:

- `resultados.csv`;
- `resultados_matriz_final.csv`;
- `resultados_metadados.json`.

Na comparação:

- `resultados.csv`;
- `resultados_sem_controle.csv`;
- `resultados_com_controle.csv`;
- `resultados_metadados.json`.

O limiar usado para contar uma célula como “queimando” é `0.01` e pode ser
alterado em `configuracoes.py`.

## GIF e consumo de memória

Para evitar consumo excessivo em simulações longas, são armazenados no máximo
aproximadamente 250 quadros. O intervalo de amostragem é calculado
automaticamente a partir do total de iterações. A simulação numérica continua
executando todas as iterações; apenas a gravação visual é subamostrada.

## Testes

Os testes usam apenas `unittest`, disponível na biblioteca padrão:

```bash
python -m unittest discover -s tests -v
```

Eles verificam a soma de Von Neumann, um passo de cada modelo e o registro do
histórico.

## Recomendações para apresentação

- Use uma semente fixa, como `42`, para repetir a mesma demonstração.
- Comece com 50×50 e velocidade entre 30 e 60 ms.
- Mostre primeiro o modelo sem controle.
- Em seguida, mantenha os mesmos parâmetros e use a comparação lado a lado.
- Explique que `γ` atua como termo quadrático de redução da intensidade.
- Exporte previamente um GIF como contingência para o caso de problemas na
  projeção ou no computador do evento.
