# 🚀 Guia de Deploy - Agentic LGPD

Este guia explica como fazer o deploy do Agentic LGPD no **Hugging Face Spaces** (recomendado para Gradio).

---

## 📋 Pré-requisitos

1. Conta no [Hugging Face](https://huggingface.co/join)
2. Conta na [OpenAI](https://platform.openai.com) com API key
3. Python 3.10+ instalado localmente

---

## 🔧 Passo 1: Gerar os Dados Localmente

Antes de fazer o deploy, você precisa gerar os arquivos de dados:

```bash
# Clone o repositório
git clone https://github.com/ma-serra/AgenticLGPD.git
cd AgenticLGPD

# Instale as dependências
pip install -r requirements.txt
pip install beautifulsoup4 requests

# Execute o script de setup
python setup_data.py
```

Isso vai gerar 4 arquivos:
- `lei_chunks_com_metadados_lei.json`
- `lei_chunks_com_metadados_jurisprudencia.json`
- `lei_faiss_lei.index`
- `lei_faiss_jurisprudencia.index`

---

## 🤗 Passo 2: Criar Space no Hugging Face

### 2.1 Criar novo Space

1. Acesse [huggingface.co/new-space](https://huggingface.co/new-space)
2. Preencha:
   - **Space name**: `agentic-lgpd`
   - **License**: MIT
   - **SDK**: Gradio
   - **Hardware**: CPU basic (gratuito)
3. Clique em **Create Space**

### 2.2 Clonar o Space

```bash
git clone https://huggingface.co/spaces/SEU_USERNAME/agentic-lgpd
cd agentic-lgpd
```

### 2.3 Copiar os arquivos

Copie todos os arquivos do repositório AgenticLGPD para a pasta do Space:

```bash
# Copie os arquivos principais
cp /caminho/AgenticLGPD/app.py .
cp /caminho/AgenticLGPD/requirements.txt .

# Copie os arquivos de dados gerados
cp /caminho/AgenticLGPD/lei_chunks_com_metadados_lei.json .
cp /caminho/AgenticLGPD/lei_chunks_com_metadados_jurisprudencia.json .
cp /caminho/AgenticLGPD/lei_faiss_lei.index .
cp /caminho/AgenticLGPD/lei_faiss_jurisprudencia.index .
```

### 2.4 Fazer commit e push

```bash
git add .
git commit -m "Deploy Agentic LGPD"
git push
```

---

## 🔑 Passo 3: Configurar API Key da OpenAI

1. No Hugging Face, vá para **Settings** do seu Space
2. Role até **Repository secrets**
3. Clique em **New secret**
4. Adicione:
   - **Name**: `OPENAI_API_KEY`
   - **Value**: `sk-...` (sua API key)
5. Clique em **Add new secret**

---

## ✅ Passo 4: Verificar Deploy

1. Aguarde o build completar (pode levar alguns minutos)
2. Acesse: `https://huggingface.co/spaces/SEU_USERNAME/agentic-lgpd`
3. Teste fazendo uma pergunta sobre LGPD!

---

## 🔄 Alternativa: Deploy via GitHub Actions

Você pode automatizar o deploy criando um workflow. Adicione o arquivo `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Hugging Face

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install huggingface_hub
      
      - name: Generate data files
        run: python setup_data.py
      
      - name: Push to Hugging Face
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          huggingface-cli login --token $HF_TOKEN
          huggingface-cli upload SEU_USERNAME/agentic-lgpd . --repo-type space
```

Para usar, adicione `HF_TOKEN` nos secrets do GitHub.

---

## 🐛 Troubleshooting

### Erro: "Module not found"
Verifique se o `requirements.txt` contém todas as dependências.

### Erro: "OPENAI_API_KEY not found"
Configure o secret no Hugging Face Space.

### Erro: "File not found" para JSON/index
Execute `python setup_data.py` antes do deploy.

### Build muito lento
O primeiro build pode demorar até 10 minutos devido ao download dos modelos.

---

## 📁 Estrutura Final do Space

```
agentic-lgpd/
├── app.py
├── requirements.txt
├── setup_data.py
├── lei_chunks_com_metadados_lei.json
├── lei_chunks_com_metadados_jurisprudencia.json
├── lei_faiss_lei.index
└── lei_faiss_jurisprudencia.index
```

---

## 🎯 Próximos Passos

Após o deploy bem-sucedido, você pode:

1. **Personalizar a interface**: Edite o título e descrição no `app.py`
2. **Adicionar mais dados**: Inclua mais jurisprudência no `setup_data.py`
3. **Melhorar os prompts**: Ajuste os prompts do GPT para respostas mais precisas
4. **Monitorar uso**: Acompanhe métricas no painel do Hugging Face

---

## 📞 Suporte

- Documentação Gradio: https://gradio.app/docs
- Documentação HF Spaces: https://huggingface.co/docs/hub/spaces
- Issues do projeto: https://github.com/ma-serra/AgenticLGPD/issues
