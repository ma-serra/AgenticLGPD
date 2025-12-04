"""
Script para gerar os arquivos de dados necessários para o Agentic LGPD.

Este script:
1. Baixa o texto da LGPD do site do Planalto
2. Processa e divide em chunks
3. Gera embeddings e índices FAISS
4. Salva os arquivos JSON e índices

Uso:
    python setup_data.py
"""

import json
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------
# Configuração
# ---------------------------
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE = 500  # caracteres por chunk
CHUNK_OVERLAP = 50  # sobreposição entre chunks

# ---------------------------
# Texto da LGPD (Lei 13.709/2018) - Versão resumida para exemplo
# ---------------------------
LGPD_TEXT = """
CAPÍTULO I - DISPOSIÇÕES PRELIMINARES

Art. 1º Esta Lei dispõe sobre o tratamento de dados pessoais, inclusive nos meios digitais, por pessoa natural ou por pessoa jurídica de direito público ou privado, com o objetivo de proteger os direitos fundamentais de liberdade e de privacidade e o livre desenvolvimento da personalidade da pessoa natural.

Parágrafo único. As normas gerais contidas nesta Lei são de interesse nacional e devem ser observadas pela União, Estados, Distrito Federal e Municípios.

Art. 2º A disciplina da proteção de dados pessoais tem como fundamentos:
I - o respeito à privacidade;
II - a autodeterminação informativa;
III - a liberdade de expressão, de informação, de comunicação e de opinião;
IV - a inviolabilidade da intimidade, da honra e da imagem;
V - o desenvolvimento econômico e tecnológico e a inovação;
VI - a livre iniciativa, a livre concorrência e a defesa do consumidor;
VII - os direitos humanos, o livre desenvolvimento da personalidade, a dignidade e o exercício da cidadania pelas pessoas naturais.

Art. 3º Esta Lei aplica-se a qualquer operação de tratamento realizada por pessoa natural ou por pessoa jurídica de direito público ou privado, independentemente do meio, do país de sua sede ou do país onde estejam localizados os dados, desde que:
I - a operação de tratamento seja realizada no território nacional;
II - a atividade de tratamento tenha por objetivo a oferta ou o fornecimento de bens ou serviços ou o tratamento de dados de indivíduos localizados no território nacional;
III - os dados pessoais objeto do tratamento tenham sido coletados no território nacional.

Art. 4º Esta Lei não se aplica ao tratamento de dados pessoais:
I - realizado por pessoa natural para fins exclusivamente particulares e não econômicos;
II - realizado para fins exclusivamente jornalísticos, artísticos ou acadêmicos;
III - realizado para fins exclusivos de segurança pública, defesa nacional, segurança do Estado ou atividades de investigação e repressão de infrações penais;
IV - provenientes de fora do território nacional e que não sejam objeto de comunicação, uso compartilhado de dados com agentes de tratamento brasileiros ou objeto de transferência internacional de dados com outro país que não o de proveniência.

Art. 5º Para os fins desta Lei, considera-se:
I - dado pessoal: informação relacionada a pessoa natural identificada ou identificável;
II - dado pessoal sensível: dado pessoal sobre origem racial ou étnica, convicção religiosa, opinião política, filiação a sindicato ou a organização de caráter religioso, filosófico ou político, dado referente à saúde ou à vida sexual, dado genético ou biométrico, quando vinculado a uma pessoa natural;
III - dado anonimizado: dado relativo a titular que não possa ser identificado;
IV - banco de dados: conjunto estruturado de dados pessoais;
V - titular: pessoa natural a quem se referem os dados pessoais;
VI - controlador: pessoa natural ou jurídica que toma decisões referentes ao tratamento de dados pessoais;
VII - operador: pessoa natural ou jurídica que realiza o tratamento de dados pessoais em nome do controlador;
VIII - encarregado: pessoa indicada pelo controlador e operador para atuar como canal de comunicação entre o controlador, os titulares dos dados e a Autoridade Nacional de Proteção de Dados (ANPD);
IX - agentes de tratamento: o controlador e o operador;
X - tratamento: toda operação realizada com dados pessoais;
XI - anonimização: utilização de meios técnicos razoáveis e disponíveis no momento do tratamento;
XII - consentimento: manifestação livre, informada e inequívoca pela qual o titular concorda com o tratamento de seus dados pessoais para uma finalidade determinada;
XIII - bloqueio: suspensão temporária de qualquer operação de tratamento;
XIV - eliminação: exclusão de dado ou de conjunto de dados armazenados em banco de dados;
XV - transferência internacional de dados: transferência de dados pessoais para país estrangeiro ou organismo internacional do qual o país seja membro;
XVI - uso compartilhado de dados: comunicação, difusão, transferência internacional, interconexão de dados pessoais ou tratamento compartilhado de bancos de dados pessoais por órgãos e entidades públicos;
XVII - relatório de impacto à proteção de dados pessoais: documentação do controlador que contém a descrição dos processos de tratamento de dados pessoais;
XVIII - órgão de pesquisa: órgão ou entidade da administração pública direta ou indireta ou pessoa jurídica de direito privado sem fins lucrativos legalmente constituída;
XIX - autoridade nacional: órgão da administração pública responsável por zelar, implementar e fiscalizar o cumprimento desta Lei em todo o território nacional.

CAPÍTULO II - DO TRATAMENTO DE DADOS PESSOAIS

Art. 6º As atividades de tratamento de dados pessoais deverão observar a boa-fé e os seguintes princípios:
I - finalidade: realização do tratamento para propósitos legítimos, específicos, explícitos e informados ao titular;
II - adequação: compatibilidade do tratamento com as finalidades informadas ao titular;
III - necessidade: limitação do tratamento ao mínimo necessário para a realização de suas finalidades;
IV - livre acesso: garantia, aos titulares, de consulta facilitada e gratuita sobre a forma e a duração do tratamento;
V - qualidade dos dados: garantia, aos titulares, de exatidão, clareza, relevância e atualização dos dados;
VI - transparência: garantia, aos titulares, de informações claras, precisas e facilmente acessíveis;
VII - segurança: utilização de medidas técnicas e administrativas aptas a proteger os dados pessoais;
VIII - prevenção: adoção de medidas para prevenir a ocorrência de danos em virtude do tratamento de dados pessoais;
IX - não discriminação: impossibilidade de realização do tratamento para fins discriminatórios ilícitos ou abusivos;
X - responsabilização e prestação de contas: demonstração, pelo agente, da adoção de medidas eficazes.

Art. 7º O tratamento de dados pessoais somente poderá ser realizado nas seguintes hipóteses:
I - mediante o fornecimento de consentimento pelo titular;
II - para o cumprimento de obrigação legal ou regulatória pelo controlador;
III - pela administração pública, para o tratamento e uso compartilhado de dados necessários à execução de políticas públicas;
IV - para a realização de estudos por órgão de pesquisa;
V - quando necessário para a execução de contrato ou de procedimentos preliminares relacionados a contrato do qual seja parte o titular;
VI - para o exercício regular de direitos em processo judicial, administrativo ou arbitral;
VII - para a proteção da vida ou da incolumidade física do titular ou de terceiro;
VIII - para a tutela da saúde, exclusivamente, em procedimento realizado por profissionais de saúde, serviços de saúde ou autoridade sanitária;
IX - quando necessário para atender aos interesses legítimos do controlador ou de terceiro;
X - para a proteção do crédito.

CAPÍTULO III - DOS DIREITOS DO TITULAR

Art. 17. Toda pessoa natural tem assegurada a titularidade de seus dados pessoais e garantidos os direitos fundamentais de liberdade, de intimidade e de privacidade.

Art. 18. O titular dos dados pessoais tem direito a obter do controlador, em relação aos dados do titular por ele tratados, a qualquer momento e mediante requisição:
I - confirmação da existência de tratamento;
II - acesso aos dados;
III - correção de dados incompletos, inexatos ou desatualizados;
IV - anonimização, bloqueio ou eliminação de dados desnecessários, excessivos ou tratados em desconformidade;
V - portabilidade dos dados a outro fornecedor de serviço ou produto;
VI - eliminação dos dados pessoais tratados com o consentimento do titular;
VII - informação das entidades públicas e privadas com as quais o controlador realizou uso compartilhado de dados;
VIII - informação sobre a possibilidade de não fornecer consentimento e sobre as consequências da negativa;
IX - revogação do consentimento.

Art. 19. A confirmação de existência ou o acesso a dados pessoais serão providenciados, mediante requisição do titular:
I - em formato simplificado, imediatamente; ou
II - por meio de declaração clara e completa, que indique a origem dos dados, a inexistência de registro, os critérios utilizados e a finalidade do tratamento, no prazo de até 15 dias.

Art. 20. O titular dos dados tem direito a solicitar a revisão de decisões tomadas unicamente com base em tratamento automatizado de dados pessoais que afetem seus interesses.

CAPÍTULO IV - DO TRATAMENTO DE DADOS PESSOAIS PELO PODER PÚBLICO

Art. 23. O tratamento de dados pessoais pelas pessoas jurídicas de direito público deverá ser realizado para o atendimento de sua finalidade pública, na persecução do interesse público, com o objetivo de executar as competências legais ou cumprir as atribuições legais do serviço público.

CAPÍTULO V - DA TRANSFERÊNCIA INTERNACIONAL DE DADOS

Art. 33. A transferência internacional de dados pessoais somente é permitida nos seguintes casos:
I - para países ou organismos internacionais que proporcionem grau de proteção de dados pessoais adequado ao previsto nesta Lei;
II - quando o controlador oferecer e comprovar garantias de cumprimento dos princípios, dos direitos do titular e do regime de proteção de dados previstos nesta Lei;
III - quando a transferência for necessária para a cooperação jurídica internacional entre órgãos públicos de inteligência, de investigação e de persecução;
IV - quando a transferência for necessária para a proteção da vida ou da incolumidade física do titular ou de terceiro;
V - quando a autoridade nacional autorizar a transferência;
VI - quando a transferência resultar em compromisso assumido em acordo de cooperação internacional;
VII - quando a transferência for necessária para a execução de política pública ou atribuição legal do serviço público;
VIII - quando o titular tiver fornecido o seu consentimento específico e em destaque para a transferência;
IX - quando necessário para atender às hipóteses previstas nos incisos II, V e VI do art. 7º desta Lei.

CAPÍTULO VI - DOS AGENTES DE TRATAMENTO DE DADOS PESSOAIS

Art. 37. O controlador e o operador devem manter registro das operações de tratamento de dados pessoais que realizarem, especialmente quando baseado no legítimo interesse.

Art. 38. A autoridade nacional poderá determinar ao controlador que elabore relatório de impacto à proteção de dados pessoais, inclusive de dados sensíveis, referente a suas operações de tratamento de dados.

Art. 41. O controlador deverá indicar encarregado pelo tratamento de dados pessoais.

Art. 42. O controlador ou o operador que, em razão do exercício de atividade de tratamento de dados pessoais, causar a outrem dano patrimonial, moral, individual ou coletivo, em violação à legislação de proteção de dados pessoais, é obrigado a repará-lo.

CAPÍTULO VII - DA SEGURANÇA E DAS BOAS PRÁTICAS

Art. 46. Os agentes de tratamento devem adotar medidas de segurança, técnicas e administrativas aptas a proteger os dados pessoais de acessos não autorizados e de situações acidentais ou ilícitas de destruição, perda, alteração, comunicação ou qualquer forma de tratamento inadequado ou ilícito.

Art. 48. O controlador deverá comunicar à autoridade nacional e ao titular a ocorrência de incidente de segurança que possa acarretar risco ou dano relevante aos titulares.

Art. 50. Os controladores e operadores, no âmbito de suas competências, pelo tratamento de dados pessoais, individualmente ou por meio de associações, poderão formular regras de boas práticas e de governança que estabeleçam as condições de organização, o regime de funcionamento, os procedimentos, incluindo reclamações e petições de titulares, as normas de segurança, os padrões técnicos, as obrigações específicas para os diversos envolvidos no tratamento, as ações educativas, os mecanismos internos de supervisão e de mitigação de riscos e outros aspectos relacionados ao tratamento de dados pessoais.

CAPÍTULO VIII - DA FISCALIZAÇÃO

Art. 52. Os agentes de tratamento de dados, em razão das infrações cometidas às normas previstas nesta Lei, ficam sujeitos às seguintes sanções administrativas aplicáveis pela autoridade nacional:
I - advertência, com indicação de prazo para adoção de medidas corretivas;
II - multa simples, de até 2% do faturamento da pessoa jurídica de direito privado, grupo ou conglomerado no Brasil no seu último exercício, excluídos os tributos, limitada, no total, a R$ 50.000.000,00 por infração;
III - multa diária, observado o limite total a que se refere o inciso II;
IV - publicização da infração após devidamente apurada e confirmada a sua ocorrência;
V - bloqueio dos dados pessoais a que se refere a infração até a sua regularização;
VI - eliminação dos dados pessoais a que se refere a infração;
VII - suspensão parcial do funcionamento do banco de dados a que se refere a infração pelo período máximo de 6 meses, prorrogável por igual período, até a regularização da atividade de tratamento pelo controlador;
VIII - suspensão do exercício da atividade de tratamento dos dados pessoais a que se refere a infração pelo período máximo de 6 meses, prorrogável por igual período;
IX - proibição parcial ou total do exercício de atividades relacionadas a tratamento de dados.

CAPÍTULO IX - DA AUTORIDADE NACIONAL DE PROTEÇÃO DE DADOS (ANPD)

Art. 55-A. Fica criada, sem aumento de despesa, a Autoridade Nacional de Proteção de Dados (ANPD), órgão da administração pública federal, integrante da Presidência da República.

Art. 55-J. Compete à ANPD:
I - zelar pela proteção dos dados pessoais;
II - zelar pela observância dos segredos comercial e industrial;
III - elaborar diretrizes para a Política Nacional de Proteção de Dados Pessoais e da Privacidade;
IV - fiscalizar e aplicar sanções em caso de tratamento de dados realizado em descumprimento à legislação;
V - apreciar petições de titular contra controlador;
VI - promover na população o conhecimento das normas e das políticas públicas sobre proteção de dados pessoais;
VII - promover e elaborar estudos sobre as práticas nacionais e internacionais de proteção de dados pessoais e privacidade;
VIII - estimular a adoção de padrões para serviços e produtos que facilitem o exercício de controle dos titulares sobre seus dados pessoais;
IX - promover ações de cooperação com autoridades de proteção de dados pessoais de outros países;
X - dispor sobre as formas de publicidade das operações de tratamento de dados pessoais;
XI - solicitar, a qualquer momento, às entidades do poder público que realizem operações de tratamento de dados pessoais informe específico sobre o âmbito, a natureza dos dados e os demais detalhes do tratamento realizado;
XII - elaborar relatórios de gestão anuais acerca de suas atividades;
XIII - editar regulamentos e procedimentos sobre proteção de dados pessoais e privacidade;
XIV - ouvir os agentes de tratamento e a sociedade em matérias de interesse relevante;
XV - arrecadar e aplicar suas receitas e publicar, no relatório de gestão a que se refere o inciso XII do caput deste artigo, o detalhamento de suas receitas e despesas;
XVI - realizar auditorias, ou determinar sua realização, no âmbito da atividade de fiscalização de que trata o inciso IV e com a devida observância do disposto no inciso II do caput deste artigo;
XVII - celebrar, a qualquer momento, compromisso com agentes de tratamento para eliminar irregularidade, incerteza jurídica ou situação contenciosa no âmbito de processos administrativos;
XVIII - editar normas, orientações e procedimentos simplificados e diferenciados, inclusive quanto aos prazos, para que microempresas e empresas de pequeno porte, bem como iniciativas empresariais de caráter incremental ou disruptivo que se autodeclarem startups ou empresas de inovação, possam adequar-se a esta Lei;
XIX - garantir que o tratamento de dados de idosos seja efetuado de maneira simples, clara, acessível e adequada ao seu entendimento;
XX - deliberar, na esfera administrativa, em caráter terminativo, sobre a interpretação desta Lei, as suas competências e os casos omissos.
"""

# ---------------------------
# Jurisprudência de exemplo
# ---------------------------
JURISPRUDENCIA_TEXT = """
JURISPRUDÊNCIA SOBRE LGPD

CASO 1 - VAZAMENTO DE DADOS
Tribunal: TJSP
Processo: 1000000-00.2021.8.26.0000
Ementa: RESPONSABILIDADE CIVIL - VAZAMENTO DE DADOS PESSOAIS - LGPD - Autor que teve seus dados pessoais vazados pela empresa ré - Falha na segurança da informação - Violação dos artigos 46 e 48 da Lei 13.709/2018 - Dano moral configurado - Indenização devida.
Decisão: A empresa foi condenada ao pagamento de indenização por danos morais no valor de R$ 10.000,00, em razão do vazamento de dados pessoais do autor, caracterizando violação à LGPD.

CASO 2 - COMPARTILHAMENTO INDEVIDO DE DADOS
Tribunal: TJRJ  
Processo: 2000000-00.2022.8.19.0000
Ementa: AÇÃO INDENIZATÓRIA - LGPD - COMPARTILHAMENTO INDEVIDO DE DADOS PESSOAIS - Empresa que compartilhou dados do consumidor com terceiros sem consentimento - Violação do artigo 7º, inciso I, da Lei 13.709/2018 - Dano moral in re ipsa.
Decisão: Condenação da empresa ao pagamento de R$ 15.000,00 por danos morais, por compartilhamento de dados pessoais sem o consentimento do titular.

CASO 3 - DIREITO DE ACESSO AOS DADOS
Tribunal: TJMG
Processo: 3000000-00.2022.8.13.0000
Ementa: OBRIGAÇÃO DE FAZER - LGPD - DIREITO DE ACESSO AOS DADOS - Titular que requereu acesso aos seus dados pessoais tratados pela empresa - Recusa injustificada - Violação do artigo 18, inciso II, da Lei 13.709/2018.
Decisão: Empresa condenada a fornecer ao autor, no prazo de 15 dias, declaração completa sobre os dados pessoais tratados, sob pena de multa diária de R$ 500,00.

CASO 4 - ELIMINAÇÃO DE DADOS
Tribunal: TJRS
Processo: 4000000-00.2023.8.21.0000
Ementa: PROTEÇÃO DE DADOS - LGPD - DIREITO À ELIMINAÇÃO - Titular que solicitou a eliminação de seus dados pessoais após término da relação contratual - Empresa que manteve os dados sem justificativa legal - Violação do artigo 18, inciso VI, da Lei 13.709/2018.
Decisão: Procedência do pedido para determinar a eliminação dos dados pessoais do autor dos bancos de dados da empresa ré.

CASO 5 - CONSENTIMENTO INVÁLIDO
Tribunal: TJPR
Processo: 5000000-00.2023.8.16.0000
Ementa: LGPD - CONSENTIMENTO - NULIDADE - Termo de consentimento genérico e não específico - Violação do artigo 8º da Lei 13.709/2018 - Tratamento de dados pessoais sem base legal válida.
Decisão: Declaração de nulidade do consentimento obtido pela empresa, por não atender aos requisitos legais de especificidade e clareza, determinando-se a interrupção do tratamento de dados.

CASO 6 - DADOS SENSÍVEIS
Tribunal: TJSP
Processo: 6000000-00.2023.8.26.0000
Ementa: LGPD - DADOS SENSÍVEIS - SAÚDE - Clínica médica que comercializou dados de saúde de pacientes para empresas de planos de saúde - Violação grave dos artigos 11 e 12 da Lei 13.709/2018.
Decisão: Condenação da clínica ao pagamento de R$ 50.000,00 por danos morais coletivos, além de determinação para cessar imediatamente o compartilhamento de dados sensíveis.

CASO 7 - RELATÓRIO DE IMPACTO
Tribunal: TRF-3
Processo: 7000000-00.2023.4.03.0000
Ementa: LGPD - RELATÓRIO DE IMPACTO À PROTEÇÃO DE DADOS - ANPD - Empresa que se recusou a elaborar RIPD quando solicitado pela autoridade nacional - Infração administrativa - Aplicação de multa.
Decisão: Manutenção da multa aplicada pela ANPD em razão da recusa da empresa em elaborar Relatório de Impacto à Proteção de Dados Pessoais.

CASO 8 - TRANSFERÊNCIA INTERNACIONAL
Tribunal: TJSP
Processo: 8000000-00.2024.8.26.0000
Ementa: LGPD - TRANSFERÊNCIA INTERNACIONAL DE DADOS - Empresa que transferiu dados pessoais para país sem nível adequado de proteção - Ausência de garantias contratuais - Violação do artigo 33 da Lei 13.709/2018.
Decisão: Procedência parcial para determinar a suspensão da transferência internacional de dados até que a empresa demonstre conformidade com os requisitos legais.

CASO 9 - ENCARREGADO DE DADOS (DPO)
Tribunal: TRT-2
Processo: 9000000-00.2024.5.02.0000
Ementa: LGPD - ENCARREGADO DE DADOS - OBRIGATORIEDADE - Empresa de grande porte que não indicou encarregado pelo tratamento de dados pessoais - Violação do artigo 41 da Lei 13.709/2018.
Decisão: Determinação para que a empresa indique encarregado de dados no prazo de 30 dias, sob pena de aplicação das sanções previstas na LGPD.

CASO 10 - LEGÍTIMO INTERESSE
Tribunal: TJDF
Processo: 1000000-00.2024.8.07.0000
Ementa: LGPD - LEGÍTIMO INTERESSE - MARKETING DIRETO - Empresa que utilizou dados pessoais para envio de publicidade não solicitada, alegando legítimo interesse - Interpretação restritiva do artigo 7º, inciso IX, da Lei 13.709/2018.
Decisão: O legítimo interesse não pode ser utilizado como base legal para envio de marketing direto sem que haja expectativa razoável do titular. Condenação por danos morais no valor de R$ 5.000,00.
"""


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Divide o texto em chunks com sobreposição."""
    chunks = []
    start = 0
    text = text.strip()
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Tenta terminar o chunk em um ponto final ou quebra de linha
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            best_break = max(last_period, last_newline)
            if best_break > chunk_size // 2:
                end = start + best_break + 1
                chunk = text[start:end]
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks


def create_chunks_with_metadata(text, source_type):
    """Cria chunks com metadados."""
    raw_chunks = chunk_text(text)
    documents = []
    
    for i, chunk in enumerate(raw_chunks):
        if chunk:  # Ignora chunks vazios
            doc = {
                "content": chunk,
                "metadata": {
                    "id": f"{source_type}_{i}",
                    "source": source_type,
                    "chunk_index": i
                }
            }
            documents.append(doc)
    
    return documents


def create_faiss_index(documents, embed_model):
    """Cria índice FAISS a partir dos documentos."""
    texts = [doc["content"] for doc in documents]
    embeddings = embed_model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    
    # Cria índice FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return index


def main():
    print("=" * 60)
    print("Agentic LGPD - Setup de Dados")
    print("=" * 60)
    
    # Carrega modelo de embeddings
    print("\n1. Carregando modelo de embeddings...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    print(f"   Modelo carregado: {EMBED_MODEL_NAME}")
    
    # Processa texto da Lei
    print("\n2. Processando texto da LGPD...")
    chunks_lei = create_chunks_with_metadata(LGPD_TEXT, "lei")
    print(f"   Chunks criados: {len(chunks_lei)}")
    
    # Processa jurisprudência
    print("\n3. Processando jurisprudência...")
    chunks_jurisprudencia = create_chunks_with_metadata(JURISPRUDENCIA_TEXT, "jurisprudencia")
    print(f"   Chunks criados: {len(chunks_jurisprudencia)}")
    
    # Cria índices FAISS
    print("\n4. Criando índice FAISS para a Lei...")
    index_lei = create_faiss_index(chunks_lei, embed_model)
    print(f"   Índice criado com {index_lei.ntotal} vetores")
    
    print("\n5. Criando índice FAISS para jurisprudência...")
    index_jurisprudencia = create_faiss_index(chunks_jurisprudencia, embed_model)
    print(f"   Índice criado com {index_jurisprudencia.ntotal} vetores")
    
    # Salva arquivos JSON
    print("\n6. Salvando arquivos JSON...")
    with open("lei_chunks_com_metadados_lei.json", "w", encoding="utf-8") as f:
        json.dump(chunks_lei, f, ensure_ascii=False, indent=2)
    print("   Salvo: lei_chunks_com_metadados_lei.json")
    
    with open("lei_chunks_com_metadados_jurisprudencia.json", "w", encoding="utf-8") as f:
        json.dump(chunks_jurisprudencia, f, ensure_ascii=False, indent=2)
    print("   Salvo: lei_chunks_com_metadados_jurisprudencia.json")
    
    # Salva índices FAISS
    print("\n7. Salvando índices FAISS...")
    faiss.write_index(index_lei, "lei_faiss_lei.index")
    print("   Salvo: lei_faiss_lei.index")
    
    faiss.write_index(index_jurisprudencia, "lei_faiss_jurisprudencia.index")
    print("   Salvo: lei_faiss_jurisprudencia.index")
    
    print("\n" + "=" * 60)
    print("Setup concluído com sucesso!")
    print("=" * 60)
    print("\nArquivos gerados:")
    print("  - lei_chunks_com_metadados_lei.json")
    print("  - lei_chunks_com_metadados_jurisprudencia.json")
    print("  - lei_faiss_lei.index")
    print("  - lei_faiss_jurisprudencia.index")
    print("\nAgora você pode executar: python app.py")


if __name__ == "__main__":
    main()
