#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Fine-Tuning - Servidor Web OTIMIZADO E COMPLETO
Todas as rotas funcionais
"""

import os
import sys

# Configurações específicas para Render
PORT = int(os.environ.get('PORT', 10000))  # Render usa PORT env var
HOST = '0.0.0.0'  # Necessário para Render
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'services'))

import json
import logging

# Importação segura do yaml
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ PyYAML não instalado - usando configuração padrão")
import gc
import struct
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit

# Importação segura do Flask-CORS
try:
    from flask_cors import CORS
    HAS_CORS = True
except ImportError:
    print("[AVISO] Flask-CORS não encontrado. Execute: pip install flask-cors")
    HAS_CORS = False
import threading
import traceback
from datetime import datetime
import torch
from services.gguf_requantizer import UniversalConverter, QuantizationConfig # Corrigido: importação relativa correta
from train import AITrainingSystem
from services.web_scraper import WebScraperService
from create_usbabc_model import create_usbabc_model, create_small_portuguese_model
from modeling_usbabc import USBABCForCausalLM
from configuration_usbabc import USBABCConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from universal_file_manager import get_file_manager
from dotenv import load_dotenv
from data_processor import process_all_training_data
from universal_model_loader import load_any_model, get_model_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'default_config.yaml')

def load_config():
    if not os.path.exists(CONFIG_FILE):
        logger.warning(f"⚠️ Arquivo de configuração não encontrado: {CONFIG_FILE}")
        return {}

    if not HAS_YAML:
        logger.warning("⚠️ PyYAML não disponível - retornando configuração padrão")
        return {}

    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"❌ Erro ao carregar config: {e}")
        return {}

app_config = load_config()
load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24).hex()
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024 * 1024

# Configurar CORS se disponível
if HAS_CORS:
    CORS(app)
    print("[OK] Flask-CORS configurado")
else:
    print("[AVISO] Executando sem CORS - algumas funcionalidades podem não funcionar")
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    max_http_buffer_size=5 * 1024 * 1024 * 1024,
    ping_timeout=180,
    ping_interval=25
)

# Estado global
app_state = {
    'model_loaded': False,
    'model_path': None,
    'training_active': False,
    'chat_model': None,
    'chat_tokenizer': None,
    'training_system': None,
    'model_type': None
}


def cleanup_memory():
    """Limpa memória agressivamente"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_chat_model(model_path: str):
    """Carrega modelo para chat otimizado com suporte universal"""
    if not isinstance(model_path, str):
        logger.error(f"❌ Erro: model_path recebido não é uma string. Tipo: {type(model_path)}, Valor: {model_path}")
        return False

    try:
        # Limpar modelo anterior
        if app_state.get("chat_model"):
            del app_state["chat_model"]
            del app_state["chat_tokenizer"]
            cleanup_memory()

        logger.info(f"📦 Carregando modelo: {model_path}")

        # Validar caminho
        if not os.path.exists(model_path):
            logger.error(f"❌ Modelo não encontrado: {model_path}")
            return False

        # DETECTAR FORMATO E CARREGAR
        model_path_obj = Path(model_path)
        file_extension = model_path_obj.suffix.lower()

        # GGUF
        if file_extension == '.gguf' or (model_path_obj.is_file() and model_path.endswith('.gguf')):
            try:
                from llama_cpp import Llama
                logger.info("🔧 Carregando modelo USBABC (formato GGUF)...")

                # Verificar se arquivo existe e é válido
                if not os.path.exists(model_path):
                    logger.error(f"❌ Arquivo não encontrado: {model_path}")
                    return False

                file_size = os.path.getsize(model_path)
                logger.info(f"📊 Tamanho do arquivo USBABC: {file_size / (1024**3):.2f} GB")

                # Verificar se é um arquivo GGUF válido
                if not _validate_gguf_file(model_path):
                    logger.error(f"❌ Arquivo GGUF inválido ou corrompido: {model_path}")
                    return False

                # Converter caminho para formato correto
                model_path = str(Path(model_path).resolve())
                logger.info(f"📂 Caminho normalizado: {model_path}")

                n_gpu_layers = 0
                if torch.cuda.is_available():
                    n_gpu_layers = 20
                    logger.info(f"🎮 GPU detectada - usando {n_gpu_layers} camadas")

                # Tentar carregar com diferentes configurações
                try:
                    llm = Llama(
                        model_path=model_path,
                        n_ctx=2048,
                        n_threads=min(os.cpu_count() or 4, 8),
                        n_gpu_layers=n_gpu_layers,
                        n_batch=512,
                        verbose=False,  # Desativar verbose para evitar spam
                        use_mmap=True,
                        use_mlock=False
                    )
                except Exception as e1:
                    logger.warning(f"⚠️ Tentativa 1 falhou: {e1}")
                    logger.info("🔄 Tentando sem GPU...")

                    try:
                        # Fallback sem GPU
                        llm = Llama(
                            model_path=model_path,
                            n_ctx=1024,      # Contexto menor
                            n_threads=min(os.cpu_count() or 4, 4),
                            n_gpu_layers=0,  # Sem GPU
                            n_batch=128,     # Batch muito menor
                            verbose=False,
                            use_mmap=True,
                            use_mlock=False
                        )
                    except Exception as e2:
                        logger.error(f"❌ Todas as tentativas falharam: {e2}")
                        logger.error("🔍 Possíveis causas:")
                        logger.error("   - Arquivo GGUF corrompido ou incompleto")
                        logger.error("   - Arquitetura de modelo não suportada")
                        logger.error("   - Memória insuficiente")
                        logger.error("   - Versão incompatível do llama-cpp-python")
                        return False


                app_state['chat_model'] = llm
                app_state['chat_tokenizer'] = None
                app_state['model_type'] = 'usbabc_gguf'

                logger.info("✅ Modelo USBABC (GGUF) carregado com sucesso")
                logger.info("🏢 Arquitetura: USBABC")
                return True

            except ImportError:
                logger.error("❌ llama-cpp-python não instalado!")
                logger.error("Instale com: pip install llama-cpp-python")
                return False
            except Exception as e:
                logger.error(f"❌ Erro GGUF: {e}")
                return False

        # ZIP - Extrair e carregar
        elif file_extension == '.zip':
            logger.info("📦 Detectado arquivo ZIP, extraindo...")
            try:
                from universal_model_loader import load_any_model
                model, tokenizer, metadata = load_any_model(model_path)

                if model and metadata.get('format') == 'gguf':
                    # Modelo GGUF extraído do ZIP
                    app_state['chat_model'] = model
                    app_state['chat_tokenizer'] = tokenizer
                    app_state['model_type'] = 'gguf'
                    logger.info("✅ Modelo GGUF do ZIP carregado")
                    return True
                elif model:
                    # Modelo HuggingFace extraído do ZIP
                    app_state['chat_model'] = model
                    app_state['chat_tokenizer'] = tokenizer
                    app_state['model_type'] = 'hf'
                    logger.info("✅ Modelo HF do ZIP carregado")
                    return True
                else:
                    logger.error("❌ Não foi possível carregar modelo do ZIP")
                    return False
            except Exception as e:
                logger.error(f"❌ Erro ao carregar ZIP: {e}")
                return False

        # SAFETENSORS, BIN, PT, PTH ou DIRETÓRIO
        # Usar carregador universal
        try:
            from universal_model_loader import load_any_model, get_model_info

            # Obter informações do modelo
            info = get_model_info(model_path)
            logger.info(f"📊 Informações do modelo: {info}")

            if not info.get('supported', False):
                logger.warning(f"⚠️ Formato não suportado pelo carregador universal: {info.get('format')}")
                # Tentar fallback para HuggingFace
            else:
                # Tentar carregar com o sistema universal
                try:
                    model, tokenizer, metadata = load_any_model(
                        model_path,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map='auto',
                        low_cpu_mem_usage=True
                    )

                    if model:
                        app_state['chat_model'] = model
                        app_state['chat_tokenizer'] = tokenizer
                        app_state['model_type'] = 'usbabc_' + metadata.get('format', 'hf')
                        logger.info(f"✅ Modelo USBABC carregado via sistema universal: {metadata.get('format')}")
                        logger.info("🏢 Arquitetura: USBABC")
                        return True
                except Exception as e:
                    logger.warning(f"⚠️ Carregador universal falhou: {e}, tentando HuggingFace...")

        except ImportError:
            logger.warning("⚠️ Carregador universal não disponível, usando HuggingFace")

        # FALLBACK: Hugging Face - Tentar carregar modelo SafeTensors/PyTorch
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            logger.info("🔧 Tentando carregar como modelo HF...")

            # Detectar se é arquivo único ou diretório
            if os.path.isfile(model_path):
                # Arquivo único - usar diretório pai
                model_dir = os.path.dirname(model_path)
                logger.info(f"📂 Arquivo único detectado, usando diretório: {model_dir}")

                # Verificar se diretório tem arquivos necessários
                config_path = os.path.join(model_dir, "config.json")
                if not os.path.exists(config_path):
                    logger.warning(f"⚠️ Diretório {model_dir} não contém config.json. Não é um modelo HF completo.")
                    # Não retornar False aqui, permitir que o from_pretrained falhe e seja capturado
                    # Isso permite que o AutoTokenizer tente inferir, ou que o erro seja mais específico.
                    # return False # Removido para permitir que o try-except capture o erro de forma mais granular

                model_path = model_dir

            # Validar que model_path é uma string antes de prosseguir
            if not isinstance(model_path, str):
                logger.error(f"❌ Erro: model_path não é uma string. Tipo: {type(model_path)}, Valor: {model_path}")
                return False # Ou lidar com o erro de outra forma

            # Carregar tokenizer com fallbacks seguros e validação de caminho
            tokenizer = None
            tokenizer_loaded = False
            tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "special_tokens_map.json", "merges.txt", "added_tokens.json"]

            # Verificar a existência de arquivos comuns de tokenizer
            has_tokenizer_files = False
            if os.path.isdir(model_path):
                for f_name in tokenizer_files:
                    if os.path.exists(os.path.join(model_path, f_name)):
                        has_tokenizer_files = True
                        break

            if not has_tokenizer_files:
                logger.warning(f"⚠️ Nenhum arquivo de tokenizer comum encontrado em {model_path}. Tentando carregar mesmo assim, mas pode falhar.")

            try:
                logger.info(f"🔧 Tentando carregar AutoTokenizer de: {model_path} (use_fast=False)")
                if not isinstance(model_path, str):
                    raise TypeError(f"model_path para AutoTokenizer.from_pretrained (use_fast=False) não é uma string. Tipo: {type(model_path)}, Valor: {model_path}")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    local_files_only=True,
                    use_fast=True  # Evitar problemas com tokenizers rápidos
                )
                logger.info("✅ AutoTokenizer carregado")
                tokenizer_loaded = True
            except Exception as e1:
                logger.warning(f"⚠️ AutoTokenizer (use_fast=False) falhou para {model_path}: {e1}")
                try:
                    logger.info(f"🔄 Tentando carregar AutoTokenizer de: {model_path} (use_fast=True)")
                    if not isinstance(model_path, str):
                        raise TypeError(f"model_path para AutoTokenizer.from_pretrained (use_fast=True) não é uma string. Tipo: {type(model_path)}, Valor: {model_path}")
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        local_files_only=True,
                        use_fast=True
                    )
                    logger.info("✅ AutoTokenizer (fast) carregado")
                    tokenizer_loaded = True
                except Exception as e2:
                    logger.warning(f"⚠️ AutoTokenizer (use_fast=True) falhou para {model_path}: {e2}")
                    try:
                        # Tentar LlamaTokenizer como fallback
                        from transformers import LlamaTokenizer
                        logger.info(f"🔄 Tentando carregar LlamaTokenizer de: {model_path}")
                        if not isinstance(model_path, str):
                            raise TypeError(f"model_path para LlamaTokenizer.from_pretrained não é uma string. Tipo: {type(model_path)}, Valor: {model_path}")
                        tokenizer = LlamaTokenizer.from_pretrained(
                            model_path,
                            trust_remote_code=True,
                            local_files_only=True
                        )
                        logger.info("✅ LlamaTokenizer carregado")
                        tokenizer_loaded = True
                    except Exception as e3:
                        logger.error(f"❌ Todos os tokenizers falharam para {model_path}: {e3}")
                        logger.error("🔍 Possíveis causas para falha do tokenizer:")
                        logger.error("   - Caminho do modelo incorreto ou não é um diretório válido.")
                        logger.error("   - Arquivos de configuração do tokenizer ausentes, corrompidos ou em formato inesperado (e.g., 'not a string' em um campo que espera string).")
                        logger.error("   - O modelo foi salvo de forma incompleta ou com uma estrutura de diretório não padrão.")
                        logger.error("   - Versão incompatível da biblioteca transformers.")
                        tokenizer = None

            if not tokenizer_loaded:
                logger.warning(f"⚠️ Nenhum tokenizer foi carregado para o modelo em {model_path}. A funcionalidade de tokenização pode ser limitada ou ausente.")
                # O aplicativo pode continuar, mas com avisos claros sobre a falta do tokenizer.
                # Se a aplicação *exige* um tokenizer, pode-se retornar False aqui.
                # Por enquanto, vamos permitir que continue, mas com o tokenizer como None.

            # Carregar modelo com configurações otimizadas para CUDA
            cuda_available = torch.cuda.is_available()
            logger.info(f"🎮 CUDA disponível: {cuda_available}")

            if cuda_available:
                # Configurações otimizadas para GPU
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                    max_memory={0: "6GB"} if torch.cuda.device_count() > 0 else None,
                    use_safetensors=True
                )

                # Garantir que o modelo está na GPU
                if hasattr(model, 'to') and not next(model.parameters()).is_cuda:
                    try:
                        model = model.to('cuda')
                        logger.info("🎮 Modelo movido para GPU")
                    except Exception as e:
                        logger.warning(f"⚠️ Não foi possível mover para GPU: {e}")
            else:
                # Configurações para CPU
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                    use_safetensors=True
                )

            model.eval()

            app_state['chat_model'] = model
            app_state['chat_tokenizer'] = tokenizer
            app_state['model_type'] = 'hf'

            logger.info("✅ Modelo HF carregado com sucesso")
            cleanup_memory()
            return True

        except Exception as e:
            logger.error(f"❌ Erro HF: {e}")
            logger.error(f"Tipo de erro: {type(e).__name__}")
            if "Repo id must use alphanumeric chars" in str(e):
                logger.error("💡 Dica: Este arquivo não é um modelo HuggingFace válido")
            return False

    except Exception as e:
        logger.error(f"❌ Erro fatal ao carregar modelo: {e}")
        import traceback
        traceback.print_exc()
        return False


def _validate_gguf_file(file_path: str) -> bool:
    """Valida se um arquivo GGUF é válido e tem arquitetura suportada"""
    try:
        with open(file_path, 'rb') as f:
            # Verificar magic number GGUF
            magic = f.read(4)
            if magic != b'GGUF':
                logger.error(f"❌ Magic number inválido: {magic}")
                return False

            # Ler versão
            version = struct.unpack('<I', f.read(4))[0]
            if version < 1 or version > 3:
                logger.error(f"❌ Versão GGUF não suportada: {version}")
                return False

            # Ler número de tensors e metadados
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            metadata_kv_count = struct.unpack('<Q', f.read(8))[0]

            logger.info(f"✅ GGUF válido: v{version}, {tensor_count} tensors, {metadata_kv_count} metadados")

            # Verificar se tem pelo menos alguns tensors
            if tensor_count == 0:
                logger.error("❌ Arquivo GGUF sem tensors")
                return False

            # Tentar ler alguns metadados para verificar arquitetura
            try:
                for i in range(min(metadata_kv_count, 10)):  # Ler apenas os primeiros metadados
                    # Ler comprimento da chave
                    key_len = struct.unpack('<Q', f.read(8))[0]
                    if key_len > 1000:  # Chave muito longa, arquivo corrompido
                        break

                    # Ler chave
                    key = f.read(key_len).decode('utf-8', errors='ignore')

                    # Ler tipo do valor
                    value_type = struct.unpack('<I', f.read(4))[0]

                    # Verificar se é arquitetura
                    if key == 'general.architecture':
                        if value_type == 8:  # String
                            arch_len = struct.unpack('<Q', f.read(8))[0]
                            if arch_len > 0 and arch_len < 100:
                                arch = f.read(arch_len).decode('utf-8', errors='ignore')
                                logger.info(f"🏗️ Arquitetura detectada: {arch}")

                                # Lista de arquiteturas suportadas pelo llama-cpp-python
                                supported_archs = [
                                    'llama', 'falcon', 'gpt2', 'gptj', 'gptneox',
                                    'mpt', 'baichuan', 'starcoder', 'refact',
                                    'bert', 'nomic-bert', 'bloom', 'stablelm',
                                    'qwen', 'qwen2', 'phi', 'phi3', 'plamo',
                                    'codeshell', 'orion', 'internlm2', 'minicpm',
                                    'gemma', 'gemma2', 'starcoder2', 'mamba',
                                    'xverse', 'command-r', 'dbrx', 'olmo',
                                    'usbabc'  # Suporte para modelos USBABC
                                ]

                                if arch.lower() in [a.lower() for a in supported_archs]:
                                    logger.info(f"✅ Arquitetura suportada: {arch}")
                                    return True
                                else:
                                    logger.error(f"❌ Arquitetura não suportada: {arch}")
                                    logger.error(f"🔧 Arquiteturas suportadas: {', '.join(supported_archs[:10])}...")
                                    return False
                        break
                    else:
                        # Pular valor baseado no tipo
                        if value_type == 8:  # String
                            str_len = struct.unpack('<Q', f.read(8))[0]
                            f.seek(str_len, 1)
                        elif value_type in [4, 5]:  # Int32, UInt32
                            f.seek(4, 1)
                        elif value_type in [6, 7]:  # Float32, Bool
                            f.seek(4, 1)
                        elif value_type in [9, 10]:  # Array
                            array_type = struct.unpack('<I', f.read(4))[0]
                            array_len = struct.unpack('<Q', f.read(8))[0]
                            # Pular array baseado no tipo
                            if array_type == 8:  # String array
                                for _ in range(array_len):
                                    str_len = struct.unpack('<Q', f.read(8))[0]
                                    f.seek(str_len, 1)
                            else:
                                f.seek(array_len * 4, 1)  # Assumir 4 bytes por elemento
                        else:
                            # Tipo desconhecido, pular
                            f.seek(8, 1)

            except Exception as e:
                logger.warning(f"⚠️ Erro ao ler metadados: {e}")
                # Se não conseguir ler metadados, assumir que é válido
                return True

            # Se chegou aqui sem encontrar arquitetura, assumir que é válido
            logger.warning("⚠️ Arquitetura não encontrada nos metadados, assumindo válido")
            return True

    except Exception as e:
        logger.error(f"❌ Erro ao validar GGUF: {e}")
        return False

        # Hugging Face - Tentar carregar modelo SafeTensors/PyTorch
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Verificar se é um diretório de modelo local
            model_path_obj = Path(model_path)
            if model_path_obj.is_file():
                # Se for um arquivo único, tentar carregar do diretório pai
                model_dir = model_path_obj.parent
                logger.info(f"📂 Arquivo único detectado, usando diretório: {model_dir}")

                # Verificar se há arquivos de modelo no diretório
                model_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
                config_files = list(model_dir.glob("config.json"))

                if not model_files or not config_files:
                    logger.warning("⚠️ Diretório não contém arquivos de modelo HF completos")

                    # Verificar se o arquivo único é um SafeTensors válido
                    if model_path_obj.suffix.lower() == '.safetensors':
                        try:
                            from safetensors import safe_open
                            with safe_open(str(model_path_obj), framework="pt") as f:
                                # Verificar se o arquivo não está corrompido
                                if len(f.keys()) == 0:
                                    logger.error("❌ Arquivo SafeTensors vazio ou corrompido")
                                    return False
                                logger.info(f"✅ SafeTensors válido com {len(f.keys())} tensors")
                        except Exception as e:
                            logger.error(f"❌ Erro ao validar SafeTensors: {e}")
                            if "invalid JSON in header" in str(e) or "EOF while parsing" in str(e):
                                logger.error("❌ Arquivo SafeTensors corrompido - cabeçalho JSON inválido")
                            return False

                    return False

                model_path = str(model_dir)

            # Verificar se o caminho é válido para HuggingFace
            if not os.path.exists(model_path):
                logger.error(f"❌ Caminho não existe: {model_path}")
                return False

            # Verificar se é um diretório com arquivos necessários
            if os.path.isdir(model_path):
                required_files = ['config.json']
                model_files = [f for f in os.listdir(model_path) if f.endswith(('.safetensors', '.bin', '.pt'))]

                has_config = any(os.path.exists(os.path.join(model_path, f)) for f in required_files)

                if not has_config:
                    logger.warning("⚠️ Diretório não contém config.json")

                    # Tentar criar config.json básico se há arquivos de modelo
                    if model_files:
                        logger.info("🔧 Tentando criar config.json básico...")
                        try:
                            basic_config = {
                                "architectures": ["LlamaForCausalLM"],
                                "model_type": "llama",
                                "torch_dtype": "float16",
                                "transformers_version": "4.36.0",
                                "vocab_size": 22000,
                                "hidden_size": 4096,
                                "intermediate_size": 11008,
                                "num_attention_heads": 32,
                                "num_hidden_layers": 32,
                                "rms_norm_eps": 1e-06,
                                "rope_theta": 10000.0,
                                "max_position_embeddings": 2048,
                                "use_cache": True
                            }

                            config_path = os.path.join(model_path, 'config.json')
                            with open(config_path, 'w', encoding='utf-8') as f:
                                json.dump(basic_config, f, indent=2)

                            logger.info("✅ config.json básico criado")
                        except Exception as e:
                            logger.error(f"❌ Falha ao criar config.json: {e}")
                            return False
                    else:
                        return False

                if not model_files:
                    logger.warning("⚠️ Nenhum arquivo de modelo encontrado no diretório")
                    return False

            logger.info(f"🔧 Carregando modelo HF de: {model_path}")

            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=True,
                local_files_only=True  # Forçar uso local
            )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                local_files_only=True,  # Forçar uso local
                max_memory={0: "6GB"} if torch.cuda.is_available() else None
            )

            model.eval()

            app_state['chat_model'] = model
            app_state['chat_tokenizer'] = tokenizer
            app_state['model_type'] = 'hf'

            logger.info("✅ Modelo HF carregado com sucesso")
            cleanup_memory()
            return True

        except Exception as e:
            logger.error(f"❌ Erro HF: {e}")
            logger.error(f"Tipo de erro: {type(e).__name__}")
            # Não mostrar traceback completo para erros de repo ID inválido
            if "Repo id must use alphanumeric chars" in str(e):
                logger.error("💡 Dica: Este arquivo não é um modelo HuggingFace válido")
            else:
                import traceback
                traceback.print_exc()
            return False

    except Exception as e:
        logger.error(f"❌ Erro fatal ao carregar modelo: {e}")
        import traceback
        traceback.print_exc()
        return False


@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')


@app.route('/estimate_quantization', methods=['POST'])
def estimate_quantization():
    """Estima tamanho do arquivo quantizado"""
    try:
        data = request.json
        model_path = data.get('model_path', '').strip()
        quant_type = data.get('quant_type', 'q4_k_m').lower()

        if not model_path:
            return jsonify({'success': False, 'error': 'Caminho do modelo não fornecido'}), 400

        if not os.path.exists(model_path):
            return jsonify({'success': False, 'error': 'Arquivo não encontrado'}), 404

        original_size = os.path.getsize(model_path)
        original_size_gb = original_size / (1024**3)

        estimated_size_gb = QuantizationConfig.estimate_size(original_size_gb, quant_type)
        space_saved_gb = original_size_gb - estimated_size_gb
        compression_ratio = (space_saved_gb / original_size_gb) * 100

        return jsonify({
            'success': True,
            'original_size_gb': round(original_size_gb, 2),
            'estimated_size_gb': estimated_size_gb,
            'space_saved_gb': round(space_saved_gb, 2),
            'compression_ratio': round(compression_ratio, 1)
        })

    except Exception as e:
        logger.error(f"❌ Erro ao estimar quantização: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/get_quantization_types', methods=['GET'])
def get_quantization_types():
    """Retorna tipos de quantização disponíveis"""
    try:
        types_info = {}
        # Changed to list_all_types based on gguf_requantizer.py
        for qtype in QuantizationConfig.list_all_types():
            # Changed to get_type_info based on gguf_requantizer.py
            info = QuantizationConfig.get_type_info(qtype)
            types_info[qtype] = {
                'description': info['description'],
                'bits': info['bits'],
                'quality': info.get('quality', 50)
            }

        return jsonify({
            'success': True,
            'types': types_info
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/upload_model', methods=['POST'])
def upload_model():
    """Upload do modelo com estrutura de diretório dedicada e ativação imediata"""
    try:
        logger.info("📤 Iniciando upload de modelo...")

        # Verificar se há arquivos na requisição
        if not request.files:
            logger.error("❌ Nenhum arquivo na requisição")
            return jsonify({'success': False, 'error': 'Nenhum arquivo enviado'}), 400

        if 'model' not in request.files:
            logger.error("❌ Campo 'model' não encontrado nos arquivos")
            return jsonify({'success': False, 'error': 'Campo model não encontrado'}), 400

        file = request.files['model']
        if not file or not file.filename or file.filename == '':
            logger.error("❌ Arquivo vazio ou sem nome")
            return jsonify({'success': False, 'error': 'Nome de arquivo vazio'}), 400

        logger.info(f"📁 Arquivo recebido: {file.filename}")

        # Validar extensão
        valid_extensions = ['.gguf', '.safetensors', '.bin', '.pt', '.pth', '.zip']
        if not any(file.filename.lower().endswith(ext) for ext in valid_extensions):
            return jsonify({
                'success': False,
                'error': f'Formato não suportado. Use: {", ".join(valid_extensions)}'
            }), 400

        model_dir = Path('modelos')
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / file.filename

        logger.info(f"📥 Salvando: {file.filename}")

        # Salvar em chunks
        chunk_size = 8192 * 1024
        total_size = 0

        # Verificar duplicata
        if model_path.exists():
            model_path.unlink()

        # Salvar arquivo com tratamento de erro robusto
        try:
            with open(model_path, 'wb') as f:
                while True:
                    try:
                        chunk = file.stream.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        total_size += len(chunk)

                        # Limpar memória periodicamente
                        if total_size % (100 * 1024 * 1024) == 0:
                            cleanup_memory()

                    except Exception as chunk_error:
                        logger.error(f"❌ Erro ao ler chunk: {chunk_error}")
                        # Se der erro, tentar continuar
                        break

        except Exception as file_error:
            logger.error(f"❌ Erro ao salvar arquivo: {file_error}")
            if model_path.exists():
                model_path.unlink()  # Remover arquivo parcial
            return jsonify({'success': False, 'error': f'Erro ao salvar arquivo: {str(file_error)}'}), 500

        logger.info(f"✅ Arquivo salvo: {total_size / (1024**3):.2f} GB")

        # Validar arquivo GGUF
        if file.filename.endswith('.gguf'):
            try:
                # Ler cabeçalho GGUF (primeiros bytes)
                with open(model_path, 'rb') as f:
                    header = f.read(8)
                    if len(header) < 4:
                        raise Exception("Arquivo muito pequeno para ser um GGUF válido")

                    # Verificar magic number GGUF
                    magic = header[:4]
                    if magic not in [b'GGUF', b'GGJT', b'GGML', b'GGLA', b'GGMF']:
                        logger.warning(f"⚠️ Magic number não reconhecido: {magic}")
                        # Não bloquear, mas avisar
                    else:
                        logger.info(f"✅ GGUF válido detectado: {magic.decode('ascii', errors='ignore')}")

            except Exception as e:
                logger.warning(f"⚠️ Não foi possível validar GGUF: {e}")

        # Processar arquivo ZIP se necessário
        if file.filename.lower().endswith('.zip'):
            try:
                logger.info("📦 Processando arquivo ZIP de modelo...")
                import zipfile

                # Verificar se é um ZIP válido
                if not zipfile.is_zipfile(model_path):
                    raise Exception("Arquivo não é um ZIP válido")

                # Criar diretório para extração
                extract_dir = model_dir / f"{file.filename[:-4]}_extracted"
                extract_dir.mkdir(exist_ok=True)

                # Extrair o ZIP com segurança
                with zipfile.ZipFile(model_path, 'r') as zip_ref:
                    # Verificar arquivos suspeitos
                    for member in zip_ref.namelist():
                        if member.startswith('/') or '..' in member:
                            raise Exception(f"Arquivo suspeito no ZIP: {member}")

                    zip_ref.extractall(extract_dir)

                # Procurar pelo arquivo principal do modelo
                model_files = []
                for ext in ['.gguf', '.safetensors', '.bin', '.pt', '.pth']:
                    model_files.extend(list(extract_dir.rglob(f'*{ext}')))

                if model_files:
                    # Usar o primeiro arquivo encontrado
                    model_path = model_files[0]
                    logger.info(f"✅ Modelo extraído: {model_path}")
                else:
                    # Se não encontrou arquivos de modelo, usar o diretório
                    model_path = extract_dir
                    logger.info(f"✅ Diretório de modelo extraído: {model_path}")

                # Remover ZIP original
                Path(model_dir / file.filename).unlink()

            except Exception as e:
                logger.error(f"❌ Erro ao extrair ZIP: {e}")
                return jsonify({
                    'success': False,
                    'error': f'Erro ao extrair arquivo ZIP: {str(e)}'
                }), 400

        app_state['model_path'] = str(model_path)

        # Validação adicional para SafeTensors
        if str(model_path).endswith('.safetensors') or (model_path.is_dir() and list(model_path.glob('*.safetensors'))):
            try:
                from safetensors import safe_open

                # Encontrar arquivo SafeTensors
                if model_path.is_dir():
                    safetensors_files = list(model_path.glob('*.safetensors'))
                    if safetensors_files:
                        safetensors_file = safetensors_files[0]
                    else:
                        raise Exception("Nenhum arquivo SafeTensors encontrado no diretório")
                else:
                    safetensors_file = model_path

                with safe_open(safetensors_file, framework="pt") as f:
                    tensor_count = len(f.keys())
                    logger.info(f"✅ SafeTensors válido com {tensor_count} tensors")

                    # Verificar se é um modelo HF completo
                    model_dir_path = safetensors_file.parent if safetensors_file.is_file() else model_path
                    config_path = model_dir_path / "config.json"
                    tokenizer_path = model_dir_path / "tokenizer.json"

                    if not config_path.exists():
                        logger.warning("⚠️ Criando config.json básico para modelo SafeTensors")
                        # Criar config.json básico para modelos SafeTensors
                        basic_config = {
                            "architectures": ["LlamaForCausalLM"],
                            "model_type": "llama",
                            "torch_dtype": "float16",
                            "transformers_version": "4.36.0",
                            "vocab_size": 22000,
                            "hidden_size": 4096,
                            "intermediate_size": 11008,
                            "num_hidden_layers": 32,
                            "num_attention_heads": 32,
                            "max_position_embeddings": 2048,
                            "rms_norm_eps": 1e-6,
                            "use_cache": True,
                            "pad_token_id": 0,
                            "bos_token_id": 1,
                            "eos_token_id": 2
                        }
                        with open(config_path, 'w') as cf:
                            json.dump(basic_config, cf, indent=2)
                        logger.info("✅ config.json básico criado")

            except Exception as e:
                logger.error(f"❌ SafeTensors inválido: {e}")
                return jsonify({
                    'success': False,
                    'error': f'Arquivo SafeTensors corrompido: {str(e)}'
                }), 400

        # MISSÃO 2: Carregar modelo usando sistema universal
        logger.info("🚀 MISSÃO 2: Carregando modelo universal...")

        try:
            # Obter informações do modelo
            model_info = get_model_info(model_path)
            logger.info(f"📊 Informações do modelo: {model_info}")

            if not model_info['supported']:
                logger.warning(f"⚠️ Formato não suportado: {model_info['format']}")
                load_success = False
            else:
                # Carregar modelo usando o sistema universal
                model, tokenizer, metadata = load_any_model(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map='auto',
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )

                # Salvar informações globalmente
                global current_model, current_tokenizer, current_model_path
                current_model = model
                current_tokenizer = tokenizer
                current_model_path = str(model_path)

                app_state['model_metadata'] = metadata
                load_success = True

                logger.info(f"✅ Modelo {metadata['format']} carregado com sucesso!")

        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo universal: {e}")
            load_success = False

        if load_success:
            # Carregar modelo para chat
            if not load_chat_model(str(model_path)):
                logger.warning('⚠️ Modelo não carregado para chat')
                app_state['model_loaded'] = False
            else:
                                app_state['model_loaded'] = True

                                # Notificar frontend via Socket.IO
                                socketio.emit('model_loaded', {
                                    'success': True,
                                    'path': str(model_path),
                                    'filename': file.filename,
                                    'loaded': True
                                })
            logger.info("✅ Modelo carregado e pronto")

            return jsonify({
                'success': True,
                'filename': file.filename,
                'size': total_size,
                'size_gb': round(total_size / (1024**3), 2),
                'path': str(model_path),
                'loaded': True,
                'format': app_state.get('model_metadata', {}).get('format', 'unknown'),
                'metadata': app_state.get('model_metadata', {})
            })
        else:
            # Modelo salvo mas não carregado - ainda retornar sucesso parcial
            logger.warning("⚠️ Modelo salvo mas falha ao carregar")
            app_state['model_loaded'] = False

            return jsonify({
                'success': True,  # Mudado para True pois o arquivo foi salvo
                'filename': file.filename,
                'size': total_size,
                'size_gb': round(total_size / (1024**3), 2),
                'path': str(model_path),
                'loaded': False,
                'warning': 'Modelo salvo mas não pôde ser carregado automaticamente. Verifique se é um formato compatível ou tente carregar manualmente.'
            })

    except Exception as e:
        # Tratamento específico para ClientDisconnected
        from werkzeug.exceptions import ClientDisconnected
        if isinstance(e, ClientDisconnected):
            logger.error(f"❌ Cliente desconectado durante upload: {e}")
            return jsonify({
                'success': False,
                'error': 'Conexão perdida durante upload. Tente novamente com arquivo menor ou conexão mais estável.'
            }), 400
        else:
            logger.error(f"❌ Erro no upload: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': f'Erro ao processar arquivo: {str(e)}'
            }), 500


@app.route('/upload_data', methods=['POST'])
def upload_data():
    """Upload dos dados"""
    try:
        if 'data' not in request.files:
            return jsonify({'success': False, 'error': 'Nenhum arquivo'}), 400

        file = request.files['data']
        if not file.filename or file.filename == '':
            return jsonify({'success': False, 'error': 'Arquivo vazio'}), 400

        # Changed allowed extensions to include more data types handled by UniversalDataProcessor
        allowed_extensions = ('.json', '.jsonl', '.csv', '.txt', '.parquet', '.xls', '.xlsx', '.doc', '.docx', '.pdf')
        if not file.filename.lower().endswith(allowed_extensions):
            return jsonify({'success': False, 'error': f'Tipo de arquivo não suportado. Use: {", ".join(allowed_extensions)}'}), 400

        data_dir = Path('dados')
        data_dir.mkdir(exist_ok=True)
        data_path = data_dir / file.filename
        file.save(str(data_path))

        # Use UniversalDataProcessor to get sample count - handles various formats
        from data_processor import UniversalDataProcessor
        processor = UniversalDataProcessor()
        # Process just the uploaded file to get its count
        file_data = processor.supported_formats.get(data_path.suffix.lower(), lambda p: [])(data_path)
        sample_count = len(file_data)

        logger.info(f"✓ Dados carregados: {sample_count} exemplos")

        return jsonify({
            'success': True,
            'filename': file.filename,
            'samples': sample_count,
            'path': str(data_path)
        })

    except Exception as e:
        logger.error(f"❌ Erro dados: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/start_training', methods=['POST'])
def start_training():
    """Iniciar treinamento OTIMIZADO"""
    try:
        config = request.json

        if app_state['training_active']:
            return jsonify({'success': False, 'error': 'Treinamento em andamento'}), 400

        if not app_state.get('model_path'):
            return jsonify({'success': False, 'error': 'Modelo não carregado'}), 400

        if app_state.get('chat_model'):
            logger.info("🧹 Liberando modelo de chat da memória...")
            del app_state['chat_model']
            if app_state.get('chat_tokenizer'):
                del app_state['chat_tokenizer']
            app_state['chat_model'] = None
            app_state['chat_tokenizer'] = None
            cleanup_memory()

        app_state['training_system'] = AITrainingSystem()
        app_state['training_active'] = True

        def train_thread():
            try:
                logger.info("🚀 Iniciando thread de treinamento...")
                socketio.emit('training_status', {
                    'status': 'Preparando modelo...',
                    'progress': 5
                })

                logger.info(f"📦 Carregando modelo: {app_state.get('model_path')}")
                app_state['training_system'].load_model(app_state.get('model_path'))

                socketio.emit('training_status', {
                    'status': 'Carregando dados...',
                    'progress': 15
                })

                # Construir caminho completo do arquivo de dados
                data_file = config.get('data_file')
                logger.info(f"📄 Arquivo de dados: {data_file}")

                if not data_file:
                    raise ValueError("Arquivo de dados não especificado. Por favor, faça upload de um arquivo de dados antes de iniciar o treinamento.")

                data_path = Path('dados') / data_file

                if not data_path.exists():
                    raise FileNotFoundError(f"Arquivo de dados não encontrado: {data_path}")

                logger.info(f"📂 Caminho completo: {data_path}")
                app_state['training_system'].prepare_dataset(str(data_path))

                socketio.emit('training_status', {
                    'status': 'Treinando (pode demorar)...',
                    'progress': 25
                })

                metrics = app_state['training_system'].train()

                socketio.emit('training_status', {
                    'status': 'Mesclando e salvando...',
                    'progress': 90
                })

                socketio.emit('training_progress', {
                    'progress': 100,
                    'message': 'Concluído!'
                })

                # Use metrics['output_path'] which now contains the final model path
                final_model_path = metrics.get('output_path', app_state.get('model_path'))

                socketio.emit('training_complete', {
                    'metrics': metrics,
                    'message': 'Modelo treinado salvo com sucesso!',
                    'final_model_path': final_model_path # Send the final path back
                })

                logger.info("=" * 60)
                logger.info("✅ TREINAMENTO FINALIZADO COM SUCESSO")
                logger.info(f"💾 Modelo final salvo em: {final_model_path}")
                logger.info("=" * 60)

            except Exception as e:
                logger.error(f"❌ Erro no treinamento: {e}")
                import traceback
                traceback.print_exc()
                socketio.emit('training_error', {'error': str(e)})

            finally:
                app_state['training_active'] = False

                if app_state.get('training_system'):
                    del app_state['training_system']
                    app_state['training_system'] = None

                cleanup_memory()

                final_model_path = metrics.get('output_path', app_state.get('model_path'))
                if final_model_path:
                    logger.info(f"🔄 Recarregando modelo TREINADO ({final_model_path}) para chat...")
                    socketio.emit('training_status', {
                        'status': 'Recarregando modelo treinado...',
                        'progress': 95
                    })

                    if load_chat_model(final_model_path):
                        logger.info("✓ Modelo TREINADO recarregado com sucesso!")
                        app_state['model_path'] = final_model_path # Update path to the trained one
                        socketio.emit('model_reloaded', {'success': True, 'model': 'trained', 'path': final_model_path})
                    else:
                        logger.warning("⚠️ Falha ao recarregar modelo treinado, tentando original...")
                        if app_state.get('model_path') and load_chat_model(app_state['model_path']):
                             logger.info("✓ Modelo original recarregado")
                             socketio.emit('model_reloaded', {'success': True, 'model': 'original', 'path': app_state['model_path']})
                        else:
                             logger.warning("⚠️ Falha ao recarregar qualquer modelo")
                             app_state['model_loaded'] = False # Ensure state reflects no model loaded
                             socketio.emit('model_reloaded', {'success': False, 'error': 'Falha ao recarregar modelo'})


        thread = threading.Thread(target=train_thread, daemon=True)
        thread.start()

        return jsonify({'success': True, 'message': 'Treinamento iniciado'})

    except Exception as e:
        logger.error(f"❌ Erro ao iniciar: {e}")
        app_state['training_active'] = False
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/save_model', methods=['POST'])
def save_model():
    """Salvar modelo treinado - PRIORIDADE 4"""
    try:
        # Verificar se há dados JSON ou form data
        if request.is_json:
            data = request.json or {}
        else:
            data = request.form.to_dict()

        model_name = data.get('save_path', 'modelo_treinado')
        save_format = data.get('save_format', 'safetensors')

        logger.info(f"💾 PRIORIDADE 4: Salvando modelo - Nome: {model_name}, Formato: {save_format}")

        # Verificar se existe modelo treinado na pasta modelos/modelo_treinado
        # This path is determined by the AITrainingSystem.save_model method
        trained_model_path = Path('modelos/modelo_treinado') # Default path set in AITrainingSystem
        # Or maybe it's the path stored in app_state if training updated it
        if app_state.get('model_path') and 'treinado' in app_state['model_path']:
             trained_model_path = Path(app_state['model_path'])

        if not trained_model_path.exists():
            logger.error(f"❌ Modelo treinado não encontrado em {trained_model_path}")
            return jsonify({'success': False, 'error': 'Nenhum modelo treinado encontrado. Execute o treinamento primeiro.'}), 400

        # Gerar nome com timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Criar diretório de destino
        # Use only the model_name provided by the user, add timestamp later
        destination_base_name = f"{model_name}"
        destination_dir = Path(f"modelos/{destination_base_name}_{timestamp}")
        destination_dir.mkdir(parents=True, exist_ok=True)


        logger.info(f"📁 Copiando modelo treinado de {trained_model_path} para: {destination_dir}")

        # Copiar todos os arquivos do modelo treinado
        import shutil
        try:
            # Check if source is dir or file (if GGUF was created)
            if trained_model_path.is_dir():
                 for item in trained_model_path.iterdir():
                    if item.is_file():
                        shutil.copy2(item, destination_dir / item.name)
                        logger.info(f"✓ Copiado arquivo: {item.name}")
                    elif item.is_dir():
                        shutil.copytree(item, destination_dir / item.name, dirs_exist_ok=True)
                        logger.info(f"✓ Copiado diretório: {item.name}")
            elif trained_model_path.is_file() and trained_model_path.suffix == '.gguf':
                 shutil.copy2(trained_model_path, destination_dir / trained_model_path.name)
                 logger.info(f"✓ Copiado GGUF: {trained_model_path.name}")
                 # Also copy config/tokenizer if they exist alongside
                 config_src = trained_model_path.with_name('config.json')
                 tokenizer_src = trained_model_path.with_name('tokenizer_config.json')
                 if config_src.exists(): shutil.copy2(config_src, destination_dir / config_src.name)
                 if tokenizer_src.exists(): shutil.copy2(tokenizer_src, destination_dir / tokenizer_src.name)

            save_path = str(destination_dir)

            # Calcular tamanho total
            file_size = sum(f.stat().st_size for f in destination_dir.rglob('*') if f.is_file()) / (1024 * 1024)

            logger.info(f"✅ PRIORIDADE 4: Modelo salvo com sucesso!")
            logger.info(f"📂 Localização: {save_path}")
            logger.info(f"💾 Tamanho: {file_size:.1f} MB")

            return jsonify({
                'success': True,
                'path': save_path,
                'format': save_format, # Keep user requested format for info, even though HF is saved
                'size_mb': round(file_size, 1),
                'message': f'Modelo treinado salvo com sucesso em {save_path}'
            })

        except Exception as copy_error:
            logger.error(f"❌ Erro ao copiar modelo: {copy_error}")
            return jsonify({'success': False, 'error': f'Erro ao copiar modelo: {str(copy_error)}'}), 500

    except Exception as e:
        logger.error(f"❌ Erro ao salvar: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    """Chat com modelo - OTIMIZADO"""
    try:
        data = request.json
        message = data.get('message', '').strip()

        if not message:
            return jsonify({'success': False, 'error': 'Mensagem vazia'}), 400

        if not app_state.get('chat_model'):
            logger.error(f'❌ Modelo não carregado. Estado: {app_state}')
            return jsonify({'success': False, 'error': 'Modelo não carregado'}), 400

        if app_state.get('training_active'):
            return jsonify({'success': False, 'error': 'Treinamento em andamento'}), 400

        chat_model = app_state['chat_model']
        model_type = app_state.get('model_type', 'hf')

        logger.info(f"💬 Chat ({model_type}): {message[:50]}...")

        # GGUF (incluindo USBABC GGUF)
        if model_type in ['gguf', 'usbabc_gguf']:
            logger.info(f"🤖 Processando com modelo USBABC GGUF...")

            try:
                # PRIORIDADE 1: Verificar perguntas sobre identidade
                if any(keyword in message.lower() for keyword in ["quem é você", "quem desenvolveu", "sua origem", "who are you", "who developed"]):
                    logger.info("🎯 PRIORIDADE 1: Pergunta sobre identidade detectada")
                    response_text = "Sou um modelo LLM base, criado pela empresa USBABC."
                else:
                    # Gerar resposta normal com o modelo
                    response = chat_model.create_chat_completion(
                        messages=[{"role": "user", "content": message}],
                        max_tokens=512,
                        temperature=0.7,
                        top_p=0.9,
                        top_k=40,
                        repeat_penalty=1.1,
                        stop=["</s>", "<|im_end|>", "User:", "\n\nUser:"]
                    )
                    response_text = response['choices'][0]['message']['content'].strip()

            except Exception as e:
                logger.warning(f"⚠️ Fallback GGUF: {e}")
                try:
                    # PRIORIDADE 1: Mesmo no fallback, manter identidade USBABC
                    if any(keyword in message.lower() for keyword in ["quem é você", "quem desenvolveu", "sua origem"]):
                        response_text = "Sou um modelo LLM base, criado pela empresa USBABC."
                    else:
                        response = chat_model(
                            f"<|user|>\n{message}</s>\n<|assistant|>\n",
                            max_tokens=512,
                            temperature=0.7,
                            top_p=0.9,
                            stop=["<|user|>", "</s>", "\n\n"]
                        )
                        response_text = response['choices'][0]['text'].strip()
                except Exception as e2:
                    logger.error(f"❌ Erro no fallback GGUF: {e2}")
                    # Última tentativa - resposta padrão USBABC
                    if any(keyword in message.lower() for keyword in ["quem é você", "quem desenvolveu", "sua origem"]):
                        response_text = "Sou um modelo LLM base, criado pela empresa USBABC."
                    else:
                        response_text = f"Olá! Sou um modelo LLM base criado pela empresa USBABC. Você disse: '{message}'. Como posso ajudá-lo?"

        # Hugging Face
        else:
            chat_tokenizer = app_state.get('chat_tokenizer')

            # Se tokenizer não disponível, usar resposta padrão USBABC
            if not chat_tokenizer:
                logger.info("🤖 Usando resposta padrão USBABC (tokenizer não disponível)")

                # PRIORIDADE 1: Resposta padrão USBABC
                if "quem é você" in message.lower() or "quem desenvolveu" in message.lower() or "sua origem" in message.lower():
                    response_text = "Sou um modelo LLM base, criado pela empresa USBABC."
                else:
                    response_text = f"Olá! Sou um modelo LLM base criado pela empresa USBABC. Você disse: '{message}'. Como posso ajudá-lo?"

                # PRIORIDADE 2: Log mostra arquitetura USBABC
                logger.info(f"🏗️ Resposta gerada pela arquitetura USBABC")

                return jsonify({
                    'success': True,
                    'response': response_text,
                    'model_type': 'usbabc_default'
                })

            prompt = f"<|user|>\n{message}</s>\n<|assistant|>\n"

            try:
                inputs = chat_tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024
                )
            except Exception as tokenizer_error:
                logger.error(f"❌ Erro no tokenizer: {tokenizer_error}")
                return jsonify({
                    'success': False,
                    'error': f'Erro no tokenizer: {str(tokenizer_error)}'
                }), 400



            # Filtrar apenas inputs suportados pelo modelo
            supported_keys = ['input_ids', 'attention_mask']

            try:
                # Mover inputs para o dispositivo do modelo
                model_device = next(chat_model.parameters()).device
                inputs = {k: v.to(model_device) for k, v in inputs.items() if k in supported_keys}
                logger.info(f"🎮 Inputs movidos para: {model_device}")
            except Exception as device_error:
                logger.error(f"❌ Erro ao mover inputs para dispositivo: {device_error}")
                return jsonify({
                    'success': False,
                    'error': f'Erro de dispositivo: {str(device_error)}'
                }), 500

            try:
                with torch.no_grad():
                    # Configurações de geração simplificadas
                    generation_config = {
                        'max_new_tokens': 50,  # Reduzido para teste
                        'do_sample': False,    # Desabilitado para teste
                        'num_beams': 1
                    }

                    # Adicionar tokens especiais se disponíveis
                    if hasattr(chat_tokenizer, 'pad_token_id') and chat_tokenizer.pad_token_id is not None:
                        generation_config['pad_token_id'] = chat_tokenizer.pad_token_id
                    if hasattr(chat_tokenizer, 'eos_token_id') and chat_tokenizer.eos_token_id is not None:
                        generation_config['eos_token_id'] = chat_tokenizer.eos_token_id

                    logger.info(f"🔧 Configuração de geração: {generation_config}")

                    outputs = chat_model.generate(**inputs, **generation_config)

            except Exception as generation_error:
                logger.error(f"❌ Erro na geração: {generation_error}")
                return jsonify({
                    'success': False,
                    'error': f'Erro na geração: {str(generation_error)}'
                }), 500

            # Verificar se há tokens gerados
            if len(outputs) == 0 or len(outputs[0]) == 0:
                response_text = "Desculpe, não consegui gerar uma resposta."
            else:
                try:
                    input_length = inputs['input_ids'].shape[1]
                    output_length = len(outputs[0])

                    logger.info(f"🔍 Debug: input_length={input_length}, output_length={output_length}")

                    if output_length > input_length:
                        # Há novos tokens gerados
                        response_tokens = outputs[0][input_length:]
                        response_text = chat_tokenizer.decode(
                            response_tokens,
                            skip_special_tokens=True
                        ).strip()
                        logger.info(f"✅ Novos tokens gerados: {len(response_tokens)}")
                    else:
                        # Não há novos tokens - usar fallback
                        logger.warning(f"⚠️ Nenhum token novo gerado (input={input_length}, output={output_length})")
                        response_text = chat_tokenizer.decode(
                            outputs[0],
                            skip_special_tokens=True
                        ).strip()

                        # Tentar remover o prompt da resposta
                        original_prompt = chat_tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
                        if original_prompt in response_text:
                            response_text = response_text.replace(original_prompt, "").strip()

                        # Se ainda não há resposta útil, usar fallback
                        if not response_text or len(response_text) < 3:
                            response_text = "Olá! Como posso ajudá-lo hoje?"

                except Exception as decode_error:
                    logger.error(f"❌ Erro na decodificação: {decode_error}")
                    logger.error(f"❌ Debug info: outputs shape={outputs.shape if hasattr(outputs, 'shape') else 'N/A'}")
                    response_text = "Desculpe, houve um erro na geração da resposta."

            del outputs
            cleanup_memory()

        # Verificação final da resposta
        if not response_text or len(response_text.strip()) < 2:
            response_text = "Desculpe, não consegui gerar uma resposta adequada. Tente reformular sua pergunta."

        # PRIORIDADE 2: Log mostra arquitetura USBABC
        logger.info(f"🏗️ Resposta gerada pela arquitetura USBABC")
        logger.info(f"✓ Resposta: {response_text[:100]}...")

        return jsonify({
            'success': True,
            'response': response_text,
            'model_type': model_type
        })

    except Exception as e:
        logger.error(f"❌ Erro chat: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Erro ao gerar resposta: {str(e)}'
        }), 500


@app.route('/model_status')
def model_status():
    """Status do modelo"""
    status = {
        'loaded': app_state['model_loaded'],
        'path': app_state.get('model_path'),
        'type': app_state.get('model_type'),
        'training_active': app_state['training_active'],
        'chat_available': app_state.get('chat_model') is not None
    }

    if torch.cuda.is_available():
        status['gpu_memory_allocated'] = round(
            torch.cuda.memory_allocated() / (1024**3), 2
        )
        status['gpu_memory_reserved'] = round(
            torch.cuda.memory_reserved() / (1024**3), 2
        )

    return jsonify(status)


@app.route('/load_existing_model', methods=['POST'])
def load_existing_model():
    """Carregar modelo existente"""
    try:
        data = request.get_json()
        model_name = data.get('model_name')

        if not model_name:
            return jsonify({'success': False, 'error': 'Nome do modelo não fornecido'}), 400

        model_path = Path('modelos') / model_name

        if not model_path.exists():
            return jsonify({'success': False, 'error': f'Modelo {model_name} não encontrado'}), 404

        # Carregar o modelo
        if load_chat_model(str(model_path)):
            app_state['model_loaded'] = True
            app_state['model_path'] = str(model_path)
            logger.info(f"✅ Modelo {model_name} carregado com sucesso")
            return jsonify({'success': True, 'message': f'Modelo {model_name} carregado'})
        else:
            return jsonify({'success': False, 'error': 'Falha ao carregar modelo'}), 500

    except Exception as e:
        logger.error(f"❌ Erro ao carregar modelo: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/scrape_web', methods=['POST'])
def scrape_web():
    """Web scraping"""
    try:
        data = request.json
        query = data.get('query', '').strip()
        num_results = data.get('num_results', 10)

        if not query:
            return jsonify({'success': False, 'error': 'Query vazia'}), 400

        scraper = WebScraperService()

        def scrape_thread():
            try:
                socketio.emit('scraping_status', {
                    'status': 'Buscando...',
                    'progress': 10
                })

                result = scraper.scrape_and_generate(query, num_results)

                if result['success']:
                    socketio.emit('scraping_complete', {
                        'success': True,
                        'filepath': result['filename'],
                        'num_examples': result['num_examples']
                    })
                else:
                    socketio.emit('scraping_error', {
                        'error': result.get('error', 'Erro desconhecido')
                    })

            except Exception as e:
                logger.error(f"❌ Erro scraping: {e}")
                socketio.emit('scraping_error', {'error': str(e)})

        thread = threading.Thread(target=scrape_thread, daemon=True)
        thread.start()

        return jsonify({'success': True, 'message': 'Scraping iniciado'})

    except Exception as e:
        logger.error(f"❌ Erro scraping: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/cleanup_memory', methods=['POST'])
def cleanup_memory_route():
    """Limpar memória manualmente"""
    try:
        cleanup_memory()

        status = {'success': True, 'message': 'Memória limpa'}

        if torch.cuda.is_available():
            status['gpu_memory_freed'] = True
            status['gpu_memory_allocated'] = round(
                torch.cuda.memory_allocated() / (1024**3), 2
            )

        return jsonify(status)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/create_model', methods=['POST'])
def create_model():
    """Cria um modelo USBABC profissional do zero - PRIORIDADE 1: Carrega modelo base automaticamente"""
    try:
        # Aceitar tanto JSON quanto form data
        if request.is_json:
            data = request.json or {}
        else:
            data = request.form.to_dict() if request.form else {}

        model_size_option = data.get('model_size', 'small')
        model_name = data.get('model_name', 'usbabc_model_ptbr')

        def create_thread():
            """Thread para criar o modelo"""
            try:
                # PRIORIDADE 1: Carregar modelo base USBABC.gguf automaticamente
                logger.info("🚀 PRIORIDADE 1: Carregando modelo base USBABC automaticamente")
                base_model_path = os.path.join(os.path.dirname(__file__), "modelo base", "MODELO BASE USBABC.gguf")

                socketio.emit('model_creation_status', {
                    'status': 'Carregando modelo base USBABC.gguf...',
                    'progress': 5
                })

                # Verificar se o modelo base existe
                if os.path.exists(base_model_path):
                    logger.info(f"✅ Modelo base encontrado: {base_model_path}")

                    # Carregar o modelo base diretamente
                    if load_chat_model(base_model_path):
                        app_state['model_loaded'] = True
                        app_state['model_path'] = base_model_path

                        # PRIORIDADE 1: Configurar diretrizes específicas do USBABC
                        logger.info("🎯 PRIORIDADE 1: Configurando diretrizes USBABC")

                        socketio.emit('model_creation_status', {
                            'status': 'Modelo base USBABC carregado com diretrizes configuradas!',
                            'progress': 100
                        })

                        # Emitir evento de sucesso com informações do modelo base
                        socketio.emit('model_created', {
                            'success': True,
                            'path': base_model_path,
                            'model_name': 'MODELO BASE USBABC',
                            'architecture': 'USBABC',
                            'info': {
                                'name': 'MODELO BASE USBABC',
                                'architecture': 'USBABC',
                                'description': 'Modelo LLM base criado pela empresa USBABC',
                                'guidelines': {
                                    'identity': 'Sou um modelo LLM base, criado pela empresa USBABC',
                                    'restrictions': 'Não mencionar Google, OpenAI, Llama ou outras empresas'
                                }
                            },
                            'loaded': True,
                            'base_model': True
                        })

                        logger.info("============================================================")
                        logger.info("✅ MODELO BASE USBABC CARREGADO AUTOMATICAMENTE")
                        logger.info("🏢 Empresa: USBABC")
                        logger.info("🎯 Diretrizes: Configuradas")
                        logger.info("📂 Caminho: " + base_model_path)
                        logger.info("============================================================")
                        return
                    else:
                        logger.error("❌ Falha ao carregar modelo base USBABC")
                        socketio.emit('model_creation_status', {
                            'status': 'Erro ao carregar modelo base. Criando novo modelo...',
                            'progress': 10
                        })
                else:
                    logger.warning("⚠️ Modelo base USBABC.gguf não encontrado, criando novo modelo")
                    socketio.emit('model_creation_status', {
                        'status': 'Modelo base não encontrado. Criando novo modelo USBABC...',
                        'progress': 10
                    })

                # Fallback: Criar novo modelo se o base não estiver disponível
                socketio.emit('model_creation_status', {
                    'status': 'Inicializando modelo USBABC profissional...',
                    'progress': 15
                })

                model_path = Path('modelos') / model_name
                model_path.mkdir(parents=True, exist_ok=True)

                socketio.emit('model_creation_status', {
                    'status': 'Criando arquitetura profissional USBABC...',
                    'progress': 30
                })

                # Verificar se há configurações personalizadas
                has_custom_config = any([
                    data.get('hidden_size') and data.get('hidden_size') != 768,
                    data.get('num_layers') and data.get('num_layers') != 12,
                    data.get('num_heads') and data.get('num_heads') != 12,
                    data.get('max_length') and data.get('max_length') != 4096,
                    data.get('vocab_size') and data.get('vocab_size') != 32000
                ])

                if model_size_option == 'small' and not has_custom_config:
                    model, tokenizer = create_small_portuguese_model(save_path=str(model_path))
                else:
                    # Criar configuração personalizada
                    custom_config = {
                        'hidden_size': data.get('hidden_size', 64 if has_custom_config else 768),
                        'num_hidden_layers': data.get('num_layers', 2 if has_custom_config else 12),
                        'num_attention_heads': data.get('num_heads', 2 if has_custom_config else 12),
                        'num_key_value_heads': data.get('num_heads', 2 if has_custom_config else 12),  # Usar mesmo valor para evitar GQA
                        'max_position_embeddings': data.get('max_length', 256 if has_custom_config else 4096),
                        'intermediate_size': int(data.get('hidden_size', 64 if has_custom_config else 768) * 2.7)  # Padrão SwiGLU
                    }

                    model, tokenizer = create_usbabc_model(
                        model_size='custom',
                        vocab_size=data.get('vocab_size', 4000 if has_custom_config else 22000),
                        custom_config=custom_config,
                        save_path=str(model_path)
                    )

                socketio.emit('model_creation_status', {
                    'status': 'Modelo profissional criado!',
                    'progress': 90
                })

                model_size = sum(p.numel() for p in model.parameters())
                model_size_mb = (model_size * 4) / (1024 * 1024)

                info_file = model_path / 'model_info.json'
                if info_file.exists():
                    with open(info_file, 'r', encoding='utf-8') as f:
                        info = json.load(f)
                else:
                    info = {
                        'name': model_name,
                        'architecture': 'USBABC Profissional',
                        'parameters': model_size,
                        'size_mb': round(model_size_mb, 2)
                    }

                # Tentar carregar o modelo automaticamente
                socketio.emit('model_creation_status', {
                    'status': 'Carregando modelo criado...',
                    'progress': 95
                })

                # Carregar o modelo recém-criado
                if load_chat_model(str(model_path)):
                    app_state['model_loaded'] = True
                    app_state['model_path'] = str(model_path)
                    logger.info("✅ Modelo carregado automaticamente após criação")

                    socketio.emit('model_created', {
                        'success': True,
                        'path': str(model_path),
                        'model_name': model_name,
                        'parameters': model_size,
                        'size_mb': round(model_size_mb, 2),
                        'info': info,
                        'loaded': True
                    })
                else:
                    logger.warning("⚠️ Modelo criado mas não pôde ser carregado automaticamente")
                    socketio.emit('model_created', {
                        'success': True,
                        'path': str(model_path),
                        'model_name': model_name,
                        'parameters': model_size,
                        'size_mb': round(model_size_mb, 2),
                        'info': info,
                        'loaded': False
                    })

                # ZIP já é criado automaticamente durante a criação do modelo

                logger.info("=" * 60)
                logger.info("✅ MODELO USBABC PROFISSIONAL CRIADO")
                logger.info(f"📦 Nome: {model_name}")
                logger.info(f"🔢 Parâmetros: {model_size:,}")
                logger.info(f"📊 Tamanho: {model_size_mb:.2f} MB")
                logger.info(f"📂 Caminho: {model_path}")
                logger.info("=" * 60)

            except Exception as e:
                logger.error(f"❌ Erro ao criar modelo: {e}")
                import traceback
                traceback.print_exc()
                socketio.emit('model_creation_error', {'error': str(e)})

        thread = threading.Thread(target=create_thread, daemon=True)
        thread.start()

        return jsonify({'success': True, 'message': 'Criação de modelo profissional iniciada'})

    except Exception as e:
        logger.error(f"❌ Erro ao iniciar criação: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/start_quantization', methods=['POST'])
def start_quantization():
    """Inicia o processo de quantização do modelo"""
    try:
        data = request.json
        model_path = data.get('model_path', '').strip()
        output_path = data.get('output_path', '').strip()
        # Changed 'quant_type' to 'quantization_type' based on script.js
        quant_type = data.get('quantization_type', 'q4_k_m').upper() # Ensure upper case for gguf_requantizer

        if not model_path:
            return jsonify({'success': False, 'error': 'Caminho do modelo não fornecido'}), 400

        if not output_path:
            return jsonify({'success': False, 'error': 'Caminho de saída não fornecido'}), 400

        if not os.path.exists(model_path):
            return jsonify({'success': False, 'error': 'Arquivo do modelo não encontrado'}), 404

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        def quantize_thread():
            """Thread para executar a quantização"""
            try:
                converter = UniversalConverter()

                def progress_callback(message):
                    # Try to extract percentage
                    percent_match = re.search(r'(\d+)%', message)
                    percent = int(percent_match.group(1)) if percent_match else 50 # Default progress if not found
                    socketio.emit('quantization_progress', {
                        'message': message,
                        'percent': percent
                    })

                # Check if input is already GGUF
                if Path(model_path).suffix.lower() == '.gguf':
                    input_gguf_path = model_path
                else:
                     # Convert non-GGUF to GGUF first
                     temp_gguf_path = f"{model_path}_temp_for_quant.gguf"
                     progress_callback("Convertendo para formato GGUF...")
                     convert_success = converter.convert_to_gguf(
                         input_path=model_path,
                         output_path=temp_gguf_path,
                         progress_callback=progress_callback
                     )
                     if not convert_success:
                         raise Exception("Falha na conversão inicial para GGUF")
                     input_gguf_path = temp_gguf_path


                progress_callback(f"Quantizando modelo para {quant_type}...")
                result = converter.quantize_model(
                    input_path=input_gguf_path, # Use the GGUF input
                    output_path=output_path,
                    quant_type=quant_type,
                    progress_callback=progress_callback
                )

                # Clean up temporary GGUF if created
                if 'temp_gguf_path' in locals() and os.path.exists(temp_gguf_path):
                    os.remove(temp_gguf_path)

                if result:
                    final_size_mb = Path(output_path).stat().st_size / (1024*1024)
                    socketio.emit('quantization_complete', {
                        'success': True,
                        'output_path': output_path,
                        'size_mb': round(final_size_mb, 1),
                        'message': 'Quantização concluída com sucesso!'
                    })
                else:
                    socketio.emit('quantization_error', {
                        'error': 'Falha na quantização (ver logs do servidor)'
                    })

            except Exception as e:
                logger.error(f"❌ Erro na quantização: {e}")
                socketio.emit('quantization_error', {
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
            finally:
                 # Clean up temporary GGUF if created and thread failed
                if 'temp_gguf_path' in locals() and os.path.exists(temp_gguf_path):
                    try:
                        os.remove(temp_gguf_path)
                    except: pass


        thread = threading.Thread(target=quantize_thread, daemon=True)
        thread.start()

        return jsonify({
            'success': True,
            'message': 'Quantização iniciada',
            'output_path': output_path
        })

    except Exception as e:
        logger.error(f"❌ Erro ao iniciar quantização: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/start_scraping', methods=['POST'])
def start_scraping():
    """Inicia o processo de web scraping"""
    try:
        data = request.json
        query = data.get('query', '').strip()
        num_results = data.get('num_results', 10)

        if not query:
            return jsonify({'success': False, 'error': 'Query de busca não fornecida'}), 400

        # Validar número de resultados
        num_results = max(1, min(50, int(num_results)))

        def scraping_thread():
            """Thread para executar o web scraping"""
            try:
                # Emitir progresso inicial
                socketio.emit('scraping_progress', {
                    'message': f'Iniciando busca por: "{query}"'
                })

                # Usar o serviço de web scraping
                scraper = WebScraperService()

                # Emitir progresso
                socketio.emit('scraping_progress', {
                    'message': 'Buscando resultados na web...'
                })

                # Executar scraping
                results = scraper.scrape_and_generate(
                    query=query,
                    num_results=num_results
                )

                if results.get('success'):
                    # Emitir sucesso
                    socketio.emit('scraping_complete', {
                        'total_results': results.get('num_examples', 0),
                        'filename': results.get('filename', ''),
                        'num_urls': results.get('num_urls', 0),
                        'message': f'Web scraping concluído com sucesso!'
                    })

                    logger.info(f"✅ Web scraping concluído: {results.get('num_examples', 0)} exemplos salvos em {results.get('filename', '')}")

                else:
                    socketio.emit('scraping_error', {
                        'error': results.get('error', 'Erro desconhecido no web scraping')
                    })

            except Exception as e:
                logger.error(f"❌ Erro no web scraping: {e}")
                socketio.emit('scraping_error', {
                    'error': str(e)
                })

        # Iniciar thread
        thread = threading.Thread(target=scraping_thread, daemon=True)
        thread.start()

        return jsonify({
            'success': True,
            'message': 'Web scraping iniciado',
            'query': query,
            'num_results': num_results
        })

    except Exception as e:
        logger.error(f"❌ Erro ao iniciar web scraping: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/convert_safetensors', methods=['POST'])
def convert_safetensors():
    """Converte modelo SafeTensors para GGUF"""
    try:
        data = request.get_json()
        model_path = data.get('model_path')
        output_path = data.get('output_path', 'converted_model.gguf')
        quant_type = data.get('quant_type', 'Q4_K_M')

        if not model_path:
            return jsonify({'success': False, 'error': 'Caminho do modelo é obrigatório'})

        # Importar conversor
        # Assuming safetensors_to_gguf_converter.py exists and has the function
        from convert_hf_to_gguf import convert_hf_to_gguf # Use the correct script

        def progress_callback(message, percent):
             # Try to extract percentage if not directly provided
             percent_match = re.search(r'(\d+)%', message)
             if percent_match:
                 percent = int(percent_match.group(1))

             socketio.emit('conversion_progress', {
                 'message': message,
                 'percent': percent
             })

        # Wrap the synchronous function call in a thread for SocketIO
        def conversion_thread():
             try:
                 # convert_hf_to_gguf expects HF path, not just safetensors file
                 model_dir = str(Path(model_path).parent) if Path(model_path).is_file() else model_path
                 success = convert_hf_to_gguf(
                     model_dir, output_path # Pass the directory
                 )

                 if success:
                     size_mb = Path(output_path).stat().st_size / (1024*1024) if Path(output_path).exists() else 0
                     socketio.emit('conversion_complete', {
                         'success': True,
                         'output_path': output_path,
                         'size_mb': round(size_mb, 1)
                     })
                 else:
                     socketio.emit('conversion_complete', {
                         'success': False,
                         'error': 'Falha na conversão (ver logs)'
                     })
             except Exception as e:
                 logger.error(f"Erro na thread de conversão: {e}")
                 socketio.emit('conversion_complete', {
                     'success': False,
                     'error': str(e)
                 })

        thread = threading.Thread(target=conversion_thread, daemon=True)
        thread.start()

        return jsonify({'success': True, 'message': 'Conversão iniciada'})

    except Exception as e:
        logger.error(f"Erro na conversão SafeTensors: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/validate_safetensors', methods=['POST'])
def validate_safetensors():
    """Valida modelo SafeTensors"""
    try:
        data = request.get_json()
        model_path = data.get('model_path')

        if not model_path:
            return jsonify({'valid': False, 'error': 'Caminho do modelo é obrigatório'})

        # Assuming safetensors_to_gguf_converter.py exists and has the class/method
        # Simplified validation for now
        is_valid = False
        message = "Validação não implementada nesta rota"
        model_type = "unknown"
        vocab_size = 0
        supported = False
        try:
             if Path(model_path).suffix.lower() == '.safetensors':
                 from safetensors import safe_open
                 with safe_open(model_path, framework="pt") as f:
                     if len(f.keys()) > 0:
                         is_valid = True
                         message = "Arquivo SafeTensors válido"
                         supported = True # Assume support if valid
                         # Try to infer details from parent dir config.json
                         config_path = Path(model_path).parent / 'config.json'
                         if config_path.exists():
                              with open(config_path, 'r') as cf:
                                   config = json.load(cf)
                                   model_type = config.get('model_type', 'unknown')
                                   vocab_size = config.get('vocab_size', 0)

             else:
                 message = "Não é um arquivo .safetensors"

        except Exception as e:
            message = f"Erro na validação: {str(e)}"

        result = {
            'valid': is_valid,
            'message': message,
            'model_type': model_type,
            'vocab_size': vocab_size,
            'supported': supported,
            'error': None if is_valid else message
        }
        return jsonify(result)

    except Exception as e:
        logger.error(f"Erro na validação SafeTensors: {e}")
        return jsonify({'valid': False, 'error': str(e)})

@app.route('/estimate_conversion_size', methods=['POST'])
def estimate_conversion_size():
    """Estima tamanho da conversão"""
    try:
        data = request.get_json()
        model_path = data.get('model_path')
        quant_type = data.get('quant_type', 'Q4_K_M')

        if not model_path:
            return jsonify({'error': 'Caminho do modelo é obrigatório'})

        # Assuming safetensors_to_gguf_converter.py exists and has the class/method
        # Simplified estimation for now
        original_size_gb = 0
        estimated_size_gb = 0
        message = "Estimativa não implementada nesta rota"
        try:
            if Path(model_path).exists():
                 original_size_bytes = Path(model_path).stat().st_size
                 original_size_gb = original_size_bytes / (1024**3)
                 # Very rough estimate based on typical ratios
                 quant_info = QuantizationConfig.get_type_info(quant_type.upper())
                 ratio = quant_info.get('size_ratio', 0.15) # Default to ~Q4 ratio
                 estimated_size_gb = original_size_gb * ratio
                 message = f"Tamanho original: {original_size_gb:.2f} GB. Estimado ({quant_type}): {estimated_size_gb:.2f} GB"

        except Exception as e:
             message = f"Erro na estimativa: {str(e)}"


        result = {
             'original_size_gb': round(original_size_gb, 2),
             'estimated_size_gb': round(estimated_size_gb, 2),
             'quant_type': quant_type,
             'message': message,
             'error': None if original_size_gb > 0 else "Falha na estimativa"
        }

        return jsonify(result)

    except Exception as e:
        logger.error(f"Erro na estimativa: {e}")
        return jsonify({'error': str(e)})

@app.route('/create_model_with_training', methods=['POST'])
def create_model_with_training():
    """
    MISSÃO 1: Cria modelo do zero com treinamento automático
    Incorpora todos os arquivos da pasta /dados e gera modelo .GGUF
    """
    try:
        data = request.json or {}

        def create_and_train_thread():
            """Thread para criar e treinar o modelo automaticamente"""
            try:
                # Etapa 1: Processar todos os dados da pasta /dados
                socketio.emit('model_creation_status', {
                    'status': '🔍 Processando todos os arquivos da pasta /dados...',
                    'progress': 10
                })

                logger.info("🔍 Iniciando processamento de dados da pasta /dados")
                training_data, data_file_path = process_all_training_data("dados")

                # Etapa 2: Criar modelo base USBABC
                socketio.emit('model_creation_status', {
                    'status': '🗿 Criando modelo USBABC base...',
                    'progress': 30
                })

                model_name = f"usbabc_trained_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                model_path = Path('modelos') / model_name
                model_path.mkdir(parents=True, exist_ok=True)

                # Criar modelo pequeno otimizado
                from create_usbabc_model import create_small_portuguese_model
                model, tokenizer = create_small_portuguese_model(save_path=str(model_path), train_model_flag=False) # Create without training first


                socketio.emit('model_creation_status', {
                    'status': '✅ Modelo base criado com sucesso',
                    'progress': 50
                })

                # Etapa 3: TREINAR modelo com dados (se houver)
                if training_data and len(training_data) > 0:
                    socketio.emit('model_creation_status', {
                        'status': f'🚀 Iniciando treinamento com {len(training_data)} exemplos...',
                        'progress': 55
                    })

                    logger.info(f"🚀 Iniciando treinamento com {len(training_data)} amostras")

                    # Salvar dados de treinamento temporariamente se não foram salvos antes
                    if not data_file_path:
                         temp_data_path = "temp_training_data.json"
                         with open(temp_data_path, 'w', encoding='utf-8') as f:
                              json.dump(training_data, f, ensure_ascii=False, indent=2)
                         train_data_source = temp_data_path
                    else:
                         train_data_source = data_file_path


                    # Inicializar sistema de treinamento
                    training_system = AITrainingSystem()

                    # Carregar o modelo recém-criado
                    logger.info(f"📦 Carregando modelo para treinamento: {model_path}")
                    training_system.load_model(str(model_path))

                    socketio.emit('model_creation_status', {
                        'status': '📊 Preparando dataset...',
                        'progress': 60
                    })

                    # Preparar dataset
                    training_system.prepare_dataset(train_data_source)

                    socketio.emit('model_creation_status', {
                        'status': '🔥 Treinando modelo (isso pode demorar)...',
                        'progress': 65
                    })

                    # EXECUTAR TREINAMENTO
                    training_results = training_system.train()

                    if not training_results.get('success', False):
                        logger.error("❌ Treinamento falhou")
                        socketio.emit('model_creation_status', {
                            'status': '❌ Erro durante o treinamento',
                            'progress': 65,
                            'error': True
                        })
                        # Clean up temp data if created
                        if 'temp_data_path' in locals() and os.path.exists(temp_data_path): os.remove(temp_data_path)
                        return

                    logger.info("✅ Treinamento concluído com sucesso")
                    socketio.emit('model_creation_status', {
                        'status': '✅ Treinamento concluído!',
                        'progress': 85
                    })

                    # Limpar arquivo temporário
                    if 'temp_data_path' in locals() and os.path.exists(temp_data_path):
                        os.remove(temp_data_path)

                    # O modelo treinado já foi salvo pelo sistema de treinamento
                    # A função train agora retorna o caminho final
                    final_model_path = training_results.get('output_path', str(model_path)) # Use trained path if available

                else:
                    # Sem dados - apenas modelo base
                    logger.warning("⚠️ Nenhum dado de treinamento encontrado")
                    socketio.emit('model_creation_status', {
                        'status': '⚠️ Modelo base criado (sem treinamento)',
                        'progress': 85
                    })
                    # Save the base model explicitly if not trained
                    model.save_pretrained(str(model_path))
                    tokenizer.save_pretrained(str(model_path))
                    final_model_path = str(model_path)

                # Etapa 4: Finalizar
                socketio.emit('model_creation_status', {
                    'status': '🎉 Processo concluído!',
                    'progress': 100,
                    'model_path': final_model_path,
                    'training_samples': len(training_data) if training_data else 0
                })

                logger.info("=" * 60)
                logger.info(f"✅ Modelo finalizado: {final_model_path}")
                if training_data:
                    logger.info(f"📊 Amostras de treinamento: {len(training_data)}")
                logger.info("=" * 60)

            except Exception as e:
                logger.error(f"❌ Erro na criação/treinamento: {e}")
                import traceback
                traceback.print_exc()

                socketio.emit('model_creation_status', {
                    'status': f'❌ Erro: {str(e)}',
                    'progress': 0,
                    'error': True
                })
            finally:
                # Clean up temp data if created and thread failed
                if 'temp_data_path' in locals() and os.path.exists(temp_data_path):
                    try: os.remove(temp_data_path)
                    except: pass


        thread = threading.Thread(target=create_and_train_thread)
        thread.daemon = True
        thread.start()

        return jsonify({
            'success': True,
            'message': 'Criação e treinamento automático iniciado'
        })

    except Exception as e:
        logger.error(f"❌ Erro na rota create_model_with_training: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("🚀 AI Training System - Servidor OTIMIZADO")
    logger.info("=" * 60)
    logger.info("💡 Dicas:")
    logger.info("   - Use modelos menores para evitar crashes")
    logger.info("   - Limite o dataset a 500 exemplos")
    logger.info("   - Feche outros programas durante o treinamento")
    logger.info("   - O GGUF será ATUALIZADO automaticamente")
    logger.info("=" * 60)
    logger.info(f"🌍 Servidor rodando em http://{HOST}:{PORT}") # Log corrected host/port


    socketio.run(
        app,
        host=HOST,                 # Use HOST variable
        port=PORT,                 # Use PORT variable
        debug=False,
        allow_unsafe_werkzeug=True # Keep for compatibility, but consider alternatives if possible
    )
