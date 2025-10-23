#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Criador de Modelos USBABC
Sistema para criar modelos USBABC do zero com configurações personalizadas
INCLUI: Fine-tuning de modelos base
"""

import os
import json
import torch
import logging
import datetime
import zipfile
import shutil
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    PreTrainedTokenizerFast, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from torch.utils.data import Dataset

from configuration_usbabc import USBABCConfig
from modeling_usbabc import USBABCForCausalLM

logger = logging.getLogger(__name__)


# ============================================================================
# Dataset personalizado para train.json com limpeza
# ============================================================================

class CleanTextDataset(Dataset):
    """Dataset com limpeza automática de textos"""
    
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizerFast, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        logger.info(f"📂 Carregando e limpando dados: {data_path}")
        
        # Carregar dados do JSON
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"📊 Total de exemplos no arquivo: {len(data)}")
        
        # Processar e limpar cada exemplo
        skipped = 0
        for item in data:
            if isinstance(item, dict):
                text = item.get('text') or item.get('content') or item.get('instruction') or str(item)
            else:
                text = str(item)
            
            # Limpar texto
            text = self._clean_text(text)
            
            # Validar
            if not self._is_valid_text(text):
                skipped += 1
                continue
            
            # Tokenizar
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            self.examples.append({
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze()
            })
        
        logger.info(f"✅ {len(self.examples)} exemplos válidos carregados")
        if skipped > 0:
            logger.info(f"⚠️ {skipped} exemplos descartados (inválidos)")
    
    def _clean_text(self, text: str) -> str:
        """Limpa e normaliza o texto"""
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)\"\'áàâãéèêíïóôõöúçñÁÀÂÃÉÈÊÍÏÓÔÕÖÚÇÑ]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def _is_valid_text(self, text: str) -> bool:
        """Valida se o texto é adequado para treinamento"""
        if len(text) < 10:
            return False
        words = text.split()
        if len(words) < 3:
            return False
        return True
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


class DetailedLoggingCallback(TrainerCallback):
    """Callback para logs detalhados durante treinamento"""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logger.info(f"📊 Step {state.global_step}: {logs}")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        logger.info(f"✅ Época {int(state.epoch)} concluída!")


# ============================================================================
# Funções de treinamento
# ============================================================================

def train_model(
    model,
    tokenizer: PreTrainedTokenizerFast,
    train_data_path: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    warmup_steps: int = 100,
    save_steps: int = 500,
    logging_steps: int = 100,
    gradient_accumulation_steps: int = 4
):
    """Treina o modelo USBABC com os dados fornecidos"""
    logger.info("🎓 Iniciando treinamento do modelo...")
    
    # Criar dataset
    dataset = CleanTextDataset(train_data_path, tokenizer, max_length=512)
    
    if len(dataset) == 0:
        logger.error("❌ Dataset vazio após limpeza!")
        raise ValueError("Dataset vazio")
    
    # Configurar data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Configurar argumentos de treinamento
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        save_total_limit=2,
        prediction_loss_only=True,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        fp16=torch.cuda.is_available(),
        logging_dir=os.path.join(output_dir, 'logs'),
        report_to=['tensorboard'] if os.path.exists(output_dir) else [],
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        max_steps=100,  # Definir max_steps para evitar erro com dataloader sem length
    )
    
    # Criar trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=[DetailedLoggingCallback()]
    )
    
    # Treinar
    logger.info(f"🚀 Iniciando treinamento por {num_epochs} épocas...")
    train_result = trainer.train()
    
    logger.info("✅ Treinamento concluído!")
    logger.info(f"📊 Loss final: {train_result.training_loss:.4f}")
    
    return model


# ============================================================================
# Fine-tuning de modelo base existente
# ============================================================================

def finetune_base_model(
    base_model_path: str,
    train_data_path: str,
    output_dir: str,
    num_epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 3e-5,
    create_gguf: bool = True,
    create_zip: bool = True
) -> Tuple:
    """
    Fine-tuning de um modelo base existente
    
    Args:
        base_model_path: Caminho do modelo base (ex: "modelo base/Usbabc_base_llm")
        train_data_path: Caminho dos dados de treinamento
        output_dir: Diretório de saída
        num_epochs: Número de épocas
        batch_size: Tamanho do batch
        learning_rate: Taxa de aprendizado
        create_gguf: Se deve criar versão GGUF
        create_zip: Se deve criar ZIP
        
    Returns:
        Tuple[model, tokenizer]
    """
    logger.info("=" * 60)
    logger.info("🚀 FINE-TUNING DE MODELO BASE")
    logger.info("=" * 60)
    
    # Verificar se modelo base existe
    if not os.path.exists(base_model_path):
        logger.error(f"❌ Modelo base não encontrado: {base_model_path}")
        raise FileNotFoundError(f"Modelo base não encontrado: {base_model_path}")
    
    # Verificar se dados existem
    if not os.path.exists(train_data_path):
        logger.error(f"❌ Dados não encontrados: {train_data_path}")
        raise FileNotFoundError(f"Dados não encontrados: {train_data_path}")
    
    logger.info(f"📦 Modelo base: {base_model_path}")
    logger.info(f"📂 Dados: {train_data_path}")
    logger.info(f"💾 Saída: {output_dir}")
    logger.info(f"⚙️ Épocas: {num_epochs}, Batch: {batch_size}, LR: {learning_rate}")
    
    # Carregar tokenizer e modelo
    logger.info("\n📦 Carregando modelo base...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=False
        )
        logger.info("✅ Modelo carregado com sucesso")
    except Exception as e:
        logger.error(f"❌ Erro ao carregar modelo: {e}")
        raise
    
    # Configurar pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # Info
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"📊 Parâmetros: {num_params:,}")
    
    # Treinar
    model = train_model(
        model=model,
        tokenizer=tokenizer,
        train_data_path=train_data_path,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    # Salvar
    logger.info(f"\n💾 Salvando modelo em {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Salvar info
    model_info = {
        "model_type": "usbabc_finetuned",
        "base_model": base_model_path,
        "finetuned_at": str(datetime.datetime.now()),
        "training_data": train_data_path,
        "num_epochs": num_epochs,
        "parameters": num_params,
    }
    
    with open(os.path.join(output_dir, "model_info.json"), "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Modelo salvo com {num_params:,} parâmetros")
    
    # Converter para GGUF e criar ZIP
    if create_gguf:
        try:
            logger.info("\n🔄 Convertendo para GGUF...")
            gguf_path = convert_to_gguf(output_dir)
            
            if gguf_path and os.path.exists(gguf_path):
                logger.info(f"✅ GGUF criado: {gguf_path}")
                
                if create_zip:
                    logger.info("\n📦 Criando ZIP com GGUF...")
                    zip_path = create_model_zip_with_gguf(output_dir, gguf_path)
                    logger.info(f"✅ ZIP criado: {zip_path}")
            else:
                if create_zip:
                    logger.info("\n📦 Criando ZIP com SafeTensors...")
                    zip_path = create_model_zip(output_dir)
                    logger.info(f"✅ ZIP criado: {zip_path}")
        except Exception as e:
            logger.error(f"❌ Erro: {e}")
    
    # Limpar checkpoints
    logger.info("\n🧹 Limpando checkpoints temporários...")
    for item in os.listdir(output_dir):
        if item.startswith('checkpoint-'):
            try:
                shutil.rmtree(os.path.join(output_dir, item))
                logger.info(f"   ✓ Removido: {item}")
            except:
                pass
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 FINE-TUNING CONCLUÍDO!")
    logger.info("=" * 60)
    
    return model, tokenizer


# ============================================================================
# Funções auxiliares (ZIP, GGUF, etc.)
# ============================================================================

def create_model_zip(model_path: str, zip_path: str = None) -> str:
    """Cria um arquivo ZIP contendo todos os arquivos do modelo"""
    try:
        if not os.path.isdir(model_path):
            raise ValueError(f"Caminho do modelo não é um diretório: {model_path}")
        
        if zip_path is None:
            model_name = os.path.basename(model_path)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_path = os.path.join(os.path.dirname(model_path), f"{model_name}_{timestamp}.zip")
        
        logger.info(f"🗜️ Criando ZIP do modelo: {zip_path}")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, os.path.dirname(model_path))
                    zipf.write(file_path, arcname)
                    logger.info(f"  ✓ Adicionado: {arcname}")
            
            # Incorpora train.json
            current_dir = os.path.dirname(os.path.abspath(__file__))
            train_json_path = os.path.join(current_dir, "dados", "train.json")
            
            if os.path.exists(train_json_path):
                zipf.write(train_json_path, "train.json")
                logger.info(f"  ✅ train.json incorporado")
            
            # Metadados
            metadata = {
                "created_at": datetime.datetime.now().isoformat(),
                "model_name": os.path.basename(model_path),
                "model_type": "USBABC",
                "version": "1.0",
            }
            
            zipf.writestr("model_metadata.json", json.dumps(metadata, indent=2, ensure_ascii=False))
        
        zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        logger.info(f"✅ ZIP criado: {zip_path} ({zip_size_mb:.2f} MB)")
        
        return zip_path
        
    except Exception as e:
        logger.error(f"❌ Erro ao criar ZIP: {e}")
        raise


def convert_to_gguf(model_path: str) -> str:
    """Converte um modelo para formato GGUF"""
    try:
        import gguf
        from pathlib import Path
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import numpy as np
        
        logger.info(f"🔧 Iniciando conversão para GGUF: {model_path}")
        
        model_path = Path(model_path)
        gguf_path = model_path.parent / f"{model_path.name}.gguf"
        
        logger.info("📦 Carregando modelo e tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=torch.float16)
        
        logger.info("🔧 Criando arquivo GGUF...")
        gguf_writer = gguf.GGUFWriter(str(gguf_path), "llama")
        
        # Metadados
        gguf_writer.add_name("USBABC-Model")
        gguf_writer.add_description("Modelo USBABC profissional")
        gguf_writer.add_architecture()
        gguf_writer.add_context_length(2048)
        gguf_writer.add_embedding_length(model.config.hidden_size)
        gguf_writer.add_block_count(model.config.num_hidden_layers)
        gguf_writer.add_feed_forward_length(model.config.intermediate_size)
        gguf_writer.add_head_count(model.config.num_attention_heads)
        gguf_writer.add_head_count_kv(getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads))
        gguf_writer.add_layer_norm_rms_eps(model.config.rms_norm_eps)
        gguf_writer.add_rope_freq_base(getattr(model.config, 'rope_theta', 10000.0))
        
        # Vocabulário
        logger.info("📝 Adicionando vocabulário...")
        vocab = tokenizer.get_vocab()
        tokens = []
        scores = []
        for token, token_id in sorted(vocab.items(), key=lambda x: x[1]):
            tokens.append(token.encode('utf-8'))
            scores.append(0.0)
        
        gguf_writer.add_tokenizer_model("llama")
        gguf_writer.add_token_list(tokens)
        gguf_writer.add_token_scores(scores)
        
        if tokenizer.bos_token_id is not None:
            gguf_writer.add_bos_token_id(tokenizer.bos_token_id)
        if tokenizer.eos_token_id is not None:
            gguf_writer.add_eos_token_id(tokenizer.eos_token_id)
        if tokenizer.unk_token_id is not None:
            gguf_writer.add_unk_token_id(tokenizer.unk_token_id)
        if tokenizer.pad_token_id is not None:
            gguf_writer.add_pad_token_id(tokenizer.pad_token_id)
        
        # Tensores
        logger.info("🔧 Convertendo tensores...")
        state_dict = model.state_dict()
        
        for name, tensor in state_dict.items():
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float16)
            
            numpy_tensor = tensor.detach().cpu().numpy()
            
            # Mapear nomes
            gguf_name = name.replace("model.", "")
            gguf_name = gguf_name.replace("layers.", "blk.")
            gguf_name = gguf_name.replace("self_attn.", "attn_")
            gguf_name = gguf_name.replace("mlp.", "ffn_")
            gguf_name = gguf_name.replace("input_layernorm", "attn_norm")
            gguf_name = gguf_name.replace("post_attention_layernorm", "ffn_norm")
            gguf_name = gguf_name.replace("q_proj", "q")
            gguf_name = gguf_name.replace("k_proj", "k")
            gguf_name = gguf_name.replace("v_proj", "v")
            gguf_name = gguf_name.replace("o_proj", "wo")
            gguf_name = gguf_name.replace("gate_proj", "w1")
            gguf_name = gguf_name.replace("up_proj", "w3")
            gguf_name = gguf_name.replace("down_proj", "w2")
            gguf_name = gguf_name.replace("embed_tokens", "token_embd")
            gguf_name = gguf_name.replace("norm", "output_norm")
            gguf_name = gguf_name.replace("lm_head", "output")
            
            gguf_writer.add_tensor(gguf_name, numpy_tensor)
        
        logger.info("💾 Finalizando arquivo GGUF...")
        gguf_writer.write_header_to_file()
        gguf_writer.write_kv_data_to_file()
        gguf_writer.write_tensors_to_file()
        gguf_writer.close()
        
        logger.info(f"✅ Conversão GGUF concluída: {gguf_path}")
        return str(gguf_path)
            
    except Exception as e:
        logger.error(f"❌ Erro ao converter para GGUF: {e}")
        return None


def create_model_zip_with_gguf(model_path: str, gguf_path: str) -> str:
    """Cria ZIP com GGUF em vez de SafeTensors"""
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.basename(model_path)
        zip_filename = f"{model_name}_{timestamp}.zip"
        zip_path = os.path.join(os.path.dirname(model_path), zip_filename)
        
        logger.info(f"🗜️ Criando ZIP do modelo GGUF: {zip_path}")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    if file == "model.safetensors" or file.endswith(".bin"):
                        continue
                    
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, os.path.dirname(model_path))
                    zipf.write(file_path, arcname)
            
            # Adicionar GGUF
            gguf_name = os.path.basename(gguf_path)
            model_dir_name = os.path.basename(model_path)
            gguf_arcname = f"{model_dir_name}/{gguf_name}"
            zipf.write(gguf_path, gguf_arcname)
            logger.info(f"  ✓ GGUF adicionado: {gguf_arcname}")
            
            # train.json
            train_json = "dados/train.json"
            if os.path.exists(train_json):
                zipf.write(train_json, "train.json")
            
            # Metadados
            metadata = {
                "created_at": datetime.datetime.now().isoformat(),
                "model_name": os.path.basename(model_path),
                "model_type": "USBABC-GGUF",
                "format": "GGUF",
            }
            zipf.writestr("model_metadata.json", json.dumps(metadata, indent=2, ensure_ascii=False))
        
        zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        logger.info(f"✅ ZIP GGUF criado: {zip_path} ({zip_size_mb:.2f} MB)")
        
        return zip_path
        
    except Exception as e:
        logger.error(f"❌ Erro ao criar ZIP GGUF: {e}")
        raise


def create_portuguese_tokenizer(vocab_size: int = 22000, train_data_path: str = None) -> PreTrainedTokenizerFast:
    """Cria tokenizer otimizado para português"""
    logger.info(f"🔧 Criando tokenizer português com vocab_size={vocab_size}")
    
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    # Textos de treinamento
    training_texts = []
    
    if train_data_path and os.path.exists(train_data_path):
        logger.info(f"📂 Carregando textos de: {train_data_path}")
        with open(train_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                if isinstance(item, dict):
                    text = item.get('text') or item.get('content') or item.get('instruction') or str(item)
                else:
                    text = str(item)
                training_texts.append(text)
    else:
        # Texto padrão
        portuguese_text = """
        O Brasil é um país localizado na América do Sul. A língua oficial é o português.
        São Paulo é a maior cidade do Brasil. O Rio de Janeiro é conhecido por suas praias.
        """
        training_texts = [portuguese_text] * 100
    
    logger.info(f"📝 Treinando tokenizer com {len(training_texts)} textos...")
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"]
    )
    
    tokenizer.train_from_iterator(training_texts, trainer)
    
    tokenizer_fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        model_max_length=2048
    )
    
    logger.info("✅ Tokenizer português criado")
    return tokenizer_fast


def create_usbabc_model(
    model_size: str = "small",
    vocab_size: int = 22000,
    custom_config: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None,
    train_data_path: Optional[str] = None,
    train_model_flag: bool = True,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    use_base_model: bool = False,
    base_model_path: str = "modelo base/Usbabc_base_llm"
) -> Tuple[USBABCForCausalLM, PreTrainedTokenizerFast]:
    """
    Cria modelo USBABC do zero OU faz fine-tuning de modelo base
    
    Args:
        model_size: Tamanho ('small', 'medium', 'large', 'custom')
        vocab_size: Tamanho do vocabulário
        custom_config: Configurações personalizadas
        save_path: Caminho para salvar
        train_data_path: Caminho do train.json
        train_model_flag: Se deve treinar
        num_epochs: Épocas de treinamento
        batch_size: Tamanho do batch
        learning_rate: Taxa de aprendizado
        use_base_model: Se True, faz fine-tuning do modelo base
        base_model_path: Caminho do modelo base
        
    Returns:
        Tuple[model, tokenizer]
    """
    
    # OPÇÃO 1: FINE-TUNING DE MODELO BASE
    if use_base_model:
        logger.info("🎯 Modo: FINE-TUNING de modelo base")
        return finetune_base_model(
            base_model_path=base_model_path,
            train_data_path=train_data_path or "dados/train.json",
            output_dir=save_path or "modelos/usbabc_finetuned",
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
    
    # OPÇÃO 2: CRIAR MODELO DO ZERO
    logger.info(f"🚀 Criando modelo USBABC {model_size} do zero")
    
    size_configs = {
        "small": {
            "hidden_size": 512,
            "intermediate_size": 1376,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "max_position_embeddings": 2048
        },
        "medium": {
            "hidden_size": 768,
            "intermediate_size": 2048,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "num_key_value_heads": 12,
            "max_position_embeddings": 2048
        },
        "large": {
            "hidden_size": 1024,
            "intermediate_size": 2816,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "max_position_embeddings": 4096
        }
    }
    
    config_dict = size_configs.get(model_size, size_configs["small"]).copy()
    
    if custom_config:
        config_dict.update(custom_config)
    
    config_dict["vocab_size"] = vocab_size
    
    config = USBABCConfig(**config_dict)
    
    logger.info(f"📋 Configuração do modelo:")
    logger.info(f"   - Vocab size: {config.vocab_size}")
    logger.info(f"   - Hidden size: {config.hidden_size}")
    logger.info(f"   - Layers: {config.num_hidden_layers}")
    logger.info(f"   - Attention heads: {config.num_attention_heads}")
    logger.info(f"   - Max position: {config.max_position_embeddings}")
    
    # Buscar train.json
    if train_data_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        train_data_path = os.path.join(current_dir, "dados", "train.json")
        if not os.path.exists(train_data_path):
            train_data_path = "dados/train.json"
    
    # Criar tokenizer
    tokenizer = create_portuguese_tokenizer(vocab_size, train_data_path)
    
    # Criar modelo
    logger.info("🔧 Inicializando modelo...")
    model = USBABCForCausalLM(config)
    model.apply(model._init_weights)
    
    # Configurar tokenizer
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    
    # Treinar se solicitado
    if train_model_flag and os.path.exists(train_data_path):
        logger.info(f"🎓 Iniciando treinamento com: {train_data_path}")
        
        temp_train_dir = os.path.join(save_path if save_path else ".", "training_temp")
        os.makedirs(temp_train_dir, exist_ok=True)
        
        model = train_model(
            model=model,
            tokenizer=tokenizer,
            train_data_path=train_data_path,
            output_dir=temp_train_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        logger.info("✅ Modelo treinado com sucesso!")
    elif train_model_flag:
        logger.warning(f"⚠️ Arquivo não encontrado: {train_data_path}")
        logger.warning("⚠️ Modelo criado sem treinamento")
    
    # Salvar
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Salvando modelo em {save_path}")
        
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        # Informações
        model_info = {
            "model_type": "usbabc",
            "model_size": model_size,
            "vocab_size": vocab_size,
            "created_at": str(datetime.datetime.now()),
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "trained": train_model_flag and os.path.exists(train_data_path),
            "training_data": train_data_path if train_model_flag else None,
            "config": config_dict
        }
        
        with open(save_path / "model_info.json", "w", encoding="utf-8") as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Modelo salvo com {model_info['parameters']:,} parâmetros")
        
        # Converter e criar ZIP
        try:
            logger.info("🔄 Convertendo para GGUF...")
            gguf_path = convert_to_gguf(str(save_path))
            
            if gguf_path and os.path.exists(gguf_path):
                logger.info("🔄 Criando ZIP com GGUF...")
                zip_path = create_model_zip_with_gguf(str(save_path), gguf_path)
                logger.info(f"📦 ZIP GGUF criado: {zip_path}")
            else:
                logger.warning("⚠️ Conversão GGUF falhou, criando ZIP com SafeTensors...")
                zip_path = create_model_zip(str(save_path))
                logger.info(f"📦 ZIP SafeTensors criado: {zip_path}")
                
        except Exception as e:
            logger.error(f"❌ Erro: {e}")
            try:
                logger.info("🔄 Fallback: Criando ZIP com SafeTensors...")
                zip_path = create_model_zip(str(save_path))
                logger.info(f"📦 ZIP criado: {zip_path}")
            except Exception as e2:
                logger.error(f"❌ Erro no fallback: {e2}")
    
    logger.info("✅ Modelo USBABC criado com sucesso")
    return model, tokenizer


def create_small_portuguese_model(
    save_path: str = "modelos/usbabc-small-pt",
    train_data_path: str = None,
    train_model_flag: bool = True,
    num_epochs: int = 3,
    batch_size: int = 4,
    use_base_model: bool = False
) -> Tuple[USBABCForCausalLM, PreTrainedTokenizerFast]:
    """
    Cria modelo pequeno para português OU faz fine-tuning
    
    Args:
        save_path: Caminho para salvar
        train_data_path: Caminho do train.json
        train_model_flag: Se deve treinar
        num_epochs: Épocas
        batch_size: Tamanho do batch
        use_base_model: Se True, faz fine-tuning do modelo base
    """
    logger.info("🇧🇷 Criando modelo para português brasileiro")
    
    if use_base_model:
        # Fine-tuning de modelo base
        logger.info("📦 Usando modelo base para fine-tuning")
        return create_usbabc_model(
            save_path=save_path,
            train_data_path=train_data_path,
            train_model_flag=True,
            num_epochs=num_epochs,
            batch_size=batch_size,
            use_base_model=True,
            base_model_path="modelo base/Usbabc_base_llm"
        )
    else:
        # Criar do zero
        portuguese_config = {
            "hidden_size": 512,
            "intermediate_size": 1376,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "attention_dropout": 0.0,
            "hidden_act": "silu"
        }
        
        return create_usbabc_model(
            model_size="custom",
            vocab_size=22000,
            custom_config=portuguese_config,
            save_path=save_path,
            train_data_path=train_data_path,
            train_model_flag=train_model_flag,
            num_epochs=num_epochs,
            batch_size=batch_size,
            use_base_model=False
        )


def load_usbabc_model(model_path: str) -> Tuple:
    """Carrega um modelo USBABC salvo"""
    logger.info(f"📂 Carregando modelo USBABC de {model_path}")
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    
    config = USBABCConfig.from_pretrained(model_path)
    model = USBABCForCausalLM.from_pretrained(model_path, config=config)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    
    logger.info("✅ Modelo USBABC carregado com sucesso")
    return model, tokenizer


def get_model_info(model_path: str) -> Dict[str, Any]:
    """Obtém informações sobre um modelo USBABC"""
    model_path = Path(model_path)
    info_file = model_path / "model_info.json"
    
    if info_file.exists():
        with open(info_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    try:
        config = USBABCConfig.from_pretrained(model_path)
        return {
            "model_type": "usbabc",
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "max_position_embeddings": config.max_position_embeddings
        }
    except Exception as e:
        logger.error(f"Erro ao obter informações: {e}")
        return {}


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Criador de Modelos USBABC")
    parser.add_argument("--mode", choices=["create", "finetune"], default="create",
                        help="Modo: create (do zero) ou finetune (modelo base)")
    parser.add_argument("--base-model", default="modelo base/Usbabc_base_llm",
                        help="Caminho do modelo base (modo finetune)")
    parser.add_argument("--train-data", default="dados/train.json",
                        help="Caminho dos dados de treinamento")
    parser.add_argument("--output", default="modelos/usbabc_model",
                        help="Diretório de saída")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Número de épocas")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Tamanho do batch")
    parser.add_argument("--learning-rate", type=float, default=3e-5,
                        help="Taxa de aprendizado")
    parser.add_argument("--vocab-size", type=int, default=32000,
                        help="Tamanho do vocabulário (modo create)")
    parser.add_argument("--model-size", default="medium",
                        choices=["small", "medium", "large"],
                        help="Tamanho do modelo (modo create)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 CRIADOR DE MODELOS USBABC")
    print("=" * 70)
    print()
    
    if args.mode == "finetune":
        print("🎯 Modo: FINE-TUNING de modelo base")
        print(f"📦 Modelo base: {args.base_model}")
        print(f"📂 Dados: {args.train_data}")
        print(f"💾 Saída: {args.output}")
        print(f"⚙️ Épocas: {args.epochs}, Batch: {args.batch_size}, LR: {args.learning_rate}")
        print()
        
        # Verificar modelo base
        if not os.path.exists(args.base_model):
            print(f"❌ Modelo base não encontrado: {args.base_model}")
            print()
            print("📋 Coloque o modelo base em:")
            print(f"   {os.path.abspath(args.base_model)}")
            sys.exit(1)
        
        print("✅ Modelo base encontrado")
        print()
        
        resp = input("🚀 Iniciar fine-tuning? (s/N): ")
        if resp.lower() != 's':
            print("❌ Cancelado")
            sys.exit(0)
        
        print()
        
        # Fine-tuning
        model, tokenizer = create_usbabc_model(
            save_path=args.output,
            train_data_path=args.train_data,
            train_model_flag=True,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            use_base_model=True,
            base_model_path=args.base_model
        )
        
    else:  # create
        print("🎯 Modo: CRIAR modelo do zero")
        print(f"📊 Tamanho: {args.model_size}")
        print(f"📊 Vocab: {args.vocab_size}")
        print(f"📂 Dados: {args.train_data}")
        print(f"💾 Saída: {args.output}")
        print(f"⚙️ Épocas: {args.epochs}, Batch: {args.batch_size}")
        print()
        
        resp = input("🚀 Iniciar criação? (s/N): ")
        if resp.lower() != 's':
            print("❌ Cancelado")
            sys.exit(0)
        
        print()
        
        # Criar do zero
        model, tokenizer = create_usbabc_model(
            model_size=args.model_size,
            vocab_size=args.vocab_size,
            save_path=args.output,
            train_data_path=args.train_data,
            train_model_flag=True,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            use_base_model=False
        )
    
    print()
    print("=" * 70)
    print("🎉 PROCESSO CONCLUÍDO COM SUCESSO!")
    print("=" * 70)
    print()
    print(f"📂 Modelo salvo em: {args.output}")
    print()
    print("🔥 Próximos passos:")
    print("   1. Teste o modelo na interface web")
    print("   2. Carregue o modelo criado")
    print("   3. Faça perguntas e avalie as respostas")
    print()
    print("=" * 70)
