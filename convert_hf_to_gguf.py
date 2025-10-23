#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conversor HuggingFace para GGUF
Converte modelos treinados com LoRA incorporado para formato GGUF
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import json

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    import gguf
    import numpy as np
    
    # Registrar modelo USBABC
    try:
        from modeling_usbabc import USBABCConfig, USBABCForCausalLM
        AutoConfig.register("usbabc", USBABCConfig)
        AutoModelForCausalLM.register(USBABCConfig, USBABCForCausalLM)
        print("OK Modelo USBABC registrado para conversao")
    except ImportError:
        print("AVISO Modelo USBABC nao disponivel")
    
    HAS_DEPS = True
except ImportError as e:
    print(f"ERRO Dependências faltando: {e}")
    HAS_DEPS = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_hf_to_gguf(hf_model_path: str, gguf_output_path: str) -> bool:
    """
    Converte modelo HuggingFace para GGUF
    
    Args:
        hf_model_path: Caminho do modelo HuggingFace
        gguf_output_path: Caminho de saída do GGUF
        
    Returns:
        bool: True se conversão foi bem-sucedida
    """
    if not HAS_DEPS:
        logger.error("ERRO Dependências não disponíveis")
        return False
    
    try:
        logger.info(f"CONV Convertendo {hf_model_path} -> {gguf_output_path}")
        
        # Carregar modelo e tokenizer
        logger.info("IN Carregando modelo HuggingFace...")
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            hf_model_path,
            trust_remote_code=True
        )
        
        # Criar diretório de saída
        output_path = Path(gguf_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Criar writer GGUF usando a biblioteca oficial
        logger.info("CONV Criando arquivo GGUF...")
        
        # Obter configuração do modelo
        config = model.config
        
        # Detectar arquitetura real do modelo
        model_type = getattr(config, 'model_type', 'gemma')
        if hasattr(config, 'architectures') and config.architectures:
            arch_name = config.architectures[0].lower()
            if 'gemma' in arch_name:
                model_type = 'gemma'
            elif 'llama' in arch_name:
                model_type = 'llama'
        
        logger.info(f"CONV Arquitetura detectada: {model_type}")
        
        # MÉTODO CORRETO: Usar arquitetura compatível com llama-cpp-python
        # Forçar arquitetura gemma que sabemos que funciona
        logger.info("CONV Usando arquitetura 'gemma' para compatibilidade")
        writer = gguf.GGUFWriter(gguf_output_path, "gemma")
        
        # Adicionar metadados essenciais
        writer.add_name("USBABC_TRAINED")
        writer.add_description("Modelo USBABC treinado com LoRA incorporado")
        writer.add_file_type(gguf.GGMLQuantizationType.F16)
        
        # Configurações do modelo baseadas no config real
        writer.add_context_length(getattr(config, 'max_position_embeddings', 2048))
        writer.add_embedding_length(getattr(config, 'hidden_size', 256))
        writer.add_block_count(getattr(config, 'num_hidden_layers', 8))
        writer.add_head_count(getattr(config, 'num_attention_heads', 8))
        writer.add_head_count_kv(getattr(config, 'num_key_value_heads', 8))
        
        # Informações do tokenizer
        vocab_size = getattr(tokenizer, 'vocab_size', 12000)
        writer.add_vocab_size(vocab_size)
        
        # Tokens especiais
        writer.add_bos_token_id(getattr(tokenizer, 'bos_token_id', 2))
        writer.add_eos_token_id(getattr(tokenizer, 'eos_token_id', 3))
        writer.add_pad_token_id(getattr(tokenizer, 'pad_token_id', 0))
        
        # Converter tensores com verificação
        logger.info("CONV Convertendo tensores...")
        state_dict = model.state_dict()
        tensor_count = 0
        
        for name, tensor in state_dict.items():
            try:
                logger.info(f"   📦 Tensor {tensor_count+1}/{len(state_dict)}: {name} {list(tensor.shape)}")
                
                # Converter para float16 numpy
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float16)
                elif tensor.dtype == torch.float32:
                    tensor = tensor.to(torch.float16)
                
                numpy_tensor = tensor.detach().cpu().numpy().astype(np.float16)
                
                # Verificar tensor válido
                if numpy_tensor.size == 0:
                    logger.warning(f"   ⚠️ Tensor vazio ignorado: {name}")
                    continue
                
                # Adicionar tensor
                writer.add_tensor(name, numpy_tensor)
                tensor_count += 1
                
            except Exception as e:
                logger.error(f"   ❌ Erro no tensor {name}: {e}")
                continue
        
        logger.info(f"CONV Processados {tensor_count} tensores")
        
        # Finalizar arquivo CORRETAMENTE
        logger.info("💾 Finalizando arquivo GGUF...")
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        
        # Verificar arquivo criado
        if output_path.exists() and output_path.stat().st_size > 1024:
            size_mb = output_path.stat().st_size / (1024**2)
            logger.info(f"✅ Conversão concluída! Arquivo: {size_mb:.1f} MB")
            return True
        else:
            logger.error("ERRO Arquivo GGUF não foi criado corretamente")
            return False
            
    except Exception as e:
        logger.error(f"ERRO Erro na conversão: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Converte modelo HuggingFace para GGUF')
    parser.add_argument('input_path', help='Caminho do modelo HuggingFace')
    parser.add_argument('output_path', help='Caminho de saída do GGUF')
    
    args = parser.parse_args()
    
    success = convert_hf_to_gguf(args.input_path, args.output_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()