#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Processador Universal de Dados - MISSÃO 1
Sistema para processar automaticamente todos os arquivos da pasta /dados
Suporta: JSON, PARQUET, CSV, TXT, WEBDATASET, JSONL, XLS
"""

import os
import json
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Any, Union
import glob

logger = logging.getLogger(__name__)

class UniversalDataProcessor:
    """Processador universal para todos os formatos de dados"""
    
    def __init__(self, data_dir: str = "dados"):
        self.data_dir = Path(data_dir)
        self.supported_formats = {
            '.json': self._process_json,
            '.jsonl': self._process_jsonl,
            '.csv': self._process_csv,
            '.txt': self._process_txt,
            '.parquet': self._process_parquet,
            '.xls': self._process_excel,
            '.xlsx': self._process_excel,
            '.webdataset': self._process_webdataset,
            '.doc': self._process_doc,
            '.docx': self._process_docx,
            '.pdf': self._process_pdf
        }
        
    def process_all_data(self) -> List[Dict[str, Any]]:
        """
        Processa automaticamente todos os arquivos da pasta /dados
        
        Returns:
            Lista unificada de dados de treinamento
        """
        logger.info(f"🔍 Processando todos os arquivos em: {self.data_dir}")
        
        if not self.data_dir.exists():
            logger.warning(f"⚠️ Diretório não encontrado: {self.data_dir}")
            return []
        
        all_data = []
        processed_files = 0
        
        # Buscar todos os arquivos suportados
        for file_path in self.data_dir.rglob('*'):
            if file_path.is_file():
                file_ext = file_path.suffix.lower()
                
                if file_ext in self.supported_formats:
                    try:
                        logger.info(f"📄 Processando: {file_path.name}")
                        processor = self.supported_formats[file_ext]
                        file_data = processor(file_path)
                        
                        if file_data:
                            all_data.extend(file_data)
                            processed_files += 1
                            logger.info(f"✅ {file_path.name}: {len(file_data)} registros")
                        else:
                            logger.warning(f"⚠️ {file_path.name}: Nenhum dado extraído")
                            
                    except Exception as e:
                        logger.error(f"❌ Erro ao processar {file_path.name}: {e}")
                else:
                    logger.debug(f"🔄 Ignorando arquivo não suportado: {file_path.name}")
        
        logger.info(f"✅ Processamento concluído:")
        logger.info(f"   - Arquivos processados: {processed_files}")
        logger.info(f"   - Total de registros: {len(all_data)}")
        
        return all_data
    
    def _process_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """Processa arquivos JSON"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                return self._normalize_data(data)
            elif isinstance(data, dict):
                return self._normalize_data([data])
            else:
                logger.warning(f"⚠️ Formato JSON não suportado em {file_path.name}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Erro ao processar JSON {file_path.name}: {e}")
            return []
    
    def _process_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Processa arquivos JSONL (JSON Lines)"""
        try:
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(f"⚠️ Linha {line_num} inválida em {file_path.name}: {e}")
            
            return self._normalize_data(data)
            
        except Exception as e:
            logger.error(f"❌ Erro ao processar JSONL {file_path.name}: {e}")
            return []
    
    def _process_csv(self, file_path: Path) -> List[Dict[str, Any]]:
        """Processa arquivos CSV"""
        try:
            # Tentar diferentes encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                logger.error(f"❌ Não foi possível decodificar {file_path.name}")
                return []
            
            # Converter para lista de dicionários
            data = df.to_dict('records')
            return self._normalize_data(data)
            
        except Exception as e:
            logger.error(f"❌ Erro ao processar CSV {file_path.name}: {e}")
            return []
    
    def _process_txt(self, file_path: Path) -> List[Dict[str, Any]]:
        """Processa arquivos TXT"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                return []
            
            # Dividir por linhas ou parágrafos
            if '\n\n' in content:
                # Dividir por parágrafos
                chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
            else:
                # Dividir por linhas
                chunks = [line.strip() for line in content.split('\n') if line.strip()]
            
            # Criar dados de treinamento
            data = []
            for i, chunk in enumerate(chunks):
                if len(chunk) > 10:  # Ignorar chunks muito pequenos
                    data.append({
                        'input': f"Texto {i+1}:",
                        'output': chunk,
                        'source': file_path.name,
                        'type': 'text'
                    })
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Erro ao processar TXT {file_path.name}: {e}")
            return []
    
    def _process_parquet(self, file_path: Path) -> List[Dict[str, Any]]:
        """Processa arquivos Parquet"""
        try:
            df = pd.read_parquet(file_path)
            data = df.to_dict('records')
            return self._normalize_data(data)
            
        except Exception as e:
            logger.error(f"❌ Erro ao processar Parquet {file_path.name}: {e}")
            return []
    
    def _process_excel(self, file_path: Path) -> List[Dict[str, Any]]:
        """Processa arquivos Excel (XLS/XLSX)"""
        try:
            # Ler todas as planilhas
            excel_file = pd.ExcelFile(file_path)
            all_data = []
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                sheet_data = df.to_dict('records')
                
                # Adicionar informação da planilha
                for record in sheet_data:
                    record['_sheet'] = sheet_name
                
                all_data.extend(sheet_data)
            
            return self._normalize_data(all_data)
            
        except Exception as e:
            logger.error(f"❌ Erro ao processar Excel {file_path.name}: {e}")
            return []
    
    def _process_webdataset(self, file_path: Path) -> List[Dict[str, Any]]:
        """Processa arquivos WebDataset (formato TAR)"""
        try:
            # Para WebDataset, assumir que é um arquivo de texto por enquanto
            # Implementação completa requereria biblioteca webdataset
            logger.warning(f"⚠️ WebDataset não totalmente implementado para {file_path.name}")
            return self._process_txt(file_path)
            
        except Exception as e:
            logger.error(f"❌ Erro ao processar WebDataset {file_path.name}: {e}")
            return []
    
    def _normalize_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normaliza dados para formato de treinamento padrão
        Detecta automaticamente campos de input/output
        """
        if not data:
            return []
        
        normalized = []
        
        for record in data:
            if not isinstance(record, dict):
                continue
            
            # Detectar campos de input/output automaticamente
            input_field = None
            output_field = None
            
            # Campos comuns para input
            input_candidates = ['input', 'question', 'query', 'prompt', 'text', 'instruction', 'pergunta']
            # Campos comuns para output  
            output_candidates = ['output', 'answer', 'response', 'target', 'label', 'resposta', 'resultado']
            
            # Buscar campos de input
            for field in input_candidates:
                if field in record and record[field]:
                    input_field = field
                    break
            
            # Buscar campos de output
            for field in output_candidates:
                if field in record and record[field]:
                    output_field = field
                    break
            
            # Se não encontrou campos específicos, usar heurísticas
            if not input_field or not output_field:
                keys = list(record.keys())
                
                if len(keys) >= 2:
                    # Usar os dois primeiros campos
                    if not input_field:
                        input_field = keys[0]
                    if not output_field:
                        output_field = keys[1]
                elif len(keys) == 1:
                    # Usar o único campo como output
                    output_field = keys[0]
                    input_field = 'text'
                    record[input_field] = f"Conteúdo:"
            
            # Criar registro normalizado
            if input_field and output_field and record.get(input_field) and record.get(output_field):
                normalized_record = {
                    'input': str(record[input_field]).strip(),
                    'output': str(record[output_field]).strip(),
                    'source': record.get('source', 'unknown'),
                    'type': record.get('type', 'qa')
                }
                
                # Adicionar campos extras se existirem
                for key, value in record.items():
                    if key not in ['input', 'output', input_field, output_field] and value:
                        normalized_record[f'extra_{key}'] = value
                
                normalized.append(normalized_record)
        
        return normalized
    
    def _process_doc(self, file_path: Path) -> List[Dict[str, Any]]:
        """Processa arquivos DOC (Word antigo)"""
        try:
            # Tentar usar python-docx2txt para DOC
            try:
                import docx2txt
                content = docx2txt.process(str(file_path))
            except ImportError:
                logger.warning("⚠️ python-docx2txt não instalado. Tentando alternativa...")
                # Fallback: tentar ler como texto
                with open(file_path, 'rb') as f:
                    content = f.read().decode('utf-8', errors='ignore')
            
            if not content or not content.strip():
                return []
            
            # Dividir em parágrafos
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip() and len(p.strip()) > 20]
            
            data = []
            for i, paragraph in enumerate(paragraphs):
                data.append({
                    'input': f"Documento {file_path.stem} - Parágrafo {i+1}:",
                    'output': paragraph,
                    'source': file_path.name,
                    'type': 'document'
                })
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Erro ao processar DOC {file_path.name}: {e}")
            return []
    
    def _process_docx(self, file_path: Path) -> List[Dict[str, Any]]:
        """Processa arquivos DOCX (Word moderno)"""
        try:
            try:
                from docx import Document
                doc = Document(file_path)
                
                # Extrair texto de todos os parágrafos
                paragraphs = []
                for paragraph in doc.paragraphs:
                    text = paragraph.text.strip()
                    if text and len(text) > 20:
                        paragraphs.append(text)
                
            except ImportError:
                logger.warning("⚠️ python-docx não instalado. Tentando docx2txt...")
                try:
                    import docx2txt
                    content = docx2txt.process(str(file_path))
                    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip() and len(p.strip()) > 20]
                except ImportError:
                    logger.error("❌ Nenhuma biblioteca para DOCX disponível")
                    return []
            
            if not paragraphs:
                return []
            
            data = []
            for i, paragraph in enumerate(paragraphs):
                data.append({
                    'input': f"Documento {file_path.stem} - Seção {i+1}:",
                    'output': paragraph,
                    'source': file_path.name,
                    'type': 'document'
                })
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Erro ao processar DOCX {file_path.name}: {e}")
            return []
    
    def _process_pdf(self, file_path: Path) -> List[Dict[str, Any]]:
        """Processa arquivos PDF"""
        try:
            try:
                import PyPDF2
                
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text_content = []
                    
                    for page_num, page in enumerate(pdf_reader.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text and page_text.strip():
                                text_content.append(f"Página {page_num + 1}: {page_text.strip()}")
                        except Exception as e:
                            logger.warning(f"⚠️ Erro ao extrair página {page_num + 1}: {e}")
                            continue
                
            except ImportError:
                logger.warning("⚠️ PyPDF2 não instalado. Tentando pdfplumber...")
                try:
                    import pdfplumber
                    
                    text_content = []
                    with pdfplumber.open(file_path) as pdf:
                        for page_num, page in enumerate(pdf.pages):
                            try:
                                page_text = page.extract_text()
                                if page_text and page_text.strip():
                                    text_content.append(f"Página {page_num + 1}: {page_text.strip()}")
                            except Exception as e:
                                logger.warning(f"⚠️ Erro ao extrair página {page_num + 1}: {e}")
                                continue
                                
                except ImportError:
                    logger.error("❌ Nenhuma biblioteca para PDF disponível (PyPDF2 ou pdfplumber)")
                    return []
            
            if not text_content:
                return []
            
            # Dividir em seções menores se o texto for muito longo
            data = []
            for i, content in enumerate(text_content):
                # Se o conteúdo for muito longo, dividir em chunks
                if len(content) > 1000:
                    chunks = [content[j:j+1000] for j in range(0, len(content), 800)]  # Overlap de 200 chars
                    for chunk_num, chunk in enumerate(chunks):
                        if chunk.strip():
                            data.append({
                                'input': f"PDF {file_path.stem} - Seção {i+1}.{chunk_num+1}:",
                                'output': chunk.strip(),
                                'source': file_path.name,
                                'type': 'pdf'
                            })
                else:
                    data.append({
                        'input': f"PDF {file_path.stem} - Seção {i+1}:",
                        'output': content,
                        'source': file_path.name,
                        'type': 'pdf'
                    })
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Erro ao processar PDF {file_path.name}: {e}")
            return []
    
    def save_processed_data(self, data: List[Dict[str, Any]], output_path: str = "dados/processed_training_data.json") -> str:
        """Salva dados processados em arquivo JSON"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ Dados processados salvos em: {output_path}")
            logger.info(f"📊 Total de registros: {len(data)}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"❌ Erro ao salvar dados processados: {e}")
            raise
    
    def get_data_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Retorna estatísticas dos dados processados"""
        if not data:
            return {}
        
        stats = {
            'total_records': len(data),
            'sources': {},
            'types': {},
            'avg_input_length': 0,
            'avg_output_length': 0,
            'fields': set()
        }
        
        total_input_len = 0
        total_output_len = 0
        
        for record in data:
            # Contar por fonte
            source = record.get('source', 'unknown')
            stats['sources'][source] = stats['sources'].get(source, 0) + 1
            
            # Contar por tipo
            type_val = record.get('type', 'unknown')
            stats['types'][type_val] = stats['types'].get(type_val, 0) + 1
            
            # Calcular comprimentos médios
            if 'input' in record:
                total_input_len += len(str(record['input']))
            if 'output' in record:
                total_output_len += len(str(record['output']))
            
            # Coletar campos
            stats['fields'].update(record.keys())
        
        if len(data) > 0:
            stats['avg_input_length'] = total_input_len / len(data)
            stats['avg_output_length'] = total_output_len / len(data)
        
        stats['fields'] = list(stats['fields'])
        
        return stats


def process_all_training_data(data_dir: str = "dados") -> tuple[List[Dict[str, Any]], str]:
    """
    Função principal para processar todos os dados de treinamento
    
    Returns:
        Tuple com (dados_processados, caminho_arquivo_salvo)
    """
    processor = UniversalDataProcessor(data_dir)
    
    # Processar todos os dados
    all_data = processor.process_all_data()
    
    if not all_data:
        logger.warning("⚠️ Nenhum dado foi processado!")
        return [], ""
    
    # Salvar dados processados
    output_path = processor.save_processed_data(all_data)
    
    # Mostrar estatísticas
    stats = processor.get_data_statistics(all_data)
    logger.info("📊 Estatísticas dos dados processados:")
    logger.info(f"   - Total de registros: {stats['total_records']}")
    logger.info(f"   - Fontes: {list(stats['sources'].keys())}")
    logger.info(f"   - Tipos: {list(stats['types'].keys())}")
    logger.info(f"   - Comprimento médio input: {stats['avg_input_length']:.1f}")
    logger.info(f"   - Comprimento médio output: {stats['avg_output_length']:.1f}")
    
    return all_data, output_path


if __name__ == "__main__":
    # Teste do processador
    logging.basicConfig(level=logging.INFO)
    data, path = process_all_training_data()
    print(f"Processados {len(data)} registros, salvos em: {path}")