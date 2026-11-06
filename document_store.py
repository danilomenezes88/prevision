"""
Prevision AI Engine - Repositório de Conhecimento e Biblioteca de Dados
Permite upload, armazenamento persistente, listagem e seleção de documentos (PDF, TXT, MD, CSV)
para enriquecer as simulações e a construção do grafo relacional.
"""

import os
import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger("Prevision.DocumentStore")

DATA_DIR = Path(__file__).resolve().parent / "data_repository"
DATA_DIR.mkdir(exist_ok=True)
INDEX_FILE = DATA_DIR / "repository_index.json"


class DocumentStore:
    def __init__(self):
        self.data_dir = DATA_DIR
        self.index_file = INDEX_FILE
        self._ensure_index()

    def _ensure_index(self):
        if not self.index_file.exists():
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)

    def _load_index(self) -> List[Dict[str, Any]]:
        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_index(self, index: List[Dict[str, Any]]):
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    def extract_text_from_upload(self, file_name: str, file_bytes: bytes) -> str:
        """Extrai texto limpo a partir de arquivos PDF, TXT, MD ou CSV."""
        ext = file_name.split(".")[-1].lower()

        if ext == "pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                text_pages = []
                for idx, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    text_pages.append(f"--- Página {idx + 1} ---\n{page_text}")
                return "\n\n".join(text_pages).strip()
            except Exception as e:
                logger.error(f"Erro ao ler PDF {file_name}: {e}")
                return f"[Erro na leitura do PDF: {e}]"

        elif ext in ["txt", "md", "csv", "json"]:
            try:
                return file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                return file_bytes.decode("latin-1", errors="ignore")
        else:
            try:
                return file_bytes.decode("utf-8", errors="ignore")
            except Exception:
                return ""

    def add_document(self, file_name: str, content: str, category: str = "Geral") -> Dict[str, Any]:
        """Salva um novo documento no repositório com metadados."""
        doc_id = f"doc_{int(datetime.now().timestamp())}_{len(self._load_index()) + 1}"
        file_path = self.data_dir / f"{doc_id}.txt"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        doc_meta = {
            "id": doc_id,
            "name": file_name,
            "category": category,
            "created_at": datetime.now().strftime("%d/%m/%Y %H:%M"),
            "char_count": len(content),
            "word_count": len(content.split()),
            "preview": content[:200].replace("\n", " ") + "..." if len(content) > 200 else content,
            "file_path": str(file_path)
        }

        index = self._load_index()
        # Se já existir documento com mesmo nome, atualiza
        index = [d for d in index if d["name"] != file_name]
        index.append(doc_meta)
        self._save_index(index)

        logger.info(f"Documento '{file_name}' salvo no repositório (ID: {doc_id}).")
        return doc_meta

    def list_documents(self) -> List[Dict[str, Any]]:
        """Retorna todos os documentos disponíveis no repositório."""
        return self._load_index()

    def get_document_content(self, doc_id: str) -> Optional[str]:
        """Retorna o texto completo de um documento."""
        index = self._load_index()
        for doc in index:
            if doc["id"] == doc_id:
                p = Path(doc["file_path"])
                if p.exists():
                    with open(p, "r", encoding="utf-8") as f:
                        return f.read()
        return None

    def delete_document(self, doc_id: str) -> bool:
        """Remove um documento do repositório."""
        index = self._load_index()
        new_index = []
        deleted = False
        for doc in index:
            if doc["id"] == doc_id:
                deleted = True
                p = Path(doc["file_path"])
                if p.exists():
                    p.unlink()
            else:
                new_index.append(doc)
        self._save_index(new_index)
        return deleted

    def get_combined_context(self, selected_doc_ids: List[str]) -> str:
        """Combina o texto dos documentos selecionados para injetar no GraphRAG."""
        if not selected_doc_ids:
            return ""

        parts = []
        for doc_id in selected_doc_ids:
            doc_content = self.get_document_content(doc_id)
            doc_meta = next((d for d in self._load_index() if d["id"] == doc_id), None)
            doc_name = doc_meta["name"] if doc_meta else doc_id
            if doc_content:
                parts.append(f"=== DOCUMENTO DO REPOSITÓRIO: {doc_name} ===\n{doc_content}")

        return "\n\n".join(parts)

    def auto_select_relevant_docs(self, scenario_text: str, top_k: int = 3) -> List[str]:
        """Seleciona automaticamente documentos do repositório com base em palavras-chave do cenário."""
        docs = self.list_documents()
        if not docs:
            return []

        words = set([w.lower() for w in scenario_text.split() if len(w) > 3])
        scored = []
        for doc in docs:
            content = self.get_document_content(doc["id"]) or ""
            doc_words = set([w.lower() for w in content.split() if len(w) > 3])
            overlap = len(words.intersection(doc_words))
            scored.append((doc["id"], overlap))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, score in scored[:top_k] if score > 0]


# Instância global padrão do repositório
document_store = DocumentStore()
