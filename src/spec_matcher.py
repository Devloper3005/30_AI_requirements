from vector_db import VectorDB
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
import torch
import numpy as np
from typing import List, Dict, Any

class SpecMatcher:
    def __init__(self, vector_db: VectorDB = None, model_name="bert-base-uncased"):
        self.vector_db = vector_db or VectorDB()
        self.model_name = model_name
        
        # Load tokenizer and model for embeddings
        if "roberta" in model_name.lower():
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.model = RobertaModel.from_pretrained(model_name)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name)
        
        self.model.eval()

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text using BERT/RoBERTa"""
        try:
            if not text or not text.strip():
                # Return zero vector for empty text
                return np.zeros(768, dtype=np.float32)
                
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling of last hidden state
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            
            return embedding.astype(np.float32)
        except Exception as e:
            print(f"Error generating embedding for text: {e}")
            # Return zero vector as fallback
            return np.zeros(768, dtype=np.float32)

    def index_requirements(self, requirements: List[Dict[str, Any]]) -> int:
        """Index a list of requirements into the vector database"""
        embeddings = []
        metadata = []
        
        for req in requirements:
            # Combine text and supplier comment for embedding
            combined_text = req.get('text', '')
            supplier_comment = req.get('supplier_comment', '')
            if supplier_comment:
                combined_text += " " + supplier_comment
            
            # Generate embedding
            embedding = self.embed_text(combined_text)
            embeddings.append(embedding)
            
            # Store metadata
            metadata.append({
                'id': req.get('id', ''),
                'text': req.get('text', ''),
                'supplier_comment': req.get('supplier_comment', ''),
                'supplier_status': req.get('supplier_status', ''),
                'combined_text': combined_text
            })
        
        # Add to vector database
        self.vector_db.add(embeddings, metadata)
        return len(embeddings)

    def match(self, input_text: str, top_k=5) -> List[Dict[str, Any]]:
        """Find similar requirements for given input text"""
        try:
            if self.vector_db.index.ntotal == 0:
                return []
            
            if not input_text or not input_text.strip():
                return []
            
            # Generate embedding for input text
            query_embedding = self.embed_text(input_text)
            
            # Search for similar requirements
            results = self.vector_db.search(query_embedding, top_k=top_k)
            
            return results
        except Exception as e:
            print(f"Error during matching: {e}")
            return []

    def explain_match(self, input_text: str, matched_req: Dict[str, Any]) -> str:
        """Generate explanation for why a requirement was matched"""
        explanation = f"Match found with similarity score: {matched_req.get('similarity_score', 0):.3f}\n\n"
        explanation += f"Input: {input_text}\n\n"
        explanation += f"Matched Requirement (ID: {matched_req.get('id', 'N/A')}):\n"
        explanation += f"Text: {matched_req.get('text', 'N/A')}\n\n"
        
        if matched_req.get('supplier_comment'):
            explanation += f"Supplier Comment: {matched_req.get('supplier_comment')}\n\n"
        
        explanation += f"Previous Status: {matched_req.get('supplier_status', 'N/A')}"
        
        return explanation

    def get_statistics(self) -> Dict[str, Any]:
        """Get matching statistics"""
        return self.vector_db.get_stats()