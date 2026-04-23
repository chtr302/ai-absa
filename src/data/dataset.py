import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer
from typing import List
from src.data.schemas import ABSAResult, Sentiment

class ABSADataset(Dataset):
    """
    Standardized Dataset class for AI-ABSA project.
    Supports contextual sentences and quadruplet metadata.
    """
    def __init__(self, documents: List[ABSAResult], tokenizer: RobertaTokenizer, max_length: int = 128):
        self.documents = documents
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Mapping for sentiment classification (to be used by the sentiment module)
        self.sentiment_map = {
            Sentiment.NEUTRAL: 0,
            Sentiment.POSITIVE: 1,
            Sentiment.NEGATIVE: 2
        }

    def __len__(self):
        return len(self.documents)

    def _get_primary_label(self, doc: ABSAResult) -> int:
        """
        Heuristic to get a single label for baseline models.
        In ASQP, this will be replaced by multi-label or pointer-based logic.
        """
        # Placeholder logic: defaults to Neutral if no sentences/quads are present
        return self.sentiment_map[Sentiment.NEUTRAL]

    def __getitem__(self, index):
        doc = self.documents[index]
        # In the new schema, we process the first sentence as a baseline or iterate through sentences
        if not doc.sentences:
            return None
            
        target_sentence = doc.sentences[0]
        text = target_sentence['text']
        
        label = self._get_primary_label(doc)

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'parent_context': doc.parent_context
        }
