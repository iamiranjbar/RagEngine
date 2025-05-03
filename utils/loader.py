import json
import os

class DocumentLoader:
    """
    A class for loading documents from various file formats.
    """
    
    @staticmethod
    def load_documents_from_json(data_path: str) -> list[str]:
        """
        Load documents from a JSON file.
        
        Args:
            data_path: Path to the JSON file containing documents
            
        Returns:
            List of document strings
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        with open(data_path, 'r') as f:
            documents = json.load(f)
        
        return documents
