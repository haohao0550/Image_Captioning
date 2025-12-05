import torch
from PIL import Image
from pathlib import Path
from typing import Union, Dict
from transformers import AutoProcessor, BlipForConditionalGeneration


class BlipCaptioningModel:
    """
    BLIP-based Image Captioning model.
    """

    def __init__(
        self,
        model_path: str = "./best_model",
        max_length: int = 50,
        device: str = None,
    ):
        """
        Initialize the BLIP model.

        Args:
            model_path: Path to the BLIP model directory
            max_length: Maximum caption length
            device: Device to run model on ('cuda' or 'cpu'). Auto-detect if None.
        """
        self.max_length = max_length
        
        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Check if model path exists
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load model and processor
        print(f"Loading BLIP model from {model_path}...")
        self.model = BlipForConditionalGeneration.from_pretrained(str(model_path)).to(self.device)
        self.processor = AutoProcessor.from_pretrained(str(model_path))
        self.model.eval()
        
        print("✅ BLIP model loaded successfully!")

    def preprocess_image(self, image: Union[str, Image.Image]) -> Image.Image:
        """
        Preprocess image for inference.

        Args:
            image: Path to image or PIL Image object

        Returns:
            PIL Image in RGB format
        """
        if isinstance(image, str):
            if not Path(image).exists():
                raise FileNotFoundError(f"Image not found at {image}")
            image = Image.open(image)
        
        return image.convert("RGB")

    def generate_caption_greedy(self, image: Union[str, Image.Image]) -> Dict[str, str]:
        """
        Generate caption using greedy search.

        Args:
            image: Path to image or PIL Image

        Returns:
            Dictionary with 'caption' key
        """
        image = self.preprocess_image(image)
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # Generate caption with greedy decoding
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=1,  # Greedy search
                early_stopping=True,
            )
        
        caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        
        return {"caption": caption}

    def generate_caption_beam_search(
        self, 
        image: Union[str, Image.Image], 
        beam_width: int = 5,
        alpha: float = 1.0,
    ) -> Dict[str, str]:
        """
        Generate caption using beam search.

        Args:
            image: Path to image or PIL Image
            beam_width: Number of beams (beam_size)
            alpha: Length penalty factor

        Returns:
            Dictionary with 'caption' key
        """
        image = self.preprocess_image(image)
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # Generate caption with beam search
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=beam_width,
                no_repeat_ngram_size=3,
                early_stopping=True,
                length_penalty=alpha,
            )
        
        caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        
        return {"caption": caption}


# Example usage
if __name__ == "__main__":
    model = BlipCaptioningModel(model_path="../best_model")
    result = model.generate_caption_beam_search("./test/example.jpg")
    print(f"Caption: {result['caption']}")
