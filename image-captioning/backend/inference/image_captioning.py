import json
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import img_to_array
from typing import Dict, Union, Tuple


class ImageCaptioningModel:
    """
    Image Captioning model using ResNet50 encoder + Transformer decoder.
    """

    def __init__(
        self,
        weights_path: str = "./weight/best_e2e_model.weights.h5",
        vocab_path: str = "./weight/vocabulary.json",
        image_size: int = 224,
        max_length: int = 35,
        d_model: int = 128,
        num_heads: int = 2,
        ff_dim: int = 512,
        num_layers: int = 2,
    ):
        """
        Initialize the Image Captioning model.

        Args:
            weights_path: Path to model weights file
            vocab_path: Path to vocabulary JSON file
            image_size: Input image size (square)
            max_length: Maximum caption length
            d_model: Transformer dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            num_layers: Number of transformer layers
        """
        self.image_size = image_size
        self.max_length = max_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers

        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU Available: {len(gpus)} GPU(s) detected")
            for gpu in gpus:
                print(f"   - {gpu.name}")
            # Enable memory growth to avoid OOM errors
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"   Memory growth setting failed: {e}")
        else:
            print("⚠️  No GPU detected - Running on CPU")
            print("   Inference will be slower. Consider using GPU for better performance.")

        # Load vocabulary
        if not Path(vocab_path).exists():
            raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}")

        with open(vocab_path, "r") as f:
            vocab_data = json.load(f)
            self.word_to_idx = vocab_data["word_to_idx"]
            self.idx_to_word = {int(k): v for k, v in vocab_data["idx_to_word"].items()}
            self.vocab_size = vocab_data["vocab_size"]

        print(f"Loaded vocabulary: {self.vocab_size} words")

        # Build model
        self.model = self._build_model()

        # Load weights
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Model weights not found at {weights_path}")

        self.model.load_weights(weights_path)
        print(f"Loaded model weights: {weights_path}")

    def _build_encoder(self):
        """Build ResNet50 encoder"""
        from tensorflow.keras.applications import ResNet50

        base_model = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        encoder = Model(inputs=base_model.input, outputs=x, name="ResNet50_Encoder")
        return encoder

    def _get_positional_encoding(self, max_len, d_model):
        """Generate positional encoding"""
        positions = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]
        dims = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
        angle_rates = 1 / tf.pow(10000.0, (2 * (dims // 2)) / tf.cast(d_model, tf.float32))
        angle_rads = positions * angle_rates
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)[tf.newaxis, ...]
        return pos_encoding

    def _build_transformer_decoder(self, image_feature_dim=2048):
        """Build Transformer decoder"""

        class TransformerDecoderLayer(tf.keras.layers.Layer):
            def __init__(self, d_model, num_heads, ff_dim, rate=0.1):
                super().__init__()
                self.mha1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
                self.mha2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
                self.ffn = tf.keras.Sequential([
                    layers.Dense(ff_dim, activation="relu"),
                    layers.Dense(d_model),
                ])
                self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
                self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
                self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
                self.dropout1 = layers.Dropout(rate)
                self.dropout2 = layers.Dropout(rate)
                self.dropout3 = layers.Dropout(rate)

            def call(self, x, enc_output, training=None, look_ahead_mask=None):
                attn1 = self.mha1(query=x, value=x, key=x, attention_mask=look_ahead_mask, training=training)
                out1 = self.layernorm1(x + self.dropout1(attn1, training=training))

                attn2 = self.mha2(query=out1, value=enc_output, key=enc_output, training=training)
                out2 = self.layernorm2(out1 + self.dropout2(attn2, training=training))

                ffn_output = self.ffn(out2, training=training)
                out3 = self.layernorm3(out2 + self.dropout3(ffn_output, training=training))

                return out3

        image_input = layers.Input(shape=(image_feature_dim,), name="image_features")
        image_proj = layers.Dense(self.d_model, activation="relu")(image_input)
        image_proj = layers.Lambda(lambda x: tf.expand_dims(x, 1))(image_proj)

        caption_input = layers.Input(shape=(self.max_length - 1,), name="caption_input")
        caption_emb = layers.Embedding(self.vocab_size, self.d_model, mask_zero=True)(caption_input)

        pos_encoding = self._get_positional_encoding(self.max_length - 1, self.d_model)
        caption_emb += tf.cast(pos_encoding[:, :self.max_length - 1, :], tf.float32)

        x = caption_emb
        look_ahead_mask = tf.linalg.band_part(tf.ones((self.max_length - 1, self.max_length - 1)), -1, 0)

        for _ in range(self.num_layers):
            x = TransformerDecoderLayer(self.d_model, self.num_heads, self.ff_dim, rate=0.1)(
                x, image_proj, look_ahead_mask=look_ahead_mask
            )

        output = layers.Dense(self.vocab_size)(x)
        return Model(inputs=[image_input, caption_input], outputs=output)

    def _build_model(self):
        """Build end-to-end model"""
        encoder = self._build_encoder()
        decoder = self._build_transformer_decoder()

        image_input = layers.Input(shape=(224, 224, 3), name="image_input")
        caption_input = layers.Input(shape=(self.max_length - 1,), name="caption_input")

        image_features = encoder(image_input)
        output = decoder([image_features, caption_input])

        model = Model(inputs=[image_input, caption_input], outputs=output, name="E2E_ImageCaptioning")
        return model

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for inference.

        Args:
            image: PIL Image object

        Returns:
            Preprocessed image array
        """
        image = image.resize((self.image_size, self.image_size))
        img_array = img_to_array(image)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        return np.expand_dims(img_array, 0)

    def generate_caption_greedy(self, image: Union[str, Image.Image]) -> Dict[str, str]:
        """
        Generate caption using greedy search.

        Args:
            image: Path to image or PIL Image

        Returns:
            Dictionary with 'caption' key
        """
        if isinstance(image, str):
            if not Path(image).exists():
                raise FileNotFoundError(f"Image not found at {image}")
            image = Image.open(image).convert("RGB")

        image_input = self.preprocess_image(image)

        start_token = self.word_to_idx['<SOS>']
        end_token = self.word_to_idx['<EOS>']
        pad_token = self.word_to_idx['<PAD>']

        caption_tokens = [start_token]

        for _ in range(self.max_length - 1):
            input_seq = caption_tokens + [pad_token] * (self.max_length - 1 - len(caption_tokens))
            input_seq = np.array([input_seq[:self.max_length - 1]])

            predictions = self.model.predict([image_input, input_seq], verbose=0)
            next_token = np.argmax(predictions[0][len(caption_tokens) - 1])

            if next_token == end_token:
                break

            caption_tokens.append(int(next_token))

        caption_words = [self.idx_to_word.get(token, '<UNK>') 
                        for token in caption_tokens[1:] 
                        if token not in [start_token, end_token, pad_token]]

        return {"caption": ' '.join(caption_words)}

    def generate_caption_beam_search(
        self, 
        image: Union[str, Image.Image], 
        beam_width: int = 5, 
        alpha: float = 0.7
    ) -> Dict[str, str]:
        """
        Generate caption using beam search.

        Args:
            image: Path to image or PIL Image
            beam_width: Number of beams
            alpha: Length penalty factor

        Returns:
            Dictionary with 'caption' key
        """
        if isinstance(image, str):
            if not Path(image).exists():
                raise FileNotFoundError(f"Image not found at {image}")
            image = Image.open(image).convert("RGB")

        image_input = self.preprocess_image(image)

        start_token = self.word_to_idx['<SOS>']
        end_token = self.word_to_idx['<EOS>']
        pad_token = self.word_to_idx['<PAD>']

        sequences = [[start_token]]
        scores = [0.0]

        for _ in range(self.max_length - 1):
            all_candidates = []

            for seq, score in zip(sequences, scores):
                if seq[-1] == end_token:
                    all_candidates.append((seq, score))
                    continue

                input_seq = seq + [pad_token] * (self.max_length - 1 - len(seq))
                input_seq = np.array([input_seq[:self.max_length - 1]])

                predictions = self.model.predict([image_input, input_seq], verbose=0)
                next_word_probs = predictions[0][len(seq) - 1]

                top_indices = np.argsort(next_word_probs)[::-1][:beam_width]

                for idx in top_indices:
                    if idx == pad_token:
                        continue

                    new_seq = seq + [int(idx)]
                    lp = ((5 + len(new_seq)) ** alpha) / ((5 + 1) ** alpha)
                    new_score = (score - np.log(next_word_probs[idx] + 1e-8)) / lp

                    all_candidates.append((new_seq, new_score))

            all_candidates.sort(key=lambda x: x[1])
            sequences = [c[0] for c in all_candidates[:beam_width]]
            scores = [c[1] for c in all_candidates[:beam_width]]

            if all(seq[-1] == end_token for seq in sequences):
                break

        best_sequence = sequences[0]
        caption_words = [self.idx_to_word.get(token, '<UNK>') 
                        for token in best_sequence[1:] 
                        if token not in [start_token, end_token, pad_token]]

        return {"caption": ' '.join(caption_words)}


# Example usage
if __name__ == "__main__":
    model = ImageCaptioningModel()
    result = model.generate_caption_beam_search("./test/example.jpg")
    print(f"Caption: {result['caption']}")
