# DINOv2 + CLIP encoders
import torch
import torch.nn as nn
from transformers import AutoModel, CLIPTextModel, AutoTokenizer
from torchvision import transforms

vision_encoder = 0
language_encoder = 0

# Convert raw images and test instructions into pretrained embeddings.
class ObservationEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = AutoModel.from_pretrained("facebook/dinov2-base")
        self.language_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        # Freeze parameters
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        for param in self.language_encoder.parameters():
            param.requires_grad = False

    def forward(self, rgb_static, gripper_images, lang_ins):
        # static images [B, 3, 200, 200]
        # gripper images [B, 3, 84, 84]
        # instructions list[str], length B
        B = rgb_static.shape[0]

        # Set both models to eval mode
        self.vision_encoder.eval()
        self.language_encoder.eval()

        # Vision embeddings are [B, 768]
        # Scale images to values between [0, 1] before passing them in and make sure they are float32
        rgb_scaled = (rgb_static / 255).to(torch.float32)
        gripper_scaled = (gripper_images / 255).to(torch.float32)

        s, g = self.encode_images(rgb_scaled, gripper_scaled)
        l = self.encode_language(lang_ins)

        # return static features [B, 768], gripper features [B, 768], and language instructions [B, 512]
        assert s.shape == (B, 768)
        assert g.shape == (B, 768)
        assert l.shape == (B, 512)

        return s, g, l

    def encode_images(self, rgb_static, gripper_images):
        # Resize to 224 x 224
        B = rgb_static.shape[0]
        resized_rgb = nn.functional.interpolate(rgb_static, mode="bilinear", size=(224, 224), align_corners=False)
        resized_gripper = nn.functional.interpolate(gripper_images, mode="bilinear", size=(224, 224), align_corners=False)

        assert resized_rgb.shape == (B, 3, 224, 224)
        assert resized_gripper.shape == (B, 3, 224, 224)

        # Concatenate along batch dimension
        batched_images = torch.cat([resized_rgb, resized_gripper], dim=0)

        assert batched_images.shape == (2 * B, 3, 224, 224)

        # Normalize using mean/std
        transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        images_normalized = transform(batched_images)

        # Pass into shared DINOv2 frozen encoder
        outputs = self.vision_encoder(images_normalized)

        # Extract CLS token (global representation of the image)
        features = outputs.last_hidden_state[:, 0, :]
        assert features.shape == (2 * B, 768)

        # Split resulting features after
        static_features, gripper_features = torch.chunk(features, dim=0, chunks=2)

        return static_features, gripper_features

    def encode_language(self, language_instructions):
        B = len(language_instructions)

        # Tokenize with AutoTokenizer, padding/truncation to 77 tokens. 
        tokens = self.tokenizer(language_instructions, return_tensors="pt", padding="max_length", truncation=True, max_length=77)

        # Obtain token IDs and attention masks, each [B,77]. 
        ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        assert ids.shape == (B, 77)
        assert attention_mask.shape == (B, 77)
        
        # Pass them through the frozen CLIPTextModel. 
        outputs = self.language_encoder(ids, attention_mask)
        
        # Extract .pooler_output, which is [B,512].
        language_features = outputs.pooler_output

        assert language_features.shape == (B, 512)

        return language_features
    