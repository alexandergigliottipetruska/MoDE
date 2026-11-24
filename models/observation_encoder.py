# DINOv2 + CLIP encoders
import torch
import torch.nn as nn
from transformers import AutoModel, CLIPTextModelWithProjection, AutoTokenizer
from torchvision.transforms import v2, Resize, InterpolationMode
from torchvision.models import resnet50

class FiLMLayer(nn.Module):
    def __init__(self, channel_dims=256, lang_dim=512):
        super().__init__()
        self.channel_dims = channel_dims
        self.lang_dim = lang_dim

        # Scale projection
        self.scale_proj = nn.Linear(self.lang_dim, self.channel_dims)

        for layer in self.scale_proj.parameters():
            nn.init.zeros_(layer)

        # Shift projection
        self.shift_proj = nn.Linear(self.lang_dim, self.channel_dims)

        for layer in self.shift_proj.parameters():
            nn.init.zeros_(layer)

    def forward(self, x, l):
        """
        x: Feature map produced by ResNet [B, C, H, W]
        l: Language Embedding [B, 512]
        """
        # Scale param
        gamma = self.scale_proj(l)

        # Shift param
        beta = self.shift_proj(l)

        # Shift and scale operation
        y = (torch.ones_like(gamma[:, :, None, None]) + gamma[:, :, None, None]) * x + beta[:, :, None, None]

        return y

class FiLMResNet50Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet50_encoder = resnet50(weights=None)
        self.film_lay1 = FiLMLayer(channel_dims=256)
        self.film_lay2 = FiLMLayer(channel_dims=512)
        self.film_lay3 = FiLMLayer(channel_dims=1024)
        self.film_lay4 = FiLMLayer(channel_dims=2048)

        # ResNet Stem
        self.relu = self.resnet50_encoder.relu
        self.maxpool = self.resnet50_encoder.maxpool
        self.avg_pool = self.resnet50_encoder.avgpool
        self.resnet50_encoder.fc = nn.Identity()

    def forward(self, x, l):
        # ResNet Stem
        x_1 = self.maxpool(self.relu(self.resnet50_encoder.bn1(self.resnet50_encoder.conv1(x))))

        # ResNet + FiLM stack
        H_1 = self.film_lay1(self.resnet50_encoder.layer1(x_1), l)
        H_2 = self.film_lay2(self.resnet50_encoder.layer2(H_1), l)
        H_3 = self.film_lay3(self.resnet50_encoder.layer3(H_2), l)
        H_4 = self.film_lay4(self.resnet50_encoder.layer4(H_3), l)

        y = self.avg_pool(H_4)
        return y.flatten(1)

class RandomShiftsAug(nn.Module):
    def __init__(self, padding):
        super().__init__()

        self.p = padding

    def forward(self, image):
        # x: [B, 3, H, W]
        B, _, H, W = image.shape

        # Pad edges of spatial dimensions
        padded_image = torch.nn.functional.pad(image, (self.p, self.p, self.p, self.p), mode="replicate")

        ## Unshifted Sampling Grid
        i = torch.arange(self.p, self.p + H)
        j = torch.arange(self.p, self.p + W)

        # Normalized coordinates of pixel centres
        y_coord = 2 * ((i + 0.5) / (H + 2 * self.p)) - 1
        x_coord = 2 * ((j + 0.5) / (W + 2 * self.p)) - 1

        Y, X = torch.meshgrid(y_coord, x_coord, indexing="ij")

        sampling_grid = torch.stack([X, Y], dim=-1)
        assert sampling_grid.shape == (H, W, 2)

        # Sample offsets
        delta_x = torch.randint(low=-self.p, high=self.p+1, size=(B,))
        delta_y = torch.randint(low=-self.p, high=self.p+1, size=(B,))

        delta_x_norm = 2 * delta_x / (W + 2 * self.p)
        delta_y_norm = 2 * delta_y / (H + 2 * self.p)

        norm_offsets = torch.stack([delta_x_norm, delta_y_norm], dim=-1)
        print(norm_offsets.shape)
        assert norm_offsets.shape == (B, 2)

        shifted_grid = sampling_grid.expand(B, -1, -1, -1) + norm_offsets[:, None, None, :]

        # Shifted image
        shifted_image = torch.nn.functional.grid_sample(padded_image, grid = shifted_grid, mode="bilinear", align_corners=False)

        assert shifted_image.shape == (B, 3, H, W)

        return shifted_image

# Convert raw images and test instructions into pretrained embeddings.
class ObservationEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder_gripper = FiLMResNet50Encoder()
        self.vision_encoder_static_rgb = FiLMResNet50Encoder()
        self.language_encoder = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        # Freeze parameters
        for param in self.language_encoder.parameters():
            param.requires_grad = False

        self.transforms = v2.Compose([
            v2.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711])
        ])

        self.rgb_transform = Resize((224, 224), interpolation=InterpolationMode.BILINEAR, antialias=True)
        self.gripper_transform = Resize((112, 112), interpolation=InterpolationMode.BILINEAR, antialias=True)

        self.rgb_shift = RandomShiftsAug(padding=10)
        self.gripper_shift = RandomShiftsAug(padding=4)

        
    def forward(self, rgb_static, gripper_images, lang_ins):
        # static images [B, 3, 200, 200]
        # gripper images [B, 3, 84, 84]
        # instructions list[str], length B
        B = rgb_static.shape[0]

        # Set language model to eval mode
        self.language_encoder.eval()

        l = self.encode_language(lang_ins)
        s, g = self.encode_images(rgb_static, gripper_images, l)
        
        # return static features [B, 2048], gripper features [B, 2048], and language instructions [B, 512]
        assert s.shape == (B, 2048)
        assert g.shape == (B, 2048)
        assert l.shape == (B, 512)

        return s, g, l

    def encode_images(self, rgb_static, gripper_images, language_embed):
        B = rgb_static.shape[0]

        # Resize to 224 x 224, Bilinear Mode, Antialias true
        resized_rgb = self.rgb_transform(rgb_static)
        resized_gripper = self.gripper_transform(gripper_images)

        # Convert to float32
        resized_rgb = resized_rgb.to(torch.float32)
        resized_gripper = resized_gripper.to(torch.float32)

        # Random Shifts Augment (Training time Spatial Augmentation)
        rgb_shifted = self.rgb_shift(resized_rgb)
        gripper_shifted = self.gripper_shift(resized_gripper)

        # Scale images to values between [0, 1]
        rgb_scaled = (rgb_shifted / 255.0)
        gripper_scaled = (gripper_shifted / 255.0)

        # Normalize using CLIP statistics
        rgb_normalized = self.transforms(rgb_scaled)
        gripper_normalized = self.transforms(gripper_scaled)

        assert rgb_normalized.shape == (B, 3, 224, 224)
        assert gripper_normalized.shape == (B, 3, 112, 112)

        assert rgb_normalized.dtype == torch.float32
        assert gripper_normalized.dtype == torch.float32

        assert torch.isfinite(rgb_normalized).all()
        assert torch.isfinite(gripper_normalized).all()

        # Pass into ResNet50 Encoders
        gripper_features = self.vision_encoder_gripper(gripper_normalized, language_embed)
        static_features = self.vision_encoder_static_rgb(rgb_normalized, language_embed)

        # Each is [B, 2048]
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
        
        # Extract .text_embeds, which is [B,512].
        language_features = outputs.text_embeds

        assert language_features.shape == (B, 512)

        return language_features
    