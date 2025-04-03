import torch
from PIL import Image
from torchvision import transforms
from transformers import ProcessorMixin, BatchEncoding
from transformers.image_processing_utils import BatchFeature
import numpy as np

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

def make_list_of_images(x):
    if not isinstance(x, list):
        return [x]
    return x

def get_image_transform(config=None):
    # config = config.vision_config
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            # transforms.CenterCrop(224),
            transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)  # assume image
        ]
    )
    return transform


def load_and_transform_image(image_path, transform):
    image = Image.open(image_path).convert('RGB') if isinstance(image_path, str) else image_path
    image_outputs = transform(image)
    return image_outputs

def generate_mask(bbox, shape):
    mask = np.zeros(shape, dtype=np.float32)

    x_min = int(bbox['x'])
    y_min = int(bbox['y'])
    x_max = int(bbox['x'] + bbox['w'])
    y_max = int(bbox['y'] + bbox['h'])

    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, shape[1])
    y_max = min(y_max, shape[0])

    mask[y_min:y_max, x_min:x_max] = 0.0
    return mask

def apply_mask_to_image(image, mask):
    if isinstance(image, Image.Image):
        image = np.array(image) 
    
    if len(image.shape) == 2:  
        image = np.expand_dims(image, axis=-1)
    elif len(image.shape) == 3 and image.shape[-1] not in [1, 3]:  
        raise ValueError("Invalid image shape.")

    mask = np.expand_dims(mask, axis=-1)
    masked_image = image * mask  
    return Image.fromarray(masked_image.astype(np.uint8))

def load_and_transform_image_bbox(image_path, bbox, transform):
    image = Image.open(image_path).convert('RGB') if isinstance(image_path, str) else image_path
    image_outputs = transform(image)
    H, W = image.size[::-1]

    # import pdb; pdb.set_trace()
    mask = generate_mask(bbox, (H, W))
    masked_image = apply_mask_to_image(image, mask)
    m_l = transform(masked_image)

    # mask = torch.tensor(mask, dtype=torch.float32, device=image_outputs.device)
    # mask = mask.unsqueeze(0).expand(1, C, -1, -1)

    m_w = image_outputs
    # m_l = (1 - mask) * image_outputs

    m_w = m_w.unsqueeze(1)
    m_l = m_l.unsqueeze(1)

    m_w = m_w.repeat(1, 8, 1, 1)
    m_l = m_l.repeat(1, 8, 1, 1)

    image_outputs = [m_w, m_l]

    return image_outputs

class LanguageBindImageProcessor(ProcessorMixin):
    attributes = []
    tokenizer_class = ("LanguageBindImageTokenizer")

    def __init__(self, config, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.transform = get_image_transform(config)
        self.image_processor = load_and_transform_image
        self.image_processor_bbox = load_and_transform_image_bbox
        self.tokenizer = tokenizer
        self.image_mean = OPENAI_DATASET_MEAN
        self.crop_size = {'height': 224, 'width': 224}

    def __call__(self, images=None, text=None, bbox=None, context_length=77, return_tensors=None, **kwargs):
        if bbox is None:
            if text is None and images is None:
                raise ValueError("You have to specify either text or images. Both cannot be none.")

            if text is not None:
                encoding = self.tokenizer(text, max_length=context_length, padding='max_length',
                                        truncation=True, return_tensors=return_tensors, **kwargs)

            if images is not None:
                images = make_list_of_images(images)
                image_features = [self.image_processor(image, self.transform) for image in images]
                image_features = torch.stack(image_features)

            if text is not None and images is not None:
                encoding["pixel_values"] = image_features
                return encoding
            elif text is not None:
                return encoding
            else:
                return {"pixel_values": image_features}
        else:
            if text is None and images is None:
                raise ValueError("You have to specify either text or images. Both cannot be none.") #generate_mask

            if text is not None:
                encoding = self.tokenizer(text, max_length=context_length, padding='max_length',
                                        truncation=True, return_tensors=return_tensors, **kwargs)

            if images is not None:
                images = make_list_of_images(images)
                image_features = [self.image_processor_bbox(image, bbox, self.transform) for image in images]
                # import pdb; pdb.set_trace()
                # image_features = torch.stack(image_features)

            if text is not None and images is not None:
                encoding["pixel_values"] = image_features
                return encoding
            elif text is not None:
                return encoding
            else:
                # return {"pixel_values": image_features}
                return image_features[0][0], image_features[0][1]

    def preprocess(self, images, bboreturn_tensors):
        return self.__call__(images=images, return_tensors=return_tensors)

    def batch_decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)

    def decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)
