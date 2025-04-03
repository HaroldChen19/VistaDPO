
import torch
import copy, os, random
import cv2
import decord
from decord import VideoReader, cpu
decord.bridge.set_bridge('torch')
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import ProcessorMixin, BatchEncoding
from transformers.image_processing_utils import BatchFeature
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda, ToTensor, Resize
from torchvision.transforms._transforms_video import NormalizeVideo, RandomCropVideo, RandomHorizontalFlipVideo, CenterCropVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample

import torch.nn.functional as F


OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

def get_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image

def load_frames(frames_dir):
    results = []
    frame_names = os.listdir(frames_dir)
    frame_names.sort()
    # frame_files = [x for x in os.listdir(frames_dir) if x.endswith('jpg')]
    # # sort frame by name, converted to int
    # frame_files = sorted(frame_files, key=lambda x: int(x.split('.')[0]))
    for frame_name in frame_names:
        image_path = f"{frames_dir}/{frame_name}"
        results.append(image_path)
    return results

# def sample_frames(frames, num_segments):
#     if len(frames) <= num_segments:
#         return frames
#     frame_indices = list(range(len(frames)))
#     cand_indices = copy.deepcopy(frame_indices)
#     intervals = np.linspace(start=0, stop=len(frame_indices), num=num_segments + 1).astype(int)
#     ranges = []

#     for idx, interv in enumerate(intervals[:-1]):
#         ranges.append((interv, intervals[idx + 1] - 1))

#     try:
#         frame_indices = [cand_indices[random.choice(range(x[0], x[1]))] for x in ranges]
#     except:
#         frame_indices = [cand_indices[x[0]] for x in ranges]

#     sampled_frames = [frames[indice] for indice in frame_indices]

#     return sampled_frames

def sample_frames(frames, num_segments):
    duration = len(frames)
    frame_id_array = np.linspace(0, duration-1, num_segments, dtype=int)
    frame_id_list = frame_id_array.tolist()

    sampled_frames = []
    for frame_idx in frame_id_list:
        # import pdb; pdb.set_trace()
        image_path = frames[frame_idx]
        image = get_image(image_path)
        sampled_frames.append(image)
    return sampled_frames

def make_list_of_images(x):
    if not isinstance(x, list):
        return [x]
    return x

class ResizeVideo:
    def __init__(self, size):

        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, video):

        T, C, H, W = video.shape
        video = video.permute(1, 0, 2, 3) 
        video = F.interpolate(video, size=self.size, mode='bilinear', align_corners=False)
        video = video.permute(1, 0, 2, 3)  
        return video

def get_video_transform(video_decode_backend, num_frames=8):
    if video_decode_backend == 'pytorchvideo':
        transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                    ShortSideScale(size=224),
                    CenterCropVideo(224),
                    RandomHorizontalFlipVideo(p=0.5),
                ]
            ),
        )
    elif video_decode_backend == 'decord':
        transform = Compose(
            [
                # UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=224),
                CenterCropVideo(224),
                # RandomHorizontalFlipVideo(p=0.5),
            ]
        )

    elif video_decode_backend == 'opencv':
        transform = Compose(
            [
                # UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=224),
                CenterCropVideo(224),
                # RandomHorizontalFlipVideo(p=0.5),
            ]
        )
    elif video_decode_backend == 'frames':
        transform = Compose(
            [
                NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=224),
                # CenterCropVideo(224),
                ResizeVideo(size=(224, 224)),
                # RandomHorizontalFlipVideo(p=0.5),
            ]
        )
    else:
        raise NameError('video_decode_backend should specify in (pytorchvideo, decord, opencv)')
    return transform


def load_and_transform_video(
        video_path,
        transform,
        video_decode_backend='opencv',
        clip_start_sec=0.0,
        clip_end_sec=None,
        num_frames=8,
):
    # import pdb; pdb.set_trace()
    if video_decode_backend == 'pytorchvideo':
        #  decord pyav
        video = EncodedVideo.from_path(video_path, decoder="decord", decode_audio=False)
        duration = video.duration
        start_sec = clip_start_sec  # secs
        end_sec = clip_end_sec if clip_end_sec is not None else duration  # secs
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
        video_outputs = transform(video_data)
    
    elif video_decode_backend == 'decord':
        decord.bridge.set_bridge('torch')
        decord_vr = VideoReader(video_path, ctx=cpu(0))
        ori_duration = len(decord_vr)
        # frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)
        fps_vid = decord_vr.get_avg_fps()
        valid_duration = min(int(fps_vid * 10), ori_duration)
        frame_id_list = np.linspace(0, valid_duration-1, num_frames, dtype=int)
        video_data = decord_vr.get_batch(frame_id_list)
        video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        video_outputs = transform(video_data)

    # elif video_decode_backend == 'decord':
    #     decord.bridge.set_bridge('torch')
    #     decord_vr = VideoReader(video_path, ctx=cpu(0))
    #     duration = len(decord_vr)
    #     frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)
    #     video_data = decord_vr.get_batch(frame_id_list)
    #     video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
    #     video_outputs = transform(video_data)
    
    elif video_decode_backend == 'frames':
        frames = load_frames(video_path)
        frames = sample_frames(frames, num_frames)
        to_tensor = ToTensor()
        video_data = torch.stack([to_tensor(_) for _ in frames]).permute(1, 0, 2, 3) # (T, C, H, W) -> (C, T, H, W)
        video_outputs = transform(video_data)
        # import pdb; pdb.set_trace()

    elif video_decode_backend == 'opencv':
        cv2_vr = cv2.VideoCapture(video_path)
        duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)
        # frame_id_list = np.linspace(0, duration-5, num_frames, dtype=int)

        video_data = []
        for frame_idx in frame_id_list:
            cv2_vr.set(1, frame_idx)
            ret, frame = cv2_vr.read()
            if not ret:
                raise ValueError(f'video error at {video_path} for frame {frame_idx}')
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_data.append(torch.from_numpy(frame).permute(2, 0, 1))
        cv2_vr.release()
        video_data = torch.stack(video_data, dim=1)
        video_outputs = transform(video_data)
    else:
        raise NameError('video_decode_backend should specify in (pytorchvideo, decord, opencv)')
    return video_outputs

def generate_mask(bbox, frame_shape):

    H, W = frame_shape
    mask = np.zeros((H, W), dtype=np.float32)
    
    x, y, w, h = int(bbox["x"]), int(bbox["y"]), int(bbox["w"]), int(bbox["h"])
    mask[y:y+h, x:x+w] = 1.0  
    
    return mask

def load_and_transform_video_bbox(
        video_path,
        bbox,
        transform,
        video_decode_backend='opencv',
        clip_start_sec=0.0,
        clip_end_sec=None,
        num_frames=8,
        beta=0.5,
):
    # # import pdb; pdb.set_trace()
    # if video_decode_backend == 'pytorchvideo':
    #     #  decord pyav
    #     video = EncodedVideo.from_path(video_path, decoder="decord", decode_audio=False)
    #     duration = video.duration
    #     start_sec = clip_start_sec  # secs
    #     end_sec = clip_end_sec if clip_end_sec is not None else duration  # secs
    #     video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
    #     video_outputs = transform(video_data)
    
    # elif video_decode_backend == 'decord':
    #     decord.bridge.set_bridge('torch')
    #     decord_vr = VideoReader(video_path, ctx=cpu(0))
    #     ori_duration = len(decord_vr)
    #     # frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)
    #     fps_vid = decord_vr.get_avg_fps()
    #     valid_duration = min(int(fps_vid * 10), ori_duration)
    #     frame_id_list = np.linspace(0, valid_duration-1, num_frames, dtype=int)
    #     video_data = decord_vr.get_batch(frame_id_list)
    #     video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
    #     video_outputs = transform(video_data)
    
    if video_decode_backend == 'frames':
        frames = load_frames(video_path)
        frames = sample_frames(frames, num_frames)
        to_tensor = ToTensor()
        video_data = torch.stack([to_tensor(_) for _ in frames]).permute(1, 0, 2, 3) # (T, C, H, W) -> (C, T, H, W)

        C, T, H, W = video_data.shape
        m_w = torch.zeros_like(video_data, dtype=torch.float32)
        m_l = torch.zeros_like(video_data, dtype=torch.float32)
        mask = generate_mask(bbox, (H, W))
        mask = torch.tensor(mask, dtype=torch.float32, device=video_data.device) 
        mask = mask.unsqueeze(0)
        mask = mask.expand(C, -1, -1)
        
        for t in range(T):
            frame = video_data[:, t, :, :]
            m_w[:, t, :, :] = mask * frame + (1 - mask) * beta * frame
            m_l[:, t, :, :] = (1 - mask) * frame
        # print("video_data:", video_data.shape)
        # print("mask:", mask.shape)
        # print("m_w:", m_w.shape)
        # print("m_l:", m_l.shape)
        # import pdb; pdb.set_trace()

        video_w = transform(m_w)
        video_l = transform(m_l)
        # video_w = torch.stack(video_w)
        # video_l = torch.stack(video_l)
        image_features = [video_w, video_l]
        # import pdb; pdb.set_trace()

    # elif video_decode_backend == 'opencv':
    #     cv2_vr = cv2.VideoCapture(video_path)
    #     duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
    #     frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)
    #     # frame_id_list = np.linspace(0, duration-5, num_frames, dtype=int)

    #     video_data = []
    #     for frame_idx in frame_id_list:
    #         cv2_vr.set(1, frame_idx)
    #         ret, frame = cv2_vr.read()
    #         if not ret:
    #             raise ValueError(f'video error at {video_path} for frame {frame_idx}')
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         video_data.append(torch.from_numpy(frame).permute(2, 0, 1))
    #     cv2_vr.release()
    #     video_data = torch.stack(video_data, dim=1)
    #     video_outputs = transform(video_data)
    else:
        raise NameError('video_decode_backend should specify in (pytorchvideo, decord, opencv)')
    return image_features

class LanguageBindVideoProcessor(ProcessorMixin):
    attributes = []
    tokenizer_class = ("LanguageBindVideoTokenizer")

    def __init__(self, config, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.config.vision_config.video_decode_backend = 'frames'
        video_decode_backend = self.config.vision_config.video_decode_backend
        num_frames = self.config.vision_config.num_frames
        self.transform = get_video_transform(video_decode_backend, num_frames)
        self.image_processor = load_and_transform_video
        self.image_processor_bbox = load_and_transform_video_bbox
        self.tokenizer = tokenizer
        

    def __call__(self, video_path=None, text=None, bbox=None, context_length=77, return_tensors=None, **kwargs):
        images = video_path
        if bbox is None:
            if text is None and images is None:
                raise ValueError("You have to specify either text or images. Both cannot be none.")
            if text is not None:
                encoding = self.tokenizer(text, max_length=context_length, padding='max_length',
                                        truncation=True, return_tensors=return_tensors, **kwargs)

            if images is not None:
                if 'video_decode_backend' in kwargs:
                    video_decode_backend = kwargs['video_decode_backend']
                    num_frames = kwargs.get('num_frames', 8)
                    transform_function = get_video_transform(video_decode_backend, num_frames)
                else:
                    video_decode_backend = self.config.vision_config.video_decode_backend
                    transform_function = self.transform
                images = make_list_of_images(images)
                image_features = [self.image_processor(image, transform_function,
                                                    video_decode_backend=video_decode_backend,
                                                    num_frames=self.config.vision_config.num_frames) for image in images]
                # image_features = [torch.rand(3, 8, 224, 224) for image in images]
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
                raise ValueError("You have to specify either text or images. Both cannot be none.")
            if text is not None:
                encoding = self.tokenizer(text, max_length=context_length, padding='max_length',
                                        truncation=True, return_tensors=return_tensors, **kwargs)

            if images is not None:
                if 'video_decode_backend' in kwargs:
                    video_decode_backend = kwargs['video_decode_backend']
                    num_frames = kwargs.get('num_frames', 8)
                    transform_function = get_video_transform(video_decode_backend, num_frames)
                else:
                    video_decode_backend = self.config.vision_config.video_decode_backend
                    transform_function = self.transform
                images = make_list_of_images(images)
                image_features = [self.image_processor_bbox(image, bbox, transform_function,
                                                    video_decode_backend=video_decode_backend,
                                                    num_frames=self.config.vision_config.num_frames) for image in images]
                # import pdb; pdb.set_trace()
                # image_features = [torch.rand(3, 8, 224, 224) for image in images]
                # image_features = torch.stack(image_features)
            if text is not None and images is not None:
                encoding["pixel_values"] = image_features
                return encoding
            elif text is not None:
                return encoding
            else:
                # import pdb; pdb.set_trace()
                return image_features[0][0], image_features[0][1]

    def preprocess(self, images, return_tensors):
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
