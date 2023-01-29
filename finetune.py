import torch
import torch.nn as nn
import torch.optim as optim
import clip
import json
import argparse
from os.path import exists, basename, join, dirname
from os import mkdir
from typing import Callable, Dict, List
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator

class ImageTitleDataset(Dataset):
    def __init__(self, tsv_index_filename: str, preprocess: Callable[[Image.Image], torch.Tensor]):
        """
        Initializes a dataset for CLIP fine tuning.
        The core information stores in the file tsv_index_filename. It's a tsv file with two columns. The first column is the path to the images (relative to the tsv file).
        And the second column is the tags separated by commas. Natural language will also work.
        """
        self.preprocess = preprocess
        self.rootdir = dirname(tsv_index_filename)
        lines = [x.strip().split('\t') for x in open(tsv_index_filename)]
        self.image_paths = [join(self.rootdir, x[0]) for x in lines]
        texts = ['a photo of ' + ' '.join(x[1].split(',')) for x in lines]
        self.titles = clip.tokenize(texts, truncate=True)

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        image = self.preprocess(Image.open(self.image_paths[idx])) # Image from PIL module
        title = self.titles[idx]
        return image, title

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

def fine_tune(config: Dict, accelerator: Accelerator):
    if not exists(config['output_dir']):
        mkdir(config['output_dir'])
    if accelerator.is_main_process:
        open(join(config['output_dir'], 'hyp.json'), 'w').write(json.dumps(config, indent=4))
    model, preprocess = clip.load(config['clip_model'], jit=False) # Must set jit=False for training
    model, preprocess = accelerator.prepare(model, preprocess)
    model.train()
    clip.model.convert_weights(model) # to fp16

    dataset = ImageTitleDataset('TouhouCLIP/tags.tsv', preprocess)
    train_dataloader = DataLoader(dataset, batch_size = config['batch_size'], num_workers=8)

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
    optimizer, train_dataloader = accelerator.prepare(optimizer, train_dataloader)

    # add your own code to track the training progress.
    progress_bar = tqdm(total=config['epochs'] * len(train_dataloader), disable=not accelerator.is_main_process)
    for epoch in range(config['epochs']):
        for images, texts in train_dataloader :
            optimizer.zero_grad()
            logits_per_image, logits_per_text = model(images, texts)
            ground_truth = torch.arange(len(images), dtype=torch.long, device=accelerator.device)
            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            accelerator.backward(total_loss)
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)
            progress_bar.update(1)
            progress_bar.set_postfix(loss=total_loss.item(), epoch=epoch)

        if accelerator.is_main_process and epoch % config['save_epoch'] == 0:
            accelerator.save_state(join(config['output_dir'], f"epoch_{epoch}"))
    
    if accelerator.is_main_process:
        accelerator.save_state(join(config['output_dir'], f"epoch_{epoch}"))
    return accelerator.unwrap_model(model)

def predict(config: Dict,
    accelerator: Accelerator,
    modelfn: str,
    imgfn: str,
    tags: List[str]):
    """
    An example showing how to load the model, extract features, and perform NN search.
    """
    device = accelerator.device
    model, preprocess = clip.load(config['clip_model'], jit=False)
    model.load_state_dict(torch.load(modelfn))
    model.eval()
    model = model.to(device)
    image = preprocess(Image.open(imgfn))
    image = image.to(device).unsqueeze(dim=0)
    tags = ['a photo of ' + x for x in tags]
    texts = clip.tokenize(tags, truncate=True)
    texts = texts.to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(texts)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)
    for value, index in zip(values, indices):
        tag = tags[index]
        print(f'{tag}: {value}')

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser(description='A command line utility to fine tune CLIP models.')
    parser.add_argument('--batch_size', help='Batch size of the training job, before gradient accumulation.', default=128, type=int)
    parser.add_argument('--epochs', help='Max epochs to do fine tuning.', default=50, type=int)
    parser.add_argument('--save_epoch', help='How many epochs between saving checkpoints.', default=10, type=int)
    parser.add_argument('--lr', help='The learning rate', default=2e-6)
    parser.add_argument('--clip_model', help='The clip variant to be fine tuned.', default='ViT-B/32')
    parser.add_argument('--output_dir', help='The directory holding the result models', default='model')
    args = parser.parse_args()
    accelerator = Accelerator()
    model = fine_tune(args.__dict__, accelerator)
    predict(args.__dict__,
        accelerator,
        f'./{args.output_dir}/epoch_{args.epochs - 1}/pytorch_model.bin',
        '../Dreambooth-Anything/data/TouhouNSFW/98145/98145.jpg',
        [x.strip() for x in open('TouhouCLIP/allTags.tsv')])