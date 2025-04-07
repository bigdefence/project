import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
import json
import os
from tqdm import tqdm

# Create output directory if it doesn't exist
output_dir = "generated_audio"
os.makedirs(output_dir, exist_ok=True)

file_path = "dataset.jsonl"

try:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                json_obj = json.loads(line)
                if 'input' in json_obj:
                    data.append(json_obj['input'])
    
    print(f"Loaded {len(data)} input texts")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)

    wav, sampling_rate = torchaudio.load("assets/exampleaudio.mp3")
    speaker = model.make_speaker_embedding(wav, sampling_rate)

    torch.manual_seed(421)

    for idx, text in enumerate(tqdm(data, desc="Generating audio")):
        try:
            cond_dict = make_cond_dict(
                text=text,  
                speaker=speaker,                 
                language="ko"                   
            )
            conditioning = model.prepare_conditioning(cond_dict)
            codes = model.generate(conditioning)
            wavs = model.autoencoder.decode(codes).cpu()
            
            output_path = os.path.join(output_dir, f"sample_{idx+1}.wav")
            torchaudio.save(output_path, wavs[0], model.autoencoder.sampling_rate)
            
        except Exception as e:
            print(f"Error processing text {idx+1}: {str(e)}")
            continue

    print(f"Audio generation completed. Files saved in '{output_dir}' directory")

except Exception as e:
    print(f"An error occurred: {str(e)}")
