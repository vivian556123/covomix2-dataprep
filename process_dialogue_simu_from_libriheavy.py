import librosa
import soundfile as sf
import numpy as np
import random
import os
import glob
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment
import tqdm


seed = 0
random.seed(seed)
np.random.seed(seed)

class Utterance:
    def __init__(self, filepath):
        self.parent_utt_id = os.path.basename(filepath).split('_')[0]  # Speaker ID (1 or 2)
        with open(filepath.replace('.wav', '.txt'), 'r') as f:
            lines = f.readlines()
        text_list = []
        for line in lines: 
            spk, content, start, end = line.strip().split('|')
            text_list.append(content)
        self.spk = spk
        text = ' '.join(text_list)
        self.text = text        # Transcription text
        self.filepath = filepath  # Path to audio file
        self.audio, self.sr = match_loudness(filepath)  # Load audio file
        self.duration = int(len(self.audio) / self.sr * 1000)  # Duration in miliseconds
        self.start_time = None
        self.end_time = None


def match_loudness(audio_path, target_loudness_db=-20, peak_limit_db = -1):
    y, sr = librosa.load(audio_path, sr=None)
    # Compute RMS loudness of the original audio
    rms = np.sqrt(np.mean(y**2))
    rms_db = librosa.amplitude_to_db([rms])[0]
    # Compute gain needed to reach the target loudness
    gain_db = target_loudness_db - rms_db
    y_adjusted = y * librosa.db_to_amplitude(gain_db)
    # Compute peak amplitude after gain adjustment
    peak_amplitude = np.max(np.abs(y_adjusted))
    peak_db = librosa.amplitude_to_db([peak_amplitude])[0]
    # Check if peak exceeds safe limit
    if peak_db > peak_limit_db:
        peak_reduction_db = peak_db - peak_limit_db
        y_adjusted *= librosa.db_to_amplitude(-peak_reduction_db)
    return y_adjusted, sr

def generate_dialogue_dense(utt_list, output_dir="output_dialogues", simulated_numbers = 10000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i in tqdm.tqdm(range(simulated_numbers)):
        try:
            interleaved_utterances = random.sample(utt_list, 2)
            interleaved_utterances = [Utterance(utt) for utt in interleaved_utterances]
            parent_id1 = interleaved_utterances[0].parent_utt_id
            parent_id2 = interleaved_utterances[1].parent_utt_id
            spk1_id = interleaved_utterances[0].spk
            spk2_id = interleaved_utterances[1].spk
            if parent_id1 == parent_id2 and spk1_id == spk2_id: # the chosen 2 utts are of the same speaker
                new_wav = random.choice(utt_list)
                new_utt = Utterance(new_wav)
                if new_utt.spk == spk1_id:
                    continue
                else:
                    # append new wav in the middle of the two utterances
                    interleaved_utterances_three = [interleaved_utterances[0], new_utt, interleaved_utterances[1]]
                saved_id = f"{parent_id1}_{parent_id2}_{new_utt.parent_utt_id}_{i}"
                save_dialogue_dense(interleaved_utterances_three, interleaved_utterances_three[0].sr, f"{output_dir}/{saved_id}.wav", f"{output_dir}/{saved_id}.txt")
            else:
                saved_id = f"{parent_id1}_{parent_id2}_{i}"
                save_dialogue_dense(interleaved_utterances, interleaved_utterances[0].sr, f"{output_dir}/{saved_id}.wav", f"{output_dir}/{saved_id}.txt")
        except Exception as e:
            print(f"Error: {e}")
            continue


def generate_dialogue(utt_list, output_dir="output_dialogues", simulated_numbers = 10000, overlap_prob = 0.0, silence_prob = 0.0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(f"{output_dir}_overlap"):
        os.makedirs(f"{output_dir}_overlap")
    if not os.path.exists(f"{output_dir}_silence"):
        os.makedirs(f"{output_dir}_silence")
    if not os.path.exists(f"{output_dir}_dense"):
        os.makedirs(f"{output_dir}_dense")

    for i in tqdm.tqdm(range(simulated_numbers)):
        try:
            interleaved_utterances = random.sample(utt_list, 2)
            interleaved_utterances = [Utterance(utt) for utt in interleaved_utterances]
            parent_id1 = interleaved_utterances[0].parent_utt_id
            parent_id2 = interleaved_utterances[1].parent_utt_id
            spk1_id = interleaved_utterances[0].spk
            spk2_id = interleaved_utterances[1].spk
            if parent_id1 == parent_id2 and spk1_id == spk2_id: # the chosen 2 utts are of the same speaker
                new_wav = random.choice(utt_list)
                new_utt = Utterance(new_wav)
                if new_utt.spk == spk1_id:
                    continue
                else:
                    # append new wav in the middle of the two utterances
                    interleaved_utterances_three = [interleaved_utterances[0], new_utt, interleaved_utterances[1]]
                saved_id = f"{parent_id1}_{parent_id2}_{new_utt.parent_utt_id}_{i}"
                interleaved_utterances = interleaved_utterances_three
            else:
                saved_id = f"{parent_id1}_{parent_id2}_{i}"

            random_prob = random.random()
            if random_prob < overlap_prob:
                save_dialogue_overlap(interleaved_utterances, interleaved_utterances[0].sr, f"{output_dir}_overlap/{saved_id}.wav", f"{output_dir}/{saved_id}.txt")
            elif random_prob < overlap_prob + silence_prob:
                save_dialogue_silence(interleaved_utterances, interleaved_utterances[0].sr, f"{output_dir}_silence/{saved_id}.wav", f"{output_dir}/{saved_id}.txt")
            else:
                save_dialogue_dense(interleaved_utterances, interleaved_utterances[0].sr, f"{output_dir}_dense/{saved_id}.wav", f"{output_dir}/{saved_id}.txt")
        except Exception as e:
            print(f"Error: {e}")
            continue



def save_dialogue_dense(dialogue, sample_rate, output_wav, output_txt):
    if len(dialogue) == 2:
        wav1 = dialogue[0].audio
        wav2 = dialogue[1].audio
        mixed_audio = np.concatenate((wav1, wav2), axis=0)
        sf.write(output_wav, mixed_audio, sample_rate)
        # save transcription
        transcription_lines = []
        transcription_lines.append(f"{0}|{dialogue[0].text}|{0}|{dialogue[0].duration}")
        transcription_lines.append(f"{1}|{dialogue[1].text}|{dialogue[0].duration}|{dialogue[0].duration + dialogue[1].duration}")
        with open(output_txt, "w") as f:
            f.write("\n".join(transcription_lines))
    elif len(dialogue) == 3:
        wav1 = dialogue[0].audio
        wav2 = dialogue[1].audio
        wav3 = dialogue[2].audio
        mixed_audio = np.concatenate((wav1, wav2, wav3), axis=0)
        sf.write(output_wav, mixed_audio, sample_rate)
        # save transcription
        transcription_lines = []
        transcription_lines.append(f"{0}|{dialogue[0].text}|{0}|{dialogue[0].duration}")
        transcription_lines.append(f"{1}|{dialogue[1].text}|{dialogue[0].duration}|{dialogue[0].duration + dialogue[1].duration}")
        transcription_lines.append(f"{0}|{dialogue[2].text}|{dialogue[0].duration + dialogue[1].duration}|{dialogue[0].duration + dialogue[1].duration + dialogue[2].duration}")



def save_dialogue_overlap(dialogue, sample_rate, output_wav, output_txt):
    if len(dialogue) == 2:
        wav1 = dialogue[0].audio
        wav2 = dialogue[1].audio
        overlap_samples = int(random.choice(range(1000, dialogue[0].sr))) # have a 0.1-1s overlap
        if len(wav1) < overlap_samples or len(wav2) < overlap_samples:
            return   
        # Extract the overlapping parts
        overlap_wav1 = wav1[-overlap_samples:]
        overlap_wav2 = wav2[:overlap_samples]
        # Combine the overlapping parts 
        combined_overlap = overlap_wav1 + overlap_wav2
        # Concatenate the non-overlapping parts with the combined overlap
        mixed_audio = np.concatenate((wav1[:-overlap_samples], combined_overlap, wav2[overlap_samples:]), axis=0)
        sf.write(output_wav, mixed_audio, sample_rate)
        overlap_millisecond = int(overlap_samples/dialogue[0].sr * 1000)
        # save transcription
        transcription_lines = []
        transcription_lines.append(f"{0}|{dialogue[0].text}|{0}|{dialogue[0].duration}")
        transcription_lines.append(f"{1}|{dialogue[1].text}|{dialogue[0].duration - overlap_millisecond}|{dialogue[0].duration + dialogue[1].duration - overlap_millisecond}")
        with open(output_txt, "w") as f:
            f.write("\n".join(transcription_lines))
    elif len(dialogue) == 3:
        wav1 = dialogue[0].audio
        wav2 = dialogue[1].audio
        wav3 = dialogue[2].audio

        overlap_samples = int(random.choice(range(1000, dialogue[0].sr))) # have a 0.1-1s overlap
        overlap_millisecond = int(overlap_samples/dialogue[0].sr * 1000)
        if len(wav1) < overlap_samples or len(wav2) < overlap_samples or len(wav3) < overlap_samples:
            return   
        # Extract the overlapping parts
        if random.random() < 0.5:
            overlap_wav1 = wav1[-overlap_samples:]
            overlap_wav2 = wav2[:overlap_samples]
            # Combine the overlapping parts 
            combined_overlap = overlap_wav1 + overlap_wav2
            mixed_audio = np.concatenate((wav1[:-overlap_samples], combined_overlap, wav2[overlap_samples:], wav3), axis=0)
            transcription_lines = []
            transcription_lines.append(f"{0}|{dialogue[0].text}|{0}|{dialogue[0].duration}")
            transcription_lines.append(f"{1}|{dialogue[1].text}|{dialogue[0].duration - overlap_millisecond}|{dialogue[0].duration + dialogue[1].duration - overlap_millisecond}")
            transcription_lines.append(f"{0}|{dialogue[2].text}|{dialogue[0].duration + dialogue[1].duration - overlap_millisecond}|{dialogue[0].duration + dialogue[1].duration + dialogue[2].duration - overlap_millisecond}")
        else: 
            overlap_wav2 = wav2[-overlap_samples:]
            overlap_wav3 = wav3[:overlap_samples]
            # Combine the overlapping parts 
            combined_overlap = overlap_wav2 + overlap_wav3
            mixed_audio = np.concatenate((wav1, wav2[:-overlap_samples], combined_overlap, wav3[overlap_samples:]), axis=0)
            transcription_lines = []
            transcription_lines.append(f"{0}|{dialogue[0].text}|{0}|{dialogue[0].duration}")
            transcription_lines.append(f"{1}|{dialogue[1].text}|{dialogue[0].duration}|{dialogue[0].duration + dialogue[1].duration}")
            transcription_lines.append(f"{0}|{dialogue[2].text}|{dialogue[0].duration + dialogue[1].duration - overlap_millisecond}|{dialogue[0].duration + dialogue[1].duration + dialogue[2].duration - overlap_millisecond}")
        sf.write(output_wav, mixed_audio, sample_rate)



def save_dialogue_silence(dialogue, sample_rate, output_wav, output_txt):
    if len(dialogue) == 2:
        wav1 = dialogue[0].audio
        wav2 = dialogue[1].audio
        silence_samples = int(random.choice(range(1000, dialogue[0].sr))) # have a 0.1-1s overlap 
        silence = np.zeros(silence_samples, dtype=wav1.dtype)
        mixed_audio = np.concatenate((wav1, silence, wav2), axis=0)
        sf.write(output_wav, mixed_audio, sample_rate)
        silence_millisecond = int(silence_samples/dialogue[0].sr * 1000)
        # save transcription
        transcription_lines = []
        transcription_lines.append(f"{0}|{dialogue[0].text}|{0}|{dialogue[0].duration}")
        transcription_lines.append(f"{1}|{dialogue[1].text}|{dialogue[0].duration + silence_millisecond}|{dialogue[0].duration + dialogue[1].duration+ silence_millisecond}")
        with open(output_txt, "w") as f:
            f.write("\n".join(transcription_lines))
    elif len(dialogue) == 3:
        wav1 = dialogue[0].audio
        wav2 = dialogue[1].audio
        wav3 = dialogue[2].audio
        silence_samples = int(random.choice(range(1000, dialogue[0].sr))) # have a 0.1-1s overlap
        silence_millisecond = int(silence_samples/dialogue[0].sr * 1000)
        silence = np.zeros(silence_samples, dtype=wav1.dtype)
        # Extract the overlapping parts
        if random.random() < 0.5:
            mixed_audio = np.concatenate((wav1, silence, wav2, wav3), axis=0)
            transcription_lines = []
            transcription_lines.append(f"{0}|{dialogue[0].text}|{0}|{dialogue[0].duration}")
            transcription_lines.append(f"{1}|{dialogue[1].text}|{dialogue[0].duration  + silence_millisecond}|{dialogue[0].duration + dialogue[1].duration  + silence_millisecond}")
            transcription_lines.append(f"{0}|{dialogue[2].text}|{dialogue[0].duration + dialogue[1].duration  + silence_millisecond}|{dialogue[0].duration + dialogue[1].duration + dialogue[2].duration  + silence_millisecond}")
        else: 
            mixed_audio = np.concatenate((wav1, wav2, silence, wav3), axis=0)
            transcription_lines = []
            transcription_lines.append(f"{0}|{dialogue[0].text}|{0}|{dialogue[0].duration}")
            transcription_lines.append(f"{1}|{dialogue[1].text}|{dialogue[0].duration}|{dialogue[0].duration + dialogue[1].duration}")
            transcription_lines.append(f"{0}|{dialogue[2].text}|{dialogue[0].duration + dialogue[1].duration + silence_millisecond}|{dialogue[0].duration + dialogue[1].duration + dialogue[2].duration + silence_millisecond}")
        sf.write(output_wav, mixed_audio, sample_rate)
        


monologue_path = "YOUR_MONOLOGUE_DATA_PATH"
output_dir = "YOUR_OUTPUT_DIR"
utt_list = sorted(glob.glob("{}/*.wav".format(monologue_path)))
print("utt_list numberts", len(utt_list))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# Generate Dialogue with Overlap
generate_dialogue(utt_list,  output_dir, simulated_numbers = 50000, overlap_prob=0.5, silence_prob=0.5)
