import argparse
import glob
import json
import os
import re
from math import ceil, floor

import numpy as np
import scipy.io.wavfile as wavfile
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Podcast Data Processing")
parser.add_argument(
    "--audio_root", default=None, type=str, required=True, help="The path to the root of the audio (wav) data folder.",
)
parser.add_argument(
    "--transcript_root",
    default=None,
    type=str,
    required=True,
    help="The path to the root of the transcript data folder.",
)
parser.add_argument(
    "--dest_root", default=None, type=str, required=True, help="Path to the destination root directory.",
)

# Optional arguments
parser.add_argument(
    "--min_slice_duration", default=10.0, type=float, help="Minimum audio slice duration after processing.",
)

parser.add_argument(
    "--max_conversation_duration", default=60.0, type=float, help="maximum audio slice duration after processing.",
)

parser.add_argument(
    "--adjust_boundary", default=0, type=float, help="maximum adjust boundary for the segment (second)",
)

parser.add_argument(
    "--keep_low_conf", action="store_true", help="Keep all utterances with low confidence transcripts",
)
parser.add_argument(
    "--remove_noises", action="store_true", help="Removes transcripted noises such as [laughter].",
)
parser.add_argument(
    "--noises_to_emoji", action="store_true", help="Converts transcripts for noises to an emoji character.",
)
args = parser.parse_args()

# Total number of files before segmenting, and train/val/test splits
NUM_FILES = 5850 + 5849
TRAIN_END_IDX = int(NUM_FILES )
VAL_END_IDX = int(NUM_FILES )



def __write_sample(dest, file_id, count, file_count, sample_rate, audio, transcript, first_spk = None):
    """
    Writes one slice to the given target directory.
    Args:
        dest: the destination directory
        file_id: name of the transcript/audio file for this block
        count: the count of segments in the file so far
        file_count: the total number of filse processed so far
        sample rate: sample rate of the audio data
        audio: audio data of the current sample
        duration: audio duration of the current sample
        transcript: transcript of the current sample
    """
    partition = __partition_name(file_count)
    audio_path = os.path.join(dest, partition, f"{file_id}_{count:03}.wav")

    # Write audio
    wavfile.write(audio_path, sample_rate, audio)

    with open(audio_path.replace(".wav",".txt"), 'w') as f:
        f.write(transcript)


def create_overlap_content(combined_data, sep_token = '[spkchange]'):
    final_sequence = []
    last_speaker = None
    for _, _, speaker, transcript in combined_data:
        if speaker != last_speaker and last_speaker is not None:
            final_sequence.append("[spkchange]")
        final_sequence.append(transcript)
        last_speaker = speaker
    txt_sequence = ' '.join(final_sequence) 
    return txt_sequence

def prepare_generated_conversation_transcription(combined_data):
    final_sequence = []
    for t_start, t_end, speaker, transcript in combined_data:
        final_sequence.append(str(speaker)+'|'+transcript+'|'+str(t_start)+'|'+str(t_end))
    txt_sequence = '\n'.join(final_sequence) 
    return txt_sequence


def compute_frame_energy(wave, frame_size, hop_size):
    return np.array([np.sum(wave[i:i+frame_size]**2) for i in range(0, len(wave) - frame_size + 1, hop_size)])

def find_lowest_energy_index(energy, start_idx, end_idx):
    return np.argmin(energy[start_idx:end_idx]) + start_idx

def adjust_transcription_boundaries(wave, sr, transcript_buffer, final_t_start, final_t_end, search_range=0.5, frame_duration=0.001):
    # Convert times to sample indices
    start_time = final_t_start / 1000 # second
    end_time = final_t_end / 1000 # second
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    
    # Frame settings
    frame_size = int(frame_duration * sr) 
    hop_size = frame_size // 2  # 50% overlap

    # Extract segments for search
    start_segment_start = max(0, start_sample - int(search_range * sr))
    start_segment_end = min(start_sample + int(search_range * sr), len(wave))
    end_segment_start = max(0, end_sample - int(search_range * sr))
    end_segment_end = min(end_sample + int(search_range * sr), len(wave))
    # Compute energy for the segments
    start_segment = wave[start_segment_start:start_segment_end]
    end_segment = wave[end_segment_start:end_segment_end]
    
    start_energy = compute_frame_energy(start_segment, frame_size, hop_size)
    end_energy = compute_frame_energy(end_segment, frame_size, hop_size)
    
    # Find the lowest energy frames
    start_lowest_frame = find_lowest_energy_index(start_energy, 0, len(start_energy))
    end_lowest_frame = find_lowest_energy_index(end_energy, 0, len(end_energy))
    # Convert frame indices back to sample indices
    adjusted_start = start_segment_start + start_lowest_frame * hop_size
    adjusted_end = end_segment_start + end_lowest_frame * hop_size + frame_size

    transcript_buffer.sort(key=lambda x: x[0])
    transcript_buffer[0] = (int(adjusted_start / sr * 1000),) + transcript_buffer[0][1:]
    transcript_buffer.sort(key=lambda x: x[1])
    transcript_buffer[-1] = transcript_buffer[-1][:1] + (int(adjusted_end / sr * 1000),) + transcript_buffer[-1][2:]
    # transcript_buffer[-1][1] = int(adjusted_end / sr * 1000) # ms
    # print(f"Adjusted start: {adjusted_start}, Adjusted end: {adjusted_end}", "original start: ", start_sample, "original end: ", end_sample)
    return transcript_buffer

def check_transcription_error(phrases):
    words = phrases.get("words",[])
    min_start = None
    max_end = None
    spk_set = set()
    sentence_spk = phrases.get('speaker', 0)
    for idx, phrase in enumerate(words, start=1):
        t_start = float(phrase.get('start', 0))
        t_end = float(phrase.get('end', 0))
        spk = phrase.get('speaker', 0)
        if min_start is None:
            min_start = t_start
            max_end = t_end
        min_start = min(min_start, t_start)
        max_end = max(max_end, t_end)
        spk_set.add(spk)
    sentence_start = float(phrases.get('start', 0))
    sentence_end = float(phrases.get('end', 0))
    error_start = abs(sentence_start - min_start)
    error_end = abs(sentence_end - max_end)
    if len(spk_set) > 1 :
        print(f"SPLIT PHRASE: Multiple speakers in {sentence_start} {sentence_end},  spk_set: {spk_set}, sentence_spk: {sentence_spk}")
        return 1 # The phrases need to be splited
    if error_start > 0.1 or error_end > 0.1:
        print(f"CLEAR BUFFER: Transcription error: {error_start} {error_end}, min_start: {min_start}, max_end: {max_end}, sentence_start: {sentence_start}, sentence_end: {sentence_end}")
        return 2 # There is error in the transcription
    else: 
        return 0


def split_phrases_for_different_speakers(phrases, trans_path): # if there is multiple speakers in a phrase, we split speaker'sentence to becomes a new phrase. 
    words = phrases.get("words",[])
    phrases_list = []
    phrase_buffer = []
    last_spk = None
    for idx, phrase in enumerate(words, start=1):
        spk = phrase.get('speaker', 0)
        content = phrase.get('word', '')
        start = float(phrase.get('start', 0))
        end = float(phrase.get('end', 0))
        if last_spk is None:
            last_spk = spk
        if spk == last_spk:
            phrase_buffer.append((start, end, spk, content))
        else:
            # min_start in the phrase_buffer
            min_start = phrase_buffer[0][0]
            max_end = phrase_buffer[-1][1]
            combined_content = ' '.join([x[3] for x in phrase_buffer])
            phrases_list.append((min_start, max_end, last_spk, combined_content))
            # renew buffer 
            phrase_buffer = [(start, end, spk, content)]
            last_spk = spk
    if len(phrase_buffer) > 0:
        min_start = phrase_buffer[0][0]
        max_end = phrase_buffer[-1][1]
        combined_content = ' '.join([x[3] for x in phrase_buffer])
        phrases_list.append((min_start, max_end, last_spk, combined_content))
    # print("phrases_list",phrases_list, "original phrase", phrases, "trans_path", trans_path)

    return phrases_list



def __process_one_file(
    trans_path,
    sample_rate,
    audio_data,
    file_id,
    dst_root,
    min_slice_duration,
    file_count,
    keep_low_conf,
    rem_noises,
    emojify,
    max_conversation_duration = 60,
    minimum_speech_turns = 2,
    maximum_speaker = 2, 
    minimum_speaker = 2, 
    adjust_boundary = 0,
):
    """
    Creates one block of audio slices and their corresponding transcripts.
    Args:
        trans_path: filepath to transcript
        sample_rate: sample rate of the audio
        audio_data: numpy array of shape [samples, channels]
        file_id: identifying label, e.g. 'fe_03_01102'
        dst_root: path to destination directory
        min_slice_duration: min number of seconds for an audio slice
        file_count: total number of files processed so far
        keep_low_conf: keep utterances with low-confidence transcripts
        rem_noises: remove noise symbols
        emojify: convert noise symbols into emoji characters
    """
    count = 0

    with open(trans_path, 'r') as fin:
        data = json.load(fin)

        transcript_buffer = []
        spk_set = set()
        audio_buffers = [[], []]
        buffer_durations = [0.0]

        
        # Extract and format the relevant fields
        phrases = data.get("results",{}).get('utterances',[])
        for idx, phrase in enumerate(phrases, start=1):
            # Check for transcription errors
            check_type = check_transcription_error(phrase)
            if check_type == 2:
                # print(f"CLEAR BUFFER: Transcription error in {file_id} at {t_start}-{t_end} phrase: {phrase}")
                # clear buffer and continue
                transcript_buffer = []
                spk_set = set()
                continue    
            elif check_type == 1:
                print("phrases need to be splited")
                current_utt_list = split_phrases_for_different_speakers(phrase, trans_path)
            else: 
                pass           
                t_start = float(phrase.get('start', 0))
                t_end = float(phrase.get('end', 0))
                content = phrase.get('transcript', '')
                spk_id = phrase.get('speaker', 0)
                current_utt_list = [(t_start, t_end, spk_id, content)]

            for current_utt in current_utt_list:
                #print("current_utt", current_utt)
                t_start, t_end, spk_id, content  = current_utt
                t_start = int(t_start * 1000) # ms
                t_end = int(t_end * 1000) # ms
                if content is None or not content:
                    #print(f"Empty content{content}")
                    continue
                # We save previous N utterances when we are processing the N+1 utterances. 
                if transcript_buffer != []:
                    transcript_buffer.sort(key=lambda x: x[0])
                    final_t_start = transcript_buffer[0][0]
                    first_spk_index = transcript_buffer[0][2]
                    transcript_buffer.sort(key=lambda x: x[1])
                    final_t_end = transcript_buffer[-1][1]      

                    if t_start < final_t_end: # The next utterance is overlapped with our current buffer
                        transcript_buffer.append((t_start, t_end, spk_id, content))
                        spk_set.add(spk_id)
                    elif (final_t_end - final_t_start) > max_conversation_duration:
                        if len(transcript_buffer) > 1 and len(spk_set) <= 1:  
                            # There are multiple sentences of only one speaker, we only preserve the last sentence in this buffer
                            transcript_buffer = transcript_buffer[-1:]
                            transcript_buffer.append((t_start, t_end, spk_id, content))
                            spk_set.add(spk_id)
                        else: 
                            # Clear buffers, do not save this wav because it is so long
                            transcript_buffer = []
                            spk_set = set()
                            transcript_buffer.append((t_start, t_end, spk_id, content))
                            spk_set.add(spk_id)

                            # print("The conversation is too long", f"transcript_buffer: {transcript_buffer} Final t_start: {final_t_start}, Final t_end: {final_t_end}")
                    
                    elif len(spk_set) < minimum_speaker: # if there are less than minimum_speakers in this buffer
                        transcript_buffer.append((t_start, t_end, spk_id, content))
                        spk_set.add(spk_id)

                    elif (final_t_end - final_t_start) < min_slice_duration: 
                        # The current buffer is too short
                        transcript_buffer.append((t_start, t_end, spk_id, content))
                        spk_set.add(spk_id)

                    elif len(spk_set) > maximum_speaker: # The next utterance has more than max speakers
                        # Clear buffers
                        transcript_buffer = []
                        spk_set = set()
                        transcript_buffer.append((t_start, t_end, spk_id, content))
                        spk_set.add(spk_id)

                    else: # The next utterance is not overlapped with our current buffer and there are more than min speakers in this buffer
                        # Save wavs
                        if adjust_boundary > 0:
                            # print(f"Adjusting the boundaries for {file_id}")
                            transcript_buffer = adjust_transcription_boundaries(wave=audio_data, sr=sample_rate, transcript_buffer=transcript_buffer, 
                                                                                final_t_start=final_t_start, final_t_end=final_t_end, 
                                                                                search_range=adjust_boundary, frame_duration=0.01)
                        content_to_write =  prepare_generated_conversation_transcription(transcript_buffer)
                        audio_to_write = audio_data[floor(final_t_start * sample_rate * 0.001) : ceil(final_t_end * sample_rate * 0.001)] # because the timestep is ms
                        # Write out segment and transcript
                        count += 1
                        __write_sample(
                            dst_root,
                            file_id,
                            count,
                            file_count,
                            sample_rate,
                            audio_to_write,
                            content_to_write,
                            first_spk_index,
                        )
                        
                        # Clear buffers
                        transcript_buffer = []
                        spk_set = set()
                        transcript_buffer.append((t_start, t_end, spk_id, content))
                        spk_set.add(spk_id)
                        
                # Note: We drop any shorter "scraps" at the end of the file, if
                #   they end up shorter than min_slice_duration.
                # Append utterance to buffer
                else:
                    transcript_buffer.append((t_start, t_end, spk_id, content))
                    spk_set.add(spk_id)

def __partition_name(file_count):
    return "train"


def __process_data(
    audio_root, transcript_root, dst_root, min_slice_duration, file_count, keep_low_conf, rem_noises, 
    emojify, max_conversation_duration, adjust_boundary,
):
    """
    Converts Fisher wav files to numpy arrays, segments audio and transcripts.
    Args:
        audio_root: source directory with the wav files
        transcript_root: source directory with the transcript files
            (can be the same as audio_root)
        dst_root: where the processed and segmented files will be stored
        min_slice_duration: minimum number of seconds for a slice of output
        file_count: total number of files processed so far
        keep_low_conf: whether or not to keep low confidence transcriptions
        rem_noises: whether to remove noise symbols
        emojify: whether to convert noise symbols to emoji, lower precedence
    Assumes:
        1. There is exactly one transcripts directory in data_folder
        2. Audio files are all: <audio_root>/audio-wav/fe_03_xxxxx.wav
    """
    transcript_list = glob.glob(os.path.join(transcript_root, "*.json"))
    transcript_list = sorted(transcript_list)
    print("Found {} transcripts.".format(len(transcript_list)))

    count = file_count

    # Grab audio file associated with each transcript, and slice
    for trans_path in tqdm(transcript_list, desc="Matching and segmenting"):
        file_id, _ = os.path.splitext(os.path.basename(trans_path))
        audio_path = os.path.join(audio_root, file_id + ".wav")

        sample_rate, audio_data = wavfile.read(audio_path)

        # Create a set of segments (a block) for each file
        try:
            __process_one_file(
                trans_path,
                sample_rate,
                audio_data,
                file_id,
                dst_root,
                min_slice_duration * 1000, # ms
                count,
                keep_low_conf,
                rem_noises,
                emojify,
                max_conversation_duration * 1000,
                adjust_boundary = adjust_boundary, # second
            )
            count += 1
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

    return count


def main():
    # Arguments to the script
    audio_root = args.audio_root
    transcript_root = args.transcript_root
    dest_root = args.dest_root

    min_slice_duration = args.min_slice_duration
    keep_low_conf = args.keep_low_conf
    rem_noises = args.remove_noises
    emojify = args.noises_to_emoji

    adjust_boundary = args.adjust_boundary
    print("adjust_boundary", adjust_boundary)

    max_conversation_duration = args.max_conversation_duration

    print(f"Expected number of files to segment: {NUM_FILES}")
    print("With a 100/0/0 split:")
    print(f"Number of training files: {TRAIN_END_IDX}")
    print(f"Number of validation files: {VAL_END_IDX - TRAIN_END_IDX}")
    print(f"Number of test files: {NUM_FILES - VAL_END_IDX}")

    if not os.path.exists(os.path.join(dest_root, 'train/')):
        os.makedirs(os.path.join(dest_root, 'train/'))
        os.makedirs(os.path.join(dest_root, 'val/'))
        os.makedirs(os.path.join(dest_root, 'test/'))
    else:
        # Wipe manifest contents first
        open(os.path.join(dest_root, "manifest_train.json"), 'w').close()
        open(os.path.join(dest_root, "manifest_val.json"), 'w').close()
        open(os.path.join(dest_root, "manifest_test.json"), 'w').close()

    file_count = 0

    file_count = __process_data(
        audio_root,
        transcript_root,
        dest_root,
        min_slice_duration,
        file_count,
        keep_low_conf,
        rem_noises,
        emojify,
        max_conversation_duration = max_conversation_duration,
        adjust_boundary = adjust_boundary
    )

    print(f"Total file count so far: {file_count}")


if __name__ == "__main__":
    main()
