import os
import glob
import requests
import tqdm

wav_files = glob.glob("YOUR_WAV_DIR")
length = len(wav_files)

# API 请求的 URL 和 Token
api_url = "https://api.deepgram.com/v1/listen?model=nova-3&smart_format=false&filler_words=true&punctuate=true&utterances=true&diarize=true&utt_split=0.8"  # 替换为实际的 API URL
token = "YOUR_TOKEN"
headers = {
    "Authorization": f"Token {token}",
    "Content-Type": "audio/wav"
}

for wav in tqdm.tqdm(wav_files):
    json_file = wav.replace(".wav", ".json")
    if os.path.exists(json_file):
        continue
    # run the following command 
    # 打开 WAV 文件并发送 POST 请求
    with open(wav, "rb") as audio_file:
        response = requests.post(api_url, headers=headers, data=audio_file)
    
    # 检查响应状态
    if response.status_code == 200:
        # 将结果保存为 JSON 文件
        with open(json_file, "w") as json_file:
            json_file.write(response.text)
        print(f"Processed {wav} and saved result to {json_file}")
    else:
        print(f"Failed to process {wav}. Status code: {response.status_code}, Response: {response.text}")

