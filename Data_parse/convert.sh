#!/bin/bash

input_dir="audio"
output_dir="audio_wav"

mkdir -p "$output_dir"

for sph_file in "$input_dir"/*.sph; do
    base_name=$(basename "$sph_file" .sph)
    
    output_wav_A="$output_dir/${base_name}A.wav"
    output_wav_B="$output_dir/${base_name}B.wav"

    ffmpeg -y -i "$sph_file" -filter_complex "[0:a]channelsplit=channel_layout=stereo:channels=FL[left]" -map "[left]" -acodec pcm_s16le -ar 16000 "$output_wav_A"
    ffmpeg -y -i "$sph_file" -filter_complex "[0:a]channelsplit=channel_layout=stereo:channels=FR[right]" -map "[right]" -acodec pcm_s16le -ar 16000 "$output_wav_B"
    
    echo "✅ Converted: $sph_file --> $output_wav_A / $output_wav_B"
done

echo "🎯 All files converted!"