cd ../aea/aea_processed

for dir in */; do
    id="${dir%/}"
    echo "Processing ID: ${id}"
    mkdir -p "${id}/video_merged"

    ffmpeg \
        -i "${id}/video/${id}.mp4" \
        -i "${id}/audio/${id}.wav" \
        -c:v copy \
        -c:a aac \
        -ar 48000 \
        -ac 2 \
        -map 0:v:0 \
        -map 1:a:0 \
        "${id}/video_merged/${id}.mp4" \
        -y
done