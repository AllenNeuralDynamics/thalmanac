mkdir -p /data/abc_atlas

rclone mount :s3:allen-brain-cell-atlas /data/abc_atlas \
    --allow-non-empty \
    --allow-other \
    --s3-provider AWS \
    --s3-region us-west-2 \
    --read-only \
    --vfs-cache-mode full \
    --vfs-cache-max-size 100g \
    --vfs-read-ahead 64m \
    --buffer-size 32m \
    --dir-cache-time 1h \
    --no-checksum \
    --no-modtime \
    --poll-interval 0 \
    --checkers 36