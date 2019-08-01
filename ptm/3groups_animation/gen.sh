OUT=ptm_3groups.mp4

[ -f $OUT ] && rm $OUT 

# 1130 x 743 in
ffmpeg -r 15 -start_number 5 -i cfgB%04d.png  -filter:v "crop=1128:740:0:0" -c:v libx264 -preset slow  -profile:v high -level:v 4.0 -pix_fmt yuv420p -crf 22 $OUT

