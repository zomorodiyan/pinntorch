#!/bin/bash

# Generate the palette for better color representation
ffmpeg -i fig_%03d.png -vf "palettegen" -y palette.png

# Create the GIF using the palette
ffmpeg -framerate 10 -i fig_%03d.png -i palette.png -lavfi "scale=1600:-1:flags=lanczos [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5" -y output.gif

# Clean up the palette file
rm palette.png

