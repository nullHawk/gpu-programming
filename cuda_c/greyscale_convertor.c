#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// stb_image is a single header library: the implementation only gets compiled
// into the translation unit that defines STB_IMAGE_IMPLEMENTATION first.
#define STB_IMAGE_IMPLEMENTATION
#include "../third_party/stb_image.h"

#define CHANNELS 3 // we force the loader to give us RGB, so 3 bytes per pixel

__global__ void colorToGrayscaleConvertion(unsigned char *Pout, unsigned char *Pin, int width, int hieght){

}

int main(){
    const char *imgPath = "../sample_img/color.jpg";
    int width, hieght, channelsInFile;

    // stbi_load decodes the jpeg and hands back one big row major array of bytes.
    // Layout is interleaved: [R0 G0 B0][R1 G1 B1]... so pixel (row, col) starts
    // at index (row * width + col) * CHANNELS.
    // The last argument forces the output to CHANNELS regardless of what the file
    // actually holds, so a greyscale or RGBA jpeg still lands as plain RGB.
    unsigned char *Pin_h = stbi_load(imgPath, &width, &hieght, &channelsInFile, CHANNELS);
    if(Pin_h == NULL){
        fprintf(stderr, "failed to load %s: %s\n", imgPath, stbi_failure_reason());
        return 1;
    }

    int pixels = width * hieght;
    printf("loaded %s -> %d x %d, %d channels in file, %d channels in array\n",
           imgPath, width, hieght, channelsInFile, CHANNELS);
    printf("colour array size: %d bytes\n", pixels * CHANNELS);

    // sanity check: print the first few pixels straight out of the array
    for(int i = 0; i < 3 && i < pixels; i++){
        unsigned char r = Pin_h[i * CHANNELS + 0];
        unsigned char g = Pin_h[i * CHANNELS + 1];
        unsigned char b = Pin_h[i * CHANNELS + 2];
        printf("pixel %d -> R:%3u G:%3u B:%3u\n", i, r, g, b);
    }

    // the greyscale output is one byte per pixel instead of three
    unsigned char *Pout_h = (unsigned char*) malloc(pixels);
    if(Pout_h == NULL){
        fprintf(stderr, "failed to allocate output buffer\n");
        stbi_image_free(Pin_h);
        return 1;
    }

    // TODO: copy Pin_h to the device, launch colorToGrayscaleConvertion,
    // and copy the result back into Pout_h.

    free(Pout_h);
    stbi_image_free(Pin_h); // must be stbi_image_free, not free, for stb allocations
    return 0;
}
