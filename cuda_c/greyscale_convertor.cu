#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// stb_image is a single header library: the implementation only gets compiled
// into the translation unit that defines STB_IMAGE_IMPLEMENTATION first.
#define STB_IMAGE_IMPLEMENTATION
#include "../third_party/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../third_party/stb_image_write.h"

#define CHANNELS 3 // we force the loader to give us RGB, so 3 bytes per pixel

__global__ void colorToGrayscaleKernel(unsigned char *Pout, unsigned char *Pin, int width, int hieght){
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;

    if (col < width && row < hieght){
        int grayOffset = row*width + col;

        int rgbOffset= grayOffset*CHANNELS;

        unsigned char r = Pin[rgbOffset ];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];

        Pout[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
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

    unsigned char *Pin_d, *Pout_d;
    size_t input_size = width * hieght * CHANNELS * sizeof(unsigned char); // we are serializing size of 3-d array because internally compiler flattens 3-d array and then store in the memory
    size_t output_size = width * hieght * sizeof(unsigned char);

    cudaMalloc((void**) &Pin_d, input_size);
    cudaMalloc((void**) &Pout_d, output_size);

    cudaMemcpy(Pin_d, Pin_h, input_size, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid(
        (width + dimBlock.x - 1) /  dimBlock.x,
        (hieght + dimBlock.y - 1) / dimBlock.y
    );

    // the kernel takes Pout first, then Pin
    colorToGrayscaleKernel<<<dimGrid, dimBlock>>>(Pout_d, Pin_d, width, hieght);

    // the launch itself is async and never returns an error code, so ask for it
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // cudaMemcpy blocks until the kernel is done, so no explicit sync is needed
    cudaMemcpy(Pout_h, Pout_d, output_size, cudaMemcpyDeviceToHost);

    cudaFree(Pin_d);
    cudaFree(Pout_d);

    // write the result back out. png is lossless, so what we compare on disk is
    // exactly what the kernel produced. 1 channel, and the stride between rows
    // is just width because there is one byte per pixel.
    const char *outPath = "../sample_img/grey.png";
    if(stbi_write_png(outPath, width, hieght, 1, Pout_h, width) == 0){
        fprintf(stderr, "failed to write %s\n", outPath);
        free(Pout_h);
        stbi_image_free(Pin_h);
        return 1;
    }
    printf("wrote %s -> %d x %d, 1 channel\n", outPath, width, hieght);
    printf("grey pixel 0 -> %u (from R:%u G:%u B:%u)\n",
           Pout_h[0], Pin_h[0], Pin_h[1], Pin_h[2]);

    free(Pout_h);
    stbi_image_free(Pin_h); // must be stbi_image_free, not free, for stb allocations
    return 0;
}
