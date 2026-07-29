// this is non-tiled version of img blur
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
#define BLUR_SIZE 3

__global__ void imgblr_kernel(unsigned char *Pout, unsigned char *Pin, int width, int hieght){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= width || row >= hieght)
        return;

    // the image is interleaved RGB, so a pixel is CHANNELS bytes wide and each
    // channel has to be averaged against its own neighbours
    for(int c = 0; c < CHANNELS; ++c){
        int pixVal = 0;
        int pixels = 0;

        for(int blurRow=-BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow){
            for(int blurCol=-BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol){
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                // pixels near the border have a smaller window, which is why we
                // count them instead of dividing by the fixed patch size
                if (curRow>=0 && curRow<hieght && curCol>=0 && curCol<width){
                    pixVal += Pin[(curRow*width + curCol)*CHANNELS + c];
                    ++pixels;
                }
            }
        }
        // the write belongs outside both loops, once the sum is complete
        Pout[(row*width + col)*CHANNELS + c] = (unsigned char) (pixVal/pixels);
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

    // the blur keeps all three channels, so the output is the same size as the input
    unsigned char *Pout_h = (unsigned char*) malloc(pixels * CHANNELS);
    if(Pout_h == NULL){
        fprintf(stderr, "failed to allocate output buffer\n");
        stbi_image_free(Pin_h);
        return 1;
    }

    unsigned char *Pin_d, *Pout_d;
    size_t size = width * hieght * CHANNELS * sizeof(unsigned char); // we are serializing size of 3-d array because internally compiler flattens 3-d array and then store in the memory

    cudaMalloc((void**) &Pin_d, size);
    cudaMalloc((void**) &Pout_d, size);

    cudaMemcpy(Pin_d, Pin_h, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid(
        (width + dimBlock.x - 1) /  dimBlock.x,
        (hieght + dimBlock.y - 1) / dimBlock.y
    );


    imgblr_kernel<<<dimGrid, dimBlock>>>(Pout_d, Pin_d, width, hieght);

    // the launch itself is async and never returns an error code, so ask for it
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(err));
        free(Pout_h);
        stbi_image_free(Pin_h);
        return 1;
    }

    // cudaMemcpy blocks until the kernel is done, so no explicit sync is needed
    cudaMemcpy(Pout_h, Pout_d, size, cudaMemcpyDeviceToHost);

    cudaFree(Pin_d);
    cudaFree(Pout_d);

    // write the result back out. png is lossless, so what we compare on disk is
    // exactly what the kernel produced. CHANNELS bytes per pixel, so the stride
    // between rows is width * CHANNELS.
    const char *outPath = "../output/blur.png";
    if(stbi_write_png(outPath, width, hieght, CHANNELS, Pout_h, width * CHANNELS) == 0){
        fprintf(stderr, "failed to write %s\n", outPath);
        free(Pout_h);
        stbi_image_free(Pin_h);
        return 1;
    }
    printf("wrote %s -> %d x %d, %d channels\n", outPath, width, hieght, CHANNELS);
    printf("blur pixel 0 -> R:%u G:%u B:%u (was R:%u G:%u B:%u)\n",
           Pout_h[0], Pout_h[1], Pout_h[2], Pin_h[0], Pin_h[1], Pin_h[2]);

    free(Pout_h);
    stbi_image_free(Pin_h); // must be stbi_image_free, not free, for stb allocations
    return 0;
}
