// function for vector addition on host machine.
void tradVecAdd(float* A_h, float* B_h, float* C_h, int n){
    for(int i = 0; i < n; i++){
        C_h[i] = A_h[i] + B_h[i];
    }
}

void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    cudaMalloc((void**) &A_d, size);
    cudaMalloc((void**) &B_d, size);
    cudaMalloc((void**) &C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // Vector Addition kernal launch
    vecAddKernel<<ceil(n/256.0), 256>>(A_d, B_d, C_d, n); // we use ceil here to pad extra threads if there is remainder.

    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

__global__
void vecAddKernel(float* A, float* B, float* C, int n){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < n) { // skips computaton of extra threads added by ceil command
        C[i] = A[i] + B[i];
    }
}

int main(){
    vecAdd(A, B, C, N);
}