//extern "C" {
#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include <omp.h>

#ifndef GPU_DEVICE
#define GPU_DEVICE 2
#endif

struct twoPointers {
  double* pointers[2];
  double* rad;
  int lastused;
};


__device__
void kernelPrintMatrix(int r, int c, double* A) {
  int i, j;
  for(i = 0; i < r; i++) {
    for(j = 0; j < c; j++) {
      printf("%lf ", A[i*c + j]);
    }
    printf("\n");
  }
  printf("\n");
}

__global__
void reference(int N, struct twoPointers mydata, int nthreads, int loops) {
  int i, r, c;
  int new_val = mydata.lastused;
  int old;

  for (i = 0; i < loops; i++){
    old = mydata.lastused;
    if (new_val == 0) { mydata.lastused = 1; }
    else { mydata.lastused = 0; }
    new_val = mydata.lastused;

    for(r = 1; r < N-1; r++) {
      for(c = 1; c < N-1; c++) {
        double val;
        val = mydata.pointers[old][(r-1)*N + c];
        val += mydata.pointers[old][(r+1)*N + c];
        val += mydata.pointers[old][r*N + c-1];
        val += mydata.pointers[old][r*N + c+1];
        val += mydata.rad[r*N + c];
        val = val * 0.25;
        mydata.pointers[new_val][r*N + c] = val;
      }
    }
  }
}

__global__
void naive(int N, struct twoPointers mydata, int nthreads, int loops) {
  int x = blockIdx.x * blockDim.x + threadIdx.x; 
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= 1 && x < N-1 && y >= 1 && y < N - 1) {
    int new_val;
    int old;


    old = mydata.lastused;
    if (old == 0) { new_val = 1; }
    else { new_val = 0; }

    double val;
    int myPos = y*N + x;
    val  = mydata.pointers[old][myPos - N];
    val += mydata.pointers[old][myPos + N];
    val += mydata.pointers[old][myPos - 1];
    val += mydata.pointers[old][myPos + 1];
    val += mydata.rad[myPos];
    val = val * 0.25;
    mydata.pointers[new_val][myPos] = val;
  }
}

__global__
void duo(int N, struct twoPointers mydata, struct twoPointers mydata_gpu2, int nthreads, int loops, int yMin, int yMax, int gpuId) {
  int x = blockIdx.x * blockDim.x + threadIdx.x; 
  int y = blockIdx.y * blockDim.y + threadIdx.y + (N/2)*gpuId;
  if(x >= 1 && x < N-1 && y >= yMin && y < yMax) {
    int new_val;
    int old;
    int split = N*N/2;

    old = mydata.lastused;
    if (old == 0) { new_val = 1; }
    else { new_val = 0; }
 

    double val;
    int myPos = y*N + x;
    if(gpuId == 1 && myPos - N < split) { val = mydata_gpu2.pointers[old][myPos - N]; }
    else { val  = mydata.pointers[old][myPos - N]; }
    
    if(gpuId == 1 && myPos - 1 < split) { val += mydata_gpu2.pointers[old][myPos - 1]; }
    else { val += mydata.pointers[old][myPos - 1]; }

    if(gpuId == 0 && myPos + N >= split) { val += mydata_gpu2.pointers[old][myPos + N]; }
    else { val += mydata.pointers[old][myPos + N]; }

    if(gpuId == 0 && myPos + 1 >= split) { val += mydata_gpu2.pointers[old][myPos + 1]; }
    else { val += mydata.pointers[old][myPos + 1]; }

    val += mydata.rad[myPos];
    val = val * 0.25;
    mydata.pointers[new_val][myPos] = val;
  }
}

extern "C" {
void printMatrix(int r, int c, double* A) {
  int i, j;
  for(i = 0; i < r; i++) {
    for(j = 0; j < c; j++) {
      printf("%lf ", A[i*c + j]);
    }
    printf("\n");
  }
  printf("\n");
}

void writeMatrix(int N, double* data) {
  FILE *fpout;
  fpout = fopen("data.dat", "wb+");
    if (fpout == NULL) {
      printf("Error opening outfile!\n");
      exit(1);
  }

  int r, c;
  fprintf(fpout, "#x y z\n");
  for(r = 0; r < N; r++) {
    for(c = 0; c < N; c++) {
      double x = r * 2.0/((double) (N-1)) - 1;
      double y = c * 2.0/((double) (N-1)) - 1;
      fprintf(fpout, "%lf %lf %lf\n", x, y, data[r*N + c]);
    }
  }
  fclose(fpout);
}

int switchGpu(int gpu1, int gpu2, int currentGpu) {
  if(currentGpu == gpu1) { return gpu2; }
  return gpu1;
}

int main(int argc, char *argv[]) {
  int N = 3;
  //int i;
  int nthreads;
  int completedLoops = -1;
  int gpu1 = 0;
  int gpu2 = 1;

  cudaSetDevice(gpu1);

  #pragma omp parallel
  { if(omp_get_thread_num()==0) { nthreads = omp_get_num_threads(); } } //end parallel
  if(argc >= 2) { N = atoi(argv[1]); }
  int mat_size = N*N*sizeof(double);

  int loops = N * N * log(N);
  
  double spacing = 2.0/((double) N);
  char* funcname = "ref";
  if(argc >= 3) { funcname = argv[2]; }
  if (argc >=4){ loops =atoi(argv[3]);}

  //ALLOCATION HOST
  double *h_matrix;
  double *h_matrix2;
  double *h_radiator;

  double *d_gpu2_matrix;
  double *d_gpu2_matrix2;
  double *d_gpu2_radiator;

  if ( (h_matrix = (double*)calloc( N*N, sizeof(double) )) == NULL ) 
    { perror("main(__LINE__), allocation failed"); exit(1); }
  if ( (h_matrix2 = (double*)calloc( N*N, sizeof(double) )) == NULL ) 
    { perror("main(__LINE__), allocation failed"); exit(1); }
  if ( (h_radiator = (double*)calloc( N*N, sizeof(double) )) == NULL ) 
    { perror("main(__LINE__), allocation failed"); exit(1); }

  //ALLOCATION DEVICE
  double *d_matrix;
  double *d_matrix2;
  double *d_radiator;

  cudaMalloc ((void**) &d_matrix, mat_size); 
  if (d_matrix == NULL) { fprintf(stderr, "device memory allocation failed!\n"); return 1; }
  cudaMalloc ((void**) &d_matrix2, mat_size); 
  if (d_matrix2 == NULL) { fprintf(stderr, "device memory allocation failed!\n"); return 1; }
  cudaMalloc ((void**) &d_radiator, mat_size); 
  if (d_radiator == NULL) { fprintf(stderr, "device memory allocation failed!\n"); return 1; }

  double hotWall = 20.0;
  double coldWall = 0.0;
  //Initialization Outer
  int r,c;

  for(c = 0; c < N; c++) {
    h_matrix[0 + c]        = hotWall;
    h_matrix2[0 + c]       = hotWall;
    h_matrix[(N-1)*N  + c] = hotWall;
    h_matrix2[(N-1)*N + c] = hotWall;
  }

  for(r = 0; r < N; r++) {
    h_matrix[r*N + 0]      = coldWall;
    h_matrix2[r*N + 0]     = coldWall;
    h_matrix[r*N + (N-1)]  = hotWall;
    h_matrix2[r*N + (N-1)] = hotWall;
  }

  //Initialization Inner
  double initialGuess = 5.0; 
  for(r = 1; r < N-1; r++) {
    for(c = 1; c < N-1; c++) {
      h_matrix[r*N + c]   = initialGuess;
    }
  }

  //Initialization radiator
  for(r = 1; r < N-1; r++) {
    for(c = 1; c < N-1; c++) {
      double x = r * 2.0/((double) (N-1)) - 1;
      double y = c * 2.0/((double) (N-1)) - 1;
      if(x >= 0 && x <= 1.0/3.0 && y >= -2.0/3.0 && y <= -1.0/3.0) { h_radiator[r*N + c]= 200.0*spacing*spacing; }
      else {h_radiator[r*N + c]= 0.0;}
    }
  }

  //TRANSFER HOST->DEVICE
  cudaMemcpy (d_matrix, h_matrix, mat_size, cudaMemcpyHostToDevice);
  cudaMemcpy (d_matrix2, h_matrix2, mat_size, cudaMemcpyHostToDevice);
  cudaMemcpy (d_radiator, h_radiator, mat_size, cudaMemcpyHostToDevice);

  struct twoPointers mydata;
  mydata.pointers[0] = d_matrix;
  mydata.pointers[1] = d_matrix2;
  mydata.rad = d_radiator;
  mydata.lastused = 0;

  struct twoPointers mydata_gpu2;

  int factor = 16;
  int grid_x = (N-1)/factor + 1;
  int grid_y = (N-1)/factor + 1;
  int block_x = factor;
  int block_y = factor;
  int i;

  //TIME JACOBI
  struct timeval t1, t2;
  double elapsedTime;
  gettimeofday(&t1, NULL);
  if(strcmp(funcname, "ref") == 0) { 
    reference<<<1,1>>>(N, mydata, nthreads, loops); 
    cudaDeviceSynchronize();
  }
  else { 
    if(strcmp(funcname, "nai") == 0) { 
      dim3 dimGrid(grid_x, grid_y, 1);
      dim3 dimBlock(block_x, block_y, 1);
      for (i = 0; i < loops; i++){ 
        naive<<<dimGrid, dimBlock>>>(N, mydata, nthreads, loops);
        cudaDeviceSynchronize();
        if(mydata.lastused == 0) { mydata.lastused = 1; }
        else { mydata.lastused = 0; } 
      } 
    }
    else{
      if(strcmp(funcname, "duo") == 0) {
        cudaSetDevice(gpu2);

        cudaMalloc ((void**) &d_gpu2_matrix, mat_size); 
        if (d_gpu2_matrix == NULL) { fprintf(stderr, "device memory allocation failed!\n"); return 1; }
        cudaMalloc ((void**) &d_gpu2_matrix2, mat_size); 
        if (d_gpu2_matrix2 == NULL) { fprintf(stderr, "device memory allocation failed!\n"); return 1; }
        cudaMalloc ((void**) &d_gpu2_radiator, mat_size); 
        if (d_gpu2_radiator == NULL) { fprintf(stderr, "device memory allocation failed!\n"); return 1; }
          //TRANSFER HOST->DEVICE 2
        cudaMemcpy (d_gpu2_matrix, h_matrix, mat_size, cudaMemcpyHostToDevice);
        cudaMemcpy (d_gpu2_matrix2, h_matrix2, mat_size, cudaMemcpyHostToDevice);
        cudaMemcpy (d_gpu2_radiator, h_radiator, mat_size, cudaMemcpyHostToDevice);

        mydata_gpu2.pointers[0] = d_gpu2_matrix;
        mydata_gpu2.pointers[1] = d_gpu2_matrix2;
        mydata_gpu2.rad = d_gpu2_radiator;
        mydata_gpu2.lastused = 0;

        int grid_x = (N-1)/(factor*1) + 1;
        int grid_y = (N-1)/(factor*2) + 1;
        int block_x = factor;
        int block_y = factor;

        dim3 dimGrid(grid_x, grid_y, 1);
        dim3 dimBlock(block_x, block_y, 1);
        cudaSetDevice(gpu1);
        cudaDeviceEnablePeerAccess(gpu2, 0);
        cudaSetDevice(gpu2);
        cudaDeviceEnablePeerAccess(gpu1, 0);

        for (i = 0; i < loops; i++){ 
          cudaSetDevice(gpu1);
          duo<<<dimGrid, dimBlock>>>(N, mydata, mydata_gpu2, nthreads, loops, 1, N/2, gpu1);
          cudaSetDevice(gpu2);
          duo<<<dimGrid, dimBlock>>>(N, mydata_gpu2, mydata, nthreads, loops, N/2, N-1, gpu2);
          cudaDeviceSynchronize();
          if(mydata_gpu2.lastused == 0) { mydata_gpu2.lastused = 1; }
          else { mydata_gpu2.lastused = 0; } 

          cudaSetDevice(gpu1);
          cudaDeviceSynchronize();
          if(mydata.lastused == 0) { mydata.lastused = 1; }
          else { mydata.lastused = 0; } 
        } 
      } 
    }
  }
  //___TIME JACOBI
  gettimeofday(&t2, NULL);
  elapsedTime = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec)/1000000.0;

  cudaSetDevice(gpu1);
  cudaMemcpy (h_matrix, mydata.pointers[mydata.lastused], mat_size, cudaMemcpyDeviceToHost);
  if(strcmp(funcname, "duo") == 0) {
    cudaSetDevice(gpu2);
    cudaMemcpy (h_matrix2, mydata_gpu2.pointers[mydata_gpu2.lastused], mat_size, cudaMemcpyDeviceToHost);
    for(i = (N*N)/2; i < N*N; i++) { 
      h_matrix[i] = h_matrix2[i]; 
    }
  }
  //printMatrix(N, N, h_matrix);
  writeMatrix(N, h_matrix);

  //FREE HOST
  free(h_matrix);
  free(h_matrix2);
  free(h_radiator);

  //FREE DEVICE
  cudaFree(d_matrix);
  cudaFree(d_matrix2);
  cudaFree(d_radiator);

  if(strcmp(funcname, "duo") == 0) {
    cudaFree(d_gpu2_matrix);
    cudaFree(d_gpu2_matrix2);
    cudaFree(d_gpu2_radiator);
  }

  double size = -1;
  double jacobiMflops = -1;

  if(strcmp(funcname, "ref") == 0) {
    size = ((double) (N*N*sizeof(double) * 3.0))/1024.0;
    jacobiMflops = (5.0*(N-2)*(N-2)*loops/1000000.0) / elapsedTime;
  }

  if(strcmp(funcname, "nai") == 0) {
    size = ((double) (N*N*sizeof(double) * 3.0))/1024.0;
    jacobiMflops = (5.0*(N-2)*(N-2)*loops/1000000.0) / elapsedTime;
  }

  if(strcmp(funcname, "duo") == 0) {
    size = ((double) (N*N*sizeof(double) * 3.0))/1024.0;
    jacobiMflops = (5.0*(N-2)*(N-2)*loops/1000000.0) / elapsedTime;
  }
  printf("%lf %lf %s %lf\n", size, jacobiMflops, funcname, elapsedTime);
  return 0;
}
}
//}

