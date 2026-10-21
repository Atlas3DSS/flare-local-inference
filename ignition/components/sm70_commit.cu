// Sol-12 direct recurrent-state copy. The reviewed commit_vector12 body is unchanged.
#include <cuda_runtime.h>
#include <stdint.h>
#include <limits.h>

struct Sol12CommitLayer {
  uint32_t *state;
  int64_t row_stride;
  int width, slots;
};
static_assert(sizeof(Sol12CommitLayer) == 24 && alignof(Sol12CommitLayer) == 8,
              "SM70 commit descriptor ABI must remain 24 bytes, aligned to 8");

template<int R> __global__ void commit_vector12(const Sol12CommitLayer *layers,
    const int64_t *src,const int64_t *dst,const uint8_t *active,int requests) {
  Sol12CommitLayer d=layers[blockIdx.y];
  int c=(blockIdx.x*blockDim.x+threadIdx.x)*4;
  uint4 values[R];
  bool live[R];int64_t destinations[R];
  #pragma unroll
  for(int r=0;r<R;r++) {
    live[r]=r<requests && active[r] && dst[r]>=0 && dst[r]<d.slots;
    destinations[r]=r<requests?dst[r]:0;
    if(live[r] && c<d.width) {
      if(((uintptr_t)d.state%16)==0 && d.row_stride%4==0 && d.width%4==0)
        values[r]=*reinterpret_cast<const uint4*>(d.state+src[r]*d.row_stride+c);
      else {
        uint32_t *v=reinterpret_cast<uint32_t*>(&values[r]);
        #pragma unroll
        for(int j=0;j<4;j++)v[j]=c+j<d.width?d.state[src[r]*d.row_stride+c+j]:0;
      }
    }
  }
  asm volatile("" ::: "memory");
  #pragma unroll
  for(int r=0;r<R;r++)if(live[r] && c<d.width) {
    if(((uintptr_t)d.state%16)==0 && d.row_stride%4==0 && d.width%4==0)
      *reinterpret_cast<uint4*>(d.state+destinations[r]*d.row_stride+c)=values[r];
    else {
      uint32_t *v=reinterpret_cast<uint32_t*>(&values[r]);
      #pragma unroll
      for(int j=0;j<4;j++)if(c+j<d.width)d.state[destinations[r]*d.row_stride+c+j]=v[j];
    }
  }
}
extern "C" int sol12_commit(const Sol12CommitLayer*d,int layers,int width,const int64_t*s,const int64_t*t,const uint8_t*a,int r,void*stream) {
  if(!d||!s||!t||!a||layers<1||layers>65535||width<1||width>INT_MAX-1023||r<1||r>16)return cudaErrorInvalidValue;
  if(r<=4)commit_vector12<4><<<dim3((width+1023)/1024,layers),256,0,(cudaStream_t)stream>>>(d,s,t,a,r);
  else commit_vector12<16><<<dim3((width+1023)/1024,layers),256,0,(cudaStream_t)stream>>>(d,s,t,a,r);
  return cudaGetLastError();
}
