#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <vector>
using namespace std;
__constant__ int cb[576];
struct Task{int a,b,n,m;size_t off;};
__global__ void pairKernel(const Task*tasks,const int*so,const int*letters,float*out,float*M,float*X,float*Y,float go,float ge){
 int tid=threadIdx.x,tno=blockDim.x;Task t=tasks[blockIdx.x];int n=t.n,m=t.m,W=m+1;size_t o=t.off;
 float*mm=M+o,*x=X+o,*y=Y+o;const int*a=letters+so[t.a],*b=letters+so[t.b];
 const float neg=-1e20f;
 for(int diag=0;diag<=n+m;diag++){
  int lo=max(0,diag-m),hi=min(n,diag);
  for(int i=lo+tid;i<=hi;i+=tno){int j=diag-i;size_t z=(size_t)i*W+j;
   if(i==0&&j==0){mm[z]=0;x[z]=y[z]=neg;continue;}
   if(j==0){mm[z]=y[z]=neg;x[z]=0;continue;}
   if(i==0){mm[z]=x[z]=neg;y[z]=0;continue;}
   size_t d=z-W-1,u=z-W,l=z-1;float v=max(mm[d],max(x[d],y[d]))+cb[a[i-1]*24+b[j-1]];mm[z]=v;
   x[z]=max(mm[u]-go,max(x[u]-ge,y[u]-go));y[z]=max(mm[l]-go,max(y[l]-ge,x[l]-go));
  }
  __syncthreads();
 }
 float best=neg;
 for(int j=tid;j<=m;j+=tno){size_t z=(size_t)n*W+j;best=max(best,max(mm[z],max(x[z],y[z])));}
 for(int i=tid;i<n;i+=tno){size_t z=(size_t)i*W+m;best=max(best,max(mm[z],max(x[z],y[z])));}
 __shared__ float red[256];red[tid]=best;__syncthreads();
 for(int s=tno/2;s;s>>=1){if(tid<s)red[tid]=max(red[tid],red[tid+s]);__syncthreads();}
 if(tid==0)out[blockIdx.x]=red[0];
}
extern "C" int gpu_pair_scores(const int*letters,const int*so,int ns,const int*bmat,float*out){
 if(ns<2||cudaSetDevice(0)!=cudaSuccess)return 0;
 vector<Task>all;all.reserve(ns*(ns-1)/2);
 for(int i=0;i<ns;i++)for(int j=0;j<i;j++)all.push_back({i,j,so[i+1]-so[i],so[j+1]-so[j],0});
 const size_t maxCells=60000000;size_t begin=0,totalLen=so[ns];
 int*dl=nullptr,*dso=nullptr;float*dM=nullptr,*dX=nullptr,*dY=nullptr,*dout=nullptr;Task*dt=nullptr;vector<float>hs;
 auto cleanup=[&](){cudaFree(dl);cudaFree(dso);cudaFree(dM);cudaFree(dX);cudaFree(dY);cudaFree(dout);cudaFree(dt);};
 bool ok=cudaMalloc(&dl,totalLen*sizeof(int))==cudaSuccess&&cudaMalloc(&dso,(ns+1)*sizeof(int))==cudaSuccess;
 if(!ok){cleanup();return 0;}
 if(cudaMemcpy(dl,letters,totalLen*sizeof(int),cudaMemcpyHostToDevice)!=cudaSuccess||cudaMemcpy(dso,so,(ns+1)*sizeof(int),cudaMemcpyHostToDevice)!=cudaSuccess||cudaMemcpyToSymbol(cb,bmat,576*sizeof(int))!=cudaSuccess){cleanup();return 0;}
 while(begin<all.size()){
  size_t cells=0,end=begin;
  while(end<all.size()){
   size_t z=(size_t)(all[end].n+1)*(all[end].m+1);if(z>maxCells){cleanup();return 0;}
   if(end>begin&&cells+z>maxCells)break;cells+=z;all[end].off=cells-z;end++;
  }
  size_t nt=end-begin;hs.resize(nt);
  if(cudaMalloc(&dM,cells*sizeof(float))!=cudaSuccess||cudaMalloc(&dX,cells*sizeof(float))!=cudaSuccess||cudaMalloc(&dY,cells*sizeof(float))!=cudaSuccess||cudaMalloc(&dout,nt*sizeof(float))!=cudaSuccess||cudaMalloc(&dt,nt*sizeof(Task))!=cudaSuccess){cleanup();return 0;}
  if(cudaMemcpy(dt,all.data()+begin,nt*sizeof(Task),cudaMemcpyHostToDevice)!=cudaSuccess){cleanup();return 0;}
  pairKernel<<<nt,128>>>(dt,dso,dl,dout,dM,dX,dY,14.f,.6f);
  if(cudaGetLastError()!=cudaSuccess||cudaMemcpy(hs.data(),dout,nt*sizeof(float),cudaMemcpyDeviceToHost)!=cudaSuccess){cleanup();return 0;}
  for(size_t q=0;q<nt;q++){auto&t=all[begin+q];out[t.a*ns+t.b]=out[t.b*ns+t.a]=hs[q];}
  cudaFree(dM);dM=nullptr;cudaFree(dX);dX=nullptr;cudaFree(dY);dY=nullptr;cudaFree(dout);dout=nullptr;cudaFree(dt);dt=nullptr;begin=end;
 }
 cleanup();return 1;
}
