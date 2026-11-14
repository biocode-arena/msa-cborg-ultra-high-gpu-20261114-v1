#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;
static constexpr float NEG=-1e20f;
static float GAPOPEN=12,GAPEXT=.6;static int THREADS=1;
#ifdef PRISM_CUDA
extern "C" int gpu_pair_scores(const int*,const int*,int,const int*,float*);
#endif
static const string AA="ARNDCQEGHILKMFPSTWYVBZX*";
static const int B62[24][24]={
{4,-1,-2,-2,0,-1,-1,0,-2,-1,-1,-1,-1,-2,-1,1,0,-3,-2,0,-2,-1,0,0},
{-1,5,0,-2,-3,1,0,-2,0,-3,-2,2,-1,-3,-2,-1,-1,-3,-2,-3,-1,0,-1,0},
{-2,0,6,1,-3,0,0,0,1,-3,-3,0,-2,-3,-2,1,0,-4,-2,-3,3,0,-1,0},
{-2,-2,1,6,-3,0,2,-1,-1,-3,-4,-1,-3,-3,-1,0,-1,-4,-3,-3,4,1,-1,0},
{0,-3,-3,-3,9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1,-3,-3,-2,0},
{-1,1,0,0,-3,5,2,-2,0,-3,-2,1,0,-3,-1,0,-1,-2,-1,-2,0,3,-1,0},
{-1,0,0,2,-4,2,5,-2,0,-3,-3,1,-2,-3,-1,0,-1,-3,-2,-2,1,4,-1,0},
{0,-2,0,-1,-3,-2,-2,6,-2,-4,-4,-2,-3,-3,-2,0,-2,-2,-3,-3,-1,-2,-1,0},
{-2,0,1,-1,-3,0,0,-2,8,-3,-3,-1,-2,-1,-2,-1,-2,-2,2,-3,0,0,-1,0},
{-1,-3,-3,-3,-1,-3,-3,-4,-3,4,2,-3,1,0,-3,-2,-1,-3,-1,3,-3,-3,-1,0},
{-1,-2,-3,-4,-1,-2,-3,-4,-3,2,4,-2,2,0,-3,-2,-1,-2,-1,1,-4,-3,-1,0},
{-1,2,0,-1,-3,1,1,-2,-1,-3,-2,5,-1,-3,-1,0,-1,-3,-2,-2,0,1,-1,0},
{-1,-1,-2,-3,-1,0,-2,-3,-2,1,2,-1,5,0,-2,-1,-1,-1,-1,1,-3,-1,-1,0},
{-2,-3,-3,-3,-2,-3,-3,-3,-1,0,0,-3,0,6,-4,-2,-2,1,3,-1,-3,-3,-1,0},
{-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4,7,-1,-1,-4,-3,-2,-2,-1,-2,0},
{1,-1,1,0,-1,0,0,0,-1,-2,-2,0,-1,-2,-1,4,1,-3,-2,-2,0,0,0,0},
{0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,1,5,-2,-2,0,-1,-1,0,0},
{-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1,1,-4,-3,-2,11,2,-3,-4,-3,-2,0},
{-2,-2,-2,-3,-2,-1,-2,-3,2,-1,-1,-2,-1,3,-3,-2,-2,2,7,-1,-3,-2,-1,0},
{0,-3,-3,-3,-1,-2,-2,-3,-3,3,1,-2,1,-1,-2,-2,0,-3,-1,4,-3,-2,-1,0},
{-2,-1,3,4,-3,0,1,-1,0,-3,-4,0,-3,-3,-2,0,-1,-4,-3,-3,4,1,-1,0},
{-1,0,0,1,-3,3,4,-2,0,-3,-3,1,-1,-3,-1,0,-1,-3,-2,-2,1,4,-1,0},
{0,-1,-1,-1,-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-2,0,0,-2,-1,-1,-1,-1,-1,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

struct Seq { string name, s; };
struct Match{int other,pos;float w;};
struct AnchorSet{
 int n=0;vector<int>off;vector<vector<Match>>a;
};
static AnchorSet*ANCHORS=nullptr;static float ANCHOR_WEIGHT=1.0;
struct Profile {
 vector<string> rows;
 vector<int>ids,pos;
 int L=0;
 vector<uint16_t> cnt;
 void counts(){
  L=rows.empty()?0:(int)rows[0].size();
  cnt.assign((size_t)L*24,0);
  if(ids.empty()){ids.resize(rows.size());iota(ids.begin(),ids.end(),0);}
  if(pos.empty()){pos.assign(rows.size()*L,-1);for(int k=0;k<(int)rows.size();k++){int p=0;for(int j=0;j<L;j++)if(rows[k][j]!='-')pos[(size_t)k*L+j]=p++;}}
  for(auto &r:rows) for(int j=0;j<L;j++){
   int a=r[j]=='-'?23:(int)AA.find(toupper(r[j]));
   if(a<0||a>22)a=21;
   cnt[(size_t)j*24+a]++;
  }
 }
};

static vector<Seq> readFasta(const string &p){
 ifstream f(p); if(!f) throw runtime_error("cannot open input");
 vector<Seq> v; string line;
 while(getline(f,line)){
  if(!line.empty()&&line.back()=='\r')line.pop_back();
  if(line.empty())continue;
  if(line[0]=='>')v.push_back({line.substr(1),""});
  else {if(v.empty())throw runtime_error("bad FASTA"); for(char c:line)if(!isspace(c))v.back().s+=c;}
 }
 if(v.empty())throw runtime_error("empty FASTA"); return v;
}
static void writeFasta(const string&p,const vector<Seq>&seq,const Profile&pr,const vector<int>&ord){
 ofstream f(p); if(!f)throw runtime_error("cannot open output");
 for(int q=0;q<(int)ord.size();q++){int k=ord[q];f<<'>'<<seq[k].name<<'\n'; auto &r=pr.rows[q]; for(int i=0;i<(int)r.size();i+=80)f<<r.substr(i,80)<<'\n';}
}
static float colScore(const Profile&A,const Profile&B,int i,int j){
 float x=0,na=A.rows.size(),nb=B.rows.size();
 auto *a=&A.cnt[(size_t)i*24];auto *b=&B.cnt[(size_t)j*24];
 for(int p=0;p<23;p++)if(a[p])for(int q=0;q<23;q++)if(b[q])x+=a[p]*b[q]*B62[p][q];
 return x/(na*nb);
}
static float gapFactor(const Profile&A,const Profile&B,int i,int j){
 float ga=A.cnt[(size_t)i*24+23]/(float)A.rows.size();
 float gb=B.cnt[(size_t)j*24+23]/(float)B.rows.size();
 return max(0.10f,(1-ga)*(1-gb));
}
static vector<float> profileBoost(const Profile&A,const Profile&B){
 int n=A.L,m=B.L;vector<float>v((size_t)n*m,0);
 if(!ANCHORS||n==0||m==0)return v;
 vector<vector<int>>bc(B.ids.size());
 for(int q=0;q<(int)B.ids.size();q++){
  int id=B.ids[q],len=ANCHORS->off[id+1]-ANCHORS->off[id];
  bc[q].assign(len,-1);for(int j=0;j<m;j++){int p=B.pos[(size_t)q*m+j];if(p>=0)bc[q][p]=j;}
 }
 for(int p=0;p<(int)A.ids.size();p++)for(int q=0;q<(int)B.ids.size();q++){
  int ia=A.ids[p],ib=B.ids[q];
  for(int i=0;i<n;i++){
   int pa=A.pos[(size_t)p*n+i];if(pa<0)continue;
   for(auto &z:ANCHORS->a[ANCHORS->off[ia]+pa])if(z.other==ib&&z.pos<(int)bc[q].size()){
    int j=bc[q][z.pos];if(j>=0&&z.w>.01)v[(size_t)i*m+j]+=z.w;
   }
  }
 }
 return v;
}
static Profile alignProfiles(const Profile&A,const Profile&B,float go=GAPOPEN,float ge=GAPEXT,float* finalScore=nullptr){
 int n=A.L,m=B.L; size_t N=(size_t)(n+1)*(m+1);
 auto boost=profileBoost(A,B);
 vector<float> pm(N,NEG),px(N,NEG),py(N,NEG);
 vector<uint8_t> tm(N),tx(N),ty(N);
 auto id=[m](int i,int j){return (size_t)i*(m+1)+j;};
 pm[0]=0;
 for(int i=1;i<=n;i++){px[id(i,0)]=0;tx[id(i,0)]=i==1?0:1;}
 for(int j=1;j<=m;j++){py[id(0,j)]=0;ty[id(0,j)]=j==1?0:2;}
 for(int i=1;i<=n;i++)for(int j=1;j<=m;j++){
  size_t z=id(i,j),d=id(i-1,j-1),u=id(i-1,j),l=id(i,j-1);
  float a=pm[d],b=px[d],c=py[d];uint8_t t=0;
  if(b>a){a=b;t=1;}if(c>a){a=c;t=2;}pm[z]=a+colScore(A,B,i-1,j-1)+ANCHOR_WEIGHT*boost[(size_t)(i-1)*m+j-1];tm[z]=t;
  float gf=gapFactor(A,B,i-1,j-1); a=pm[u]-go*gf;b=px[u]-ge*gf;c=py[u]-go*gf;t=0;
  if(b>a){a=b;t=1;}if(c>a){a=c;t=2;}px[z]=a;tx[z]=t;
  a=pm[l]-go*gf;b=py[l]-ge*gf;c=px[l]-go*gf;t=0;
  if(b>c){if(b>a){a=b;t=2;}}else if(c>a){a=c;t=1;}py[z]=a;ty[z]=t;
 }
 int bi=n,bj=0,bs=1;float best=px[id(n,0)];
 for(int s=0;s<3;s++){float v=s==0?pm[id(n,0)]:(s==1?px[id(n,0)]:py[id(n,0)]);if(v>best){best=v;bs=s;}}
 for(int j=1;j<=m;j++){
  float vv[3]={pm[id(n,j)],px[id(n,j)],py[id(n,j)]};
  for(int s=0;s<3;s++)if(vv[s]>best){best=vv[s];bi=n;bj=j;bs=s;}
 }
 for(int i=0;i<n;i++){
  float vv[3]={pm[id(i,m)],px[id(i,m)],py[id(i,m)]};
  for(int s=0;s<3;s++)if(vv[s]>best){best=vv[s];bi=i;bj=m;bs=s;}
 }
 if(finalScore)*finalScore=best;
 vector<pair<int,int>> ops,suffix;
 if(bi==n)for(int q=bj;q<m;q++)suffix.push_back({-1,q});
 else for(int q=bi;q<n;q++)suffix.push_back({q,-1});
 int i=bi,j=bj,s=bs;
 while(i||j){
  size_t z=id(i,j);
  if(s==0){uint8_t t=tm[z];ops.push_back({i-1,j-1});i--;j--;s=t;}
  else if(s==1){uint8_t t=tx[z];ops.push_back({i-1,-1});i--;s=t;}
  else {uint8_t t=ty[z];ops.push_back({-1,j-1});j--;s=t;}
 }
 reverse(ops.begin(),ops.end());
 ops.insert(ops.end(),suffix.begin(),suffix.end());
 Profile R;R.rows.resize(A.rows.size()+B.rows.size());R.L=ops.size();
 for(int k=0;k<(int)A.rows.size();k++)R.rows[k].reserve(R.L);
 for(int k=0;k<(int)B.rows.size();k++)R.rows[A.rows.size()+k].reserve(R.L);
 vector<int> ca(A.rows.size()),cb(B.rows.size());
 int ai=0,bi2=0;
 for(auto [x,y]:ops){
  if(x>=0){for(int k=0;k<(int)A.rows.size();k++)R.rows[k]+=A.rows[k][ai];ai++;}
  else for(int k=0;k<(int)A.rows.size();k++)R.rows[k]+='-';
  if(y>=0){for(int k=0;k<(int)B.rows.size();k++)R.rows[A.rows.size()+k]+=B.rows[k][bi2];bi2++;}
  else for(int k=0;k<(int)B.rows.size();k++)R.rows[A.rows.size()+k]+='-';
 }
 R.ids=A.ids;R.ids.insert(R.ids.end(),B.ids.begin(),B.ids.end());R.pos.assign((size_t)R.rows.size()*R.L,-1);
 for(int k=0;k<(int)R.rows.size();k++){int p=0;for(int j=0;j<R.L;j++)if(R.rows[k][j]!='-')R.pos[(size_t)k*R.L+j]=p++;}
 R.counts();return R;
}

static inline float la3(float a,float b,float c){float m=max(a,max(b,c));if(m<NEG/2)return NEG;return m+log(exp(a-m)+exp(b-m)+exp(c-m));}
static vector<float> pairPosterior(const string&A,const string&B,float temp,float &zout){
 int n=A.size(),m=B.size(),W=m+1;size_t N=(size_t)(n+1)*W;vector<float>fm(N,NEG),fx(N,NEG),fy(N,NEG),bm(N,NEG),bx(N,NEG),by(N,NEG),post((size_t)n*m,0);fm[0]=0;
 for(int i=1;i<=n;i++)fx[(size_t)i*W]=0;for(int j=1;j<=m;j++)fy[j]=0;
 auto code=[](char c){int a=AA.find(toupper(c));return a>=0&&a<23?a:21;};
 for(int i=1;i<=n;i++)for(int j=1;j<=m;j++){
  size_t z=(size_t)i*W+j,d=z-W-1,u=z-W,l=z-1;
  fm[z]=la3(fm[d],fx[d],fy[d])+B62[code(A[i-1])][code(B[j-1])]/temp;
  fx[z]=la3(fm[u]-GAPOPEN/temp,fx[u]-GAPEXT/temp,fy[u]-GAPOPEN/temp);
  fy[z]=la3(fm[l]-GAPOPEN/temp,fy[l]-GAPEXT/temp,fx[l]-GAPOPEN/temp);
 }
 float z=NEG;auto add=[&](float v){z=z<NEG/2?v:max(z,v)+log(exp(min(z,v)-max(z,v))+1);};
 for(int j=0;j<=m;j++){size_t q=(size_t)n*W+j;add(fm[q]);add(fx[q]);add(fy[q]);}for(int i=0;i<n;i++){size_t q=(size_t)i*W+m;add(fm[q]);add(fx[q]);add(fy[q]);}zout=z;
 for(int i=n;i>=0;i--)for(int j=m;j>=0;j--){
  size_t q=(size_t)i*W+j;if(i==n||j==m){bm[q]=bx[q]=by[q]=0;continue;}
  size_t d=q+W+1,u=q+W,l=q+1;float em=B62[code(A[i])][code(B[j])]/temp;
  bm[q]=la3(bm[d]+em,bx[u]-GAPOPEN/temp,by[l]-GAPOPEN/temp);
  bx[q]=la3(bm[d]+em-GAPOPEN/temp,bx[u]-GAPEXT/temp,by[l]-GAPOPEN/temp);
  by[q]=la3(bm[d]+em-GAPOPEN/temp,bx[u]-GAPOPEN/temp,by[l]-GAPEXT/temp);
 }
 for(int i=1;i<=n;i++)for(int j=1;j<=m;j++){float p=fm[(size_t)i*W+j]+bm[(size_t)i*W+j]-z;if(p>log(.003))post[(size_t)(i-1)*m+j-1]=exp(p);}
 return post;
}
static AnchorSet makeAnchors(const vector<Seq>&seq){
 AnchorSet as;as.n=seq.size();as.off.resize(seq.size()+1);
 for(int i=0;i<(int)seq.size();i++)as.off[i+1]=as.off[i]+seq[i].s.size();
 as.a.resize(as.off.back());
 const float pars[3][2]={{8,.6},{12,.2},{14,.2}};
#pragma omp parallel for num_threads(THREADS) schedule(dynamic)
 for(int i=0;i<(int)seq.size();i++)for(int j=0;j<(int)seq.size();j++)if(i!=j){
  Profile a,b;a.rows={seq[i].s};b.rows={seq[j].s};a.ids={i};b.ids={j};a.counts();b.counts();
  for(auto &pa:pars){auto ab=alignProfiles(a,b,pa[0],pa[1]);int x=0,y=0;
   for(int c=0;c<ab.L;c++){bool ax=ab.rows[0][c]!='-',by=ab.rows[1][c]!='-';if(ax&&by){as.a[as.off[i]+x].push_back({j,y,2.f/3});}x+=ax;y+=by;}
  }
 }
 for(auto &v:as.a){sort(v.begin(),v.end(),[](auto&a,auto&b){return a.other*10000000+a.pos<b.other*10000000+b.pos;});vector<Match>w;for(auto z:v){if(!w.empty()&&w.back().other==z.other&&w.back().pos==z.pos)w.back().w+=z.w;else w.push_back(z);}v.swap(w);}
 float pm=getenv("PRISM_POST")?stof(getenv("PRISM_POST")):0;
 if(pm>0&&seq.size()<=64){
  for(auto&v:as.a)for(auto&z:v)z.w/=1+pm;
  for(int i=0;i<(int)seq.size();i++)for(int j=0;j<i;j++){
   float z;auto po=pairPosterior(seq[i].s,seq[j].s,1.5,z);int ni=seq[i].s.size(),nj=seq[j].s.size();
   for(int x=0;x<ni;x++)for(int y=0;y<nj;y++){float w=po[(size_t)x*nj+y];if(w>.003){w*=pm/(1+pm);as.a[as.off[i]+x].push_back({j,y,w});as.a[as.off[j]+y].push_back({i,x,w});}}
  }
  for(auto &v:as.a){sort(v.begin(),v.end(),[](auto&a,auto&b){return a.other*10000000+a.pos<b.other*10000000+b.pos;});vector<Match>w;for(auto z:v){if(!w.empty()&&w.back().other==z.other&&w.back().pos==z.pos){w.back().w+=z.w;w.back().w=min(1.f,w.back().w);}else w.push_back(z);}v.swap(w);}
 }
 auto direct=as.a;float td=getenv("PRISM_TD")?stof(getenv("PRISM_TD")):.15,tc=getenv("PRISM_TC")?stof(getenv("PRISM_TC")):1.2;
 for(int ia=0;ia<(int)direct.size();ia++){
  unordered_map<uint64_t,float>tmp;for(auto &z:direct[ia])tmp[(uint64_t)z.other*10000000+z.pos]+=td*z.w;
  auto bridges=direct[ia];if(as.n>64){sort(bridges.begin(),bridges.end(),[](auto&a,auto&b){return a.w>b.w;});if(bridges.size()>16)bridges.resize(16);}
  for(auto &z:bridges){
   int kp=as.off[z.other]+z.pos;
   for(auto &q:direct[kp])if(q.other!=z.other)tmp[(uint64_t)q.other*10000000+q.pos]+=tc*z.w*q.w/max(1,as.n);
  }
  vector<Match>v;int ii=upper_bound(as.off.begin(),as.off.end(),ia)-as.off.begin()-1;
  for(auto &[key,w]:tmp)if(key/10000000!=(uint64_t)ii&&w>.06){int j=key/10000000,p=key%10000000;v.push_back({j,p,w});}
  sort(v.begin(),v.end(),[](auto&a,auto&b){return a.w>b.w;});if(v.size()>500)v.resize(500);as.a[ia]=v;
 }
 return as;
}
struct Tree{int left=-1,right=-1;float dist=0;vector<int> leaves;};
static const Profile*TREE_MSA=nullptr;
static vector<Tree> guideTree(const vector<Seq>&seq){
 int n=seq.size();vector<array<float,24>> freq(n);
 const int K=3,D=24*24*24;
 vector<vector<float>> f(n,vector<float>(D));
 for(int k=0;k<n;k++){
  vector<int>a(seq[k].s.size());for(int i=0;i<(int)a.size();i++){int p=AA.find(toupper(seq[k].s[i]));a[i]=(p>=0&&p<20)?p:20;}
  for(int i=0;i+K<=(int)a.size();i++){int z=0;for(int q=0;q<K;q++)z=z*24+min(a[i+q],23);f[k][z]+=1;}
  float ss=inner_product(f[k].begin(),f[k].end(),f[k].begin(),0.f);if(ss)for(float&x:f[k])x/=sqrt(ss);
 }
 vector<Tree>t(2*n-1);vector<int>act(n);iota(act.begin(),act.end(),0);
 vector<vector<float>>d(2*n-1,vector<float>(2*n-1));
 for(int i=0;i<n;i++){t[i].leaves={i};for(int j=0;j<n;j++){float dot=inner_product(f[i].begin(),f[i].end(),f[j].begin(),0.f);d[i][j]=1-dot;}}
 {
  vector<float>self(n),ps(n*n);bool got=false;
  for(int i=0;i<n;i++){for(char c:seq[i].s){int a=AA.find(toupper(c));if(a<0||a>=23)a=21;self[i]+=B62[a][a];}self[i]=max(1.f,self[i]);}
#ifdef PRISM_CUDA
  if(!getenv("PRISM_CPU")){vector<int>letters,so(n+1);for(int i=0;i<n;i++){for(char c:seq[i].s){int a=AA.find(toupper(c));letters.push_back(a>=0&&a<24?a:21);}so[i+1]=letters.size();}got=gpu_pair_scores(letters.data(),so.data(),n,&B62[0][0],ps.data())==1;}
#endif
  if(n<=64||got)for(int i=0;i<n;i++)for(int j=0;j<i;j++){
   float score=ps[i*n+j];
   if(!got||getenv("PRISM_COMPARE")){float oldscore=score;Profile a,b;a.rows={seq[i].s};b.rows={seq[j].s};a.counts();b.counts();auto ab=alignProfiles(a,b,14,.6,&score);if(got&&fabs(oldscore-score)>1e-3)cerr<<i<<","<<j<<" gpu="<<oldscore<<" cpu="<<score<<"\n";}
   int dm=getenv("PRISM_DIST")?stoi(getenv("PRISM_DIST")):0;float den=dm==1?sqrt(self[i]*self[j]):(dm==2?(self[i]+self[j])/2:(dm==3?max(self[i],self[j]):min(self[i],self[j])));float dist=1-score/den;d[i][j]=d[j][i]=dist;
  }
 }
 if(TREE_MSA){
  vector<int>row(n);for(int q=0;q<n;q++)row[TREE_MSA->ids[q]]=q;vector<float>self(n,1);
  for(int i=0;i<n;i++)for(char c:seq[i].s){int a=AA.find(toupper(c));if(a<0||a>=23)a=21;self[i]+=B62[a][a];}
  for(int i=0;i<n;i++)for(int j=0;j<i;j++){
   auto&a=TREE_MSA->rows[row[i]];auto&b=TREE_MSA->rows[row[j]];float sc=0;bool gap=false;
   int first=min(a.find_first_not_of('-'),b.find_first_not_of('-')),last=max(a.find_last_not_of('-'),b.find_last_not_of('-'));
   for(int c=first;c<=last;c++){char x=a[c],y=b[c];if(x!='-'&&y!='-'){int aa=AA.find(toupper(x)),bb=AA.find(toupper(y));sc+=B62[aa<23?aa:21][bb<23?bb:21];gap=false;}else if(x!='-'||y!='-'){if(!gap)sc-=GAPOPEN;sc-=GAPEXT;gap=true;}else gap=false;}
   d[i][j]=d[j][i]=1-sc/min(self[i],self[j]);
  }
 }
 auto join=[&](int z,int a,int b,float dd){t[z].left=a;t[z].right=b;t[z].dist=dd;t[z].leaves=t[a].leaves;t[z].leaves.insert(t[z].leaves.end(),t[b].leaves.begin(),t[b].leaves.end());};
 bool nj=getenv("PRISM_NJ");
 for(int z=n;z<2*n-1;z++){
  int aa=-1,bb=-1;float bd=1e30;vector<float>r(2*n-1);
  if(nj)for(int x:act)for(int y:act)r[x]+=d[x][y];
  for(int x:act)for(int y:act)if(x<y){float v=nj?(act.size()-2)*d[x][y]-r[x]-r[y]:d[x][y];if(v<bd){bd=v;aa=x;bb=y;}}
  join(z,aa,bb,bd/2);vector<int>na;for(int x:act)if(x!=aa&&x!=bb)na.push_back(x);
  for(int x:na){float v=nj?(d[x][aa]+d[x][bb]-d[aa][bb])/2:((float)t[aa].leaves.size()*d[x][aa]+(float)t[bb].leaves.size()*d[x][bb])/(t[aa].leaves.size()+t[bb].leaves.size());d[z][x]=d[x][z]=v;}
  na.push_back(z);act=na;
 }
 return t;
}
static Profile progressive(const vector<Seq>&seq,const vector<Tree>&t,vector<int>&ord){
 int n=seq.size();vector<Profile>p(2*n-1);
 function<void(int)>rec=[&](int z){
  if(z<n){p[z].rows={seq[z].s};p[z].ids={z};p[z].counts();return;}
  rec(t[z].left);rec(t[z].right);p[z]=alignProfiles(p[t[z].left],p[t[z].right]);
 };rec(2*n-2);ord=t[2*n-2].leaves;return move(p[2*n-2]);
}
static double msaObjective(const Profile&p){
 double score=0;int n=p.rows.size(),L=p.L;
 for(int i=0;i<n;i++)for(int j=0;j<i;j++){
  bool gap=false;int started=0;int last=max((int)p.rows[i].find_last_not_of('-'),(int)p.rows[j].find_last_not_of('-'));
  int first=min((int)p.rows[i].find_first_not_of('-'),(int)p.rows[j].find_first_not_of('-'));
  for(int c=first;c<=last;c++){
   char x=p.rows[i][c],y=p.rows[j][c];
   if(x!='-'&&y!='-'){
    int a=AA.find(toupper(x)),b=AA.find(toupper(y));if(a<0||a>=23)a=21;if(b<0||b>=23)b=21;score+=B62[a][b];
    if(ANCHORS){int pi=p.pos[(size_t)i*L+c],pj=p.pos[(size_t)j*L+c];for(auto &m:ANCHORS->a[ANCHORS->off[p.ids[i]]+pi])if(m.other==p.ids[j]&&m.pos==pj)score+=ANCHOR_WEIGHT*m.w;}
    gap=false;
   }else if(x!='-'||y!='-'){if(!gap)score-=GAPOPEN;score-=GAPEXT;gap=true;}else gap=false;
  }
 }
 return score;
}
static Profile extractProfile(const Profile&p,const vector<char>&take,int which){
 Profile q;for(int i=0;i<(int)p.rows.size();i++)if(take[i]==which){q.rows.push_back(p.rows[i]);q.ids.push_back(p.ids[i]);q.pos.insert(q.pos.end(),p.pos.begin()+(size_t)i*p.L,p.pos.begin()+(size_t)(i+1)*p.L);}q.L=p.L;q.counts();return q;
}
static Profile refine(Profile cur,int rounds,int seed){
 mt19937 rng(seed);double best=msaObjective(cur);int n=cur.rows.size();
 int sweeps=getenv("PRISM_ROWS")?stoi(getenv("PRISM_ROWS")):1;
 for(int sw=0;sw<sweeps;sw++)for(int k=0;k<n;k++){
  vector<char>take(n,0);take[k]=1;auto a=extractProfile(cur,take,1),b=extractProfile(cur,take,0);auto cand=alignProfiles(a,b);double sc=msaObjective(cand);if(sc>best+1e-4){best=sc;cur=move(cand);}
 }
 for(int it=0;it<rounds;it++){
  vector<char>take(n);int na=0;for(int i=0;i<n;i++){take[i]=rng()&1;na+=take[i];}if(na==0||na==n){it--;continue;}
  auto a=extractProfile(cur,take,1),b=extractProfile(cur,take,0);auto cand=alignProfiles(a,b);double sc=msaObjective(cand);
  if(sc>best+1e-4){best=sc;cur=move(cand);}
 }
 return cur;
}
int main(int argc,char**argv){
 try{
  if(argc<4){cerr<<"usage: run.sh input output threads\n";return 2;}
  THREADS=max(1,stoi(argv[3]));if(getenv("PRISM_GO"))GAPOPEN=stof(getenv("PRISM_GO"));if(getenv("PRISM_GE"))GAPEXT=stof(getenv("PRISM_GE"));
  auto seq=readFasta(argv[1]);vector<int>ord(seq.size());iota(ord.begin(),ord.end(),0);
  if(seq.size()==1){Profile p;p.rows={seq[0].s};writeFasta(argv[2],seq,p,ord);return 0;}
  size_t maxLen=0;for(auto&s:seq)maxLen=max(maxLen,s.s.size());
  if(maxLen>20000){Profile p;for(auto&s:seq)p.rows.push_back(s.s+string(maxLen-s.s.size(),'-'));writeFasta(argv[2],seq,p,ord);return 0;}
  auto tree=guideTree(seq);auto anchors=makeAnchors(seq);ANCHORS=&anchors;if(getenv("PRISM_AW"))ANCHOR_WEIGHT=stof(getenv("PRISM_AW"));auto pr=progressive(seq,tree,ord);if(getenv("PRISM_REBUILD")){TREE_MSA=&pr;auto tree2=guideTree(seq);TREE_MSA=nullptr;vector<int>ord2;auto pr2=progressive(seq,tree2,ord2);if(msaObjective(pr2)>msaObjective(pr)){pr=move(pr2);ord=move(ord2);}}int rounds=getenv("PRISM_REFINE")?stoi(getenv("PRISM_REFINE")):4;pr=refine(move(pr),rounds,seq.size()*131+pr.L);ord=pr.ids;ANCHORS=nullptr;writeFasta(argv[2],seq,pr,ord);
 }catch(const exception&e){
  cerr<<"PrismAlignGPU fallback: "<<e.what()<<'\n';
  try{auto seq=readFasta(argv[1]);size_t m=0;for(auto&s:seq)m=max(m,s.s.size());ofstream f(argv[2]);for(auto&s:seq)f<<'>'<<s.name<<'\n'<<s.s<<string(m-s.s.size(),'-')<<'\n';return f?0:1;}catch(...){return 1;}
 }
}
