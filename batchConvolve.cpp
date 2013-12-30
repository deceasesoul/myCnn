#include "mex.h"
#include <math.h>
inline int getIndex(int row,int col,int rowSize)
{
    return col*rowSize+row;
}

void conv2(
        double *c,	/* Result matrix (ma+mb-1)-by-(na+nb-1) */
        double *a,	/* Larger matrix */
        double *b,	/* Smaller matrix */
        int ma,		/* Row size of a */
        int na,		/* Column size of a */
        int mb,		/* Row size of b */
        int nb,		/* Column size of b */             
        double bias
        )
{   
    double w;		/* Weight (element of 'b' matrix) */
    int mc,nc;
    int k,l,i,j; 
    int indexc;
    int indexa;
    int indexb;
    
    mc = ma-mb+1;
    nc = na-nb+1;
    
    /* Perform convolution */
    for(i=0; i<nc; i++)
    {
        for(j=0; j<mc; j++)
        {
            indexc=getIndex(j,i,mc);
            w=0.0;
            for(k=0; k<nb; k++)
            {
                for(l=0; l<mb; l++)
                {
                    indexb=getIndex(l,k,mb);
                    indexa=getIndex(j+l,k+i,ma);
                    w+=a[indexa]*b[indexb];
                }
            }
            c[indexc]=w+bias;
        }
    }
    
}
void convolve(double picMatrix[],int picRowSize,int picColSize ,int numPics,double filterMatrix[],int filterRowSize,int filterColSize,int numFilters,double bias[],double picOut[],int oRowSize,int oColSize)
{
    int picLen=picRowSize*picColSize;
    int filterLen=filterRowSize*filterColSize;
    int oLen=oRowSize*oColSize;
    int i,j;
    double *p=picMatrix;
    double *f=filterMatrix;
    double *o=picOut;
    for(i=0; i<numPics; i++)
    {        
        f=filterMatrix;
        for(j=0; j<numFilters; j++)
        {
            conv2(o,p,f,picRowSize,picColSize,filterRowSize,filterColSize,bias[j]);            
            o+=oLen;
            f+=filterLen;
        }        
        p+=picLen;
    }            
}

void mexFunction( int nlhs, mxArray *plhs[],  int nrhs, const mxArray *prhs[] )
{
    /*meanPool(pics,picRow,picCol,poolDim,picNum);
     *
     *con = batchConvolve(pics,filters,picRowSize, picColSize, filterRowSize, filterColSize,bias);
     */
    double *pics,*filters,*dpicCol,*dpicRow,*dfilterCol,*dfilterRow,*bias,*picOut;
    int picRow,picCol,filterCol,filterRow,picNum, filterNum,outRow,outCol;
    /* Check for proper number of arguments. */
    if(nrhs!=7)
    {
        mexErrMsgTxt("Seven inputs required.");
    }
    else if(nlhs>1) {
        mexErrMsgTxt("Too many output arguments");
    }
       
    picNum = mxGetN(prhs[0]);
    
    filterNum = mxGetN(prhs[1]);
    
    pics = mxGetPr(prhs[0]);
    filters=mxGetPr(prhs[1]);
    dpicRow=mxGetPr(prhs[2]);
    dpicCol=mxGetPr(prhs[3]);
    dfilterRow=mxGetPr(prhs[4]);
    dfilterCol=mxGetPr(prhs[5]);   
    bias=mxGetPr(prhs[6]);   
    
    picRow=(int)*dpicRow;
    picCol=(int)*dpicCol;
    filterRow=(int)*dfilterRow;
    filterCol=(int)*dfilterCol;
    
    outRow=picRow-filterRow+1;
    outCol=picCol-filterCol+1;        
//     printf("%d %d %d %d\n",outRow,outCol,picNum,filterNum);
//    printf("%d   %d\n",outRow*outCol,picNum*filterNum);
//    mexErrMsgTxt("Too many output arguments");
    plhs[0] = mxCreateDoubleMatrix(outRow*outCol,picNum*filterNum, mxREAL);
    picOut = mxGetPr(plhs[0]);
//   convolve(double picMatrix[],int picRowSize,int picColSize ,int numPics,double filterMatrix[],int filterRowSize,int filterColSize,int numFilters,double bias[],double picOut[],int oRowSize,int oColSize)
    
    convolve(pics,picRow,picCol,picNum,filters,filterRow,filterCol,filterNum,bias,picOut,outRow,outCol);
}