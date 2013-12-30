#include "mex.h"
#include <math.h>
inline int getIndex(int row,int col,int rowSize)
{
    return col*rowSize+row;
}

void meanPool(double pic[],int picRow, int picCol, int poolDim, int picNum,double picOut[])
{    
        int indext;
        int indexo;
        int colSize=picCol/poolDim;
        int rowSize=picRow/poolDim;
        int picOutLen=colSize*rowSize;
        int picLen=picCol*picRow;
        int indexoBase;
        int indextBase;
        double temp;
        for(int pici=0; pici<picNum; pici++)
        {
            indextBase=pici*picLen;
            indexoBase=pici*picOutLen;            
            for(int col=0; col<colSize; col++)
            {
                for(int row=0; row<rowSize; row++)
                {                    
                    temp=0;
                    indexo=getIndex(row,col,rowSize);
                    indexo+=indexoBase;
                    for(int i=0; i<poolDim; i++)
                    {
                        for(int j=0; j<poolDim; j++)
                        {
                            indext=getIndex(poolDim*row+j,poolDim*col+i,picRow);
                            indext+=indextBase;
                            temp+=pic[indext];
                        }
                    }
                    temp=temp/(double)(poolDim*poolDim);
                    picOut[indexo]=temp;
                }
            }
        }
}

void mexFunction( int nlhs, mxArray *plhs[],  int nrhs, const mxArray *prhs[] )
{
    /*meanPool(pics,picRow,picCol,poolDim,picNum);
     *
     *
     */
    double *pics,*dpicRow,*dpicCol,*dpoolDim,*dpicNum,*picOut;
    int picRow,picCol,poolDim,picNum,outDim,mrows,ncols;
    /* Check for proper number of arguments. */
    if(nrhs!=4)
    {
        mexErrMsgTxt("Four inputs required.");
    }
    else if(nlhs>1) {
        mexErrMsgTxt("Too many output arguments");
    }
    
    mrows = mxGetM(prhs[0]);
    ncols = mxGetN(prhs[0]);

    pics = mxGetPr(prhs[0]);
    dpicRow=mxGetPr(prhs[1]);
    dpicCol=mxGetPr(prhs[2]);
    dpoolDim=mxGetPr(prhs[3]);
    
    
    picRow=(int)*dpicRow;
    picCol=(int)*dpicCol;
    poolDim=(int)*dpoolDim;
    
    
    outDim=(picRow/poolDim)*(picCol/poolDim);
//     printf("%d\n",ncols);
//     mexErrMsgTxt("Too many output arguments");
    plhs[0] = mxCreateDoubleMatrix(outDim,ncols, mxREAL);
    picOut = mxGetPr(plhs[0]);
    meanPool(pics,picRow ,picCol,poolDim,ncols,picOut);
}