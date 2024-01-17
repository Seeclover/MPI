#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int t,n,m,D1,D2;
int A[10004*10004];
int K[10004*10004];

int main(int argc, char *argv[])
{
    int rank,size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(rank==0)
    {
        char file_name[100000];
        scanf("%s" ,file_name);
        if(file_name==NULL)
        {
            MPI_Finalize();
            return 0;
        }
        FILE *file=fopen(file_name, "r");
        fscanf(file ,"%d %d %d",&t,&n,&m);
        for(int i=0;i<n;i++)
        {
            for(int i2=0;i2<m;i2++)
            {
                fscanf(file, "%d", &A[(i*m)+i2]);
            }
        }
        fscanf(file, "%d %d",&D1,&D2);
        for(int i=0;i<D1;i++){
            for(int i2=0;i2<D2;i2++)
            {
                fscanf(file, "%d",&K[(i*D2)+i2]);
            }
        }
    }
    
    MPI_Bcast(&t,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&m,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&D1,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&D2,1,MPI_INT,0,MPI_COMM_WORLD);
    
    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            MPI_Send(K, D1 * D2, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    }
    if(rank!=0)
    {
        MPI_Recv(K, D1 * D2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } 
    
    int x, y;
    // 給兩個空間
    
    int n1 = n;
    if(n%size != 0)
        n1+=(size-(n%size));
    //把不能平均分配的矩陣變成可以平均分配(會計算到不需要的陣列部分)
    
    n1/=size;
    int subA[n1*m];
    
    for(int i0=0;i0<t;i0++)
    {
    	if (rank == 0) {
            for (int i = 1; i < size; i++) {
                MPI_Send(A, n * m, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
        }
        if(rank!=0)
        {
            MPI_Recv(A, n * m, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        for(int i=0;i<n1;i++)
        {
            for(int i2=0;i2<m;i2++)
            {
            	subA[i*m+i2]=0;
            	
            	for(int i3=(-((D1-1)/2));i3<=((D1-1)/2);i3++)
            	{
            	    for(int i4=(-((D2-1)/2));i4<=((D2-1)/2);i4++)
            	    {
            	    	x = i + i3 + (rank*n1);
                        while (x < 0)
                        {
                            x += n;
                        }
                        x %= n;
                        // 59~64行:調整x座標

                        y = i2 + i4;
                        while (y < 0)
                        {
                            y += m;
                        }
                        y %= m;
                        // 67~72行:調整y座標
                        
                        subA[(i*m)+i2] += (K[(((D1 - 1) / 2) + i3)*D2+(((D2 - 1) / 2) + i4)] * A[(x*m)+y]);
            	    }
            	}
            	subA[(i*m)+i2]/=(D1*D2);
            }
        }
        
        MPI_Gather(subA, n1 * m, MPI_INT, A, n1 * m, MPI_INT, 0, MPI_COMM_WORLD);
        //合併t時間的A
    }
    
    if (rank == 0)
    {
        for (int i = 0; i < n; i++)
        {
            for (int i2 = 0; i2 < m; i2++)
            {
                printf("%d ", A[i*m+i2]);
            }
        }
        // 印出A
    }

    MPI_Finalize();
}
