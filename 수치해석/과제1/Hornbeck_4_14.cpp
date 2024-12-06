#include <stdio.h>
#include <math.h>
 
 /* nd: number of data */
 /* n = nd-1           */
 
#define nd 5
double xd[nd]={0.0,1.0,2.0,3.8,5.0};
double fd[nd]={0, 0.569, 0.791, 0.224, -0.185};


main()
{
	FILE *out1;
	out1=fopen("Hornbeck_4_14_result","w");

    int n,i,j;
    double p,sum,f_x;
    double x,y;
   
   	n = nd-1;
 	x = 4.3;
 	sum=0.0;
 	f_x=0.0;
    
    for(i=0;i<=n;i++)
    {
        p=1;
        for(j=0;j<=n;j++)
        {
            if(i!=j)
            p=p*(x-xd[j])/(xd[i]-xd[j]);
        }
            sum=sum+p*fd[i];
    }
   
   f_x = sum;
   
   fprintf(out1,"  x = %f : f(x) = %f", x, f_x);
 
   fclose(out1);
}	                                           	
