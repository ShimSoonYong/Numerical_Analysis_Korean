#include <stdio.h>
#include <math.h>


#define func(x) x * log(sin(x))

double fx[1001];

main ()
{
	
	FILE *out1;
	out1=fopen("Hornbeck_8_26_b.dat","w");

    int i,j,n;
    double x,x0,xn,delx,area_pre_even,area_pre_odd,area_integ;
    
    n=4;           /* number of span  */
    
    x0=0+1e-10;
    xn=3.141592 / 2;
    delx=(xn-x0)/float(n);
    
	for(i=0;i<=n;i++) 
	{
		x=x0+delx*float(i);
		fx[i]=func(x);
		
	}	
	
	
	area_pre_odd=0.0;
	area_pre_even=0.0;	

    for(i=1;i<=n/2;i++)
	{
		j=i*2-1;
		area_pre_odd=area_pre_odd+4.0*fx[j];
	}
	
	for(i=1;i<=n/2-1;i++)
	{
		j=i*2;
		area_pre_even=area_pre_even+2.0*fx[j];
	}
	
	area_integ=(delx/3.0)*(fx[0]+fx[n]+area_pre_odd+area_pre_even);
	
	
	for(i=0;i<=n;i++)
	{
		fprintf(out1,"fx[%f] = %f \n", x0+i*delx, fx[i]);
	}
	   
        fprintf(out1,"\n");
		fprintf(out1,"integration using simpson method = %f \n", area_integ);			

	
	fclose(out1);
}
